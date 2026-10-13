"""RSL-RL ``VecEnv`` around ``RidgebackDREnv``: an RL policy drives the base, the IK/QP drives the arm.

Per policy step the wrapper holds the base velocity command for ``decimation`` physics steps. Every ``ik_every``
physics steps the (unchanged, NumPy) ``InverseKinematicsController`` of each env computes new arm joint position
targets for the world-frame reference trajectory. The QP knows nothing about the base: because the target is fixed
in the world frame, base motion shows up as a tracking error that the QP corrects, and the policy has to learn
where the base must be so that the arm's COM shift does not tip the platform.

Episodes end when the platform tips (or its COM leaves the footprint) or on time-out. Finished envs are reset
individually with freshly randomized dynamics (base mass, COM height, payload; the footprint is fixed per env
slot) and a fresh reference trajectory.
"""

from __future__ import annotations

import ctypes
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from hybrid_control.controllers.ik_qp.controller import IKQPConfig, InverseKinematicsController, PoseTarget
from hybrid_control.robots.state import Pose
from hybrid_control.sim_interfaces.isaac.ridgeback.arm_model import ArmBatch
from hybrid_control.sim_interfaces.isaac.ridgeback.params import DomainRandomization

from .qp_pool import QPPool
from .observations import ObservationCfg, Snapshot, observe, to_base_frame
from .rewards import RewardCfg, compute_rewards
from .trajectories import ReferenceTrajectories, TrajectoryCfg

PANDA_VEL_LIMIT = np.array([2.175] * 4 + [2.61] * 3)  # rad/s, joints 1-7


@dataclass(frozen=True)
class IKCfg:
    kp_pos: float = 25.0  # the QP is P-only, so a moving reference lags by speed / kp_pos; 25 keeps that within a few cm
    kp_ori: float = 5.0
    position_weight: float = 1.0
    orientation_weight: float = 0.1  # only used with track_orientation
    track_orientation: bool = False  # position-only task: the hand orientation is free
    damping: float = 1e5  # same scale as the MuJoCo experiments (the QP multiplies the task term by 1e8)


@dataclass(frozen=True)
class StabilizationEnvCfg:
    decimation: int = 4  # physics steps per policy step (120 Hz physics -> 30 Hz policy)
    ik_every: int = 2  # physics steps per QP update (-> 60 Hz); must divide ``decimation``
    episode_length_s: float = 15.0
    max_base_speed: tuple[float, float, float] = (0.5, 0.5, 0.6)  # |vx|, |vy| [m/s], |wz| [rad/s] at action = +-1
    terminate_on_left_footprint: bool = True  # also end the episode when the COM leaves the footprint
    domain_randomization: DomainRandomization = field(default_factory=DomainRandomization)
    trajectory: TrajectoryCfg = field(default_factory=TrajectoryCfg)
    observation: ObservationCfg = field(default_factory=ObservationCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    ik: IKCfg = field(default_factory=IKCfg)
    gravity_compensation: bool = True  # feedforward arm gravity torques, as the Franka controller does internally
    qp_workers: int = 0  # worker processes for the per-env QPs (0 = solve serially in this process)


@contextmanager
def _silence_stdout():
    """OSQP's C library prints "Polishing not needed ..." on every solve; mute fd 1 around a batch of solves."""
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()  # OSQP may write via Python's sys.stdout, the C stdio buffer or both; drain both while
        libc.fflush(None)  # fd 1 still points to /dev/null
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


class BaseStabilizationEnv(VecEnv):
    def __init__(self, env, cfg: StabilizationEnvCfg | None = None, seed: int = 0):
        """``env`` is a ``RidgebackDREnv`` whose ``start()`` has already been called."""
        cfg = cfg or StabilizationEnvCfg()
        if cfg.decimation % cfg.ik_every:
            raise ValueError("ik_every must divide decimation")
        self.env, self.scfg = env, cfg
        self.cfg = {"env": cfg}  # rsl_rl's logger stores this next to the training config
        self.device = env.sim.device
        self.num_envs, self.num_actions = env.num_envs, 3
        self.dt = env.dt * cfg.decimation
        self.max_episode_length = int(round(cfg.episode_length_s / self.dt))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.max_vel = torch.tensor(cfg.max_base_speed, device=self.device)
        self.arm = ArmBatch(env, track_orientation=cfg.ik.track_orientation)
        ik_cfg = IKQPConfig(
            velocity_lower=-PANDA_VEL_LIMIT, velocity_upper=PANDA_VEL_LIMIT, kp_pos=cfg.ik.kp_pos,
            kp_ori=cfg.ik.kp_ori, position_weight=cfg.ik.position_weight,
            orientation_weight=cfg.ik.orientation_weight, damping=cfg.ik.damping,
        )
        self.models = self.arm.models()
        self.controllers = [InverseKinematicsController(m, ik_cfg) for m in self.models]
        self.ik_dt = env.dt * cfg.ik_every
        self.ik_failures = 0
        self.auto_reset = True  # False keeps finished envs where they are (e.g. to record a video)
        self.pool = None
        if cfg.qp_workers > 0:
            self.pool = QPPool(
                self.num_envs, self.arm.joint_names, self.arm.limits, ik_cfg, self.ik_dt, self.arm.ee_frame,
                cfg.qp_workers,
            )

        self.traj = ReferenceTrajectories(self.num_envs, self.device, cfg.trajectory)
        self.lookahead = torch.tensor(cfg.observation.lookahead, device=self.device)
        self.ori_target = np.tile([0.0, 0.0, 0.0, 1.0], (self.num_envs, 1))
        self.last_action = torch.zeros(self.num_envs, 3, device=self.device)
        self.q_cmd = env.q_fold.clone()
        self.param_vec = torch.zeros(self.num_envs, 6, device=self.device)  # footprint, masses, payload, COM height
        self.centre_dist_prev = torch.zeros(self.num_envs, device=self.device)
        self._ep_sums: dict[str, torch.Tensor] = {}
        self.extras: dict = {}

        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._obs = self._compute_obs()

    # ------------------------------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------------------------------
    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor):
        env, cfg = self.env, self.scfg
        action = actions.to(self.device).clamp(-1.0, 1.0)
        base_cmd = action * self.max_vel
        for k in range(cfg.decimation):
            if k % cfg.ik_every == 0:
                self._update_arm_command(k * env.dt)
            effort = env.gravity_compensation() if cfg.gravity_compensation else None
            env.step(self.q_cmd, base_cmd, effort)
        self.episode_length_buf += 1

        timeouts = self.episode_length_buf >= self.max_episode_length
        failed = env.tipped | env.left_footprint if cfg.terminate_on_left_footprint else env.tipped.clone()
        terminated = failed & ~timeouts  # a time-out on the same step is not counted as a failure

        t = self.episode_length_buf.float() * self.dt
        rho = env.com_offset / env.half_length  # COM offset as a fraction of the half footprint
        track_err = (self.arm.ee_pos_t() - env.origins - self.traj.position(t)).norm(dim=-1)
        centre_dist = self._centre_dist()
        base_vel = to_base_frame(env.yaw, env.robot.data.body_com_lin_vel_w.torch[:, env.base_id])[:, :2]
        yaw_rate = env.robot.data.body_com_ang_vel_w.torch[:, env.base_id, 2]
        reward, terms = compute_rewards(
            cfg.reward, rho, env.roll, env.pitch, centre_dist, self.centre_dist_prev, self.dt, base_vel, yaw_rate,
            env.yaw, track_err, action, self.last_action, terminated,
        )
        self.centre_dist_prev = centre_dist
        for name, value in terms.items():
            self._ep_sums[name] = self._ep_sums.get(name, torch.zeros_like(value)) + value
        self.last_action = action

        dones = failed | timeouts
        log = {}
        reset_ids = dones.nonzero(as_tuple=False).flatten()
        if len(reset_ids) and self.auto_reset:
            for name, total in self._ep_sums.items():
                log[f"Episode_Reward/{name}"] = (total[reset_ids] / cfg.episode_length_s).mean()
            log["Episode/failed_fraction"] = failed[reset_ids].float().mean()
            log["Episode/length_s"] = (self.episode_length_buf[reset_ids].float() * self.dt).mean()
            self._reset_idx(reset_ids)
        log["Metrics/com_rho"] = rho.mean()
        log["Metrics/ee_track_error"] = track_err.mean()
        log["Metrics/ik_failures_total"] = float(self.ik_failures)
        log["Curriculum/difficulty"] = float(self.traj.cfg.difficulty)

        self.extras = {"time_outs": timeouts.float(), "failed": failed & dones, "log": log}
        self._obs = self._compute_obs()
        return self._obs, reward, dones.long(), self.extras

    # ------------------------------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------------------------------
    def _reset_idx(self, ids: torch.Tensor, dynamics: list | None = None):
        """Reset ``ids`` with freshly sampled dynamics, or the given ``BaseParams`` (``dynamics[k]`` for ``ids[k]``)."""
        env, dr = self.env, self.scfg.domain_randomization
        env.reset(ids, dynamics=dynamics or [dr.sample(self.rng) for _ in range(len(ids))])
        for i in ids.tolist():
            p = env.params[i]
            self.param_vec[i, :5] = torch.tensor([*p.footprint, p.base_mass, p.base_com_z, p.payload])
        self.param_vec[ids, 5] = env.mass_total[ids]
        self.arm.refresh()
        ids_np = ids.cpu().numpy()
        self.ori_target[ids_np] = self.arm.ee_quat[ids_np]
        self.traj.reset(ids, self.arm.ee_pos_t()[ids] - env.origins[ids])
        self.q_cmd[ids] = env.q_fold[ids]
        self.arm.q_cmd[ids_np] = env.q_fold[ids].cpu().numpy()
        self.episode_length_buf[ids] = 0
        self.last_action[ids] = 0.0
        self.centre_dist_prev[ids] = self._centre_dist()[ids]
        for total in self._ep_sums.values():
            total[ids] = 0.0
        for i in ids_np:
            self.controllers[i].reset()

    # ------------------------------------------------------------------------------------------
    # Arm QP
    # ------------------------------------------------------------------------------------------
    def _update_arm_command(self, substep_time: float):
        """One QP per active env against the world-frame reference; writes ``self.q_cmd``.

        ``substep_time`` is the time [s] since the start of the current policy step.
        """
        env, arm = self.env, self.arm
        arm.refresh()
        t = self.episode_length_buf.float() * self.dt + substep_time
        target_w = (self.traj.position(t) + env.origins).cpu().numpy()
        active = (~env.tipped).cpu().numpy()
        if self.pool is not None:
            jac = arm.jacobian if arm.track_orientation else np.concatenate(
                [arm.jacobian[:, :3], np.zeros_like(arm.jacobian[:, 3:])], axis=1
            )
            q_cmd, failed = self.pool.solve(arm.q, arm.ee_pos, arm.ee_quat, jac, target_w, self.ori_target, active)
            self.ik_failures += failed
        else:
            q_cmd = arm.q.copy()  # inactive or failed envs hold their current joint positions
            with _silence_stdout():
                for i in np.flatnonzero(active):
                    target = PoseTarget(arm.ee_frame, Pose(target_w[i], self.ori_target[i]))
                    try:
                        q_cmd[i] = self.controllers[i].compute(target, self.ik_dt).positions
                    except RuntimeError:  # OSQP failure
                        self.ik_failures += 1
        arm.q_cmd[:] = q_cmd
        self.q_cmd = torch.as_tensor(q_cmd, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------------------------------
    def _centre_xy(self) -> torch.Tensor:
        """Footprint centre (N, 2), env-local: the platform is centred MOUNT_X ahead of the base frame."""
        from hybrid_control.sim_interfaces.isaac.ridgeback.robot import MOUNT_X

        env = self.env
        c, s = torch.cos(env.yaw), torch.sin(env.yaw)
        return env.base_pos[:, :2] + torch.stack([c * MOUNT_X, s * MOUNT_X], dim=-1)

    def _centre_dist(self) -> torch.Tensor:
        """Planar distance (N,) from the footprint centre to the centre of the reference trajectory."""
        return (self._centre_xy() - self.traj.centre[:, :2]).norm(dim=-1)

    def _half_size(self) -> torch.Tensor:
        return torch.stack([self.env.half_length, self.env.half_width], dim=-1)

    def _snapshot(self, centre: torch.Tensor | None = None) -> Snapshot:
        env, robot = self.env, self.env.robot
        centre = self._centre_xy() if centre is None else centre
        t = self.episode_length_buf.float() * self.dt
        origins = env.origins
        return Snapshot(
            yaw=env.yaw, roll=env.roll, pitch=env.pitch, centre_xy=centre,
            lin_vel_w=robot.data.body_com_lin_vel_w.torch[:, env.base_id],
            ang_vel_w=robot.data.body_com_ang_vel_w.torch[:, env.base_id],
            com_total=env.com_total - origins, com_arm=env.com_arm - origins,
            ee_pos=self.arm.ee_pos_t() - origins,
            arm_q=robot.data.joint_pos.torch[:, env.arm_ids], arm_qd=robot.data.joint_vel.torch[:, env.arm_ids],
            ref=self.traj.lookahead(t, self.lookahead), traj_centre=self.traj.centre, half_size=self._half_size(),
            params=self.param_vec,
            last_action=self.last_action, time_fraction=self.episode_length_buf.float() / self.max_episode_length,
        )

    def _compute_obs(self) -> TensorDict:
        actor, critic = observe(self._snapshot(), self.scfg.observation)
        return TensorDict({"actor": actor, "critic": critic}, batch_size=[self.num_envs])

    # ------------------------------------------------------------------------------------------
    def close(self):
        """Stop the QP worker processes (also done at garbage collection)."""
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def set_difficulty(self, difficulty: float):
        """Scale trajectory amplitude/speed for the next resets (0 = fixed points only, 1 = full range)."""
        self.traj.cfg.difficulty = float(difficulty)
