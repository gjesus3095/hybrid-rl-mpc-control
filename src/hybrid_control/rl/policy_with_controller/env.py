"""RSL-RL ``VecEnv`` around ``RidgebackDREnv``: an RL policy drives the base, a QP controller drives the arm.

Per policy step the wrapper holds the base velocity command for ``decimation`` physics steps. Every ``ik_every``
physics steps each env's arm controller (``StabilizationEnvCfg.controller``, one of the ``controllers/`` folders)
tracks the world-frame reference pose trajectory (``trajectories.py``: fixed point, sine or quintic A -> B):

- ``"ik_qp"``: the Pink differential IK/QP (``controllers/ik_qp/ik_qp.py``) computes arm joint position targets for
  ``panda_hand`` (position and orientation). It knows nothing about the base: because the target is
  fixed in the world frame, base motion shows up as a tracking error that the QP corrects.
- ``"impedance"``: the torque-level impedance QP (``controllers/impedance``) computes arm joint torques for the TCP
  ``panda_fingertip_centered``, set up as in ``scripts/isaac/base_disturbance_impedance.py``: full pose, null-space
  posture torque, manipulability floor, joint-limit barriers and the commanded base twist as feedforward. The arm's
  implicit PD gains are zeroed (pure effort control); the QP compensates gravity.

Either way the policy has to learn where the base must be so that the arm's COM shift does not tip the platform.

Episodes end when the platform tips (or its COM leaves the footprint) or on time-out. Finished envs are reset
individually with freshly randomized dynamics (base mass, COM height, payload; the footprint is fixed per env
slot) and a fresh reference trajectory.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from prettytable import PrettyTable
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from hybrid_control.sim_interfaces.isaac.ridgeback.arm_model import ArmBatch
from hybrid_control.sim_interfaces.isaac.ridgeback.params import DomainRandomization
from hybrid_control.utils.torch_rotations import quat_angle, quat_conj, quat_mul, quat_to_rotvec

from .qp_pool import CONTROLLERS, QPPool
from .observations import ObservationCfg, Snapshot, observation_terms, observe, to_base_frame
from .rewards import RewardCfg, compute_rewards, reward_term_info
from .trajectories import ReferenceTrajectories, TrajectoryCfg

PANDA_VEL_LIMIT = np.array([2.175] * 4 + [2.61] * 3)  # rad/s, joints 1-7


@dataclass(frozen=True)
class IKCfg:
    """Pink IK/QP (``controller="ik_qp"``): position and orientation tasks toward the reference pose."""

    kp_pos: float = 25.0  # the QP is P-only, so a moving reference lags by speed / kp_pos; 25 keeps that within a few cm
    kp_ori: float = 5.0
    position_weight: float = 1.0
    orientation_weight: float = 1.0
    damping: float = 1e5  # same scale as the MuJoCo experiments (the QP multiplies the task term by 1e8)
    qp_solver: str = "daqp"  # qpsolvers backend: "daqp" (Isaac Lab's PinkIKController) or "osqp"


@dataclass(frozen=True)
class ImpedanceCfg:
    """Impedance QP (``controller="impedance"``), tuned as in ``scripts/isaac/base_disturbance_impedance.py``."""

    kp_pos: float = 400.0  # 1/s², critically damped
    kp_ori: float = 200.0
    ddq_max: tuple[float, ...] = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)  # rad/s²
    tau_max: tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)  # N·m
    nullspace_posture: tuple[float, ...] = (0.0, 0.0, 0.0, -1.5, 0.0, 1.885, 0.0)
    nullspace_stiffness: float = 10.0  # N·m/rad
    nullspace_damping: float = 1.0  # N·m·s/rad
    manipulability_rows: str = "full"
    manipulability_min: float = 0.03
    joint_limit_margin: float = 0.05  # rad
    base_feedforward: bool = True  # commanded base twist -> TCP velocity/acceleration feedforward


@dataclass(frozen=True)
class StabilizationEnvCfg:
    decimation: int = 4  # physics steps per policy step (120 Hz physics -> 30 Hz policy)
    ik_every: int = 2  # physics steps per QP update (-> 60 Hz); must divide ``decimation``
    episode_length_s: float = 5.0  # = the trajectory duration (blend / spawn -> A -> B)
    max_base_speed: tuple[float, float, float] = (1.0, 1.0, 0.6)  # |vx|, |vy| [m/s], |wz| [rad/s] at action = +-1
    terminate_on_left_footprint: bool = True  # also end the episode when the COM leaves the footprint
    domain_randomization: DomainRandomization = field(default_factory=DomainRandomization)
    trajectory: TrajectoryCfg = field(default_factory=TrajectoryCfg)
    observation: ObservationCfg = field(default_factory=ObservationCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    controller: str = "ik_qp"  # arm controller: "ik_qp" (Pink IK, position targets) or "impedance" (torques)
    ik: IKCfg = field(default_factory=IKCfg)
    impedance: ImpedanceCfg = field(default_factory=ImpedanceCfg)
    gravity_compensation: bool = True  # ik_qp only: feedforward arm gravity torques (the impedance QP includes them)
    qp_workers: int = 0  # worker processes for the per-env QPs (0 = solve serially in this process)


class BaseStabilizationEnv(VecEnv):
    def __init__(self, env, cfg: StabilizationEnvCfg | None = None, seed: int = 0):
        """``env`` is a ``RidgebackDREnv`` whose ``start()`` has already been called."""
        cfg = cfg or StabilizationEnvCfg()
        if cfg.decimation % cfg.ik_every:
            raise ValueError("ik_every must divide decimation")
        if cfg.controller not in CONTROLLERS:
            raise ValueError(f"controller must be one of {CONTROLLERS}, got {cfg.controller!r}")
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
        self.base_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.ik_dt = env.dt * cfg.ik_every
        self.ik_failures = 0
        self.auto_reset = True  # False keeps finished envs where they are (e.g. to record a video)
        self.pool = None  # set below; close() must work if the setup fails
        self.impedance = cfg.controller == "impedance"
        if self.impedance:
            from hybrid_control.sim_interfaces.isaac.ridgeback.robot import franka_tcp_offset

            self.arm = ArmBatch(env, ee_frame="panda_fingertip_centered", ee_body="panda_hand",
                                ee_offset=franka_tcp_offset(), dynamics=True)
            zeros = torch.zeros(self.num_envs, len(env.arm_ids), device=self.device)  # pure effort control of the arm
            env.robot.write_joint_stiffness_to_sim_index(stiffness=zeros, joint_ids=env.arm_ids)
            env.robot.write_joint_damping_to_sim_index(damping=zeros, joint_ids=env.arm_ids)
            solver_cfg = self._impedance_solver_cfg()
        else:
            self.arm = ArmBatch(env)
            solver_cfg = {
                "joint_names": list(self.arm.joint_names), "frame": self.arm.ee_frame, "root_frame": self.arm.root_frame,
                "dt": self.ik_dt, "qp_solver": cfg.ik.qp_solver,
                "ik": {
                    "velocity_lower": (-PANDA_VEL_LIMIT).tolist(), "velocity_upper": PANDA_VEL_LIMIT.tolist(),
                    "kp_pos": cfg.ik.kp_pos, "kp_ori": cfg.ik.kp_ori, "position_weight": cfg.ik.position_weight,
                    "orientation_weight": cfg.ik.orientation_weight, "damping": cfg.ik.damping,
                },
            }
        self.pool = QPPool(cfg.controller, self.num_envs, len(env.arm_ids), solver_cfg, cfg.qp_workers)
        self.pool.v["limits"][:] = self.arm.limits_array
        self.effort = torch.zeros(self.num_envs, len(env.arm_ids), device=self.device)

        self.traj = ReferenceTrajectories(self.num_envs, self.device, cfg.trajectory)
        self.lookahead = torch.tensor(cfg.observation.lookahead, device=self.device)
        self.last_action = torch.zeros(self.num_envs, 3, device=self.device)
        self.q_cmd = env.q_fold.clone()
        self.param_vec = torch.zeros(self.num_envs, 6, device=self.device)  # footprint, masses, payload, COM height
        self.centre_dist_prev = torch.zeros(self.num_envs, device=self.device)
        self._ep_sums: dict[str, torch.Tensor] = {}
        self.extras: dict = {}

        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._obs = self._compute_obs()
        self.print_summary()

    # ------------------------------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------------------------------
    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor):
        env, cfg = self.env, self.scfg
        action = actions.to(self.device).clamp(-1.0, 1.0)
        base_cmd = action * self.max_vel
        self.base_cmd = base_cmd
        for k in range(cfg.decimation):
            if k % cfg.ik_every == 0:
                self._update_arm_command(k * env.dt)
            if self.impedance:
                effort = self.effort  # QP torques, held between QP updates
            else:
                effort = env.gravity_compensation() if cfg.gravity_compensation else None
            env.step(self.q_cmd, base_cmd, effort)
        self.episode_length_buf += 1

        timeouts = self.episode_length_buf >= self.max_episode_length
        failed = env.tipped | env.left_footprint if cfg.terminate_on_left_footprint else env.tipped.clone()
        terminated = failed & ~timeouts  # a time-out on the same step is not counted as a failure

        t = self.episode_length_buf.float() * self.dt
        rho = env.com_offset / env.half_length  # COM offset as a fraction of the half footprint
        ref_pos, ref_quat = self.traj.pose(t)
        track_err = (self.arm.ee_pos_t() - env.origins - ref_pos).norm(dim=-1)
        ori_err = quat_angle(torch.as_tensor(self.arm.ee_quat, dtype=torch.float32, device=self.device), ref_quat)
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
        log["Metrics/ee_ori_error"] = ori_err.mean()
        log["Metrics/ik_failures_total"] = float(self.ik_failures)
        log["Curriculum/difficulty"] = float(self.traj.cfg.difficulty)
        log["Trajectory/spatial_order"] = float(self.traj.cfg.spatial_order)

        self.extras = {"time_outs": timeouts.float(), "failed": failed & dones, "log": log}
        self._obs = self._compute_obs()
        return self._obs, reward, dones.long(), self.extras

    # ------------------------------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------------------------------
    def _reset_idx(self, ids: torch.Tensor, dynamics: list | None = None, kinds: torch.Tensor | None = None):
        """Reset ``ids`` with freshly sampled dynamics, or the given ``BaseParams`` (``dynamics[k]`` for ``ids[k]``).
        ``kinds`` forces the reference trajectory kinds (see ``ReferenceTrajectories.reset``)."""
        env, dr = self.env, self.scfg.domain_randomization
        env.reset(ids, dynamics=dynamics or [dr.sample(self.rng) for _ in range(len(ids))])
        self._refresh_param_vec(ids)
        self.arm.refresh()
        ids_np = ids.cpu().numpy()
        ee_quat = torch.as_tensor(self.arm.ee_quat[ids_np], dtype=torch.float32, device=self.device)
        self.traj.reset(ids, self.arm.ee_pos_t()[ids] - env.origins[ids], ee_quat, kind=kinds)
        self.arm.reset_history(ids_np)
        self.q_cmd[ids] = env.q_fold[ids]
        self.effort[ids] = 0.0
        self.base_cmd[ids] = 0.0
        self.episode_length_buf[ids] = 0
        self.last_action[ids] = 0.0
        self.centre_dist_prev[ids] = self._centre_dist()[ids]
        for total in self._ep_sums.values():
            total[ids] = 0.0
        self.pool.v["reset"][ids_np] = 1.0  # stateful controllers (impedance warm start, posture) start over

    def _refresh_param_vec(self, ids: torch.Tensor):
        env = self.env
        for i in ids.tolist():
            p = env.params[i]
            self.param_vec[i, :5] = torch.tensor([*p.footprint, p.base_mass, p.base_com_z, p.payload])
        self.param_vec[ids, 5] = env.mass_total[ids]

    def set_payload(self, ids: torch.Tensor, payloads: list[float]):
        """Change the TCP payload [kg] of ``ids`` mid-episode (no reset); the policy sees it in its physical params."""
        self.env.set_payload(ids, payloads)
        self._refresh_param_vec(ids)
        self._obs = self._compute_obs()

    # ------------------------------------------------------------------------------------------
    # Arm QP
    # ------------------------------------------------------------------------------------------
    def _impedance_solver_cfg(self) -> dict:
        c = self.scfg.impedance
        ddq = list(c.ddq_max)
        return {
            "joint_names": list(self.arm.joint_names), "frame": self.arm.ee_frame, "dt": self.ik_dt,
            "torque_limit": list(c.tau_max),
            "impedance": {
                "acceleration_lower": [-x for x in ddq], "acceleration_upper": ddq,
                "torque_lower": [-x for x in c.tau_max], "torque_upper": list(c.tau_max),
                "kp_pos": c.kp_pos, "kp_ori": c.kp_ori,
                "posture_kp": 0.0, "posture_kd": 0.0,  # the null-space torque handles the posture
                "nullspace_posture": list(c.nullspace_posture), "nullspace_stiffness": c.nullspace_stiffness,
                "nullspace_damping": c.nullspace_damping, "manipulability_rows": c.manipulability_rows,
                "manipulability_min": c.manipulability_min, "joint_limit_margin": c.joint_limit_margin,
            },
        }

    def _reference_derivatives(self, t: torch.Tensor, eps: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
        """Reference twist velocity and acceleration (N, 6), [linear, angular] in the world frame, at ``t`` by central
        differences (the angular ones on the rotation vectors of q(t +- eps) q(t)^-1)."""
        p, q = self.traj.lookahead_pose(t, torch.tensor([-eps, 0.0, eps], device=self.device))
        r_minus = quat_to_rotvec(quat_mul(q[:, 0], quat_conj(q[:, 1])))
        r_plus = quat_to_rotvec(quat_mul(q[:, 2], quat_conj(q[:, 1])))
        vel = torch.cat([(p[:, 2] - p[:, 0]) / (2 * eps), (r_plus - r_minus) / (2 * eps)], dim=-1)
        acc = torch.cat([(p[:, 2] - 2 * p[:, 1] + p[:, 0]) / eps**2, (r_plus + r_minus) / eps**2], dim=-1)
        return vel.cpu().numpy(), acc.cpu().numpy()

    def _base_feedforward(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP twist and twist acceleration (N, 6) caused by the commanded base motion (see the impedance demo).

        With the base moving at world (vx, vy, wz) about the base_link z axis (r = TCP - base axis, w = wz ez):
        V_b = [v + w x r; w] and, at constant base velocity, A_b = [w x (w x r) + w x (J_lin dq); 0].
        """
        env, arm = self.env, self.arm
        cmd = self.base_cmd.cpu().numpy()
        yaw = env.yaw.cpu().numpy()
        c, s = np.cos(yaw), np.sin(yaw)
        v = np.stack([c * cmd[:, 0] - s * cmd[:, 1], s * cmd[:, 0] + c * cmd[:, 1], np.zeros(len(cmd))], axis=-1)
        w = np.zeros_like(v)
        w[:, 2] = cmd[:, 2]
        base = env.robot.data.body_link_pos_w.torch[:, env.base_id].cpu().numpy()
        r = arm.ee_pos - base
        arm_vel = np.einsum("nij,nj->ni", arm.jacobian[:, :3], arm.qd)
        velocity = np.concatenate([v + np.cross(w, r), w], axis=1)
        acceleration = np.concatenate([np.cross(w, np.cross(w, r)) + np.cross(w, arm_vel), np.zeros_like(v)], axis=1)
        return velocity, acceleration

    def _update_arm_command(self, substep_time: float):
        """One QP per active env against the world-frame reference; writes ``self.q_cmd`` or ``self.effort``.

        ``substep_time`` is the time [s] since the start of the current policy step.
        """
        env, arm, v = self.env, self.arm, self.pool.v
        arm.refresh()
        t = self.episode_length_buf.float() * self.dt + substep_time
        ref_pos, ref_quat = self.traj.pose(t)
        v["target"][:, :3] = (ref_pos + env.origins).cpu().numpy()
        v["target"][:, 3:] = ref_quat.cpu().numpy()
        v["active"][:] = (~env.tipped).cpu().numpy()
        v["q"][:] = arm.q
        if self.impedance:
            v["dq"][:], v["ee_pose"][:], v["jac"][:], v["djac"][:] = arm.qd, arm.ee_pose, arm.jacobian, arm.jacobian_dot
            v["mass"][:], v["h"][:] = arm.mass_matrix, arm.nonlinear_forces
            v["target_vel"][:], v["target_acc"][:] = self._reference_derivatives(t)
            if self.scfg.impedance.base_feedforward:
                base_vel, base_acc = self._base_feedforward()
                v["target_vel"] -= base_vel
                v["target_acc"] -= base_acc
        else:
            v["root_pose"][:] = arm.root_pose
        command, failed = self.pool.solve()
        self.ik_failures += failed
        command = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if self.impedance:
            self.effort = command
            self.q_cmd = env.robot.data.joint_pos.torch[:, env.arm_ids].clone()  # no effect: the arm PD gains are 0
        else:
            self.q_cmd = command

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
        ref, ref_quat = self.traj.lookahead_pose(t, self.lookahead)
        return Snapshot(
            yaw=env.yaw, roll=env.roll, pitch=env.pitch, centre_xy=centre,
            lin_vel_w=robot.data.body_com_lin_vel_w.torch[:, env.base_id],
            ang_vel_w=robot.data.body_com_ang_vel_w.torch[:, env.base_id],
            com_total=env.com_total - origins, com_arm=env.com_arm - origins,
            ee_pos=self.arm.ee_pos_t() - origins,
            # arm_q=robot.data.joint_pos.torch[:, env.arm_ids], arm_qd=robot.data.joint_vel.torch[:, env.arm_ids],
            ref=ref, ref_quat=ref_quat, half_size=self._half_size(),
            # traj_centre=self.traj.centre,
            params=self.param_vec,
            last_action=self.last_action,
            # time_fraction=self.episode_length_buf.float() / self.max_episode_length,
        )

    def _compute_obs(self) -> TensorDict:
        actor, critic = observe(self._snapshot(), self.scfg.observation)
        return TensorDict({"actor": actor, "critic": critic}, batch_size=[self.num_envs])

    # ------------------------------------------------------------------------------------------
    # Summary (printed at construction, in the style of Isaac Lab's manager-based envs)
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _table(title: str, fields: list[str], rows: list[list], left: tuple[str, ...] = ("Name",)) -> str:
        table = PrettyTable()
        table.title = title
        table.field_names = fields
        for name in fields:
            table.align[name] = "l" if name in left else "r"
        for row in rows:
            table.add_row(row)
        return table.get_string()

    @staticmethod
    def _fmt(value) -> str:
        if isinstance(value, (tuple, list)):
            return "(" + ", ".join(BaseStabilizationEnv._fmt(v) for v in value) + ")"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def summary(self) -> str:
        """Environment, arm controller, observation, action, reward, termination and reset-event tables."""
        from hybrid_control.sim_interfaces.isaac.ridgeback.env import TIP_ANGLE

        cfg, env = self.scfg, self.env
        out = [
            "[INFO]: Base stabilization environment:",
            f"\tEnvironment device    : {self.device}",
            f"\tNumber of envs        : {self.num_envs}",
            f"\tPhysics step-size     : {env.dt:g} s ({1 / env.dt:.0f} Hz)",
            f"\tEnvironment step-size : {self.dt:g} s ({1 / self.dt:.0f} Hz, decimation {cfg.decimation})",
            f"\tArm QP step-size      : {self.ik_dt:g} s ({1 / self.ik_dt:.0f} Hz, every {cfg.ik_every} physics steps)",
            f"\tEpisode length        : {cfg.episode_length_s:g} s ({self.max_episode_length} steps)",
        ]

        ctrl = asdict(cfg.impedance if self.impedance else cfg.ik)
        rows = [["frame", self.arm.ee_frame], ["command", "joint torques" if self.impedance else "joint position targets"],
                ["qp_workers", cfg.qp_workers]]
        if not self.impedance:
            rows += [["velocity_limit [rad/s]", self._fmt(PANDA_VEL_LIMIT.tolist())],
                     ["gravity_compensation", cfg.gravity_compensation]]
        rows += [[k, self._fmt(v)] for k, v in ctrl.items()]
        out.append(f"[INFO] Arm controller: '{cfg.controller}'\n"
                   + self._table(f"Arm Controller: {cfg.controller}", ["Parameter", "Value"], rows, left=("Parameter",)))

        actor, critic = observation_terms(self._snapshot(), cfg.observation)
        msg = "[INFO] Observation Manager: <ObservationManager> contains 2 groups.\n"
        for group, terms in (("actor", actor), ("critic", critic)):
            dim = sum(t.shape[1] for t in terms.values())
            rows = [[i, name, (t.shape[1],)] for i, (name, t) in enumerate(terms.items())]
            msg += self._table(f"Active Observation Terms in Group: '{group}' (shape: ({dim},))",
                               ["Index", "Name", "Shape"], rows) + "\n"
        msg += (f"\tref_lookahead times [s]: {self._fmt(cfg.observation.lookahead)}\n"
                "\tref_orientation: first two columns of the reference rotation matrix (base yaw frame) per lookahead\n"
                "\tphysical_params: [length, width, base_mass, base_com_z, payload, total_mass]")
        out.append(msg)

        out.append("[INFO] Action Manager: <ActionManager> contains 1 active terms.\n" + self._table(
            f"Active Action Terms (shape: {self.num_actions})", ["Index", "Name", "Dimension", "Scale (at |a| = 1)"],
            [[0, "base_velocity (vx, vy, wz), base frame, clipped to [-1, 1]", self.num_actions,
              f"{self._fmt(cfg.max_base_speed)} m/s, m/s, rad/s"]]))

        terms = reward_term_info(cfg.reward)
        out.append(f"[INFO] Reward Manager: <RewardManager> contains {len(terms)} active terms.\n" + self._table(
            "Active Reward Terms", ["Index", "Name", "Weight", "Term"],
            [[i, name, self._fmt(float(w)), expr] for i, (name, w, expr) in enumerate(terms)], left=("Name", "Term")))

        rows = [[0, f"tipped: |roll| or |pitch| > {TIP_ANGLE:g} rad", False]]
        if cfg.terminate_on_left_footprint:
            rows.append([1, "left_footprint: total COM outside the platform footprint", False])
        rows.append([len(rows), f"time_out: {cfg.episode_length_s:g} s", True])
        out.append(f"[INFO] Termination Manager: <TerminationManager> contains {len(rows)} active terms.\n"
                   + self._table("Active Termination Terms", ["Index", "Name", "Time Out"], rows))

        dr, tr = cfg.domain_randomization, self.traj.cfg
        rows = [
            [0, "reset_robot", "folded arm, base at the env origin"],
            [1, "randomize_base_dynamics", f"base_mass {self._fmt(dr.base_mass)} kg, base_com_z {self._fmt(dr.base_com_z)} m, "
                                           f"payload {self._fmt(dr.payload)} kg, hard_bias {dr.hard_bias:g}"],
            [2, "footprint (fixed per env at spawn)", f"length {self._fmt(dr.length)} m, width {self._fmt(dr.width)} m"],
            [3, "sample_reference_trajectory", f"p (fixed, sine, quintic) {self._fmt((tr.p_fixed, tr.p_sine, tr.p_quintic))}"],
            [4, "  fixed / sine", f"difficulty {tr.difficulty:g}, amplitude {self._fmt(tr.amplitude)} m, "
                                f"frequency {self._fmt(tr.frequency)} Hz, orientation <= {tr.ori_amplitude_deg:g} deg per axis "
                                f"about a down centre, blend {tr.blend_time:g} s"],
            [5, "  quintic spawn -> A -> B", f"segments {self._fmt(tr.segment_time)} s, Bezier order {tr.spatial_order} (deviation <= {tr.curve_deviation:g} m, "
                                            f"{2 ** tr.hull_splits} hull pieces), |xy| >= {tr.inner_radius_xy:g} m, |p| <= {tr.outer_radius:g} m "
                                            f"from {self._fmt(tr.arm_base)}, z >= {tr.quintic_min_z:g} m, "
                                            f"tilt <= {tr.tilt_max_deg:g} deg, yaw {tr.yaw_centre_deg:g} +- {tr.yaw_range_deg:g} deg"],
        ]
        out.append(f"[INFO] Event Manager: <EventManager> contains {len(rows)} active terms.\n" + self._table(
            "Active Event Terms in Mode: 'reset'", ["Index", "Name", "Parameters"], rows, left=("Name", "Parameters")))
        return "\n".join(out)

    def print_summary(self):
        print(self.summary(), flush=True)

    # ------------------------------------------------------------------------------------------
    def close(self):
        """Stop the QP worker processes (also done at garbage collection)."""
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def set_difficulty(self, difficulty: float):
        """Scale trajectory amplitude/speed for the next resets (0 = fixed points only, 1 = full range)."""
        self.traj.cfg.difficulty = float(difficulty)
