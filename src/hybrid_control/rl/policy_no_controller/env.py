"""RSL-RL ``VecEnv`` around ``MovingMassEnv``: an RL policy drives the base while a moving-mass mechanism
follows a randomized cubic B-spline trajectory that shifts the whole-robot COM each episode.

Per policy step the wrapper holds the base velocity command for ``decimation`` physics steps; at every physics step
the mechanism's joint targets are read from the spline and passed directly to the actuators (no separate QP). There
is no arm controller: the 3-DoF mechanism (roll, pitch, lift) is driven purely by the spline, and the
policy has to learn where to move the base so the total COM stays inside the platform footprint.

Episodes end when the platform tips (|roll| or |pitch| exceeds the threshold) or on time-out. Finished envs are
reset individually with freshly sampled dynamics (platform mass, COM height, terminal mass; the footprint is fixed
per env slot) and a fresh random spline trajectory.
"""
from dataclasses import dataclass, field
import math
import random

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from hybrid_control.sim_interfaces.isaac.moving_mass.params import MechanismCfg, MovingMassRandomization
from .observations import ObservationCfg, Snapshot, observe, to_base_frame
from .rewards import RewardCfg, compute_rewards
from .trajectories import JointSplines, TrajectoryCfg


@dataclass(frozen=True)
class StabilizationEnvCfg:
    environment_kind: str = 'moving_mass_v2'  # version tag; checked on load to catch config/checkpoint mismatches
    physics_dt: float = 1 / 120  # 120 Hz physics
    decimation: int = 4  # physics steps per policy step (120 Hz physics -> 30 Hz policy)
    episode_length_s: float = 15.0
    max_base_speed: tuple[float, float, float] = (0.5, 0.5, 0.6)  # |vx|, |vy| [m/s], |wz| [rad/s] at action = +-1
    friction: float = 0.05  # virtual base actuator: ground friction coefficient
    base_time_constant: float = 0.15  # virtual base actuator: first-order velocity-tracking time constant [s]
    max_base_force: float = 1000.0  # virtual base actuator: maximum planar force and yaw torque [N / N·m]
    mechanism: MechanismCfg = field(default_factory=MechanismCfg)  # geometry and joint limits of the moving-mass arm
    domain_randomization: MovingMassRandomization = field(default_factory=MovingMassRandomization)
    trajectory: TrajectoryCfg = field(default_factory=TrajectoryCfg)
    observation: ObservationCfg = field(default_factory=ObservationCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)

    def __post_init__(self):
        if self.environment_kind != 'moving_mass_v2':
            raise ValueError('incompatible environment kind')
        if self.physics_dt <= 0 or self.decimation < 1 or int(self.decimation) != self.decimation:
            raise ValueError('physics_dt and integer decimation must be positive')
        if self.episode_length_s < self.physics_dt * self.decimation:
            raise ValueError('episode must contain at least one policy step')
        if self.base_time_constant <= 0 or self.max_base_force <= 0 or self.friction < 0:
            raise ValueError('invalid base actuator or friction configuration')
        if len(self.max_base_speed) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.max_base_speed):
            raise ValueError('max_base_speed must contain three positive finite values')


class BaseStabilizationEnv(VecEnv):
    def __init__(self, env, cfg=None, seed=0):
        """``env`` is a ``MovingMassEnv`` whose ``start()`` has already been called."""
        self.env, self.scfg = env, cfg or StabilizationEnvCfg()
        cfg = self.scfg
        if not math.isclose(env.dt, cfg.physics_dt) or env.mechanism != cfg.mechanism:
            raise ValueError('physics and mechanism must match the saved environment configuration')
        self.cfg = {'env': cfg}  # rsl_rl's logger stores this next to the training config
        self.device, self.num_envs, self.num_actions = env.sim.device, env.num_envs, 3
        self.dt = env.dt * cfg.decimation
        self.max_episode_length = max(1, int(round(cfg.episode_length_s / self.dt)))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rngs = [random.Random(seed + 1000003*i) for i in range(self.num_envs)]
        self.traj = JointSplines(self.num_envs, self.device, self.max_episode_length*self.dt,
                                 cfg.mechanism, cfg.trajectory, seed)
        self.max_vel = torch.tensor(cfg.max_base_speed, device=self.device)
        self.last_action = torch.zeros(self.num_envs, 3, device=self.device)
        self.param_vec = torch.zeros(self.num_envs, 6, device=self.device)  # [length, width, base_mass, base_com_z, terminal_mass, total_mass]
        self.auto_reset = True  # False keeps finished envs in place (e.g. to record a video)
        self._ep_sums = {}
        self.extras = {}
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._obs = self._compute_obs()

    # ------------------------------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------------------------------
    def get_observations(self):
        return self._obs

    # ------------------------------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------------------------------
    def _reset_idx(self, ids):
        """Reset ``ids`` with freshly sampled dynamics and a new random spline trajectory."""
        dynamics = [self.scfg.domain_randomization.sample(self.rngs[i]) for i in ids.tolist()]
        self.env.reset(ids, dynamics=dynamics)
        self.traj.reset(ids)
        self.episode_length_buf[ids] = 0
        self.last_action[ids] = 0
        for i in ids.tolist():
            p = self.env.params[i]
            self.param_vec[i] = torch.tensor([*p.footprint, p.base_mass, p.base_com_z,
                                              p.terminal_mass, float(self.env.mass_total[i])], device=self.device)
        for value in self._ep_sums.values():
            value[ids] = 0

    def step(self, actions):
        env, cfg = self.env, self.scfg
        if actions.shape != (self.num_envs, 3) or not torch.isfinite(actions).all():
            raise ValueError('actions must be finite with shape (num_envs, 3)')
        action = actions.to(self.device).clamp(-1, 1)
        start = self.episode_length_buf.float() * self.dt
        # Hold the base command for ``decimation`` physics steps; update spline targets each substep.
        for k in range(cfg.decimation):
            q, qd, _ = self.traj.evaluate(start + (k + 1) * env.dt)
            env.step(q, action * self.max_vel, qd)
        self.episode_length_buf += 1
        timeout = self.episode_length_buf >= self.max_episode_length
        failed = env.tipped.clone()
        dones = failed | timeout
        # Reward inputs.
        rho = env.com_offset / env.half_length  # COM offset as a fraction of the half footprint
        lin = to_base_frame(env.yaw, env.robot.data.body_com_lin_vel_w.torch[:, env.base_id])[:, :2]
        ang = env.robot.data.body_com_ang_vel_w.torch[:, env.base_id, 2]
        reward, terms = compute_rewards(cfg.reward, rho, env.roll, env.pitch, lin, ang, env.yaw,
                                        action, self.last_action, failed)
        for name, value in terms.items():
            self._ep_sums.setdefault(name, torch.zeros_like(value)).add_(value)
        self.last_action = action.clone()
        actual = env.robot.data.joint_pos.torch[:, env.joint_ids]
        error = (actual - q).abs()
        # Implicit drives report estimated effort, clipped to their limits by Isaac.
        effort = env.robot.actuators.applied_effort.torch[:, env.joint_ids]
        limits = torch.tensor(cfg.mechanism.effort_limit, device=self.device)
        log = {
            'Metrics/com_rho': rho.mean(),
            'Metrics/footprint_excursion': env.left_footprint.float().mean(),
            'Metrics/tilt': torch.maximum(env.roll.abs(), env.pitch.abs()).mean(),
            'Metrics/base_speed': lin.norm(dim=-1).mean(),
            'Metrics/yaw_rate': ang.abs().mean(),
            'Metrics/revolute_tracking_error': error[:, :2].mean(),
            'Metrics/lift_tracking_error': error[:, 2].mean(),
            'Metrics/actuator_saturation': (effort.abs() >= 0.99 * limits).float().mean(),
        }
        ids = dones.nonzero(as_tuple=False).flatten()
        if len(ids) and self.auto_reset:
            duration = self.episode_length_buf[ids].float() * self.dt
            for name, total in self._ep_sums.items():
                log[f'Episode_Reward/{name}'] = (total[ids] / duration).mean()
            log['Episode/failed_fraction'] = failed[ids].float().mean()
            log['Episode/length_s'] = duration.mean()
            self._reset_idx(ids)
        self.extras = {'time_outs': (timeout & ~failed).float(), 'failed': failed, 'log': log}
        self._obs = self._compute_obs()
        return self._obs, reward, dones.long(), self.extras

    # ------------------------------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------------------------------
    def _compute_obs(self):
        env, robot = self.env, self.env.robot
        snap = Snapshot(
            yaw=env.yaw, roll=env.roll, pitch=env.pitch, centre_xy=env.base_pos[:, :2],
            lin_vel_w=robot.data.body_com_lin_vel_w.torch[:, env.base_id],
            ang_vel_w=robot.data.body_com_ang_vel_w.torch[:, env.base_id],
            com_total=env.com_total-env.origins, com_mechanism=env.com_mechanism-env.origins,
            joint_pos=robot.data.joint_pos.torch[:, env.joint_ids],
            joint_vel=robot.data.joint_vel.torch[:, env.joint_ids],
            half_size=torch.stack((env.half_length, env.half_width), -1),
            params=self.param_vec, last_action=self.last_action,
        )
        actor, critic = observe(snap, self.scfg.observation)
        return TensorDict({'actor': actor, 'critic': critic}, batch_size=[self.num_envs])

    # ------------------------------------------------------------------------------------------
    def close(self):
        """No controller processes or external resources are owned by this wrapper."""
