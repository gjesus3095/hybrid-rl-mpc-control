"""Reward terms (pure torch, no Isaac imports).

The base policy is rewarded for keeping the whole-robot COM away from the platform edge while the moving-mass
mechanism follows a randomized joint spline that shifts the COM each episode. The policy is penalized for failing
and for any motion beyond what that needs: the base should make small, slow, smooth corrections, not drive around,
spin or jerk. Each term is weighted and summed; ``compute_rewards`` also returns the weighted terms for logging.

Terms with weight 0 are disabled but still computed and logged, so they can be restored by setting the weight:
``yaw`` and ``action``. Unlike the with-controller policy there is no arm tracking term (no QP) and no
trajectory-centre shaping (the base position has no single target relative to the spline).
"""
from dataclasses import dataclass
import torch

from hybrid_control.rl.policy_with_controller.rewards import RewardCfg as RidgebackRewardCfg


@dataclass(frozen=True)
class RewardCfg:
    # --- stability ---
    # exp(-(max(rho, deadzone) / sigma)^2), rho = COM offset as a fraction of the half footprint; the weight is
    # 2 exp((deadzone / sigma)^2), so the plateau inside the deadzone is 2.0 per step (update it with sigma/deadzone)
    w_com: float = RidgebackRewardCfg.w_com
    com_sigma: float = RidgebackRewardCfg.com_sigma
    com_deadzone: float = RidgebackRewardCfg.com_deadzone  # rho below which the com reward is constant (its value at rho = deadzone)
    w_tilt: float = RidgebackRewardCfg.w_tilt  # (roll^2 + pitch^2) / tilt_scale^2; penalizes platform lean before the tip threshold
    tilt_scale: float = RidgebackRewardCfg.tilt_scale  # [rad]
    termination_penalty: float = RidgebackRewardCfg.termination_penalty  # on the step the platform tips (not on time-outs)
    # --- penalties on base motion: slow, smooth corrections ---
    w_speed: float = RidgebackRewardCfg.w_speed  # (|v| / lin_speed_scale)^2, measured base speed
    lin_speed_scale: float = RidgebackRewardCfg.lin_speed_scale  # [m/s]
    w_spin: float = RidgebackRewardCfg.w_spin  # (yaw rate / yaw_rate_scale)^2, measured
    yaw_rate_scale: float = RidgebackRewardCfg.yaw_rate_scale  # [rad/s]
    w_yaw: float = RidgebackRewardCfg.w_yaw  # (yaw / yaw_scale)^2; disabled: spinning is already penalized by w_spin
    yaw_scale: float = RidgebackRewardCfg.yaw_scale  # [rad]
    w_action_rate: float = RidgebackRewardCfg.w_action_rate  # |a_t - a_{t-1}|^2, jerky commands
    w_action: float = RidgebackRewardCfg.w_action  # |a|^2; disabled: overlaps with w_speed


def compute_rewards(cfg, com_rho, roll, pitch, base_vel, yaw_rate, yaw, action, last_action, terminated):
    """Per-env inputs (N,), except ``base_vel`` (N, 2) in the base frame and the actions (N, 3).

    Returns ``(reward (N,), weighted terms)``.
    """
    terms = {
        'com': cfg.w_com * torch.exp(-(com_rho.clamp_min(cfg.com_deadzone) / cfg.com_sigma).square()),
        'tilt': cfg.w_tilt * (roll.square() + pitch.square()) / cfg.tilt_scale**2,
        'speed': cfg.w_speed * base_vel.square().sum(-1) / cfg.lin_speed_scale**2,
        'spin': cfg.w_spin * (yaw_rate / cfg.yaw_rate_scale).square(),
        'yaw': cfg.w_yaw * (yaw / cfg.yaw_scale).clamp(-3, 3).square(),
        'action_rate': cfg.w_action_rate * (action - last_action).square().sum(-1),
        'action': cfg.w_action * action.square().sum(-1),
        'termination': cfg.termination_penalty * terminated.float(),
    }
    return sum(terms.values()), terms
