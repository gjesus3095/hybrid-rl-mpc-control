"""Reward terms (pure torch, no Isaac imports).

The base policy is rewarded for keeping the whole-robot COM centred over the platform and for staying near the
trajectory it serves, and is penalized for tipping and for any motion beyond what that needs: the base should make
small, slow, smooth corrections, not drive around, spin or jerk. Each term is weighted and summed;
``compute_rewards`` also returns the weighted terms for logging.

Direction of travel is rewarded through potential-based shaping on the distance to the trajectory centre (only while
farther than ``centre_deadzone``): going back and forth or in circles earns nothing, because the shaping telescopes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RewardCfg:
    # --- stability ---
    w_com: float = 2.0  # exp(-(rho / sigma)^2), rho = COM offset as a fraction of the half footprint
    com_sigma: float = 0.5
    w_tilt: float = -1.0  # (roll^2 + pitch^2) / tilt_scale^2
    tilt_scale: float = 0.25  # [rad]
    termination_penalty: float = -10.0  # on the step an env tips or its COM leaves the footprint (not on time-outs)
    # --- stay near / go toward the trajectory (planar distance footprint centre -> trajectory centre) ---
    w_centre: float = 0.5  # exp(-(max(d - deadzone, 0) / sigma)^2)
    centre_sigma: float = 0.5  # [m]
    centre_deadzone: float = 0.35  # [m] about the horizontal distance at which the arm is comfortably bent
    w_progress: float = 1.0  # clamp(rate of approach, -1, 1) [m/s] while d > deadzone; potential-based
    # --- penalties on base motion: slow, smooth corrections ---
    w_speed: float = -0.5  # (|v| / lin_speed_scale)^2, measured base speed
    lin_speed_scale: float = 0.5  # [m/s]
    w_spin: float = -1.0  # (yaw rate / yaw_rate_scale)^2, measured
    yaw_rate_scale: float = 0.6  # [rad/s]
    w_yaw: float = -0.3  # (yaw / yaw_scale)^2: the base has no reason to turn away from where it spawned
    yaw_scale: float = 0.6  # [rad]
    w_action_rate: float = -0.3  # |a_t - a_{t-1}|^2, jerky commands
    w_action: float = -0.05  # |a|^2
    # --- end-effector tracking (the QP does the tracking, but the base can disturb it or leave the target near the
    # edge of the workspace) ---
    w_track: float = 0.5  # exp(-(e / sigma)^2), e = end-effector distance to the reference point
    track_sigma: float = 0.08  # [m]


def compute_rewards(
    cfg: RewardCfg,
    com_rho: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    centre_dist: torch.Tensor,
    centre_dist_prev: torch.Tensor,
    dt: float,
    base_vel: torch.Tensor,
    yaw_rate: torch.Tensor,
    yaw: torch.Tensor,
    track_err: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    terminated: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-env inputs (N,), except ``base_vel`` (N, 2) in the base frame and the actions (N, 3).

    Returns ``(reward (N,), weighted terms)``.
    """
    phi, phi_prev = -(centre_dist - cfg.centre_deadzone).clamp_min(0.0), -(centre_dist_prev - cfg.centre_deadzone).clamp_min(0.0)
    terms = {
        "com": cfg.w_com * torch.exp(-((com_rho / cfg.com_sigma) ** 2)),
        "tilt": cfg.w_tilt * (roll**2 + pitch**2) / cfg.tilt_scale**2,
        "centre": cfg.w_centre * torch.exp(-(((centre_dist - cfg.centre_deadzone).clamp_min(0.0) / cfg.centre_sigma) ** 2)),
        "progress": cfg.w_progress * ((phi - phi_prev) / dt).clamp(-1.0, 1.0),
        "speed": cfg.w_speed * base_vel.square().sum(-1) / cfg.lin_speed_scale**2,
        "spin": cfg.w_spin * (yaw_rate / cfg.yaw_rate_scale) ** 2,
        "yaw": cfg.w_yaw * (yaw / cfg.yaw_scale).clamp(-3.0, 3.0) ** 2,
        "action_rate": cfg.w_action_rate * (action - last_action).square().sum(-1),
        "action": cfg.w_action * action.square().sum(-1),
        "track": cfg.w_track * torch.exp(-((track_err / cfg.track_sigma) ** 2)),
        "termination": cfg.termination_penalty * terminated.float(),
    }
    return sum(terms.values()), terms
