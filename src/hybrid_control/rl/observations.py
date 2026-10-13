"""Observation and base-frame helpers (pure torch, no Isaac imports).

"Base frame" below is the yaw-aligned frame centred on the platform footprint: x forward, y left, z up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObservationCfg:
    lookahead: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5)  # reference lookahead times [s]
    physical_params_in_actor: bool = True  # footprint, masses, payload, COM height; the critic always sees them


@dataclass
class Snapshot:
    """Per-env state tensors (all in env-local world coordinates unless noted), the input to ``observe``."""

    yaw: torch.Tensor  # (N,)
    roll: torch.Tensor  # (N,)
    pitch: torch.Tensor  # (N,)
    centre_xy: torch.Tensor  # (N, 2) footprint centre
    lin_vel_w: torch.Tensor  # (N, 3) base velocity
    ang_vel_w: torch.Tensor  # (N, 3)
    com_total: torch.Tensor  # (N, 3)
    com_arm: torch.Tensor  # (N, 3)
    ee_pos: torch.Tensor  # (N, 3)
    arm_q: torch.Tensor  # (N, 7)
    arm_qd: torch.Tensor  # (N, 7)
    ref: torch.Tensor  # (N, K, 3) reference lookahead
    traj_centre: torch.Tensor  # (N, 3) centre of the reference trajectory
    half_size: torch.Tensor  # (N, 2) half footprint length, width
    params: torch.Tensor  # (N, P) [length, width, base_mass, base_com_z, payload, total_mass]
    last_action: torch.Tensor  # (N, 3)
    time_fraction: torch.Tensor  # (N,) episode progress in [0, 1]


def to_base_frame(yaw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors ``v`` (N, ..., 2 or 3) into the yaw frame (z is passed through)."""
    shape = (-1,) + (1,) * (v.dim() - 2)
    c, s = torch.cos(yaw).reshape(shape), torch.sin(yaw).reshape(shape)
    x, y = v[..., 0], v[..., 1]
    out = [c * x + s * y, -s * x + c * y]
    if v.shape[-1] == 3:
        out.append(v[..., 2])
    return torch.stack(out, dim=-1)


def com_offset_norm(snap: Snapshot) -> torch.Tensor:
    """Max-norm distance of the total COM from the footprint centre, as a fraction of the half footprint (1 = edge).

    Matches ``RidgebackDREnv.com_offset / half_length``.
    """
    rel = to_base_frame(snap.yaw, snap.com_total[:, :2] - snap.centre_xy)
    return (rel.abs() / snap.half_size).amax(dim=-1)


def observe(snap: Snapshot, cfg: ObservationCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(actor_obs, critic_obs)``."""
    n = snap.yaw.shape[0]
    centre3 = torch.cat([snap.centre_xy, torch.zeros(n, 1, device=snap.yaw.device)], dim=-1)
    ref_b = to_base_frame(snap.yaw, snap.ref - centre3[:, None]).flatten(1)
    traj_centre_b = to_base_frame(snap.yaw, snap.traj_centre - centre3)
    com_b = to_base_frame(snap.yaw, snap.com_total - centre3)
    com_norm = com_b[:, :2] / snap.half_size
    v_b = to_base_frame(snap.yaw, snap.lin_vel_w)[:, :2]
    w_b = to_base_frame(snap.yaw, snap.ang_vel_w)  # (wx, wy) = roll/pitch rates, wz = yaw rate
    ee_b = to_base_frame(snap.yaw, snap.ee_pos - centre3)

    parts = [
        ref_b,  # reference lookahead
        traj_centre_b,  # centre of the whole reference trajectory
        com_b, com_norm,  # total COM: metres (x, y, height) and fraction of the half footprint
        v_b, w_b,  # base velocity and angular rate (roll, pitch, yaw rate)
        snap.roll[:, None], snap.pitch[:, None],
        ee_b, snap.arm_q, snap.arm_qd,
        snap.last_action,
    ]
    if cfg.physical_params_in_actor:
        parts.append(snap.params)
    actor = torch.cat(parts, dim=-1)

    arm_com_b = to_base_frame(snap.yaw, snap.com_arm - centre3)
    extras = [com_offset_norm(snap)[:, None], arm_com_b, snap.time_fraction[:, None]]
    if not cfg.physical_params_in_actor:
        extras.append(snap.params)
    critic = torch.cat([actor] + extras, dim=-1)
    return actor, critic
