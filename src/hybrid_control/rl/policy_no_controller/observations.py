"""Observation and base-frame helpers (pure torch, no Isaac imports).

"Base frame" below is the yaw-aligned frame centred on the platform footprint: x forward, y left, z up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObservationCfg:
    physical_params_in_actor: bool = True  # footprint, masses, terminal mass, COM height; the critic always sees them
    com_offset_in_actor: bool = True  # com_offset_norm (rho); the critic always sees it


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
    com_mechanism: torch.Tensor  # (N, 3)
    joint_pos: torch.Tensor  # (N, 3), rad/rad/m
    joint_vel: torch.Tensor  # (N, 3), rad/s, rad/s, m/s
    half_size: torch.Tensor  # (N, 2) half footprint length, width
    params: torch.Tensor  # (N, P) [length, width, base_mass, base_com_z, terminal mass, total_mass]
    last_action: torch.Tensor  # (N, 3)


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

    One means the projected CoM is at the footprint edge.
    """
    rel = to_base_frame(snap.yaw, snap.com_total[:, :2] - snap.centre_xy)
    return (rel.abs() / snap.half_size).amax(dim=-1)


def observation_terms(snap: Snapshot, cfg: ObservationCfg) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Named observation terms ``(actor, critic)``, each an ordered dict name -> (N, d); the critic sees the actor
    terms plus privileged ones. ``observe`` concatenates them in this order."""
    n = snap.yaw.shape[0]
    centre3 = torch.cat([snap.centre_xy, torch.zeros(n, 1, device=snap.yaw.device)], dim=-1)
    com_b = to_base_frame(snap.yaw, snap.com_total - centre3)

    actor = {
        "com_total": com_b,  # total COM [m] (x, y, height)
        "com_norm": com_b[:, :2] / snap.half_size,  # total COM as a fraction of the half footprint
        "base_lin_vel": to_base_frame(snap.yaw, snap.lin_vel_w)[:, :2],
        "base_ang_vel": to_base_frame(snap.yaw, snap.ang_vel_w),  # (wx, wy) = roll/pitch rates, wz = yaw rate
        "roll": snap.roll[:, None],
        "pitch": snap.pitch[:, None],
        "joint_pos": snap.joint_pos,
        "joint_vel": snap.joint_vel,
        "last_action": snap.last_action,
    }
    if cfg.physical_params_in_actor:
        actor["physical_params"] = snap.params
    if cfg.com_offset_in_actor:
        actor["com_offset_norm"] = com_offset_norm(snap)[:, None]

    critic = dict(actor)
    if not cfg.com_offset_in_actor:
        critic["com_offset_norm"] = com_offset_norm(snap)[:, None]
    critic["mechanism_com"] = to_base_frame(snap.yaw, snap.com_mechanism - centre3)
    if not cfg.physical_params_in_actor:
        critic["physical_params"] = snap.params
    return actor, critic


def observe(snap: Snapshot, cfg: ObservationCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(actor_obs, critic_obs)``."""
    actor, critic = observation_terms(snap, cfg)
    return torch.cat(list(actor.values()), dim=-1), torch.cat(list(critic.values()), dim=-1)
