"""Batched rotation helpers in torch (pure torch, no Isaac imports). Quaternions are xyzw like every Python interface."""

from __future__ import annotations

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched xyzw quaternion product a * b."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (= inverse for unit quaternions)."""
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by xyzw quaternions q (..., 4)."""
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_from_rotvec(r: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (..., 4) of the rotation vector r (..., 3) (axis * angle)."""
    angle = r.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(a/2)/a -> 1/2 - a^2/48 for small a
    k = torch.where(angle > 1e-6, torch.sin(half) / angle.clamp_min(1e-12), 0.5 - angle**2 / 48.0)
    return torch.cat([k * r, torch.cos(half)], dim=-1)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """Rotation vector (..., 3) of the unit quaternion q, with angle in [0, pi] (shortest rotation)."""
    q = torch.where(q[..., 3:] < 0, -q, q)
    xyz, w = q[..., :3], q[..., 3:]
    s = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, w)
    # angle/sin(a/2) -> 2 + a^2/12 for small a
    k = torch.where(s > 1e-6, angle / s.clamp_min(1e-12), 2.0 + angle**2 / 12.0)
    return k * xyz


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) of unit quaternions q (..., 4)."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def quat_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation angle [rad] in [0, pi] between unit quaternions a and b."""
    return quat_to_rotvec(quat_mul(a, quat_conj(b))).norm(dim=-1)
