"""Video helpers for multi-env runs: COM spheres per env, env labels projected into the frame, overlay."""

from __future__ import annotations

import torch
import warp as wp

import isaaclab.sim as sim_utils

# name: (RGB, radius) -- platform, arm, total
COM_MARKERS = {"platform": ((0.1, 0.4, 1.0), 0.045), "arm": ((0.1, 0.9, 0.2), 0.045), "total": ((1.0, 0.1, 0.1), 0.055)}


class ComMarkers:
    """Three visual-only spheres per env (no collision), moved every recorded frame."""

    def __init__(self, num_envs: int, device):
        from isaaclab.sim.views import FrameView

        for i in range(num_envs):
            for name, (color, radius) in COM_MARKERS.items():
                cfg = sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color, emissive_color=tuple(0.5 * c for c in color), roughness=0.4
                    ),
                )
                cfg.func(f"/World/Markers/env_{i}/com_{name}", cfg, translation=(0.0, 0.0, 3.0))
        self.views = {name: FrameView(f"/World/Markers/env_.*/com_{name}", device=device) for name in COM_MARKERS}

    def update(self, env):
        for name, com in (("platform", env.com_platform), ("arm", env.com_arm), ("total", env.com_total)):
            self.views[name].set_world_poses(positions=wp.from_torch(com.contiguous(), dtype=wp.vec3f))


def project(points: torch.Tensor, eye, target, width: int, height: int, focal: float = 24.0, aperture: float = 20.955):
    """Pixel coordinates (N, 2) and in-front mask of world points seen by a z-up look-at pinhole camera."""
    eye_t = torch.tensor(eye, dtype=torch.float32, device=points.device)
    fwd = torch.tensor(target, dtype=torch.float32, device=points.device) - eye_t
    fwd = fwd / fwd.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 0.0, 1.0], device=points.device))
    right = right / right.norm()
    up = torch.linalg.cross(right, fwd)
    rel = points - eye_t
    depth = rel @ fwd
    f_px = focal / aperture * width
    u = width / 2 + f_px * (rel @ right) / depth
    v = height / 2 - f_px * (rel @ up) / depth
    return torch.stack([u, v], dim=-1), depth > 0


def overlay(frame, env, labels_px, t: float):
    """Per-env parameter tags next to each robot, legend and status line. Returns an RGB array."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=26)
    tag = ImageFont.load_default(size=17)
    w, h = image.size
    pad = 18
    kw = dict(stroke_width=2, stroke_fill=(0, 0, 0))
    for i, (u, v) in enumerate(labels_px.tolist()):
        p = env.params[i]
        tipped = bool(env.tipped[i])
        lines = [
            f"env {i}  {'TIPPED' if tipped else 'stable'}",
            f"{p.footprint[0]:.2f} x {p.footprint[1]:.2f} m  base {p.base_mass:.0f} kg",
            f"payload {p.payload:.1f} kg  com_z {p.base_com_z:.2f} m",
        ]
        for j, line in enumerate(lines):
            color = ((255, 90, 90) if tipped else (130, 255, 130)) if j == 0 else (255, 255, 255)
            draw.text((u, v + 19 * j), line, font=tag, fill=color, anchor="mm", **kw)
    for i, (name, label) in enumerate((("platform", "platform COM"), ("arm", "arm COM"), ("total", "total COM"))):
        color = tuple(int(255 * c) for c in COM_MARKERS[name][0])
        tw = draw.textlength(label, font=tag)
        y = pad + i * 28
        draw.text((w - pad - tw, y), label, font=tag, fill=(255, 255, 255), **kw)
        draw.ellipse((w - pad - tw - 26, y + 2, w - pad - tw - 8, y + 20), fill=color, outline=(0, 0, 0), width=2)
    n_tipped = int(env.tipped.sum())
    draw.text((pad, pad), f"{env.num_envs} randomized envs", font=font, fill=(255, 255, 255), **kw)
    draw.text((pad, h - pad), f"t = {t:4.1f} s   tipped {n_tipped}/{env.num_envs}", font=font,
              fill=(255, 255, 255), anchor="ld", **kw)
    return np.asarray(image)


__all__ = ["COM_MARKERS", "ComMarkers", "overlay", "project"]
