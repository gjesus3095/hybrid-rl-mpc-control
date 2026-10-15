"""Video helpers for multi-env runs: COM markers, reference path/target markers, env labels projected into the frame.

COM markers: arm COM (green) and total COM (red) as spheres, plus a dark-red disc on the platform's top plate at the
total COM's (x, y), so it is visible whether the COM lies over the footprint even when the sphere is inside the opaque
platform. Reference markers: the reference points the policy observes (its lookahead times), coloured from yellow
(now) to green (furthest ahead). They are small, so the end effector stays visible.
"""

from __future__ import annotations

import torch
import warp as wp

import isaaclab.sim as sim_utils

from .robot import STOCK_HEIGHT

# name: (RGB, radius) -- legend colours for the COM spheres
COM_MARKERS = {"arm": ((0.1, 0.9, 0.2), 0.045), "total": ((1.0, 0.1, 0.1), 0.055)}
REF_NEAR, REF_FAR = (1.0, 0.85, 0.1), (0.15, 0.75, 0.15)  # yellow (current target) -> green (furthest lookahead)
PLATE_Z = STOCK_HEIGHT + 0.004  # just above the platform's top face


def _visual_material(color):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=tuple(0.5 * c for c in color), roughness=0.4)


class ComMarkers:
    """Visual-only (no collision) markers per env, moved every recorded frame."""

    def __init__(self, num_envs: int, device):
        from isaaclab.sim.views import FrameView

        for i in range(num_envs):
            for name, (color, radius) in COM_MARKERS.items():
                cfg = sim_utils.SphereCfg(radius=radius, visual_material=_visual_material(color))
                cfg.func(f"/World/Markers/env_{i:03d}/com_{name}", cfg, translation=(0.0, 0.0, 3.0))
            disc = sim_utils.CylinderCfg(radius=0.05, height=0.004, visual_material=_visual_material((0.5, 0.0, 0.0)))
            disc.func(f"/World/Markers/env_{i:03d}/com_total_plate", disc, translation=(0.0, 0.0, 3.0))
        names = [*COM_MARKERS, "total_plate"]
        self.views = {name: FrameView(f"/World/Markers/env_.*/com_{name}", device=device) for name in names}

    def update(self, env):
        plate = env.com_total.clone()
        plate[:, 2] = PLATE_Z
        for name, pos in (("arm", env.com_arm), ("total", env.com_total), ("total_plate", plate)):
            self.views[name].set_world_poses(positions=wp.from_torch(pos.contiguous(), dtype=wp.vec3f))


def ref_colors(n: int) -> list[tuple[float, float, float]]:
    """``n`` colours blending linearly from ``REF_NEAR`` to ``REF_FAR``."""
    return [tuple(a + (b - a) * j / max(n - 1, 1) for a, b in zip(REF_NEAR, REF_FAR)) for j in range(n)]


class ReferenceMarkers:
    """``n_points`` reference dots per env at the policy's lookahead times, yellow (now) to green (furthest)."""

    def __init__(self, num_envs: int, device, n_points: int):
        from isaaclab.sim.views import FrameView

        self.num_envs, self.n_points = num_envs, n_points
        dots = [sim_utils.SphereCfg(radius=0.025, visual_material=_visual_material(c)) for c in ref_colors(n_points)]
        for i in range(num_envs):
            for j, dot in enumerate(dots):
                dot.func(f"/World/Markers/ref_{i:03d}/point_{j:02d}", dot, translation=(0.0, 0.0, 3.0))
        self.view = FrameView("/World/Markers/ref_.*/point_[0-9]+", device=device)

    def update(self, env):
        """Move the dots to the reference lookahead of ``env`` (a ``BaseStabilizationEnv``) at its current time."""
        t = env.episode_length_buf.float() * env.dt
        points = env.traj.lookahead(t, env.lookahead) + env.env.origins[:, None]  # (N, n_points, 3), env-major
        self.view.set_world_poses(positions=wp.from_torch(points.reshape(-1, 3).contiguous(), dtype=wp.vec3f))


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


def _legend(draw, width, pad, font, kw, y0):
    """Colour legend of the markers in the top-right corner."""
    entries = [("arm COM", [COM_MARKERS["arm"][0]]), ("total COM (+ plate projection)", [COM_MARKERS["total"][0]]),
               ("reference lookahead (now -> ahead)", [REF_NEAR, REF_FAR])]
    for i, (label, colors) in enumerate(entries):
        tw = draw.textlength(label, font=font)
        y = y0 + i * 28
        draw.text((width - pad - tw, y), label, font=font, fill=(255, 255, 255), **kw)
        for j, rgb in enumerate(reversed(colors)):
            x = width - pad - tw - 26 - 20 * j
            draw.ellipse((x, y + 2, x + 18, y + 20), fill=tuple(int(255 * c) for c in rgb), outline=(0, 0, 0), width=2)


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
    _legend(draw, w, pad, tag, kw, y0=pad)
    n_tipped = int(env.tipped.sum())
    draw.text((pad, pad), f"{env.num_envs} randomized envs", font=font, fill=(255, 255, 255), **kw)
    draw.text((pad, h - pad), f"t = {t:4.1f} s   tipped {n_tipped}/{env.num_envs}", font=font,
              fill=(255, 255, 255), anchor="ld", **kw)
    return np.asarray(image)


def overlay_compare(frame, env, labels_px, t: float, group_titles, group_centres_px, divider_px, tip_times, rho=None,
                    group_subtitles=None):
    """Side-by-side comparison overlay: a title over each group of envs, a divider, per-env status tags.

    ``group_titles``: [(text, RGB)] per group; ``group_centres_px``: (G, 2) title anchors; ``divider_px``: x pixel of
    the line between the two groups; ``tip_times``: per-env time of tipping (nan if upright); ``rho``: optional
    per-env CoM offset as a fraction of the half footprint (1 = at the edge), drawn as a live bar under each tag;
    ``group_subtitles``: optional text per group under its title (e.g. the QP solver).
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    big = ImageFont.load_default(size=40)
    font = ImageFont.load_default(size=26)
    tag = ImageFont.load_default(size=19)
    w, h = image.size
    pad = 18
    kw = dict(stroke_width=2, stroke_fill=(0, 0, 0))
    draw.line((divider_px, 0, divider_px, h), fill=(255, 255, 255), width=3)
    for (text, color), (u, _) in zip(group_titles, group_centres_px):
        draw.text((u, 45), text, font=big, fill=color, anchor="mm", **kw)
    for text, (u, _) in zip(group_subtitles or [], group_centres_px):
        draw.text((u, 85), text, font=tag, fill=(255, 255, 255), anchor="mm", **kw)
    for i, (u, v) in enumerate(labels_px.tolist()):
        if u == 0.0 and v == 0.0:  # no label for this env (e.g. an unused dummy env)
            continue
        p = env.params[i]
        tipped = bool(env.tipped[i])
        status = f"TIPPED at {tip_times[i]:.1f} s" if tipped else "upright"
        lines = [status, f"{p.footprint[0]:.2f} x {p.footprint[1]:.2f} m", f"{p.base_mass:.0f} kg + {p.payload:.1f} kg payload"]
        for j, line in enumerate(lines):
            color = ((255, 90, 90) if tipped else (130, 255, 130)) if j == 0 else (255, 255, 255)
            draw.text((u, v + 22 * j), line, font=tag, fill=color, anchor="mm", **kw)
        if rho is not None:
            r = float(rho[i])
            frac = min(max(r, 0.0), 1.0)
            colour = (int(255 * min(2 * frac, 1.0)), int(255 * min(2 * (1 - frac), 1.0)), 40)  # green -> yellow -> red
            y0 = v + 22 * len(lines) + 4
            draw.text((u, y0 + 6), f"CoM offset {100 * r:.0f}%", font=tag, fill=(255, 255, 255), anchor="mm", **kw)
            draw.rectangle((u - 60, y0 + 20, u + 60, y0 + 32), fill=(30, 30, 30), outline=(255, 255, 255), width=1)
            draw.rectangle((u - 60, y0 + 20, u - 60 + int(120 * frac), y0 + 32), fill=colour)
    _legend(draw, w, pad, tag, kw, y0=pad + 80)
    draw.text((pad, h - pad), f"t = {t:4.1f} s", font=font, fill=(255, 255, 255), anchor="ld", **kw)
    return np.asarray(image)


__all__ = ["COM_MARKERS", "ComMarkers", "ReferenceMarkers", "overlay", "overlay_compare", "project"]
