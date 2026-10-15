"""Moving-mass CoM markers and optional streaming video. Import after launcher setup."""
import math
from pathlib import Path

import torch
import warp as wp
import isaaclab.sim as sim_utils


class DemoView:
    """GUI markers and an optional camera; no video dependency for headless checks.

    Red: total CoM. Green: upper mechanism CoM. Orange: total CoM projected
    vertically onto the ground (not onto a possibly tilted platform).
    """
    def __init__(self, env, args):
        # These visual-only prims need USD writes for both viewport and RTX rendering.
        from isaaclab.sim.views.usd_frame_view import UsdFrameView as FrameView
        self.env, self.args = env, args
        self.camera = self.writer = None
        self.next_frame = 0.0
        self.names = ('total', 'mechanism', 'projection')
        colors = ((1, .1, .1), (.1, .9, .2), (1, .65, .05))
        self.views = {}
        if not args.no_markers:
            for i in range(env.num_envs):
                for name, color in zip(self.names, colors):
                    cfg = sim_utils.SphereCfg(radius=.025, visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color, emissive_color=tuple(.3*x for x in color)))
                    cfg.func(f'/World/MovingMassMarkers/env_{i:04d}/{name}', cfg, translation=(0, 0, 2))
            self.views = {name: FrameView(f'/World/MovingMassMarkers/env_.*/{name}', device=env.sim.device)
                          for name in self.names}
        extent = max(float(env.origins[:, 0].abs().max()), float(env.origins[:, 1].abs().max()), 1.0)
        self.target = (0.0, 0.0, .35)
        self.eye = (0.0, -2.0*extent-3, 1.6*extent+2)
        env.sim.set_camera_view(self.eye, self.target)
        if args.video:
            from isaaclab.sensors.camera import Camera, CameraCfg
            from isaaclab_physx.renderers import IsaacRtxRendererCfg
            self.camera = Camera(cfg=CameraCfg(
                prim_path='/World/MovingMassCamera', update_period=0.0,
                height=720, width=1280, data_types=['rgb'], renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(focal_length=16, horizontal_aperture=20.955,
                                                clipping_range=(.05, 1000)),
            ))

    def start(self):
        if self.camera is not None:
            import imageio.v2 as imageio
            Path(self.args.video_path).parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(self.args.video_path, fps=self.args.video_fps)
            self.camera.set_world_poses_from_view(torch.tensor([self.eye], device=self.env.sim.device),
                                                 torch.tensor([self.target], device=self.env.sim.device))

    def update(self, time, labels):
        if self.views:
            projection = self.env.com_total.clone()
            projection[:, 2] = .025
            for name, pos in zip(self.names, (self.env.com_total, self.env.com_mechanism, projection)):
                self.views[name].set_world_poses(positions=wp.from_torch(pos.contiguous(), dtype=wp.vec3f))
        if self.camera is not None and time + 1e-9 >= self.next_frame:
            self.env.sim.render()
            self.camera.update(self.env.dt, force_recompute=True)
            frame = self.camera.data.output['rgb'].torch[0, ..., :3].cpu().numpy().copy()
            self.writer.append_data(self._overlay(frame, time, labels))
            self.next_frame = (math.floor(time * self.args.video_fps + 1e-6) + 1) / self.args.video_fps

    def _overlay(self, frame, time, labels):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=20)
        draw.text((16, 12), f'Moving mass | {self.args.scenario} | t={time:.2f}s', font=font, fill='white',
                  stroke_width=2, stroke_fill='black')
        draw.text((16, 40), 'Red: total CoM  Green: mechanism CoM  Orange: ground projection',
                  font=font, fill='white', stroke_width=2, stroke_fill='black')
        # Project current base positions so labels follow translating platforms.
        points = self.env.robot.data.body_pos_w.torch[:, self.env.base_id].clone()
        points[:, 2] += .9
        eye = points.new_tensor(self.eye)
        forward = points.new_tensor(self.target) - eye
        forward /= forward.norm()
        right = torch.linalg.cross(forward, points.new_tensor([0, 0, 1]))
        right /= right.norm()
        up = torch.linalg.cross(right, forward)
        relative = points-eye
        depth = relative @ forward
        focal = 16/20.955 * image.width
        xy = torch.stack((image.width/2 + focal*(relative@right)/depth,
                          image.height/2 - focal*(relative@up)/depth), -1)
        for i, (x, y) in enumerate(xy.tolist()):
            if depth[i] <= 0 or not (0 <= x < image.width and 0 <= y < image.height):
                continue
            p = self.env.params[i]
            rho = float(self.env.com_offset[i]/self.env.half_length[i])
            status = 'TIPPED' if self.env.tipped[i] else 'upright'
            text = (f'{labels[i]}: {status}\n{p.footprint[0]:.2f}x{p.footprint[1]:.2f} m | '
                    f'base {p.base_mass:.1f} kg, upper {p.terminal_mass+p.carriage_mass+p.gimbal_mass+p.mast_mass:.1f} kg\nCoM offset {rho:.2f} x half-footprint')
            draw.multiline_text((x, y), text, font=font, fill='salmon' if self.env.tipped[i] else 'white',
                                anchor='mm', align='center', stroke_width=2, stroke_fill='black')
        return np.asarray(image)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
