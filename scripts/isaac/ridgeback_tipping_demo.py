"""Ridgeback-Franka with a FREE, velocity-driven base that tips when the total COM leaves the footprint.

The stock USD pins ``base_link`` to the world through three dummy joints (x, y, yaw), so it can
never tilt. Here the dummy bodies are deactivated right after spawn, which makes ``base_link`` a
floating root that rests on the ground by its collision geometry. Then:

* arm       : the Franka arm is NOT modified (stock link masses and inertias),
* footprint : base collision geometry is scaled to ``--footprint L W`` [m] and centred under the
              arm mount (the arm sits in the middle of the platform),
* base      : mass ``--base_mass`` [kg] at spawn; COM and solid-box inertia are set at runtime,
* payload   : ``--payload`` [kg] added at the gripper tool centre point (Panda rating: 3 kg),
* friction  : the platform has ``--base_friction`` (default 0) against the ground, i.e. ideal wheels,
* driving   : a virtual velocity actuator -- a saturated first-order velocity tracker -- applies a
              planar force and yaw torque to ``base_link`` at ground height. With the default zero
              command the base holds still while the arm reaches out, which is the failure case.

Failure criterion: the whole-robot COM leaves the platform footprint. Videos show three spheres:
blue = platform COM, green = arm COM (incl. payload), red = total COM, plus the platform as a
translucent box so the spheres are visible inside it.

Scenario: settle (arm folded) -> arm swings to a stretched pose -> hold. Use ``--case NAME`` for a
preset (see ``CASES``); individual flags override the preset.

Usage (from the Isaac Lab folder; add ``--extra video`` and ``--video`` to record):

    uv run --extra isaacsim --extra video python ~/hybrid-rl-mpc-control/scripts/isaac/ridgeback_tipping_demo.py \
        --case small_light --video
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

# name: (footprint L x W [m], base mass [kg], payload [kg])
# Tipping needs  arm_moment / total_mass > L/2.  The arm's moment about the footprint centre is about
# 0.65 kg*m folded and 4.70 kg*m stretched (7.0 with the 3 kg payload), so the presets choose L to be
# stable with the arm folded and to tip at different points of the reach.
CASES = {
    "reference": ((0.40, 0.40), 30.0, 3.0),  # large and heavy: stays upright
    "small_light": ((0.23, 0.23), 10.0, 3.0),  # small and light: tips early in the reach
    "small_heavy": ((0.17, 0.17), 40.0, 3.0),  # small but heavier: tips mid reach
    "big_light": ((0.54, 0.54), 3.0, 3.0),  # large but nearly massless: tips at full reach
    "no_payload": ((0.29, 0.29), 20.0, 0.0),  # same platform without payload: stays upright
    "with_payload": ((0.29, 0.29), 20.0, 3.0),  # the 3 kg payload alone makes it tip
}

OUT_DIR = Path(__file__).resolve().parents[2] / "out"  # <repo>/out, not the Isaac Lab folder
parser = argparse.ArgumentParser(description="Tip-over experiment with a free Ridgeback-Franka base.")
parser.add_argument("--case", choices=sorted(CASES), default=None, help="Preset footprint / base mass / payload.")
parser.add_argument("--footprint", type=float, nargs=2, default=None, metavar=("L", "W"),
                    help="Platform length (x) and width (y) [m]. Stock: 0.93 x 0.79.")
parser.add_argument("--base_mass", type=float, default=None, help="Mass of base_link [kg]. Stock: 165.")
parser.add_argument("--payload", type=float, default=None, help="Mass added at the gripper TCP [kg].")
parser.add_argument("--base_com_z", type=float, default=0.10, help="Height of the base COM above the base frame [m].")
parser.add_argument("--base_friction", type=float, default=0.0,
                    help="Platform-ground friction. 0 = ideal wheels (the velocity tracker supplies all traction).")
parser.add_argument("--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"),
                    help="Base velocity command in the base frame, applied once the arm is stretched.")
parser.add_argument("--drive", action="store_true",
                    help="Nominal-pose driving test: the arm stays folded and the base follows a gentle schedule "
                         "(forward, back, left, right, turn both ways). Ignores the reach and --cmd.")
parser.add_argument("--drive_speed", type=float, default=0.3, help="--drive: base speed [m/s] and turn rate [rad/s].")
parser.add_argument("--tau", type=float, default=0.15, help="Velocity tracking time constant [s].")
parser.add_argument("--max_force", type=float, default=1000.0, help="Planar force limit [N] (stock base: 1000).")
parser.add_argument("--settle", type=float, default=1.0, help="Seconds with the arm folded.")
parser.add_argument("--swing", type=float, default=3.0, help="Seconds to move the arm to the stretched pose.")
parser.add_argument("--hold", type=float, default=2.0, help="Seconds held stretched (driving if --cmd is nonzero).")
parser.add_argument("--post_tip", type=float, default=2.0, help="Seconds to keep recording after the tip.")
parser.add_argument("--video", action="store_true", help="Record offscreen to an mp4.")
parser.add_argument("--video_path", type=str, default=str(OUT_DIR / "ridgeback_tipping.mp4"), help="Default: <repo>/out/.")
parser.add_argument("--video_fps", type=int, default=30)
add_launcher_args(parser)
args_cli = parser.parse_args()
# Resolve preset, then explicit flags, then defaults.
_footprint, _mass, _payload = CASES[args_cli.case] if args_cli.case else ((0.40, 0.40), 30.0, 3.0)
args_cli.footprint = args_cli.footprint or list(_footprint)
args_cli.base_mass = _mass if args_cli.base_mass is None else args_cli.base_mass
args_cli.payload = _payload if args_cli.payload is None else args_cli.payload
if args_cli.video:
    args_cli.enable_cameras = True

import math

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from hybrid_control.sim_interfaces.isaac.ridgeback.params import BaseParams
from hybrid_control.sim_interfaces.isaac.ridgeback.robot import (
    MOUNT_X, PANDA_RATED_PAYLOAD, Q_STRETCHED, STOCK_HEIGHT, MassModel, build_robot_cfg, free_and_resize_base, tilt,
)

PARAMS = BaseParams(
    footprint=tuple(args_cli.footprint), base_mass=args_cli.base_mass, base_com_z=args_cli.base_com_z,
    payload=args_cli.payload,
)

TIP_ANGLE = 0.5  # roll/pitch [rad] beyond which the robot counts as tipped
# COM sphere colours (RGB): platform, arm, total
COM_MARKERS = {"platform": ((0.1, 0.4, 1.0), 0.045), "arm": ((0.1, 0.9, 0.2), 0.045), "total": ((1.0, 0.1, 0.1), 0.055)}


def spawn_com_markers():
    """Three visual-only spheres (no collision), moved every recorded frame."""
    for name, (color, radius) in COM_MARKERS.items():
        cfg = sim_utils.SphereCfg(
            radius=radius,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, emissive_color=tuple(0.5 * c for c in color), roughness=0.4
            ),
        )
        cfg.func(f"/World/Markers/com_{name}", cfg, translation=(0.0, 0.0, 3.0))


def apply_mass_model(robot: Articulation):
    """Box COM/inertia for the platform and a point payload at the TCP (arm untouched), via the shared MassModel."""
    model = MassModel(robot)
    model.set_params([PARAMS])
    return model.base_id, model.hand_id


def overlay(frame, t: float, tipped: bool, com_offset: float, half_length: float, phase: str = ""):
    """Draw the configuration and status text into the corners of an RGB frame."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=22)
    w, h = image.size
    pad = 18
    kw = dict(stroke_width=2, stroke_fill=(0, 0, 0))
    # top-left: configuration
    lines = [
        f"Base mass: {args_cli.base_mass:.1f} kg",
        f"Footprint: {args_cli.footprint[0]:.2f} x {args_cli.footprint[1]:.2f} m",
        f"Payload: {args_cli.payload:.1f} kg",
    ]
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * 36), line, font=font, fill=(255, 255, 255), **kw)
    # top-right: legend
    for i, (name, label) in enumerate((("platform", "platform COM"), ("arm", "arm COM"), ("total", "total COM"))):
        color = tuple(int(255 * c) for c in COM_MARKERS[name][0])
        tw = draw.textlength(label, font=small)
        y = pad + i * 32
        draw.text((w - pad - tw, y), label, font=small, fill=(255, 255, 255), **kw)
        draw.ellipse((w - pad - tw - 30, y + 3, w - pad - tw - 8, y + 25), fill=color, outline=(0, 0, 0), width=2)
    # bottom-left: time and status
    status, color = ("TIPPED", (255, 70, 70)) if tipped else ("STABLE", (120, 255, 120))
    draw.text((pad, h - pad), f"t = {t:4.1f} s   {status}", font=font, fill=color, anchor="ld", **kw)
    if phase:  # driving phase, centred above the bottom edge
        draw.text((w // 2, h - pad - 40), phase, font=font, fill=(255, 255, 255), anchor="mm", **kw)
    # bottom-right: the failure criterion
    ok = com_offset <= half_length
    draw.text((w - pad, h - pad), f"COM offset {com_offset:.3f} m / half-length {half_length:.3f} m",
              font=small, fill=(255, 255, 255) if ok else (255, 160, 70), anchor="rd", **kw)
    return np.asarray(image)


def run(sim: SimulationContext, robot: Articulation, camera=None):
    from isaaclab.sim.views import FrameView

    dt = sim.get_physics_dt()
    base_id = robot.find_bodies("base_link")[0][0]
    hand_id = robot.find_bodies("panda_hand")[0][0]
    arm_body_ids = [i for i, n in enumerate(robot.body_names) if n.startswith("panda_")]
    platform_body_ids = [i for i, n in enumerate(robot.body_names) if not n.startswith("panda_")]
    arm_ids, arm_names = robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
    finger_ids, _ = robot.find_joints("panda_finger_joint.*")
    q_fold = robot.data.default_joint_pos.torch[:, arm_ids].clone()
    q_str = q_fold.clone()
    for name, value in Q_STRETCHED.items():
        q_str[:, arm_names.index(name)] = value

    # Start from the configured default state (arm folded); without this the arm starts in the USD's
    # authored pose and swings in during the first half second, which tips light bases at t=0.
    robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()

    apply_mass_model(robot)
    masses = robot.data.body_mass.torch[0].clone()
    mass_total = masses.sum().item()
    mass_arm = masses[arm_body_ids].sum().item()
    inertia_z = robot.data.body_inertia.torch[0, base_id, 8].item() + 0.5 * (mass_total - args_cli.base_mass) * MOUNT_X**2
    half_length = 0.5 * args_cli.footprint[0]
    half_width = 0.5 * args_cli.footprint[1]
    print(f"[INFO] case={args_cli.case} footprint {args_cli.footprint[0]:.2f} x {args_cli.footprint[1]:.2f} m, "
          f"base {args_cli.base_mass:.1f} kg, payload {args_cli.payload:.1f} kg, arm {mass_arm - args_cli.payload:.1f} kg, "
          f"total {mass_total:.1f} kg")
    if args_cli.payload > PANDA_RATED_PAYLOAD:
        print(f"[WARN] payload {args_cli.payload:.1f} kg exceeds the Panda rating of {PANDA_RATED_PAYLOAD:.0f} kg")

    markers = {}
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([[2.2, -2.2, 1.3]], device=robot.device), torch.tensor([[0.3, 0.0, 0.45]], device=robot.device)
        )
        markers = {name: FrameView(f"/World/Markers/com_{name}", device=robot.device) for name in COM_MARKERS}
    frame_interval = max(1, round(1.0 / (args_cli.video_fps * dt)))
    frames, tipped_at, left_at = [], None, None
    t_swing, t_hold = args_cli.settle, args_cli.settle + args_cli.swing
    t_end = t_hold + args_cli.hold
    cmd = torch.tensor(args_cli.cmd, device=robot.device)
    schedule = None
    if args_cli.drive:
        v, speed = args_cli.drive_speed, args_cli.drive_speed
        # (duration [s], vx, vy, wz, description) in the base frame, as in ridgeback_demo.py
        schedule = [
            (2.0, 0.0, 0.0, 0.0, "standing still"),
            (2.0, v, 0.0, 0.0, "driving forward (+x)"),
            (2.0, -v, 0.0, 0.0, "driving backward (-x)"),
            (2.0, 0.0, v, 0.0, "sliding left (+y)"),
            (2.0, 0.0, -v, 0.0, "sliding right (-y)"),
            (2.0, 0.0, 0.0, speed, "turning counter-clockwise"),
            (2.0, 0.0, 0.0, -speed, "turning clockwise"),
        ]
        t_swing, t_hold = float("inf"), args_cli.settle  # arm stays in its nominal pose
        t_end = t_hold + sum(phase[0] for phase in schedule)
    phase_label, last_phase = "", None
    max_tilt, max_ratio = 0.0, 0.0
    step = 0
    while True:
        t = step * dt
        if t >= t_end and tipped_at is None:
            break
        if tipped_at is not None and t - tipped_at > args_cli.post_tip:
            break
        # -- arm: fold -> stretched, smoothstep
        s = 0.0 if schedule else min(max((t - t_swing) / max(args_cli.swing, 1e-6), 0.0), 1.0)
        s = s * s * (3 - 2 * s)
        robot.actuators.target_command.set_position_index(value=q_fold + s * (q_str - q_fold), joint_ids=arm_ids)
        robot.actuators.target_command.set_position_index(
            value=torch.full((robot.num_instances, len(finger_ids)), 0.0, device=robot.device), joint_ids=finger_ids
        )

        # -- base: velocity tracker -> planar force + yaw torque at ground height (off once tipped)
        pos = robot.data.body_com_pos_w.torch[0, base_id]
        vel = robot.data.body_com_lin_vel_w.torch[0, base_id]
        ang = robot.data.body_com_ang_vel_w.torch[0, base_id]
        roll, pitch, yaw = (a.item() for a in tilt(robot.data.body_quat_w.torch[0, base_id][None]))
        base_x, base_y = robot.data.body_pos_w.torch[0, base_id, :2].tolist()
        driving = t >= t_hold
        c, sn = math.cos(yaw), math.sin(yaw)
        v_des = torch.zeros(3, device=robot.device)
        w_des = 0.0
        if schedule and t >= t_hold:
            elapsed = t_hold
            for duration, vx, vy, wz, description in schedule:
                if t < elapsed + duration or description == schedule[-1][4]:
                    break
                elapsed += duration
            cmd = torch.tensor([vx, vy, wz], device=robot.device)
            phase_label = description
            if description != last_phase:
                last_phase = description
                print(f"[INFO] t={t:5.2f}s  base_link at x={base_x:+.2f} m, y={base_y:+.2f} m, yaw={yaw:+.2f} rad"
                      f" -> {description}")
        if driving:
            v_des[0], v_des[1] = c * cmd[0] - sn * cmd[1], sn * cmd[0] + c * cmd[1]
            w_des = cmd[2].item()
        force = torch.zeros(3, device=robot.device)
        torque = torch.zeros(3, device=robot.device)
        if tipped_at is None:
            force[:2] = mass_total * (v_des[:2] - vel[:2]) / args_cli.tau
            force[:2] *= min(1.0, args_cli.max_force / max(force[:2].norm().item(), 1e-9))
            torque[2] = inertia_z * (w_des - ang[2]) / args_cli.tau
        point = torch.tensor([[[pos[0].item(), pos[1].item(), 0.0]]], device=robot.device)
        robot.permanent_wrench_composer.reset()
        robot.permanent_wrench_composer.add_forces_and_torques_index(
            forces=force.view(1, 1, 3), torques=torque.view(1, 1, 3), positions=point, body_ids=[base_id], is_global=True
        )

        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        step += 1

        # -- COMs: platform group (base + lasers), arm group (Franka + payload), total
        body_com = robot.data.body_com_pos_w.torch[0]
        com_platform = (body_com[platform_body_ids] * masses[platform_body_ids, None]).sum(0) / masses[platform_body_ids].sum()
        com_arm = (body_com[arm_body_ids] * masses[arm_body_ids, None]).sum(0) / masses[arm_body_ids].sum()
        com_total = (body_com * masses[:, None]).sum(0) / mass_total
        # footprint centre (yaw frame, under the arm mount) and the total COM relative to it
        base_pos = robot.data.body_pos_w.torch[0, base_id]
        centre = base_pos[:2] + torch.tensor([c * MOUNT_X, sn * MOUNT_X], device=robot.device)
        rel = com_total[:2] - centre
        dx, dy = c * rel[0] + sn * rel[1], -sn * rel[0] + c * rel[1]  # base-frame offsets
        com_offset = max(abs(dx.item()) / half_length, abs(dy.item()) / half_width) * half_length
        t = step * dt
        max_tilt = max(max_tilt, abs(roll), abs(pitch))
        max_ratio = max(max_ratio, com_offset / half_length)
        if left_at is None and com_offset > half_length:
            left_at = t
            print(f"[INFO] t={t:5.2f}s  total COM left the footprint (offset {com_offset:.3f} m > {half_length:.3f} m)")
        if tipped_at is None and max(abs(roll), abs(pitch)) > TIP_ANGLE:
            tipped_at = t
            print(f"[INFO] t={t:5.2f}s  TIPPED (roll={roll:+.2f}, pitch={pitch:+.2f} rad)")
        if step % int(0.5 / dt) == 0:
            print(f"[INFO] t={t:5.2f}s arm={s:4.2f} roll={roll:+.2f} pitch={pitch:+.2f} "
                  f"COM offset={com_offset:.3f} m (limit {half_length:.3f}) "
                  f"arm moment about footprint centre={mass_arm * (com_arm[0] - centre[0]).item():.2f} kg*m")
        if camera is not None:
            for name, point_w in (("platform", com_platform), ("arm", com_arm), ("total", com_total)):
                markers[name].set_world_poses(positions=wp.from_torch(point_w.view(1, 3).contiguous(), dtype=wp.vec3f))
            camera.update(dt)
            if step % frame_interval == 0:
                frame = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
                frames.append(overlay(frame, t, tipped_at is not None, com_offset, half_length, phase_label))

    result = f"tipped={tipped_at is not None}" + (f" at t={tipped_at:.2f}s" if tipped_at is not None else "")
    result += f", com_left_footprint={left_at is not None}" + (f" at t={left_at:.2f}s" if left_at is not None else "")
    print(f"[RESULT] case={args_cli.case} {result}")
    if schedule:
        print(f"[RESULT] drive: max tilt {max_tilt:.3f} rad ({math.degrees(max_tilt):.1f} deg), "
              f"max COM offset {max_ratio * half_length:.3f} m of {half_length:.3f} m half-length "
              f"({100 * max_ratio:.0f}%), final base_link x={base_x:+.2f} m, y={base_y:+.2f} m, yaw={yaw:+.2f} rad")
    if frames:
        import imageio.v2 as imageio

        Path(args_cli.video_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args_cli.video_path, frames, fps=args_cli.video_fps)
        print(f"[INFO] Saved {len(frames)} frames to {args_cli.video_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([2.2, -2.2, 1.3], [0.3, 0.0, 0.45])
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)
        light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

        robot = Articulation(cfg=build_robot_cfg("/World/Robot", args_cli.base_mass))  # spawns the USD
        free_and_resize_base("/World/Robot", tuple(args_cli.footprint), args_cli.base_friction)

        camera = None
        if args_cli.video:
            # Imported here: camera.py imports pxr, which must load only after Kit has started.
            from isaaclab.sensors.camera import Camera, CameraCfg
            from isaaclab_physx.renderers import IsaacRtxRendererCfg

            spawn_com_markers()
            camera = Camera(
                cfg=CameraCfg(
                    prim_path="/World/RecordCam",
                    update_period=1.0 / args_cli.video_fps,
                    height=720,
                    width=1280,
                    data_types=["rgb"],
                    renderer_cfg=IsaacRtxRendererCfg(),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                    ),
                )
            )
        sim.reset()
        run(sim, robot, camera)


if __name__ == "__main__":
    main()
