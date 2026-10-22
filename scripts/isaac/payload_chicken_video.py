"""One long episode where only the payload at the TCP changes: the "Mercedes chicken" video.

The chicken's head (the arm's end effector) is sent to a fixed world pose and held there for the whole episode
(``--duration``, 15 s by default). Everything else stays fixed too (platform, base mass, the head target); only the
payload in the beak follows ``--payload`` (keyframes ``time:kg``, linear in between). A heavier head shifts the
whole-robot COM toward the platform edge, so the base policy moves the body to keep it over the footprint while the
arm controller keeps the head where it is: the body moves, the head does not. With ``--mode none`` the base stays
parked (QP only), the comparison without the policy.

The payload is drawn as a ball in the beak whose volume grows with its mass. The side panel shows the payload
schedule, and the base and head displacement since the head arrived (``--settle``, by default the blend-in time).

Usage (from the repo root, after the Isaac Lab setup in the root README):

    DISPLAY= PYTHONPATH=$PWD/src $ISAACLAB_PYTHON scripts/isaac/payload_chicken_video.py \\
        --checkpoint <run>/model_1250.pt --video_path out/payload_chicken.mp4
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from hybrid_control.rl.config import CONTROLLERS, saved_controller, saved_observation_cfg
from hybrid_control.sim_interfaces.isaac.ridgeback.params import VIDEO_LIGHT

OUT_DIR = Path(__file__).resolve().parents[2] / "out"
DEFAULT_SCHEDULE = ["0:0", "7.5:3", "15:0"]  # linear 0 -> 3 -> 0 kg over the 15 s
parser = argparse.ArgumentParser(description="One episode with a changing TCP payload (chicken skin).")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--title", type=str, default="CHICKEN HEAD STABILIZATION")
parser.add_argument("--mode", choices=["deterministic", "none"], default="deterministic",
                    help="none: zero base command (base parked), only the arm controller acts.")
parser.add_argument("--duration", type=float, default=15.0, help="Episode length [s].")
parser.add_argument("--payload", nargs="+", default=DEFAULT_SCHEDULE, metavar="T:KG",
                    help="Payload keyframes time [s]:mass [kg], linear in between and held after the last one. "
                         "Training range: 0..3 kg.")
parser.add_argument("--head_pos", type=float, nargs=3, default=[0.80, 0.0, 0.70], metavar=("X", "Y", "Z"),
                    help="Head target in the env frame (base spawn = origin, x forward) [m]: panda_hand for ik_qp, "
                         "the TCP for impedance.")
parser.add_argument("--orientation", choices=["forward", "spawn"], default="forward",
                    help="forward: beak ahead, 10 deg down, comb up (out of the training distribution, which points the "
                         "TCP down); spawn: keep the orientation the arm starts with.")
parser.add_argument("--footprint", type=float, nargs=2, default=[0.40, 0.40], metavar=("LENGTH", "WIDTH"))
parser.add_argument("--base_mass", type=float, default=30.0)
parser.add_argument("--base_com_z", type=float, default=0.10)
parser.add_argument("--settle", type=float, default=None,
                    help="Time [s] from which the base/head displacement is measured; default: the first payload "
                         "change, but not before the head target is blended in.")
parser.add_argument("--hold_tipped", type=float, default=1.5, help="Seconds a tipped robot stays on screen.")
parser.add_argument("--no_chicken", action="store_true", help="Plain robot instead of the chicken skin.")
parser.add_argument("--markers", action="store_true", help="Also show the arm/total COM markers.")
parser.add_argument("--qp_workers", type=int, default=0)
parser.add_argument("--controller", choices=CONTROLLERS, default=None,
                    help="Arm controller; default: the one the checkpoint was trained with (<run>/params/env.yaml).")
parser.add_argument("--view", choices=["all", "front", "left", "right", "back"], default="left",
                    help="One camera view (full size, default: the side) or the 2 x 2 grid.")
parser.add_argument("--cam_dist", type=float, default=2.0, help="Horizontal camera distance from the look point [m].")
parser.add_argument("--cam_height", type=float, default=0.45, help="Camera height above the look point [m].")
parser.add_argument("--light", type=float, default=VIDEO_LIGHT,
                    help=f"Distant light intensity (default {VIDEO_LIGHT:g}; training and the viewer scripts use 3000, "
                         "which overexposes the light ground in the camera images).")
parser.add_argument("--sky", type=float, default=0.0,
                    help="Dome light intensity for a blue sky instead of the black background (0 = off; try 50-150).")
parser.add_argument("--video_path", type=str, default=str(OUT_DIR / "payload_chicken.mp4"))
parser.add_argument("--video_fps", type=int, default=30)
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.controller = args_cli.controller or saved_controller(args_cli.checkpoint)
args_cli.enable_cameras = True

import math

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.rl.policy_with_controller.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.rl.policy_with_controller.trajectories import FIXED, Q_DOWN, TrajectoryCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import BaseParams, DomainRandomization
from hybrid_control.sim_interfaces.isaac.ridgeback.robot import TCP_Z
from hybrid_control.sim_interfaces.isaac.ridgeback.viz import ComMarkers, _visual_material
from hybrid_control.utils.torch_rotations import quat_apply, quat_from_rotvec, quat_mul

TILE = (800, 450)  # one camera view; the views form a 2 x 2 grid
PANEL_W = 480  # side panel (frame width 480 + 1600 = 2080, a multiple of 16)
FOCAL = 16.0  # [mm]
LOOK_OFFSET = (0.40, 0.0, 0.50)  # camera look point (env-local); the cameras stay fixed, the robot moves
VIEWS = [("front", (1.0, 0.0)), ("left", (0.0, 1.0)), ("right", (0.0, -1.0)), ("back", (-1.0, 0.0))]
if args_cli.view != "all":
    VIEWS = [v for v in VIEWS if v[0] == args_cli.view]
    TILE = (2 * TILE[0], 2 * TILE[1])
BALL_RADIUS = 0.04  # payload ball radius at 3 kg [m]; the volume scales with the mass
BALL_COLOR = (0.30, 0.30, 0.34)  # dark iron, so the ball stands out from the beak
SKY = (0.62, 0.74, 0.88)  # dome light colour (--sky), also the background
BALL_GAP = 0.01  # the ball hangs this far below the TCP [m] (drawn there; its mass sits at the TCP)
GOLD, CYAN, WHITE, GREY = (255, 215, 90), (90, 210, 255), (255, 255, 255), (180, 180, 180)


def parse_schedule(items: list[str]) -> tuple[np.ndarray, np.ndarray]:
    times, masses = zip(*(tuple(float(v) for v in item.split(":")) for item in items))
    times, masses = np.asarray(times), np.asarray(masses)
    if np.any(np.diff(times) < 0.0):
        raise ValueError(f"payload keyframe times must be non-decreasing, got {times.tolist()}")
    if masses.min() < 0.0:
        raise ValueError("payloads must be non-negative")
    return times, masses


SCHEDULE_T, SCHEDULE_KG = parse_schedule(args_cli.payload)


def payload_at(t: float) -> float:
    return float(np.interp(t, SCHEDULE_T, SCHEDULE_KG))


def head_quat(device) -> torch.Tensor:
    """Beak forward and 10 deg down, comb up: Q_DOWN pitched 80 deg toward +x, then yawed -90 deg about the TCP z axis
    (the ``--head_forward`` orientation of ``record_episodes_video.py``)."""
    pitch = quat_from_rotvec(torch.tensor([[0.0, math.radians(80.0), 0.0]], device=device))
    yaw = quat_from_rotvec(torch.tensor([[0.0, 0.0, math.radians(-90.0)]], device=device))
    return quat_mul(quat_mul(torch.tensor([Q_DOWN], device=device), pitch), yaw)[0]


def chart(draw, box, t_now, duration, series, y_max, label, font, kw):
    """Strip chart of ``series`` [(times, values, colour)] in ``box`` (x0, y0, x1, y1), with a cursor at ``t_now``."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(35, 35, 35), outline=GREY, width=1)
    draw.text((x0, y0 - 26), label, font=font, fill=GREY, **kw)

    def px(t, v):
        return x0 + (x1 - x0) * t / duration, y1 - (y1 - y0) * min(max(v / y_max, 0.0), 1.0)

    for times, values, colour in series:
        if len(times) > 1:
            draw.line([px(t, v) for t, v in zip(times, values)], fill=colour, width=3)
    xc = px(t_now, 0.0)[0]
    draw.line((xc, y0, xc, y1), fill=WHITE, width=1)


def compose(tiles, p, t, status, rho, payload, history, settle, controller):
    """Side panel + camera view(s)."""
    from PIL import Image, ImageDraw, ImageFont

    if len(tiles) == 1:
        grid = tiles[0]
    else:
        grid = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
    h = grid.shape[0]
    frame = np.concatenate([np.full((h, PANEL_W, 3), 25, dtype=np.uint8), grid.astype(np.uint8)], axis=1)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    big, font, tag = ImageFont.load_default(size=34), ImageFont.load_default(size=26), ImageFont.load_default(size=21)
    kw = dict(stroke_width=2, stroke_fill=(0, 0, 0))
    draw.line((PANEL_W, 0, PANEL_W, h), fill=WHITE, width=3)
    if len(tiles) > 1:
        draw.line((PANEL_W + TILE[0], 0, PANEL_W + TILE[0], h), fill=WHITE, width=2)
        draw.line((PANEL_W, TILE[1], frame.shape[1], TILE[1]), fill=WHITE, width=2)

    draw.text((20, 14), args_cli.title, font=big, fill=WHITE, **kw)
    draw.text((20, 60), f"t = {t:4.1f} s / {args_cli.duration:.0f} s", font=font, fill=WHITE, **kw)
    draw.text((20, 94), status, font=font, fill=(255, 90, 90) if status.startswith("TIPPED") else (130, 255, 130), **kw)
    draw.text((20, 144), "payload in the beak", font=tag, fill=GREY, **kw)
    draw.text((20, 168), f"{payload:.1f} kg", font=big, fill=GOLD, **kw)

    times = history["t"]
    kg_max = max(float(SCHEDULE_KG.max()), 1e-3)
    chart(draw, (20, 248, PANEL_W - 20, 338), t, args_cli.duration, [(times, history["payload"], GOLD)], kg_max * 1.1,
          f"payload, 0 .. {kg_max:.1f} kg", tag, kw)
    moved = max([5.0] + [100 * v for v in history["base"]])  # [cm]
    chart(draw, (20, 400, PANEL_W - 20, 520), t, args_cli.duration,
          [(times, [100 * v for v in history["base"]], CYAN), (times, [100 * v for v in history["head"]], WHITE)],
          moved * 1.1, f"displacement since t = {settle:.1f} s, 0 .. {moved:.0f} cm", tag, kw)
    if times and t >= settle:
        draw.text((20, 532), f"body  {100 * history['base'][-1]:5.1f} cm", font=font, fill=CYAN, **kw)
        draw.text((250, 532), f"head  {1000 * history['head'][-1]:4.0f} mm", font=font, fill=WHITE, **kw)

    draw.text((20, 590), "platform", font=tag, fill=GREY, **kw)
    draw.text((20, 614), f"{p.footprint[0]:.2f} x {p.footprint[1]:.2f} m, {p.base_mass:.0f} kg", font=tag, fill=WHITE, **kw)
    base = "parked (no policy)" if args_cli.mode == "none" else "RL policy"
    draw.text((20, 640), f"base: {base}, arm: {controller}", font=tag, fill=WHITE, **kw)
    # live CoM offset bar
    frac = min(max(rho, 0.0), 1.0)
    bar = (int(255 * min(2 * frac, 1.0)), int(255 * min(2 * (1 - frac), 1.0)), 40)
    x0, y0 = 20, h - 70
    draw.text((x0, y0 - 54), f"CoM offset {100 * rho:.0f}% of the half footprint", font=tag, fill=WHITE, **kw)
    draw.text((x0, y0 - 30), "(100% = at the edge)", font=tag, fill=GREY, **kw)
    draw.rectangle((x0, y0, x0 + 400, y0 + 22), fill=(30, 30, 30), outline=WHITE, width=2)
    draw.rectangle((x0, y0, x0 + int(400 * frac), y0 + 22), fill=bar)
    return np.asarray(image)


class PayloadBall:
    """Visual-only gold ball hanging from the TCP of env ``e``; its radius follows the payload (volume ~ mass)."""

    def __init__(self, robot, device):
        from isaaclab.sim.views import FrameView

        cfg = sim_utils.SphereCfg(radius=BALL_RADIUS, visual_material=_visual_material(BALL_COLOR))
        cfg.func("/World/Markers/payload_ball", cfg, translation=(0.0, 0.0, -3.0))
        self.view = FrameView("/World/Markers/payload_ball", device=device)
        self.robot, self.hand_id = robot, None  # body ids exist once the simulation has started
        self.offset = torch.tensor([0.0, 0.0, TCP_Z], device=device)

    def update(self, e: int, payload: float):
        if self.hand_id is None:
            self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        data = self.robot.data
        pos, quat = data.body_link_pos_w.torch[e : e + 1, self.hand_id], data.body_link_quat_w.torch[e : e + 1, self.hand_id]
        tcp = pos + quat_apply(quat, self.offset.expand_as(pos))
        scale = max(payload / 3.0, 0.0) ** (1.0 / 3.0) if payload > 1e-3 else 1e-4
        tcp[:, 2] -= BALL_GAP + BALL_RADIUS * scale
        self.view.set_world_poses(positions=wp.from_torch(tcp.contiguous(), dtype=wp.vec3f))
        self.view.set_scales(wp.from_torch(torch.full((1, 3), scale, device=tcp.device).contiguous(), dtype=wp.vec3f))


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        params = BaseParams(footprint=tuple(args_cli.footprint), base_mass=args_cli.base_mass,
                            base_com_z=args_cli.base_com_z, payload=payload_at(0.0))
        # Env 0 is an unused dummy (the first env of a scene is not reliably rendered); the episode runs in env 1.
        base_env = RidgebackDREnv([params, params], sim, spacing=8.0, num_cols=2, light_intensity=args_cli.light)
        if args_cli.sky > 0.0:  # adds to the env's distant light: keep it low or the scene overexposes
            sky = sim_utils.DomeLightCfg(intensity=args_cli.sky, color=SKY)
            sky.func("/World/Sky", sky)
        e = 1

        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        skin = None
        if not args_cli.no_chicken:
            from hybrid_control.sim_interfaces.isaac.ridgeback.chicken import ChickenSkin

            skin = ChickenSkin([f"/World/envs/env_{i}/Robot" for i in range(base_env.num_envs)])
        markers = ComMarkers(base_env.num_envs, sim.device) if args_cli.markers else None
        ball = PayloadBall(base_env.robot, sim.device)
        cameras = [
            Camera(
                cfg=CameraCfg(
                    prim_path=f"/World/RecordCam_{name}", update_period=0.0, height=TILE[1], width=TILE[0],
                    data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=FOCAL, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                    ),
                )
            )
            for name, _ in VIEWS
        ]
        base_env.start()
        cfg = StabilizationEnvCfg(
            controller=args_cli.controller, qp_workers=args_cli.qp_workers,
            observation=saved_observation_cfg(args_cli.checkpoint), trajectory=TrajectoryCfg(),
            episode_length_s=args_cli.duration,
            # the fixed dynamics above are used for every reset (the payload is then set by the schedule)
            domain_randomization=DomainRandomization(
                length=(params.footprint[0],) * 2, width=(params.footprint[1],) * 2, base_mass=(params.base_mass,) * 2,
                base_com_z=(params.base_com_z,) * 2, payload=(params.payload,) * 2),
        )
        env = BaseStabilizationEnv(base_env, cfg)
        env.auto_reset = False
        dev = env.device
        runner = OnPolicyRunner(env, make_runner_cfg(), log_dir=None, device=str(dev))
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=str(dev))

        # one fixed head target for the whole episode, blended in from the spawn pose
        ids = torch.tensor([e], device=dev)
        env._reset_idx(ids, dynamics=[params], kinds=torch.tensor([FIXED], device=dev))
        traj = env.traj
        traj.centre[e] = torch.tensor(args_cli.head_pos, device=dev)
        if args_cli.orientation == "forward":
            q = head_quat(dev)
            traj.centre_quat[e] = -q if float((q * traj.start_quat[e]).sum()) < 0 else q
        env.centre_dist_prev[ids] = env._centre_dist()[ids]
        obs = env._compute_obs()
        if args_cli.settle is None:
            changes = SCHEDULE_T[:-1][np.abs(np.diff(SCHEDULE_KG)) > 1e-6]  # start of the first changing segment
            settle = max(traj.cfg.blend_time, float(changes[0]) if len(changes) else 0.0)
        else:
            settle = args_cli.settle
        head_target = traj.centre[e].clone()
        target = base_env.origins[e] + torch.tensor(LOOK_OFFSET, device=dev)
        for cam, (_, (dx, dy)) in zip(cameras, VIEWS):
            eye = target + torch.tensor([args_cli.cam_dist * dx, args_cli.cam_dist * dy, args_cli.cam_height], device=dev)
            cam.set_world_poses_from_view(eye[None], target[None])

        import imageio.v2 as imageio

        Path(args_cli.video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args_cli.video_path, fps=args_cli.video_fps)
        history = {"t": [], "payload": [], "base": [], "head": []}
        base_ref = head_ref = None
        payload, t_tip, hold, max_head, max_base, max_rho = payload_at(0.0), None, None, 0.0, 0.0, 0.0
        try:
            with torch.inference_mode():
                while True:
                    t = float(env.episode_length_buf[e]) * env.dt
                    new_payload = payload_at(t)
                    if abs(new_payload - payload) > 1e-4:
                        payload = new_payload
                        env.set_payload(ids, [payload])
                        obs = env.get_observations()
                    a = torch.zeros(base_env.num_envs, 3, device=dev)
                    if args_cli.mode != "none" and t_tip is None:
                        a[e] = policy(obs[e : e + 1])[0]
                    obs, _, dones, extras = env.step(a)
                    t = float(env.episode_length_buf[e]) * env.dt
                    if t_tip is None and bool(dones[e]):
                        if not bool(extras["failed"][e]):
                            break  # time-out: the episode is complete
                        t_tip, hold = t, int(args_cli.hold_tipped / env.dt)
                    status = f"TIPPED at {t_tip:.1f} s" if t_tip is not None else "upright"

                    base_xy, head = base_env.base_pos[e, :2], env.arm.ee_pos_t()[e] - base_env.origins[e]
                    if base_ref is None and t >= settle:
                        base_ref, head_ref = base_xy.clone(), head.clone()
                    rho = float(base_env.com_offset[e] / base_env.half_length[e])
                    if int(env.episode_length_buf[e]) % int(round(1.0 / env.dt)) == 0:
                        err = 1000 * float((head - head_target).norm())
                        print(f"[t = {t:4.1f} s] payload {payload:.2f} kg, base {base_xy.tolist()} m, "
                              f"head error to target {err:.0f} mm, CoM offset {100 * rho:.0f}%", flush=True)
                    if base_ref is not None and t_tip is None:
                        history["t"].append(t)
                        history["payload"].append(payload)
                        history["base"].append(float((base_xy - base_ref).norm()))
                        history["head"].append(float((head - head_ref).norm()))
                        max_base, max_head = max(max_base, history["base"][-1]), max(max_head, history["head"][-1])
                        max_rho = max(max_rho, rho)
                    elif base_ref is None:  # before settling: plot the payload only
                        history["t"].append(t)
                        history["payload"].append(payload)
                        history["base"].append(0.0)
                        history["head"].append(0.0)

                    if markers is not None:
                        markers.update(base_env)
                    if skin is not None:
                        skin.update(base_env.robot)
                    ball.update(e, payload)
                    for cam in cameras:
                        cam.update(env.dt)
                    tiles = [cam.data.output["rgb"].torch[0, ..., :3].cpu().numpy() for cam in cameras]
                    writer.append_data(compose(tiles, base_env.params[e], t, status, rho, payload, history, settle,
                                               args_cli.controller))
                    if hold is not None:
                        hold -= 1
                        if hold <= 0:
                            break
        finally:
            writer.close()
        outcome = f"TIPPED at {t_tip:.1f} s" if t_tip is not None else f"completed {args_cli.duration:.0f} s upright"
        print(f"[RESULT] {outcome}; since t = {settle:.1f} s: max body displacement {100 * max_base:.1f} cm, "
              f"max head displacement {1000 * max_head:.0f} mm, max CoM offset {100 * max_rho:.0f}% of the half footprint")
        print(f"[INFO] Saved video to {args_cli.video_path}")
        env.close()


if __name__ == "__main__":
    main()
