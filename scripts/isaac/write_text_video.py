"""Let the trained base policy and its arm controller write a text ("MPCRL2026") on a wall and record a video of it.

The TCP follows a pen path on a vertical plane in front of the robot (``hybrid_control.rl.evaluation.text_path``: one stroke
per letter at constant speed, lift / hop / lower between letters, then the arm parks beyond the text). The policy only sees
the reference like in training (the lookahead of the TCP pose) and drives the base; the arm controller tracks the reference.
Ink dots are drawn on a white board where the TCP really is while the pen is down, so the text shows the actual tracking.

Video layout: a side panel (time, phase, text written so far, platform, tracking error, COM offset bar) next to four cameras
that follow the hand (writing view, side, top, wide). When the arm parks after the last letter, one big camera flies to a
full-width view of the board and holds it, so the written text is clearly visible at the end.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    $ISAACLAB_PYTHON scripts/isaac/write_text_video.py --checkpoint logs/base_policy/<run>/model_<n>.pt
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from hybrid_control.rl.config import CONTROLLERS, saved_controller, saved_observation_cfg

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Write a text with the policy and record a video.")
parser.add_argument("--checkpoint", type=str, default=str(ROOT / "logs/base_policy/impedance_quintic_run1/model_1200.pt"))
parser.add_argument("--text", type=str, default="MPCRL2026")
parser.add_argument("--controller", choices=CONTROLLERS, default=None,
                    help="Arm controller; default: the one the checkpoint was trained with (<run>/params/env.yaml).")
parser.add_argument("--plane", choices=["table", "wall"], default="table",
                    help="Pen plane: horizontal with the TCP pointing down (the orientation the policy trained on) or a "
                         "vertical wall the TCP faces (the policy keeps the base under the pen and lags sideways: poor text).")
parser.add_argument("--z_table", type=float, default=0.75, help="Height of the horizontal pen plane [m] (a sweep found 0.65-0.75 best).")
parser.add_argument("--no_pen", action="store_true", help="Do not draw the pen in the gripper.")
parser.add_argument("--footprint", type=float, nargs=2, default=(0.45, 0.40), help="Platform length and width [m].")
parser.add_argument("--base_mass", type=float, default=30.0, help="Platform mass [kg].")
parser.add_argument("--payload", type=float, default=1.5, help="Extra mass at the gripper [kg].")
parser.add_argument("--write_speed", type=float, default=0.15, help="Pen-down speed [m/s].")
parser.add_argument("--reveal_time", type=float, default=8.0, help="Seconds the final full-text view is held after the arm has parked.")
parser.add_argument("--spacing", type=float, default=8.0)
parser.add_argument("--qp_workers", type=int, default=2)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--video_path", type=str, default=str(ROOT / "out/mpcrl2026/mpcrl2026.mp4"))
parser.add_argument("--video_fps", type=int, default=30)
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.controller = args_cli.controller or saved_controller(args_cli.checkpoint)
args_cli.enable_cameras = True

import json

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.rl.evaluation.scenarios import REF_DT, ScenarioSet
from hybrid_control.rl.evaluation.scripted import ScriptedTrajectories
from hybrid_control.rl.evaluation.text_path import PHASES, TextCfg, build_text_path
from hybrid_control.rl.policy_with_controller.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.params import BaseParams
from hybrid_control.sim_interfaces.isaac.ridgeback.viz import _visual_material
from hybrid_control.utils.torch_rotations import quat_angle

TILE = (800, 450)  # one follow camera; the four views form a 2 x 2 grid
BIG = (1600, 900)  # the reveal camera fills the grid area
PANEL_W = 480
FOCAL = 16.0  # [mm]
# (name, camera offset from the hand [m], env-local), per pen plane
VIEWS_BY_PLANE = {
    "wall": [("writing view", (-1.4, 0.9, 0.35)), ("side view", (0.0, 2.2, 0.3)), ("top view", (-0.5, 0.0, 2.6)),
             ("wide view", (-3.2, -0.8, 1.6))],
    "table": [("writing view", (-1.0, 0.7, 0.8)), ("side view", (0.0, 2.0, 0.6)), ("top view", (-0.3, 0.0, 2.4)),
              ("wide view", (-3.0, -0.8, 1.8))],
}
VIEWS = VIEWS_BY_PLANE[args_cli.plane]
FOLLOW_SMOOTH = 0.2  # low-pass factor per frame on the followed point
INK_RADIUS, INK_COLOR = 0.007, (0.0, 0.15, 0.85)
PEN_LENGTH, PEN_RADIUS, PEN_COLOR = 0.16, 0.006, (0.12, 0.12, 0.14)  # visual pen along the hand z axis, its tip at the TCP
BOARD_COLOR = (0.97, 0.97, 0.97)
REVEAL_OFFSET = {"wall": (-1.9, 0.0, 0.15), "table": (-1.2, 0.0, 1.5)}  # final camera relative to the middle of the board [m]


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def compose(view, title, text, written, status, failed, t, phase, params, err_cm, rho, big):
    """Side panel + the camera view(s): a 2 x 2 grid of tiles, or one big image (``big=True``)."""
    from PIL import Image, ImageDraw, ImageFont

    if big:
        grid = view
    else:
        grid = np.concatenate([np.concatenate(view[:2], axis=1), np.concatenate(view[2:], axis=1)], axis=0)
    h = grid.shape[0]
    frame = np.concatenate([np.full((h, PANEL_W, 3), 25, dtype=np.uint8), grid.astype(np.uint8)], axis=1)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    huge, bigf, font, tag = (ImageFont.load_default(size=s) for s in (46, 34, 26, 21))
    kw = dict(stroke_width=2, stroke_fill=(0, 0, 0))
    white, grey, accent = (255, 255, 255), (150, 150, 150), (255, 215, 90)
    draw.line((PANEL_W, 0, PANEL_W, h), fill=white, width=3)
    if not big:
        for i, (name, _) in enumerate(VIEWS):
            draw.text((PANEL_W + (i % 2) * TILE[0] + 14, (i // 2) * TILE[1] + 10), name, font=tag, fill=white, **kw)
        draw.line((PANEL_W + TILE[0], 0, PANEL_W + TILE[0], h), fill=white, width=2)
        draw.line((PANEL_W, TILE[1], frame.shape[1], TILE[1]), fill=white, width=2)
    else:
        draw.text((PANEL_W + BIG[0] / 2, 40), text, font=huge, fill=white, anchor="mm", **kw)
    draw.text((20, 14), title, font=bigf, fill=white, **kw)
    draw.text((20, 64), f"t = {t:5.1f} s", font=font, fill=white, **kw)
    draw.text((20, 100), status, font=font, fill=(255, 90, 90) if failed else (130, 255, 130), **kw)
    draw.text((20, 152), "written so far", font=tag, fill=grey, **kw)
    draw.text((20, 178), text, font=bigf, fill=grey, **kw)  # the whole text dim, the written part on top
    draw.text((20, 178), text[:written], font=bigf, fill=accent, **kw)
    draw.text((20, 236), "phase", font=tag, fill=grey, **kw)
    draw.text((20, 260), phase, font=font, fill=white, **kw)
    draw.text((20, 312), "platform", font=tag, fill=grey, **kw)
    draw.text((20, 336), f"{params.footprint[0]:.2f} x {params.footprint[1]:.2f} m, {params.base_mass:.0f} kg", font=tag, fill=white, **kw)
    draw.text((20, 362), f"payload {params.payload:.1f} kg", font=tag, fill=white, **kw)
    draw.text((20, 406), "tracking error (TCP)", font=tag, fill=grey, **kw)
    draw.text((20, 430), f"{err_cm:.1f} cm", font=font, fill=white, **kw)
    frac = min(max(rho, 0.0), 1.0)
    bar = (int(255 * min(2 * frac, 1.0)), int(255 * min(2 * (1 - frac), 1.0)), 40)
    x0, y0 = 20, h - 90
    draw.text((x0, y0 - 54), f"CoM offset {100 * rho:.0f}% of the half footprint", font=tag, fill=white, **kw)
    draw.text((x0, y0 - 30), "(100% = at the edge)", font=tag, fill=grey, **kw)
    draw.rectangle((x0, y0, x0 + 400, y0 + 22), fill=(30, 30, 30), outline=white, width=2)
    draw.rectangle((x0, y0, x0 + int(400 * frac), y0 + 22), fill=bar)
    return np.asarray(image)


def main():
    path = build_text_path(TextCfg(controller=args_cli.controller, text=args_cli.text, write_speed=args_cli.write_speed, plane=args_cli.plane,
                                   z_table=args_cli.z_table))
    cfg = path.cfg
    n_table = len(path.poses)
    t_park = float(np.argmax(path.phase == PHASES.index("park"))) * REF_DT
    print(f"[TEXT] '{cfg.text}': {path.duration:.1f} s of writing, pen path on the {cfg.plane} plane, "
          f"{len(path.strokes)} strokes", flush=True)

    params = BaseParams(tuple(args_cli.footprint), args_cli.base_mass, 0.10, args_cli.payload)
    scenario = ScenarioSet("text", np.stack([path.poses, path.poses]), np.full(2, path.duration, np.float32),
                           np.zeros((2, 3), np.float32), np.array([[params.base_mass, params.base_com_z, params.payload]] * 2, np.float32))

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        # Env 0 is an unused dummy (the first env of a scene is not reliably rendered); the robot writes in env 1.
        base_env = RidgebackDREnv([params, params], sim, spacing=args_cli.spacing, num_cols=2)
        dev = sim.device
        origin = base_env.origins[1].clone()

        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab.sim.views import FrameView
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        # whiteboard behind the ink (visual only) and the ink dots, moved onto the wall one by one while writing
        board_y, board_c, board_w, board_h = path.board  # board_c: z of the middle of the wall board / x of the table board
        if cfg.plane == "wall":
            board_size, board_local = (0.01, board_w, board_h), (cfg.x_wall + 0.012, board_y, board_c)
            centre_local = (cfg.x_wall, board_y, board_c)
        else:
            board_size, board_local = (board_h, board_w, 0.01), (board_c, board_y, cfg.z_table - 0.012)
            centre_local = (board_c, board_y, cfg.z_table)
        board = sim_utils.CuboidCfg(size=board_size, visual_material=_visual_material(BOARD_COLOR))
        board.func("/World/Markers/board", board, translation=tuple((origin + torch.tensor(board_local, device=dev)).tolist()))
        n_dots = int(path.pen_down.sum() / 8) + 40  # one dot every second policy step
        dot = sim_utils.SphereCfg(radius=INK_RADIUS, visual_material=_visual_material(INK_COLOR))
        for i in range(n_dots):
            dot.func(f"/World/Markers/ink/dot_{i:05d}", dot, translation=(0.0, 0.0, -5.0))
        ink_view = FrameView("/World/Markers/ink/dot_[0-9]+", device=dev)
        ink_pos = torch.zeros(n_dots, 3, device=dev)
        ink_pos[:, 2] = -5.0

        def make_camera(name, size):
            return Camera(cfg=CameraCfg(
                prim_path=f"/World/RecordCam_{name}", update_period=0.0, height=size[1], width=size[0], data_types=["rgb"],
                renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(focal_length=FOCAL, focus_distance=400.0, horizontal_aperture=20.955,
                                                 clipping_range=(0.1, 1.0e5))))

        rig = [make_camera(name.split()[0], TILE) for name, _ in VIEWS]
        reveal = make_camera("reveal", BIG)
        if not args_cli.no_pen:
            # a visual-only pen (no collision, no mass) attached to the hand of the writing robot, its tip at the TCP
            from pxr import Usd

            from hybrid_control.sim_interfaces.isaac.ridgeback.robot import franka_tcp_offset

            stage = sim_utils.get_current_stage()
            hands = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_1"))
                     if prim.GetName() == "panda_hand" and not prim.GetPath().pathString.endswith("Looks")]
            hand = hands[0].GetPath().pathString
            tip = np.asarray(franka_tcp_offset().position, dtype=float)
            body = sim_utils.CylinderCfg(radius=PEN_RADIUS, height=PEN_LENGTH, axis="Z", visual_material=_visual_material(PEN_COLOR))
            body.func(f"{hand}/pen_body", body, translation=tuple(tip - np.array([0.0, 0.0, PEN_LENGTH / 2])))
            nib = sim_utils.SphereCfg(radius=PEN_RADIUS * 1.2, visual_material=_visual_material(INK_COLOR))
            nib.func(f"{hand}/pen_tip", nib, translation=tuple(tip))
            print(f"[TEXT] pen attached to {hand}, tip at {tip.round(3).tolist()} in the hand frame", flush=True)
        base_env.start()
        obs_cfg = saved_observation_cfg(args_cli.checkpoint)
        env = BaseStabilizationEnv(base_env, StabilizationEnvCfg(controller=args_cli.controller, qp_workers=args_cli.qp_workers,
                                                                 episode_length_s=3600.0, observation=obs_cfg), seed=args_cli.seed)
        env.auto_reset = False
        env.traj = ScriptedTrajectories(2, env.device, env.scfg.trajectory)
        env.traj.load(scenario)
        runner = OnPolicyRunner(env, make_runner_cfg(), log_dir=None, device=str(env.device))
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=str(env.device))

        import imageio.v2 as imageio

        video_path = Path(args_cli.video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=args_cli.video_fps)
        total_steps = int(round((path.duration + args_cli.reveal_time) / env.dt))
        log = {k: [] for k in ("t", "ref_pos", "ee_pos", "pos_err", "ori_err", "rho", "margin_m", "base_pos", "yaw", "pen_down", "action")}
        failed_at, follow, flight_start, dots = None, None, None, 0
        wall_centre = torch.tensor(centre_local, device=dev)
        final_eye = wall_centre + torch.tensor(REVEAL_OFFSET[cfg.plane], device=dev)
        try:
            with torch.inference_mode():  # the env keeps inference tensors, so resets must run in it too
                env._reset_idx(torch.arange(2, device=dev), dynamics=[params, params])
                env._obs = env._compute_obs()
                obs = env.get_observations()
                for step in range(total_steps):
                    a = torch.zeros(2, 3, device=dev)
                    a[1] = policy(obs[1:2])[0]
                    obs, _, _, _ = env.step(a)
                    t = float(env.episode_length_buf[1]) * env.dt
                    idx = min(int(round(t / REF_DT)), n_table - 1)
                    ee_pos, ee_quat = env.arm.ee_pose_t()
                    tip = ee_pos[1] - origin
                    ref_pos, ref_quat = env.traj.pose(env.episode_length_buf.float() * env.dt)
                    err = float((tip - ref_pos[1]).norm())
                    ori = float(quat_angle(ee_quat[1:2], ref_quat[1:2])[0])
                    rho = float(base_env.com_offset[1] / base_env.half_length[1])
                    c, sn = torch.cos(base_env.yaw[1]), torch.sin(base_env.yaw[1])
                    rel = (base_env.com_total[1, :2] - origin[:2]) - env._centre_xy()[1]  # COM from the footprint centre
                    dx, dy = c * rel[0] + sn * rel[1], -sn * rel[0] + c * rel[1]  # platform frame
                    margin = float(torch.minimum(base_env.half_length[1] - dx.abs(), base_env.half_width[1] - dy.abs()))
                    if failed_at is None and bool(base_env.tipped[1] | base_env.left_footprint[1]):
                        failed_at = t
                    # ink: where the TCP really is, projected onto the wall, every second step while the pen is down
                    pen = bool(path.pen_down[idx])
                    if pen and step % 2 == 0 and dots < n_dots:
                        on_plane = ([cfg.x_wall, float(tip[1]), float(tip[2])] if cfg.plane == "wall"
                                    else [float(tip[0]), float(tip[1]), cfg.z_table])
                        ink_pos[dots] = origin + torch.tensor(on_plane, device=dev)
                        dots += 1
                        ink_view.set_world_poses(positions=wp.from_torch(ink_pos.contiguous(), dtype=wp.vec3f))

                    follow = tip.clone() if follow is None else follow + FOLLOW_SMOOTH * (tip - follow)
                    in_reveal = t >= t_park
                    if not in_reveal:
                        for cam, (_, off) in zip(rig, VIEWS):
                            cam.set_world_poses_from_view((origin + follow + torch.tensor(off, device=dev))[None],
                                                          (origin + follow)[None])
                        for cam in rig:
                            cam.update(env.dt)
                        view = [cam.data.output["rgb"].torch[0, ..., :3].cpu().numpy() for cam in rig]
                    else:  # fly the big camera from the writing view to the full-text view while the arm parks
                        if flight_start is None:
                            flight_start = (follow.clone(), follow + torch.tensor(VIEWS[0][1], device=dev))
                        u = smoothstep((t - t_park) / cfg.park_time)
                        eye = flight_start[1] + u * (final_eye - flight_start[1])
                        target = flight_start[0] + u * (wall_centre - flight_start[0])
                        reveal.set_world_poses_from_view((origin + eye)[None], (origin + target)[None])
                        reveal.update(env.dt)
                        view = reveal.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
                    letters_done = sum(1 for s in path.strokes if s["start"] <= idx)
                    status = f"COM LEFT FOOTPRINT at {failed_at:.1f} s" if failed_at is not None else "stable"
                    writer.append_data(compose(view, "WRITING WITH THE POLICY", cfg.text, letters_done, status, failed_at is not None, t,
                                               PHASES[path.phase[idx]] + (f" ({cfg.text[path.letter[idx]]})" if path.letter[idx] >= 0 else ""),
                                               params, 100 * err, rho, in_reveal))
                    log["t"].append(t)
                    log["ref_pos"].append(ref_pos[1].cpu().numpy()), log["ee_pos"].append(tip.cpu().numpy())
                    log["pos_err"].append(err), log["ori_err"].append(ori), log["rho"].append(rho), log["margin_m"].append(margin)
                    log["base_pos"].append(base_env.base_pos[1, :2].cpu().numpy()), log["yaw"].append(float(base_env.yaw[1]))
                    log["pen_down"].append(pen), log["action"].append(a[1].clamp(-1, 1).cpu().numpy())
                    if step % 300 == 0:
                        print(f"[TEXT] t = {t:5.1f} s, phase {PHASES[path.phase[idx]]}, tracking error {100 * err:.1f} cm, rho {rho:.2f}", flush=True)
        finally:
            writer.close()  # an unclosed MP4 has no index and does not play

        arrays = {k: np.array(v) for k, v in log.items()}
        np.savez_compressed(video_path.with_suffix(".npz"), **arrays)
        pen = arrays["pen_down"]
        summary = {
            "text": cfg.text, "video": str(video_path), "checkpoint": args_cli.checkpoint, "controller": args_cli.controller,
            "platform": {"footprint": list(args_cli.footprint), "base_mass": args_cli.base_mass, "payload": args_cli.payload},
            "plane": cfg.plane, "writing_duration_s": path.duration, "failed_at_s": failed_at,
            "pen_down_pos_err_cm": {"rms": float(100 * np.sqrt(np.mean(arrays["pos_err"][pen] ** 2))),
                                    "mean": float(100 * arrays["pos_err"][pen].mean()), "max": float(100 * arrays["pos_err"][pen].max())},
            "all_steps_pos_err_cm": {"rms": float(100 * np.sqrt(np.mean(arrays["pos_err"] ** 2)))},
            "rho_max": float(arrays["rho"].max()), "ink_dots": dots,
        }
        video_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        print("[RESULT] " + json.dumps(summary), flush=True)
        print(f"[INFO] Saved video to {video_path}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
