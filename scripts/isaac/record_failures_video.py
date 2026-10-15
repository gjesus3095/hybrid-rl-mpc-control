"""Video of every failed scenario of an ``evaluate_policy.py`` run, one after the other.

Failures that happen by a hair (the COM leaving the footprint by < 1 mm) do not reproduce when a scenario is replayed
on its own, because the GPU numerics depend slightly on which envs are simulated together. So this script replays the
evaluation exactly as it ran: same benchmark, policy and arm controller, all env slots at the same spacing, the sets in
the same order, and records only the failing envs (four cameras per failing env, following the robot and zooming out
toward the reference). Each recording runs until the failure plus ``--hold_tipped`` seconds. A side panel shows the set,
scenario, platform, time, the live COM offset bar and the outcome (with the failure time of the evaluation for
comparison). The script reports every failure that did not reproduce.

The failure is the evaluation's criterion: the whole-robot COM leaves the platform footprint (``left_footprint``) or the
platform tips (roll / pitch > 0.5 rad). Left-footprint failures are often not visibly falling over yet.

Usage (from the repo root, after the Isaac Lab setup in the root README):

    $ISAACLAB_PYTHON scripts/isaac/record_failures_video.py --eval_dir out/eval/<run> \
        --checkpoint logs/base_policy/<run>/model_<n>.pt --controller impedance --current_observation
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from hybrid_control.rl.config import CONTROLLERS, saved_controller, saved_observation_cfg

parser = argparse.ArgumentParser(description="Record the failed scenarios of an evaluation run.")
parser.add_argument("--eval_dir", type=str, required=True, help="Output folder of evaluate_policy.py.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--controller", choices=CONTROLLERS, default=None,
                    help="Arm controller; default: the one the checkpoint was trained with (<run>/params/env.yaml).")
parser.add_argument("--current_observation", action="store_true",
                    help="Use the current ObservationCfg defaults instead of <run>/params/env.yaml.")
parser.add_argument("--sets", nargs="+", default=None, help="Only record the failures of these sets (all sets before them are still replayed).")
parser.add_argument("--title", type=str, default="FAILED SCENARIOS")
parser.add_argument("--spacing", type=float, default=8.0, help="Env spacing [m]; must equal the evaluation's (default 8).")
parser.add_argument("--hold_tipped", type=float, default=1.5, help="Seconds a failed robot stays on screen.")
parser.add_argument("--qp_workers", type=int, default=16, help="Worker processes for the per-env QPs (0 = serial).")
parser.add_argument("--video_path", type=str, default=None, help="Default: <eval_dir>/failures.mp4.")
parser.add_argument("--video_fps", type=int, default=30)
parser.add_argument("--seed", type=int, default=1)
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.controller = args_cli.controller or saved_controller(args_cli.checkpoint)
args_cli.enable_cameras = True

import json

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from rsl_rl.runners import OnPolicyRunner

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.rl.evaluation.scenarios import BenchmarkCfg, generate_benchmark
from hybrid_control.rl.evaluation.scripted import ScriptedTrajectories
from hybrid_control.rl.policy_with_controller.env import BaseStabilizationEnv, StabilizationEnvCfg
from hybrid_control.rl.policy_with_controller.observations import ObservationCfg
from hybrid_control.sim_interfaces.isaac.ridgeback.env import RidgebackDREnv
from hybrid_control.sim_interfaces.isaac.ridgeback.viz import ComMarkers, ReferenceMarkers, _legend

TILE = (800, 450)  # one camera view; the views form a 2 x 2 grid
PANEL_W = 480  # side panel (frame width 480 + 1600 = 2080, a multiple of 16)
FOCAL = 16.0  # [mm]
CAM_DIST, CAM_HEIGHT, CAM_SMOOTH = 3.3, 1.15, 0.12  # base distance / height of the cameras, low-pass factor per frame
VIEWS = [("front", (1.0, 0.0)), ("left", (0.0, 1.0)), ("right", (0.0, -1.0)), ("back", (-1.0, 0.0))]


def compose(tiles, status, ok, entry, t_ep, index, total, outcomes, title, rho, params, phase):
    """Side panel + 2 x 2 grid of the camera views, with all text overlays."""
    from PIL import Image, ImageDraw, ImageFont

    grid = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
    h = grid.shape[0]
    frame = np.concatenate([np.full((h, PANEL_W, 3), 25, dtype=np.uint8), grid.astype(np.uint8)], axis=1)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    big, font, tag = ImageFont.load_default(size=38), ImageFont.load_default(size=26), ImageFont.load_default(size=21)
    kw = dict(stroke_width=2, stroke_fill=(0, 0, 0))
    white, grey, accent = (255, 255, 255), (180, 180, 180), (255, 215, 90)
    for i, (name, _) in enumerate(VIEWS):
        draw.text((PANEL_W + (i % 2) * TILE[0] + 14, (i // 2) * TILE[1] + 10), f"{name} view", font=tag, fill=white, **kw)
    draw.line((PANEL_W, 0, PANEL_W, h), fill=white, width=3)
    draw.line((PANEL_W + TILE[0], 0, PANEL_W + TILE[0], h), fill=white, width=2)
    draw.line((PANEL_W, TILE[1], frame.shape[1], TILE[1]), fill=white, width=2)
    draw.text((20, 14), title, font=big, fill=white, **kw)
    draw.text((20, 62), f"failure {index} / {total}   t = {t_ep:4.1f} s", font=font, fill=white, **kw)
    draw.text((20, 98), status, font=font, fill=(130, 255, 130) if ok else (255, 90, 90), **kw)
    draw.text((20, 150), "set / scenario", font=tag, fill=grey, **kw)
    draw.text((20, 174), f"{entry['set']} #{entry['scenario']}", font=font, fill=accent, **kw)
    draw.text((20, 214), "evaluation", font=tag, fill=grey, **kw)
    draw.text((20, 238), f"failed at {entry['fail_time']:.1f} s of {entry['duration']:.1f} s", font=font, fill=accent, **kw)
    draw.text((20, 290), "platform", font=tag, fill=grey, **kw)
    draw.text((20, 314), f"{params.footprint[0]:.2f} x {params.footprint[1]:.2f} m, {params.base_mass:.0f} kg", font=tag, fill=white, **kw)
    draw.text((20, 340), f"payload {params.payload:.1f} kg", font=tag, fill=white, **kw)
    draw.text((20, 384), "reference", font=tag, fill=grey, **kw)
    draw.text((20, 408), phase, font=tag, fill=white, **kw)
    _legend(draw, PANEL_W, 20, tag, kw, y0=486)
    frac = min(max(rho, 0.0), 1.0)
    bar = (int(255 * min(2 * frac, 1.0)), int(255 * min(2 * (1 - frac), 1.0)), 40)
    x0, y0 = 20, h - 130
    draw.text((x0, y0 - 54), f"CoM offset {100 * rho:.0f}% of the half footprint", font=tag, fill=white, **kw)
    draw.text((x0, y0 - 30), "(100% = at the edge)", font=tag, fill=grey, **kw)
    draw.rectangle((x0, y0, x0 + 400, y0 + 22), fill=(30, 30, 30), outline=white, width=2)
    draw.rectangle((x0, y0, x0 + int(400 * frac), y0 + 22), fill=bar)
    for i, reproduced in enumerate(outcomes):  # one mark per replay so far: X = failed again, ? = did not reproduce
        draw.text((20 + 26 * i, h - 52), "X" if reproduced else "?", font=tag, fill=(255, 90, 90) if reproduced else (255, 215, 90), **kw)
    return np.asarray(image)


def main():
    import shutil

    import imageio.v2 as imageio

    eval_dir = Path(args_cli.eval_dir)
    summary = json.loads((eval_dir / "summary.json").read_text())
    meta = summary["meta"]
    benchmark = generate_benchmark(BenchmarkCfg(controller=args_cli.controller))
    fingerprint = benchmark.fingerprint(list(summary["sets"]))  # over the sets that were evaluated
    if fingerprint != meta["benchmark_fingerprint"]:
        raise SystemExit(f"the evaluation used benchmark {meta['benchmark_fingerprint']}, this controller's is "
                         f"{fingerprint}: wrong --controller or the benchmark changed since")
    dt, slots = meta["policy_dt"], meta["scenarios_per_set"]

    order = list(summary["sets"])  # replay order = evaluation order
    failures = {}  # set -> [entry]
    for name in order:
        log = np.load(eval_dir / f"{name}.npz")
        failures[name] = [{"set": name, "scenario": int(j), "fail_time": (int(log["fail_step"][j]) + 1) * dt,
                           "duration": float(log["duration"][j]), "n_steps": int(log["n_steps"][j])}
                          for j in np.flatnonzero(log["fail_step"] >= 0)]
    record = [n for n in order if failures[n] and (args_cli.sets is None or n in args_cli.sets)]
    if not record:
        raise SystemExit("no failures to record")
    total = sum(len(failures[n]) for n in record)
    replay = order[: order.index(record[-1]) + 1]  # sets after the last recorded one are not needed
    print(f"[VIDEO] {total} failed scenarios: "
          + ", ".join(f"{e['set']}#{e['scenario']}" for n in record for e in failures[n]), flush=True)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        base_env = RidgebackDREnv(benchmark.params(replay[0], slots), sim, spacing=args_cli.spacing)
        n = base_env.num_envs

        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        obs_cfg = (saved_observation_cfg(args_cli.checkpoint) if not args_cli.current_observation else ObservationCfg())
        markers = ComMarkers(n, sim.device)
        ref_markers = ReferenceMarkers(n, sim.device, len(obs_cfg.lookahead))
        pool_size = max(len(failures[name]) for name in record)  # camera sets: one per simultaneously recorded env
        rigs = [[Camera(cfg=CameraCfg(
            prim_path=f"/World/RecordCam_{c}_{view}", update_period=0.0, height=TILE[1], width=TILE[0], data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(focal_length=FOCAL, focus_distance=400.0, horizontal_aperture=20.955,
                                             clipping_range=(0.1, 1.0e5)))) for view, _ in VIEWS] for c in range(pool_size)]
        base_env.start()
        env = BaseStabilizationEnv(
            base_env, StabilizationEnvCfg(controller=args_cli.controller, qp_workers=args_cli.qp_workers, episode_length_s=3600.0,
                                          observation=obs_cfg), seed=args_cli.seed)
        env.auto_reset = False
        env.traj = ScriptedTrajectories(env.num_envs, env.device, env.scfg.trajectory)
        dev = env.device
        runner = OnPolicyRunner(env, make_runner_cfg(), log_dir=None, device=str(dev))
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=str(dev))

        tmp = eval_dir / "_failure_clips"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        video_path = Path(args_cli.video_path or eval_dir / "failures.mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        clips, outcomes, index = [], [], 0  # clips: temp files in recording order
        writers = {}
        try:
            with torch.inference_mode():  # the env keeps inference tensors, so resets must run in it too
                for name in replay:
                    scenario = benchmark.sets[name]
                    env.traj.load(scenario)
                    env._reset_idx(torch.arange(n, device=dev), dynamics=benchmark.params(name, slots))
                    env._obs = env._compute_obs()
                    obs = env.get_observations()
                    n_steps = np.floor(scenario.duration[:slots] / env.dt + 1e-6).astype(int)
                    todo = failures[name] if name in record else []
                    state = []  # per recorded env: camera rig, writer, smoothed look, failure bookkeeping
                    for c, entry in enumerate(todo):
                        index += 1
                        path = tmp / f"{index:03d}_{name}_{entry['scenario']}.mp4"
                        clips.append(path)
                        state.append({"entry": entry, "rig": rigs[c], "index": index, "look": None, "failed_at": None,
                                      "hold": None, "done": False, "writer": imageio.get_writer(path, fps=args_cli.video_fps)})
                    for step in range(int(n_steps.max())):
                        obs, _, _, _ = env.step(policy(obs))
                        active = [st for st in state if not st["done"]]
                        if not active:
                            continue
                        markers.update(base_env)
                        ref_markers.update(env)
                        t_all = env.episode_length_buf.float() * env.dt
                        ref_all = env.traj.pose(t_all)[0]
                        for st in active:
                            e = st["entry"]["scenario"]
                            t_ep = float(t_all[e])
                            if st["failed_at"] is None and bool(base_env.tipped[e] | base_env.left_footprint[e]):
                                st["failed_at"] = t_ep
                                st["kind"] = "TIPPED" if bool(base_env.tipped[e]) else "COM LEFT FOOTPRINT"
                                st["hold"] = int(args_cli.hold_tipped / env.dt)
                            base_xy, ref_xy = base_env.base_pos[e, :2], ref_all[e, :2]
                            target = torch.cat([0.5 * (base_xy + ref_xy), torch.tensor([0.55], device=dev)])
                            dist = min(CAM_DIST + 0.7 * float((ref_xy - base_xy).norm()), 7.0)
                            look = (target, dist) if st["look"] is None else (
                                st["look"][0] + CAM_SMOOTH * (target - st["look"][0]),
                                st["look"][1] + CAM_SMOOTH * (dist - st["look"][1]))
                            st["look"] = look
                            eye_h = CAM_HEIGHT * (look[1] / CAM_DIST) ** 0.5
                            origin = base_env.origins[e]
                            for cam, (_, (dx, dy)) in zip(st["rig"], VIEWS):
                                eye = origin + look[0] + torch.tensor([look[1] * dx, look[1] * dy, eye_h - 0.55], device=dev)
                                cam.set_world_poses_from_view(eye[None], (origin + look[0])[None])
                            for cam in st["rig"]:
                                cam.update(env.dt)
                            tiles = [cam.data.output["rgb"].torch[0, ..., :3].cpu().numpy() for cam in st["rig"]]
                            rho = float(base_env.com_offset[e] / base_env.half_length[e])
                            reproduced = st["failed_at"] is not None
                            status = f"{st['kind']} at {st['failed_at']:.1f} s" if reproduced else "upright"
                            st["writer"].append_data(compose(
                                tiles, status, not reproduced, st["entry"], t_ep, st["index"], total,
                                outcomes + [], args_cli.title, rho, base_env.params[e], env.traj.phase_name(e, t_ep)))
                            if reproduced:
                                st["hold"] -= 1
                            if (reproduced and st["hold"] <= 0) or step + 1 >= st["entry"]["n_steps"]:
                                st["done"] = True
                                st["writer"].close()
                    for st in state:
                        if not st["done"]:
                            st["writer"].close()
                        ok = st["failed_at"] is not None
                        outcomes.append(ok)
                        print(f"[VIDEO] {name}#{st['entry']['scenario']}: "
                              f"{'failed again at %.1f s' % st['failed_at'] if ok else 'DID NOT REPRODUCE'} "
                              f"(evaluation: {st['entry']['fail_time']:.1f} s)", flush=True)
            # concatenate the clips in recording order (re-encoded once)
            with imageio.get_writer(video_path, fps=args_cli.video_fps) as out:
                for clip in clips:
                    reader = imageio.get_reader(clip)
                    for frame in reader:
                        out.append_data(frame)
                    reader.close()
        finally:
            for w in writers.values():
                w.close()
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[RESULT] {sum(outcomes)} / {len(outcomes)} failures reproduced")
        print(f"[INFO] Saved video to {video_path}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
