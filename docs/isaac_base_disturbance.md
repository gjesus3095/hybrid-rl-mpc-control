# Isaac Lab base-disturbance test (Pink IK/QP)

Use the keyboard to move the Ridgeback base. The Pink IK/QP holds the Franka TCP (`panda_fingertip_centered`) at its initial world pose. The script runs in Isaac Lab's own venv and loads this repo through `PYTHONPATH`, so nothing needs to be installed.

## Run on a desktop display

```bash
# 1. Check that a display and the GPU are available (the keyboard needs the GUI window)
echo $DISPLAY && nvidia-smi --query-gpu=name --format=csv,noheader

# 2. Check that the Isaac Lab venv can import this repo (expected output: "ok")
cd ~/hybrid-rl-mpc-control
PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python -c "import hybrid_control.controllers.ik_qp.ik_qp; print('ok')"

# 3. Run it. --viz kit opens the GUI; --num_envs is 1-512 (default 16); --solver is daqp (default) or osqp
PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/isaac/with_controller/base_disturbance_keyboard.py --viz kit --num_envs 16
```

- `--workers N` solves the QPs in N processes (one per physical core, forked before Isaac Sim starts); up to 512 envs. Solver-only timings: `PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/benchmark_parallel_ik.py`.
- `PYTHONPATH=...` replaces any ROS `PYTHONPATH` for this command only.
- The first run downloads the Ridgeback-Franka and Factory Franka USDs, which can take a while.
- Click the viewport once so it receives the keys, then close the window to stop.

## Keys (`Se2Keyboard`, base frame)

| Key | Base command |
|---|---|
| ↑ / ↓ | +/− x at 0.3 m/s |
| ← / → | +/− y at 0.3 m/s |
| Z / X | +/− yaw at 0.5 rad/s |
| L | reset the command to zero |

- All environments receive the same command.
- Speeds are set by `KEY_SENSITIVITY` in the script. At 0.8 m/s the arm reaches its joint velocity limits and the TCP drifts off.

## What you see

| Marker | Meaning |
|---|---|
| Frames | TCP target and measured TCP |
| Green sphere | Arm CoM |
| Orange sphere | Base CoM |
| Red arrow | Commanded base velocity |
| Blue arrow | Commanded yaw rate, drawn as the tangential velocity 0.6 m in front of the base |

## Log

- **Console:** every second, the maximum TCP error over all environments, the one-step prediction error, active bounds, failed solves, and the mean/p95 wall-clock time of each phase:
  - `read`: GPU→CPU;
  - `solve`: all QPs, plus the slowest environment;
  - `write`: CPU→GPU;
  - `sim`: four physics substeps.
- **CSV:** `logs/isaac_base_disturbance.csv`, one row per 20 ms control step (change the path with `--log`). Plot it with:

```bash
~/IsaacLab/.venv/bin/python -c "import pandas as pd, matplotlib.pyplot as plt; d = pd.read_csv('logs/isaac_base_disturbance.csv'); d.plot(x='t', y=['pos_err_max_mm', 'ori_err_max_mrad', 't_solve_ms', 't_read_ms', 't_write_ms'], subplots=True, figsize=(8, 9)); plt.show()"
```

## Headless run and video (no keyboard)

`--schedule` replaces the keyboard with a repeating 14 s key sequence (after `ridgeback_demo.py` on `feature/ridgeback-isaac-demo`): 2 s each of still, +x, −x, +y, −y, +yaw, −yaw, at the `KEY_SENSITIVITY` speeds. `--duration` stops after that many sim seconds. Leave `DISPLAY` empty when running headless:

```bash
DISPLAY= PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/isaac/with_controller/base_disturbance_keyboard.py --schedule --duration 14
```

`--record` adds an offscreen camera on env 0 and writes a 25 fps H.264 mp4 (640×360 by default, change with `--record_width`) with an overlay (phase, command, TCP error, solve time) to `logs/isaac_base_disturbance.mp4` (change with `--record_path`). It needs `imageio-ffmpeg` in the Isaac Lab venv (once: `uv pip install --python ~/IsaacLab/.venv/bin/python imageio-ffmpeg`). With `--schedule` it stops after one cycle. Use `--num_envs 1` for a close-up; rendering slows the run down:

```bash
DISPLAY= PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/isaac/with_controller/base_disturbance_keyboard.py --num_envs 1 --schedule --record --duration 5
```

The flag is not called `--video` because Isaac Lab's launcher already uses that name (and needs MoviePy).

## Torque control: impedance QP from `main`

`scripts/isaac/with_controller/base_disturbance_impedance.py` runs the same test with main's inverse-dynamics impedance QP (`controllers/impedance/`, see its README) instead of the Pink IK. The arm's PD gains are zeroed (pure effort targets) and gravity is on; the QP compensates it. The QP runs at `--control_hz` (default 50 Hz) and holds its torques over `--physics_hz / --control_hz` physics steps (default 200 Hz, so 4). Main's MuJoCo scripts use 500 Hz, but main's gains track as well at 50 Hz: 0.06 mm steady TCP error at 0.3 m/s, ~16 mm peak at a base reversal. Enabled options: null-space posture torque towards `Q_MID`, the manipulability floor (`w >= 0.03`, full Jacobian) and joint-limit barriers. The commanded base twist is fed forward through the target's velocity/acceleration, because the controller only sees the arm joints. Same flags as above, except `--solver` (OSQP only); defaults: 6 envs, `logs/isaac_impedance.{csv,mp4}`.

```bash
# 5 s video of 6 envs (headless)
DISPLAY= PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/isaac/with_controller/base_disturbance_impedance.py --num_envs 6 --schedule --duration 5 --record
# keyboard on a desktop display
PYTHONPATH=$PWD/src ~/IsaacLab/.venv/bin/python scripts/isaac/with_controller/base_disturbance_impedance.py --viz kit
```

With 6 envs at 50/200 Hz one simulated second takes ~1.7 s (QP ~1.8 ms per env per control step, physics ~5.5 ms per step); `--physics_hz 100` brings it to ~1.15 s with ~0.1 mm steady error.
