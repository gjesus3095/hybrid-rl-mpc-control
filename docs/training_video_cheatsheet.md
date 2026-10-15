# Training & video cheatsheet

Every command for training, evaluating and recording videos of the RL base policy, with all command-line options.
Background and troubleshooting: `docs/training_cheatsheet.md`, `docs/code_overview.md` section 4.

## Setup (once per shell)

All commands run with the Isaac Lab 3 interpreter, from any folder. Unsetting `DISPLAY` keeps headless runs from
hanging.

```bash
export R=~/hybrid-rl-mpc-control
export PYTHONPATH=$R/src
export PY=~/IsaacLab/.venv/bin/python
unset DISPLAY
```

Every script also takes the Isaac Lab launcher flags:

| Flag | Default | What it does |
| --- | --- | --- |
| `--device` | `cuda:0` | `cpu`, `cuda` or `cuda:N` (pick the GPU) |
| `--viz` / `--visualizer` | none (headless) | `kit` opens the GUI; CSV of `kit,newton,rerun,viser` |
| `--max_visible_envs` | all | Caps the envs shown in the visualizer |
| `--livestream` | -1 | 0/1/2, as the `LIVESTREAM` env var |
| `--kit_args` | none | Extra raw Kit arguments |
| `--verbose` / `--info` | off | More Kit log output |

## Training

### Train: `train_base_policy.py`

```bash
# smoke test (~5 min)
$PY $R/scripts/isaac/train_base_policy.py --num_envs 64 --max_iterations 20 --qp_workers 16
# real run (~35 min, ~6M samples)
$PY $R/scripts/isaac/train_base_policy.py --num_envs 256 --qp_workers 48 --max_iterations 1000 \
    --log_dir $R/logs/base_policy/my_run
```

| Option | Default | What it does |
| --- | --- | --- |
| `--num_envs` | 64 | Parallel envs |
| `--max_iterations` | 500 | PPO iterations |
| `--num_steps_per_env` | 24 | Rollout length per iteration |
| `--controller` | `ik_qp` | Arm controller: `ik_qp` (Pink IK/QP, joint targets) or `impedance` (torques) |
| `--qp_workers` | 16 | QP worker processes; 0 = serial (~10x slower). Scale with envs |
| `--ik_every` | 2 | Physics steps per QP update; must divide 4 |
| `--difficulty` | 1.0 | Trajectory amplitude/speed scale, 0 to 1 |
| `--spacing` | 2.5 | Env spacing [m] |
| `--seed` | 0 | Random seed |
| `--resume` | none | `model_*.pt` to continue from |
| `--log_dir` | `logs/base_policy/<timestamp>` | Output folder; `model_<iter>.pt` every 50 iterations |

Monitor: `~/IsaacLab/.venv/bin/tensorboard --logdir $R/logs/base_policy`. Watch `Episode/failed_fraction` (toward 0),
`Metrics/com_rho` (~0.1) and `Metrics/ee_track_error` (~2 cm).

### Evaluate: `play_base_policy.py`

```bash
CK=$R/logs/base_policy/my_run/model_999.pt
$PY $R/scripts/isaac/play_base_policy.py --checkpoint $CK --num_envs 128 --episodes 3 --qp_workers 32
# baseline: base stands still (~59% failures)
$PY $R/scripts/isaac/play_base_policy.py --checkpoint $CK --num_envs 128 --episodes 3 --qp_workers 32 --zero_policy
```

| Option | Default | What it does |
| --- | --- | --- |
| `--checkpoint` | required | `model_*.pt` to evaluate |
| `--num_envs` | 32 | Parallel envs |
| `--episodes` | 2 | Episodes per env |
| `--controller` | from checkpoint | Overrides the arm controller in `<run>/params/env.yaml` |
| `--zero_policy` | off | Zero base command baseline |
| `--qp_workers` | 8 | QP worker processes |
| `--difficulty` | 1.0 | Trajectory scale, 0 to 1 |
| `--spacing` | 2.5 | Env spacing [m] |
| `--seed` | 1 | Random seed |

## Videos

Both scripts write H.264 mp4 files to `$R/out/` (gitignored). `--controller` defaults to the one the checkpoint was
trained with.

### Trained vs untrained side by side: `compare_policies_video.py`

```bash
$PY $R/scripts/isaac/compare_policies_video.py \
    --trained $CK --untrained $R/logs/base_policy/my_run/model_0.pt \
    --video_path $R/out/policy_comparison.mp4
```

| Option | Default | What it does |
| --- | --- | --- |
| `--trained` | required | Checkpoint on the left half |
| `--untrained` | required | Checkpoint on the right half |
| `--scenarios` | 4 | Number of side-by-side scenarios |
| `--hard_ranks` | `0 20 40 60` | Ranks picked from the pool, sorted least to most stable platform |
| `--pool` | 80 | Size of the random platform pool |
| `--kinds` | `fixed sine quintic` | Trajectory kinds, cycled over scenarios |
| `--seconds` | 15.0 | Video length [s] |
| `--spacing` | 2.4 | Env spacing [m] |
| `--seed` | 3 | Random seed |
| `--qp_workers` | 4 | QP worker processes |
| `--controller` | from `--trained` | Arm controller override |
| `--video_path` | `out/policy_comparison.mp4` | Output file |
| `--video_fps` | 30 | Frame rate |

### One policy, several episodes: `record_episodes_video.py`

```bash
# trained policy
$PY $R/scripts/isaac/record_episodes_video.py --checkpoint $CK \
    --title "TRAINED POLICY" --video_path $R/out/trained_episodes.mp4
# untrained policy, exploring
$PY $R/scripts/isaac/record_episodes_video.py --checkpoint $R/logs/base_policy/my_run/model_0.pt \
    --mode sampled --title "UNTRAINED" --video_path $R/out/untrained_episodes.mp4
# arm controller only, base parked
$PY $R/scripts/isaac/record_episodes_video.py --checkpoint $CK --mode none --title "NO POLICY"
```

| Option | Default | What it does |
| --- | --- | --- |
| `--checkpoint` | required | Policy; also picks the arm controller |
| `--title` | `POLICY` | Title in the side panel |
| `--mode` | `deterministic` | `deterministic` = mean action; `sampled` = mean + noise; `none` = zero base command |
| `--noise_std` | 0.5 | Action noise for `--mode sampled` |
| `--episodes` | 6 | Episodes in the video, each on its own platform |
| `--kinds` | `fixed sine quintic` | Trajectory kinds, cycled over episodes |
| `--hold_tipped` | 1.5 | Seconds a tipped robot stays on screen |
| `--hold_end` | 0.5 | Seconds after a completed episode |
| `--spacing` | 8.0 | Distance between per-episode envs [m] |
| `--seed` | 3 | Random seed |
| `--qp_workers` | 2 | QP worker processes |
| `--controller` | from checkpoint | Arm controller override |
| `--video_path` | `out/episodes.mp4` | Output file |
| `--video_fps` | 30 | Frame rate |

## Gotchas

- **Hangs right after the Kit log line:** `DISPLAY` is set; run with `DISPLAY=` and `kill -9` the stuck process.
- **Old checkpoint won't load:** any edit to `observations.py` changes the observation size; retrain.
- **Leftover QP workers after a crash:** `pkill -f hybrid_control.rl.qp_worker`.
- **Disk full:** `rm -rf ~/.cache/isaaclab_tmp/wp_pch_*` (~70 MB Warp cache per launch; leave the other folders).
- **Robot shows only its COM dots in a video:** first-env rendering quirk; physics is fine.
