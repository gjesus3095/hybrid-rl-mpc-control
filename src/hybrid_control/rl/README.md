# RL policies

We will train two separate policies for the Ridgeback–Franka in Isaac Lab using
PPO (RSL-RL):

- **With controller** (`policy_with_controller/`): the policy commands base
  velocities `(vx, vy, wz)` while an IK/QP or impedance arm controller tracks a
  world-frame trajectory.
- **Without controller** (`policy_no_controller/`): a separate policy trained
  without the model-based arm controller.

Training randomizes the platform and dynamics to learn balance as the arm moves.
`config.py` defines the PPO runner. The commands below currently use
`policy_with_controller/`; training, playback, and recording entry points for
the policy without a controller are not implemented yet. Its `env.py` is still
a copy of the controller-based environment, including QP logic and imports of
missing local `qp_pool` and `trajectories` modules; its observations and rewards
still include reference-tracking terms. It is not runnable as a controller-free
environment. `--zero_policy` and
`--mode none` disable the base policy while retaining the arm controller, so they
do not run the planned policy without a controller.

Before training, playing episodes, or recording videos, complete the
[Isaac Lab 3.0 / Isaac Sim 6.1 (Python 3.12) setup](../../../README.md#isaac-lab-30--isaac-sim-61-python-312)
in the root README, including the setup and validation scripts. Then run the
commands below from the repository root, setting the interpreter once per shell:

```bash
export ISAACLAB_PYTHON=~/workspace/IsaacLab/.venv/bin/python  # adjust to your path
```

## Train

```bash
$ISAACLAB_PYTHON scripts/isaac/train_base_policy.py \
    --controller ik_qp --num_envs 64 --max_iterations 500 --seed 0
```

Use `--controller impedance` for the torque-level arm controller. Reduce
`--num_envs` and `--qp_workers` (default: 16; `0` runs QPs serially) to fit your
machine. Isaac Lab 3 runs headless by default; add `--viz kit` for the GUI.

Checkpoints (`model_*.pt`), TensorBoard logs, and configuration files
(`params/env.yaml`, `params/agent.yaml`) are saved under
`logs/base_policy/<timestamp>/`, or the directory supplied with `--log_dir`.
Resume with `--resume path/to/model_499.pt`, keeping the same `--controller`.

## Play episodes

Select an existing checkpoint from your training run:

```bash
CHECKPOINT="logs/base_policy/<timestamp>/model_499.pt"  # replace <timestamp>
$ISAACLAB_PYTHON scripts/isaac/play_base_policy.py \
    --checkpoint "$CHECKPOINT" --num_envs 1 --episodes 5 --seed 1 --viz kit
```

Playback uses deterministic policy actions and reports failures, CoM offset,
and base-motion statistics. Omit `--viz kit` for headless evaluation; increase
`--num_envs` for parallel evaluation (the completion target is
`--episodes * --num_envs`). Add `--zero_policy` to evaluate a parked-base baseline.
Playback and recording recover the controller and observation configuration
from `params/env.yaml`; keep that subdirectory inside the checkpoint's run directory.

## Save episode videos

```bash
$ISAACLAB_PYTHON scripts/isaac/record_episodes_video.py \
    --checkpoint "$CHECKPOINT" --episodes 6 --seed 3 \
    --title "TRAINED POLICY" --video_path out/trained_episodes.mp4 --video_fps 30
```

The recorder enables cameras automatically and saves one MP4 with successive
episodes, camera cuts, and status/CoM overlays. It creates the output directory;
the default path is `out/episodes.mp4`. Recording requires `imageio`, an FFmpeg
backend such as `imageio-ffmpeg`, and Pillow in the Isaac Lab environment.

Use `--mode none` for a parked-base video or `--mode sampled --noise_std 0.5`
to add action noise (default: `--mode deterministic`). Reuse `--seed` and
`--episodes` to compare the same randomized scenarios across checkpoints.
