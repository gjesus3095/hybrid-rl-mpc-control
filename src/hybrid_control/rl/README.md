# RL policies

Two separate PPO policies (RSL-RL) run in Isaac Lab:

- **With controller** (`policy_with_controller/`): the policy commands base
  velocities `(vx, vy, wz)` while an IK/QP or impedance Panda controller tracks a
  world-frame trajectory.
- **Without controller** (`policy_no_controller/`): the policy commands the same
  base velocities on a floating platform with a vertical lift, two planar revolute
  joints, and a terminal mass. Random joint splines drive the mechanism through
  finite simulator actuators. There is no Panda, IK/QP, or arm controller.

The existing `train_base_policy.py`, `play_base_policy.py`, and video scripts
continue to use the controller case. Their `--zero_policy` / `--mode none`
options park the base while retaining the Panda controller. The moving-mass
case has dedicated commands below; its checkpoints are incompatible with Panda
checkpoints, and video recording/comparison support is not yet implemented.

Before training, playing episodes, or recording videos, complete the
[Isaac Lab 3.0 / Isaac Sim 6.1 (Python 3.12) setup](../../../README.md#isaac-lab-30--isaac-sim-61-python-312)
in the root README, including the setup and validation scripts. Then run the
commands below from the repository root, setting the interpreter once per shell:

```bash
export ISAACLAB_PYTHON=~/workspace/IsaacLab/.venv/bin/python  # adjust to your path
```

## Train

```bash
$ISAACLAB_PYTHON scripts/isaac/with_controller/train_base_policy.py \
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
$ISAACLAB_PYTHON scripts/isaac/with_controller/play_base_policy.py \
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
$ISAACLAB_PYTHON scripts/isaac/with_controller/record_episodes_video.py \
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

## Controller-free moving-mass policy

```bash
# Physical assertions: mass/CoM, joint tracking, partial reset, tipping and timeout.
$ISAACLAB_PYTHON scripts/isaac/no_controller/check_moving_mass_env.py --num_envs 4 --seed 0

# Train; --viz kit enables the GUI, otherwise headless.
$ISAACLAB_PYTHON scripts/isaac/no_controller/train_moving_mass_policy.py \
    --num_envs 64 --max_iterations 500 --seed 0

# Reuse the actual checkpoint produced by training.
CHECKPOINT="logs/moving_mass_policy/<timestamp>/model_499.pt"
$ISAACLAB_PYTHON scripts/isaac/no_controller/play_moving_mass_policy.py \
    --checkpoint "$CHECKPOINT" --num_envs 16 --episodes 5 --seed 1
$ISAACLAB_PYTHON scripts/isaac/no_controller/play_moving_mass_policy.py \
    --checkpoint "$CHECKPOINT" --zero_policy --num_envs 16 --episodes 5 --seed 1

# Baseline without a checkpoint; add --viz kit to inspect the mechanism.
$ISAACLAB_PYTHON scripts/isaac/no_controller/play_moving_mass_policy.py --zero_policy --num_envs 4

# Resume restores the complete saved environment and PPO configuration.
$ISAACLAB_PYTHON scripts/isaac/no_controller/train_moving_mass_policy.py \
    --resume "$CHECKPOINT" --num_envs 64 --max_iterations 500
```

The joint order is `(revolute_1, revolute_2, lift)` in radians, radians, metres;
base actions are yaw-frame `(vx, vy, wz)` in m/s, m/s, rad/s after scaling.
Defaults are 120 Hz physics, 30 Hz policy, and 15-second episodes. The platform
uses the controller case's base mass/footprint distributions. Its upper mechanism
has 0.25 m links (0.25 kg each), a 0.5 kg carriage, 0–0.4 m lift travel, and a
terminal mass sampled uniformly from 1–5 kg. Geometry is fixed at spawn. Episode
resets resample masses and trajectories. Upper bodies have visual geometry and
physical inertia, but no collision shapes; only the platform collides with the
ground; collisions between environment slots are filtered. This is an abstract disturbance mechanism, not a manufactured robot model.

Cubic B-splines span each entire episode, with approximately two-second knot
intervals and zero velocity/acceleration at both ends. Position and derivative
control-point bounds limit each joint's motion; no state teleportation occurs
during stepping. Each environment has its own seeded random stream, allowing
baseline and policy evaluations with the same seed, environment count, and
per-environment episode count to sample matching disturbances despite different
failure times. Physics reproducibility remains backend/device dependent.

The actor sees current total CoM, normalized footprint offset, base velocity,
angular velocity, tilt, joint positions/velocities, previous action, and physical
parameters (28 inputs by default). The critic also sees mechanism CoM (31 inputs).
Neither sees future spline targets. Rewards encourage stability and penalize
base motion and abrupt actions; there are no end-effector or trajectory tracking
rewards. Episodes fail at 0.5 rad roll/pitch, including a tip on the timeout step.
Footprint excursions are reported but permit recovery. A timeout without tipping
remains eligible for value bootstrapping.

Base translation cannot directly recenter a mass prescribed relative to the base.
This task evaluates dynamic disturbance rejection through acceleration and tilt.
A short successful training run only verifies the training pipeline; compare
survival, tilt, and motion against the seeded zero-command baseline before
claiming a learned stabilization benefit.

To override defaults, pass `--config path/to/env.yaml` for a new training run or
checkpoint-free baseline. Omitted fields retain defaults; unknown fields fail.
For example:

```yaml
environment_kind: moving_mass_v1
episode_length_s: 15.0
mechanism:
  link_lengths: [0.25, 0.25]
  stiffness: [1000.0, 1000.0, 10000.0]
  damping: [100.0, 100.0, 500.0]
  effort_limit: [150.0, 150.0, 500.0]
trajectory:
  difficulty: 1.0
  knot_interval: 2.0
  max_velocity: [1.0, 1.0, 0.15]
  max_acceleration: [2.0, 2.0, 0.3]
domain_randomization:
  terminal_mass: [1.0, 5.0]
  platform:
    length: [0.19, 0.52]
    width: [0.19, 0.52]
    base_mass: [5.5, 38.0]
```

Checkpoints require their `params/env.yaml` and `params/agent.yaml`; loading
restores all settings rather than accepting a conflicting `--config`. Resume
restores model/optimizer state and starts fresh seeded episodes, not saved
simulator state. `params/run.yaml` records the run seed and environment count.
`--num_steps_per_env` can explicitly override the PPO rollout length on resume.

CPU tests: `env -u PYTHONPATH uv run pytest tests/test_moving_mass.py tests/test_rl_components.py -q`.
Checkpoint tests require RSL-RL; those tests skip when it is unavailable. Isaac
integration checks must use the configured Isaac Lab interpreter. Unit tests
cover spline derivatives against SciPy, bounds, continuity, per-slot reset
isolation, observations, rewards, and saved configuration compatibility.
