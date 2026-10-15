# RL policies

Two separate PPO policies (RSL-RL) run in Isaac Lab:

- **With controller** (`policy_with_controller/`): the policy commands base
  velocities `(vx, vy, wz)` while an IK/QP or impedance Panda controller tracks a
  world-frame trajectory.
- **Without controller** (`policy_no_controller/`): the policy commands the same
  base velocities on a floating platform with a vertical lift, roll/pitch revolute
  joints (±90° each), and a terminal mass. Random joint splines drive the mechanism through
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

The joint order is `(roll, pitch, lift)` in radians, radians, metres;
base actions are yaw-frame `(vx, vy, wz)` in m/s, m/s, rad/s after scaling.
Defaults are 120 Hz physics, 30 Hz policy, and 15-second episodes. The platform
uses the controller case's base mass/footprint distributions. Its upper mechanism
has a 0.5 m upright mast (0.25 kg), a 0.25 kg roll hub, a 0.5 kg carriage,
0–0.4 m lift travel, and a
terminal mass sampled uniformly from 1–5 kg. Geometry is fixed at spawn. Episode
resets resample masses and trajectories. All bodies have rigid-body mass/inertia and collision shapes, including ground
contacts for the mast and terminal mass during a fall. Nonadjacent self-collisions
are enabled except between the carriage and the mast through its bearing; connected
joint bodies and separate environment slots are filtered. This is an abstract disturbance mechanism, not a manufactured robot model.

Cubic B-splines span each entire episode, with approximately one-second knot
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
environment_kind: moving_mass_v2
episode_length_s: 15.0
mechanism:
  mast_length: 0.5
  stiffness: [1000.0, 1000.0, 10000.0]
  damping: [100.0, 100.0, 500.0]
  effort_limit: [150.0, 150.0, 500.0]
trajectory:
  difficulty: 1.0
  knot_interval: 1.0
  min_peak_velocity: [0.9, 0.9, 0.18]
  max_velocity: [2.0, 2.0, 0.4]
  max_acceleration: [6.0, 6.0, 1.2]
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

### Visual demos and physical probes

The demo entry points call shared code in
`sim_interfaces/isaac/moving_mass/demo.py`, `viz.py`, and `diagnostics.py`.

```bash
# Analog of ridgeback_dr_demo: randomized platforms driven by episode splines.
$ISAACLAB_PYTHON scripts/isaac/no_controller/moving_mass_dr_demo.py --viz kit --plot

# Upright -> tilt/lift -> hold, making static tipping easier to inspect.
$ISAACLAB_PYTHON scripts/isaac/no_controller/moving_mass_dr_demo.py --scenario extend --viz kit

# Isolate one joint: settle -> move -> hold -> return -> hold.
$ISAACLAB_PYTHON scripts/isaac/no_controller/moving_mass_dr_demo.py \
    --scenario joint_sweep --joint lift --num_envs 1 --viz kit --plot

# Identical dynamics and spline targets: zero commands on the left, fixed commands on the right.
# --num_envs denotes pairs in this script.
$ISAACLAB_PYTHON scripts/isaac/no_controller/compare_moving_mass.py \
    --num_envs 4 --cmd 0.2 0 0 --viz kit --plot

# Replace the fixed command with an existing moving-mass checkpoint.
$ISAACLAB_PYTHON scripts/isaac/no_controller/compare_moving_mass.py \
    --checkpoint "$CHECKPOINT" --num_envs 4 --viz kit --plot

# Optional MP4 recording (imageio/FFmpeg and Pillow required).
$ISAACLAB_PYTHON scripts/isaac/no_controller/moving_mass_dr_demo.py --video --duration 5
```

Red markers show total CoM, green shows mechanism CoM, and orange shows the
vertical ground projection. Comparison rows contain a matched pair, with identical
physical parameters and complete joint-target splines; actual joint motion can
differ because the bases react differently. There are no automatic resets, so
falls remain visible. The zero-command side still uses the base velocity actuator,
which brakes drift; it is not a completely unactuated base.

Both demos save CSV traces under `out/` by default, including joint targets and
measurements, tilt, normalized CoM offset, commanded base velocity, tipping
status, and analytical CoM error. `--plot` adds a PNG with stability/motion curves;
`--video` streams frames to MP4. Use `--csv` and `--video_path` to change outputs.
`--duration` truncates a spline preview without changing the sampled trajectory;
previews longer than an episode hold its final target. For scripted probes, use
`--settle`, `--swing`, and `--hold`. Use `--no_markers` for headless physics-only runs.

The existing `check_moving_mass_env.py` additionally verifies the analytical CoM
against measured joint positions during motion. It remains the automated physics
check; the demos intentionally allow tipping and print first-tip times.

Check domain randomization independently of training:

```bash
$ISAACLAB_PYTHON scripts/isaac/no_controller/check_moving_mass_randomization.py \
    --num_envs 8 --resets 5 --seed 0
```

This checks configured bounds, actual simulator masses and CoM, physical-parameter
observations, fixed footprints across resets, fresh bounded splines, and unchanged
neighboring slots during partial resets. It prints sampled min/mean/max values;
these are diagnostics, not a statistical distribution-fit test. `--config` supports
checking custom ranges, including deliberately constant parameters.

The roll/pitch model is `moving_mass_v2`; older planar-mechanism checkpoints and
YAML configurations are rejected because their joint meanings and dynamics differ.
Start a new training run. Both angular joints span −90° to +90°. The lift stays
vertical relative to the base; roll about X and then pitch about the rotated Y
axis swing the mast radially from its upright pose.

`min_peak_velocity` sets each joint's minimum attained speed per episode (rad/s,
rad/s, m/s), scaled by difficulty. It does not impose nonzero velocity at smooth
reversals or endpoints. New defaults increase speed/acceleration limits and halve
the knot interval. Infeasible custom speed floors raise an error rather than
silently creating a slower trajectory. Individual probes accept `--joint roll`,
`--joint pitch`, or `--joint lift`.
