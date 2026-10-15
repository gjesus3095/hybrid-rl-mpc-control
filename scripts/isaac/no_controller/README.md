# Isaac Lab — Controller-Free Moving-Mass Policy

Scripts for training, evaluating, and visualizing the moving-mass base policy. The policy drives the Ridgeback base; a 3-DoF mechanism (roll, pitch, lift) follows a randomized joint spline each episode and shifts the whole-robot CoM. There is no arm controller. The mast swings on roll/pitch joints (−90° to +90° each),
and all rigid bodies have collision shapes for physical falls. This is the
`moving_mass_v3` model; older v1/v2 checkpoints require retraining.

Carriage (0.25–1 kg), roll hub (0.125–0.5 kg), mast (0.125–0.5 kg), and terminal (0.8–3 kg) masses are sampled independently each episode. Configure these ranges under `domain_randomization`.

Require `ISAACLAB_PYTHON` set and the package installed (see [Isaac Lab setup](../../../README.md#isaac-lab-30--isaac-sim-61-python-312)). Run from the project root:

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/no_controller/<script>.py [options]
```

See the [RL policy overview](../../../src/hybrid_control/rl/README.md#controller-free-moving-mass-policy) for training workflow, YAML configuration, and curriculum details.

---

## `train_moving_mass_policy.py`

Train PPO on domain-randomized moving-mass platforms. Logs and checkpoints go to `logs/moving_mass_policy/<timestamp>/`.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/no_controller/train_moving_mass_policy.py \
    --num_envs 64 --max_iterations 500
```

Monitor with `tensorboard --logdir logs`.

Key options: `--config <yaml>`, `--resume <checkpoint>`, `--seed`, `--num_steps_per_env`.

---

## `play_moving_mass_policy.py`

Evaluate a trained policy (or a zero-command baseline) on seeded episodes and report survival rate and metrics.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/no_controller/play_moving_mass_policy.py \
    --checkpoint logs/moving_mass_policy/<run>/model_499.pt
```

Add `--video` to record to `out/play_moving_mass.mp4` (override with `--video_path`). Use `--zero_policy` instead of `--checkpoint` for a zero-command baseline.

---

## `moving_mass_dr_demo.py`

Visualize randomized moving-mass platforms with scripted motion probes or episode splines.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/no_controller/moving_mass_dr_demo.py --viz kit --plot
```

Use `--scenario joint_sweep --joint roll` (or `pitch`, `lift`) to isolate an axis.
Add `--video` to record to `out/moving_mass_dr.mp4`. Use `--scenario` to select `spline` (default), `extend`, or `joint_sweep`.

---

## `compare_moving_mass.py`

Side-by-side comparison: left column runs zero commands, right column runs a fixed base command or a trained checkpoint on identical platforms and trajectories.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/no_controller/compare_moving_mass.py \
    --cmd .2 0 0 --viz kit --plot
```

Add `--video` to record to `out/moving_mass_compare.mp4`.

---

## `check_moving_mass_env.py`

Headless assertion suite: verifies physical mass/CoM, spline tracking accuracy, reset isolation, and episode termination behavior.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/no_controller/check_moving_mass_env.py --num_envs 4 --headless
```

Exits with a non-zero code on any assertion failure.

Shared platform defaults (footprint, base mass, CoM height and their randomization
ranges/bias) follow `src/hybrid_control/sim_interfaces/isaac/ridgeback/params.py`.
Common reward defaults follow `src/hybrid_control/rl/policy_with_controller/rewards.py`.
Upper-body mass ranges and joint settings remain in `moving_mass/params.py`.
Explicit YAML overrides and saved run configurations take precedence over defaults.

Terminal mass defaults and sampling bias match Ridgeback payload settings: `payload`
and `hard_bias` in `ridgeback/params.py` (biased toward heavier payloads).
