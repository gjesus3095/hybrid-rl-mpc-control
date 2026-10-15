# Isaac Lab — Ridgeback-Franka with Arm Controller

Scripts for the Ridgeback mobile base with the Franka arm controlled by IK/QP or impedance QP. The RL base policy drives the Ridgeback; the arm controller tracks a world-frame reference trajectory.

Require `ISAACLAB_PYTHON` set and the package installed (see [Isaac Lab setup](../../../README.md#isaac-lab-30--isaac-sim-61-python-312)). Run from the project root:

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/<script>.py [options]
```

See [`docs/isaac_base_disturbance.md`](../../../docs/isaac_base_disturbance.md) for full launch commands and Isaac-specific options.

---

## `train_base_policy.py`

Train the RL base policy (PPO) with a QP arm controller in the loop on domain-randomized Ridgeback-Franka environments. Logs and checkpoints go to `logs/base_policy/<timestamp>/`.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/train_base_policy.py \
    --controller ik_qp --num_envs 64 --max_iterations 500
```

`--controller` selects `ik_qp` (Pink differential IK, position targets) or `impedance` (torque-level QP).

---

## `play_base_policy.py`

Run a trained policy deterministically with its QP arm controller and report survival rate across seeded episodes.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/play_base_policy.py \
    --checkpoint logs/base_policy/<run>/model_499.pt
```

---

## `record_episodes_video.py`

Record a continuous multi-episode video of a trained (or untrained) policy on randomized platforms. Each episode gets its own environment; the camera cuts to the active one.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/record_episodes_video.py \
    --checkpoint <run>/model_999.pt --title "TRAINED POLICY" --video_path out/trained_episodes.mp4
```

`--mode` selects `deterministic`, `sampled` (adds Gaussian noise), or `none` (zero base command).

---

## `compare_policies_video.py`

Record a trained policy next to an untrained one on identical scenarios in one scene.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/compare_policies_video.py \
    --trained <run>/model_149.pt --untrained <run>/model_0.pt
```

---

## `base_disturbance_keyboard.py`

Keyboard-driven Ridgeback base with Pink IK/QP holding the Franka TCP at a world-frame pose target.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/base_disturbance_keyboard.py --viz kit
```

---

## `base_disturbance_impedance.py`

Same base-disturbance experiment with the torque-level impedance QP (default: `--control_hz 50`, `--physics_hz 200`).

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/base_disturbance_impedance.py --viz kit
```

---

## `ridgeback_dr_demo.py`

Domain-randomized Ridgeback-Franka environments side by side: each env has its own footprint, base mass, CoM height, and payload. Envs that tip over stay down; others hold upright.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/ridgeback_dr_demo.py --num_envs 8 --seed 0 --video
```

---

## `ridgeback_tipping_demo.py`

Free-floating base that tips when the total CoM leaves the platform footprint. Visualizes blue (platform CoM), green (arm CoM), and red (total CoM) markers.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/ridgeback_tipping_demo.py --case small_light --video
```

---

## `check_stabilization_env.py`

Integration checks for the RL env with the QP in the loop: Jacobian consistency, QP tracking with zero base commands, partial reset isolation, and throughput timing.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/check_stabilization_env.py --num_envs 16
```

---

## `ridgeback_base_probe.py`

Probe how to change the Ridgeback-Franka mobile base: platform mass, robot mass scale, and base geometry.

```bash
PYTHONPATH=~/workspace/hybrid-rl-mpc-control/src \
    $ISAACLAB_PYTHON scripts/isaac/with_controller/ridgeback_base_probe.py \
    --base_mass 250 --robot_mass_scale 1.2 --base_scale 1.5 1.0 1.0
```

---

## `ridgeback_demo.py`

Basic Ridgeback-Franka demo for verifying the USD asset and joint addressing (GUI only).

```bash
<IsaacLab>/isaaclab.sh -p scripts/isaac/with_controller/ridgeback_demo.py --viz kit
```
