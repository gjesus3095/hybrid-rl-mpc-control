# Training cheatsheet

How to train and evaluate the RL base policy (the arm is driven by the IK/QP in the loop). Background and design:
`docs/code_overview.md` section 4.

## Setup (once per shell)

All commands run from the Isaac Lab environment, which has Isaac Sim, torch, RSL-RL and OSQP:

```bash
cd ~/isaaclab-3.0.0
export PYTHONPATH=~/hybrid-rl-mpc-control/src
PY=.venv/bin/python
R=~/hybrid-rl-mpc-control
```

## Train

```bash
# Quick smoke test (about 5 min incl. start-up): checks the whole loop works
$PY $R/scripts/isaac/train_base_policy.py --num_envs 64 --max_iterations 20 --qp_workers 16

# Real run (about 35 min): 256 envs, 1000 iterations, about 6M samples
$PY $R/scripts/isaac/train_base_policy.py --num_envs 256 --qp_workers 48 --max_iterations 1000 \
    --log_dir $R/logs/base_policy/my_run
```

- Logs and checkpoints go to `$R/logs/base_policy/<timestamp>/` (or `--log_dir`): `model_<iter>.pt` every 50 iterations
  plus the last one, and the TensorBoard event file. `logs/` is gitignored.
- Resume: add `--resume <dir>/model_<iter>.pt`.
- Other flags: `--seed`, `--num_steps_per_env` (24), `--difficulty` (trajectory amplitude/speed, 0..1),
  `--ik_every` (physics steps per QP update, must divide 4), `--spacing`.
- Speed: about 3000 env-steps/s at 256 envs with `--qp_workers 48`. The QP is the bottleneck, so more envs need more
  workers (up to the number of free cores; the machine has 104). `--qp_workers 0` solves serially (about 10x slower).

## Watch it

```bash
.venv/bin/tensorboard --logdir $R/logs/base_policy --port 4000
```

On your PC open `http://localhost:4001` (the `z8` SSH config forwards 4001 to 4000). Useful scalars:
`Train/mean_reward`, `Episode/failed_fraction`, `Metrics/com_rho` (CoM offset, fraction of the half footprint),
`Metrics/ee_track_error` [m], and `Episode_Reward/<term>` for every reward term.

A healthy run: `failed_fraction` falls toward 0, `com_rho` toward about 0.1, tracking error about 2 cm,
`Episode_Reward/spin|speed|action_rate` stay small (slow, smooth base).

## Evaluate

```bash
CK=$R/logs/base_policy/my_run/model_999.pt
# trained policy: failures, mean CoM offset, base speed / yaw rate / distance from spawn
$PY $R/scripts/isaac/play_base_policy.py --checkpoint $CK --num_envs 128 --episodes 3 --qp_workers 32
# baseline: same envs, base commanded to stand still (expect about 59% failures on the current settings)
$PY $R/scripts/isaac/play_base_policy.py --checkpoint $CK --num_envs 128 --episodes 3 --qp_workers 32 --zero_policy
```

## Videos (written to `$R/out/`, not tracked by git)

```bash
# trained vs untrained side by side, 4 scenarios, live CoM bars
$PY $R/scripts/isaac/compare_policies_video.py --trained $CK --untrained <run>/model_0.pt
# one policy, 6 completely different episodes (add --mode sampled for exploration noise, e.g. with model_0.pt)
$PY $R/scripts/isaac/record_episodes_video.py --checkpoint $CK --title "TRAINED POLICY" --video_path $R/out/episodes.mp4
```

Copy to your PC: `scp z8:/home/u0181636/hybrid-rl-mpc-control/out/<file>.mp4 .`

## Checks and plots

```bash
# integration checks: Jacobian, static-base QP tracking, partial reset, throughput
$PY $R/scripts/isaac/check_stabilization_env.py --num_envs 128 --qp_workers 32
# plot the reference trajectories with the reach limits -> out/reference_trajectories.png
$PY $R/scripts/plot_reference_trajectories.py
# unit tests: the repo suite runs without torch, so tests/test_rl_components.py (needs torch) is skipped there;
# pytest is not installed in the Isaac Lab env, so run that file there only after installing it
cd $R && env -u PYTHONPATH uv run pytest -q
```

## Where to change things

| What | Where |
|---|---|
| Reward terms and weights | `RewardCfg` in `src/hybrid_control/rl/rewards.py` |
| Platform randomization (size, mass, payload, hardness bias) | `DomainRandomization` in `src/hybrid_control/sim_interfaces/isaac/ridgeback/params.py` |
| Reference trajectories (reach, directions, amplitude, speed) | `TrajectoryCfg` in `src/hybrid_control/rl/trajectories.py` |
| Control rate, episode length, base speed limits, QP gains | `StabilizationEnvCfg` / `IKCfg` in `src/hybrid_control/rl/env.py` |
| Observations | `observe()` in `src/hybrid_control/rl/observations.py` |
| PPO hyperparameters, network size | `make_runner_cfg()` in `src/hybrid_control/rl/config.py` |

After changing the difficulty, measure the baseline (zero policy) failure rate: a task where standing still never fails
teaches nothing, one where nothing can succeed teaches nothing either. The current settings give about 59%.

## Troubleshooting

- **Observation size changed** (any edit to `observations.py`): old checkpoints no longer load; retrain.
- **No space left on device**: Isaac leaves a ~70 MB Warp cache per launch; remove with
  `rm -rf ~/.cache/isaaclab_tmp/wp_pch_*` (not the other folders there, those are the asset cache).
- **A robot shows only its COM dots in a video**: rendering quirk of the first env in a scene, physics is fine. The video
  scripts already keep an unused dummy env at index 0.
- **QP worker processes left over after a crash**: `pkill -f hybrid_control.rl.qp_worker`.
- **Kit prints lots of warnings** ("Failed to find rigid body at .../Looks/..."): harmless.
