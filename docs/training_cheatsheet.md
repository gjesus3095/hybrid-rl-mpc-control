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
- Other flags: `--seed`, `--num_steps_per_env` (24), `--difficulty` (trajectory amplitude/speed, 0..1), `--spatial_order` (Bezier order of the quintic paths, 1 = straight; play defaults to the checkpoint's),
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

## Benchmark (deterministic evaluation set)

Five sets of 100 scenarios, identical for every policy (`src/hybrid_control/rl/evaluation/`): `fixed`, `sine`, `poly`
(training-style), `poly_far` (quintic whose B is 0.9-2 m from the arm base, outside the 0.8 m training reach, A -> B timed
at the training mean speed 0.25 m/s) and `rigid` (constant-speed straight legs, 0 / 1 / 2 rounded 90 degree bends, inside the
training workspace). Slot `j` has footprint `j` in every set; each scenario has its own mass / payload.

`fixed`, `sine` and `rigid` only contain references that the arm can follow from the spawn pose **with a fixed base**
(`evaluation/reachability.py`: continuous IK along the timed reference with orientation held, joint limits with a 0.05 rad
margin and joint speeds below 70 % of the Panda limits; candidates are redrawn until 100 pass). `poly` and `ellipse` were
checked the same way and pass; `poly_far` is outside the arm's reach on purpose. On a welded base the three filtered sets
track to 0.1-0.6 cm RMS (zero policy, impedance controller, no payload).

```bash
# print the benchmark fingerprint and statistics, plot it (no Isaac needed)
PYTHONPATH=$R/src python3 $R/scripts/generate_eval_set.py --plot $R/out/eval/benchmark.png
# play a policy on it (about 10 min); logs + summary.json go to out/eval/<run>_<model>/
$PY $R/scripts/isaac/evaluate_policy.py --checkpoint $CK --qp_workers 32
$PY $R/scripts/isaac/evaluate_policy.py --zero_policy --qp_workers 32      # baseline
# welded base (cannot move or tip, nothing terminates; COM-margin violations are counted as "would have failed")
$PY $R/scripts/isaac/evaluate_policy.py --zero_policy --fixed_base --controller impedance --qp_workers 32
# several payloads in one process, one folder per payload; --name sets the output folder out/eval/<name>
$PY $R/scripts/isaac/evaluate_policy.py --checkpoint $CK --payloads 0 1.5 3 --qp_workers 32 --name <name>
# smoke test: first 8 scenarios of one set (same scenarios as in the full run)
$PY $R/scripts/isaac/evaluate_policy.py --checkpoint $CK --sets poly_far --num_scenarios 8 --qp_workers 8
# poly / poly_far along Bezier paths of order 3 under the quintic time scaling (as training --spatial_order 3);
# default 1 = straight segments, other orders go to out/eval/<run>_<model>_order<n>/
$PY $R/scripts/isaac/evaluate_policy.py --checkpoint $CK --spatial_order 3 --sets poly poly_far --qp_workers 32
# every metric of one or more runs in one text file
python3 $R/scripts/make_eval_summary.py $R/out/eval/<run>_<model> $R/out/eval/<run>_<model>_order3 --out $R/out/eval_summary.txt
```

`<set>.npz` holds per-step arrays `(scenario, step, dim)` (see `LOG_FIELDS` in `evaluation/metrics.py`: reference, TCP pose
and error, base state, COM and its margin to the footprint edge, arm joints, actions) plus `valid` (steps that count: within
the scenario and up to the failure) and `fail_step`. Replays of the same policy are identical on this machine.

## Videos (written to `$R/out/`, not tracked by git)

```bash
# trained vs untrained side by side, 4 scenarios, live CoM bars
$PY $R/scripts/isaac/compare_policies_video.py --trained $CK --untrained <run>/model_0.pt
# one policy, 6 completely different episodes (add --mode sampled for exploration noise, e.g. with model_0.pt)
$PY $R/scripts/isaac/record_episodes_video.py --checkpoint $CK --title "TRAINED POLICY" --video_path $R/out/episodes.mp4
```

Copy to your PC: `scp z8:/home/u0181636/hybrid-rl-mpc-control/out/<file>.mp4 .`

## Chicken overlay (videos only)

`--chicken` on `record_episodes_video.py` dresses every robot as a cartoon chicken: body on the base, head on the
hand, beak on the fingers, and a neck tube that follows the arm. It is visual only, so physics, policy and QP behave
exactly as without it. The Franka meshes are hidden under the skin.

```bash
# trained policy with the chicken skin
$PY $R/scripts/isaac/record_episodes_video.py --checkpoint $CK --title "CHICKEN" --chicken \
    --video_path $R/out/chicken_episodes.mp4
# same, but the head looks ahead (TCP forward, 10 deg down, comb up) instead of the training orientation (TCP down);
# out of the training distribution, so expect worse tracking/stability
$PY $R/scripts/isaac/record_episodes_video.py --checkpoint $CK --title "CHICKEN" --chicken --head_forward \
    --kinds quintic --view front --video_path $R/out/chicken_head_forward.mp4
```

- `--head_forward` only changes the quintic and sine targets; a fixed point keeps its spawn orientation, hence
  `--kinds quintic` (or `quintic sine`) to make the head pose visible in every episode.
- Still image of the skin (no policy, arm moved to a head-forward pose) -> `out/chicken_snapshot.png`:
  `$PY $R/scripts/isaac/chicken_snapshot.py` (`--show_robot` keeps the Franka visible, `--eye`/`--target` move the camera).
- The meshes and overlay URDF (`src/hybrid_control/sim_interfaces/isaac/assets/chicken/`) are generated; after
  editing the parts in `scripts/make_chicken_assets.py`, regenerate them with
  `cd $R && uv run python scripts/make_chicken_assets.py`.

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
| Reference trajectories (reach, directions, amplitude, speed) | `TrajectoryCfg` in `src/hybrid_control/rl/policy_with_controller/trajectories.py` |
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
