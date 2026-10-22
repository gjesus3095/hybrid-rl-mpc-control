# RL base policy: observations, actions, rewards, terminations

Source: `src/hybrid_control/rl/` (`observations.py`, `rewards.py`, `env.py`). "Base frame" = yaw-aligned frame at the platform footprint centre. Policy 30 Hz, arm QP 60 Hz, physics 120 Hz, episodes 5 s.

## Actions (3)

| Term | Dim | Purpose |
|---|---|---|
| `base_velocity` | 3 | Base-frame (vx, vy, wz) in [-1, 1], scaled to (1 m/s, 1 m/s, 0.6 rad/s); the only thing the policy controls. The arm is driven by the QP (`--controller ik_qp` or `impedance`). |

## Observations

Actor (39):

| Term | Dim | Purpose |
|---|---|---|
| `ref_lookahead` | 15 | Arm reference at t + (0, 0.25, 0.5, 1, 1.5) s: lets the policy anticipate where the arm (and its mass) will go. |
| `ref_orientation` | 30 | Reference TCP orientation at the same times (first two rotation-matrix columns, base frame). |
| `com_total` | 3 | Whole-robot COM (x, y, height): the quantity to keep over the footprint. |
| `com_norm` | 2 | COM x, y as a fraction of the half footprint: same, normalized across platform sizes (±1 = edge). |
| `base_lin_vel` | 2 | Base planar velocity: needed for smooth, damped corrections. |
| `base_ang_vel` | 3 | Roll/pitch rates and yaw rate: early warning of tipping; yaw rate control. |
| `roll`, `pitch` | 1 + 1 | Platform tilt: direct tipping state. |
| `ee_pos` | 3 | End-effector position: where the arm actually is (lags the reference). |
| `last_action` | 3 | Previous command: lets the policy keep actions smooth (action-rate penalty). |
| `physical_params` | 6 | Length, width, base mass, base COM height, payload, total mass: adapt to the randomized robot. |

Critic (43) = actor + privileged:

| Term | Dim | Purpose |
|---|---|---|
| `com_offset_norm` | 1 | COM offset as a fraction of the half footprint (max-norm), the quantity rewarded/terminated on: better value estimate. |
| `arm_com` | 3 | Arm-only COM: separates the arm's contribution to the total COM. |

## Rewards (summed per policy step)

| Term | Weight | Expression | Purpose |
|---|---|---|---|
| `com` | 2.0 | exp(-(rho / 0.5)²), rho = COM offset / half footprint | Main objective: keep the COM centred over the platform. |
| `tilt` | -1.0 | (roll² + pitch²) / 0.25² | Penalize leaning before it becomes a tip-over. |
| `centre` | 0.5 | exp(-(max(d - 0.35, 0) / 0.5)²), d = footprint centre → trajectory centre | Stay near the trajectory the arm serves. |
| `progress` | 1.0 | clamp(rate of approach of d beyond 0.35 m, -1, 1) | Potential-based shaping toward the trajectory; back-and-forth earns nothing. |
| `speed` | -1.0 | (\|v_base\| / 1.0)² | Prefer slow base motion. |
| `spin` | -1.0 | (yaw rate / 0.6)² | Prevent spinning. |
| `yaw` | -0.3 | clamp(yaw / 0.6, -3, 3)² | No reason to turn away from the spawn heading. |
| `action_rate` | -0.3 | \|a_t - a_t-1\|² | Smooth commands. |
| `action` | -0.05 | \|a\|² | Small commands. |
| `track` | 0.5 | exp(-(ee error / 0.08)²) | Don't place the base where the arm can't track its reference. |
| `termination` | -10.0 | 1 on failure (not on time-out) | Make failure costly. |

## Terminations

| Term | Time-out | Purpose |
|---|---|---|
| `tipped` | no | \|roll\| or \|pitch\| > 0.5 rad: the robot has fallen. |
| `left_footprint` | no | Total COM outside the footprint: statically unstable, counted as failure. |
| `time_out` | yes | 5 s episode limit (= the trajectory duration); bootstrapped, not penalized. |

At reset: new base mass (5.5–38 kg), payload (0–3 kg) and reference trajectory (fixed point 20 %, Lissajous with an oscillating down-pointing orientation 40 %, quintic spawn → A (2.5 s) → B (2.5 s) pose path 40 %); footprint (0.19–0.52 m per side) is fixed per env at spawn.
