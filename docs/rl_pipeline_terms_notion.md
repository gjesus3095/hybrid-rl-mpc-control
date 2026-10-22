# 🤖 RL Base Policy: Terms Overview

> The policy drives the Ridgeback base. A QP controller (`ik_qp` or `impedance`) drives the Franka arm along a world-fixed reference. Goal: keep the whole-robot COM over a small, randomized platform without tipping.

**Rates:** policy 30 Hz · arm QP 60 Hz · physics 120 Hz · episode 5 s

**Base frame:** yaw-aligned frame at the platform footprint centre

---

## 🎮 Actions

- **`base_velocity`** (3): vx, vy, wz, scaled to ±1 m/s, ±1 m/s, ±0.6 rad/s. The only thing the policy controls; the arm is handled by the QP.

---

## 👀 Observations

### Actor (39)

- **`ref_lookahead`** (15): arm reference at t + 0, 0.25, 0.5, 1, 1.5 s. Lets the policy anticipate where the arm mass will go.
- **`ref_orientation`** (30): reference TCP orientation at the same times (first two rotation-matrix columns, base frame).
- **`com_total`** (3): whole-robot COM (x, y, height). The quantity to keep over the footprint.
- **`com_norm`** (2): COM x, y as a fraction of the half footprint (±1 = edge). Comparable across platform sizes.
- **`base_lin_vel`** (2): base planar velocity. For smooth, damped corrections.
- **`base_ang_vel`** (3): roll/pitch rates and yaw rate. Early warning of tipping.
- **`roll`, `pitch`** (2): platform tilt. Direct tipping state.
- **`ee_pos`** (3): actual end-effector position. Shows how far the arm lags the reference.
- **`last_action`** (3): previous command. For smooth actions.
- **`physical_params`** (6): length, width, base mass, base COM height, payload, total mass. Lets the policy adapt to the randomized robot.

### Critic only (43 = actor + 4)

- **`com_offset_norm`** (1): the COM offset that is rewarded and terminated on. Better value estimate.
- **`arm_com`** (3): arm-only COM. Separates the arm's contribution to the total COM.

---

## 🏆 Rewards

- **`com`** · **+2.0** · exp(−(ρ / 0.5)²), ρ = COM offset / half footprint. Main objective: COM centred over the platform.
- **`tilt`** · **−1.0** · (roll² + pitch²) / 0.25². Penalizes leaning before it becomes a tip.
- **`centre`** · **+0.5** · exp(−(max(d − 0.35, 0) / 0.5)²), d = footprint centre to trajectory centre. Stay near the trajectory the arm serves.
- **`progress`** · **+1.0** · clamp(approach rate of d beyond 0.35 m, −1, 1). Potential-based push toward the trajectory; oscillating earns nothing.
- **`speed`** · **−1.0** · (base speed / 1.0)². Prefer slow base motion.
- **`spin`** · **−1.0** · (yaw rate / 0.6)². Prevent spinning.
- **`yaw`** · **−0.3** · clamp(yaw / 0.6, −3, 3)². No reason to turn away from the spawn heading.
- **`action_rate`** · **−0.3** · ‖aₜ − aₜ₋₁‖². Smooth commands.
- **`action`** · **−0.05** · ‖a‖². Small commands.
- **`track`** · **+0.5** · exp(−(EE error / 0.08)²). Don't put the base where the arm can't track.
- **`termination`** · **−10.0** · 1 on failure, not on time-out. Makes failure costly.

---

## 🛑 Terminations

- **`tipped`** (failure): abs(roll) or abs(pitch) > 0.5 rad. The robot has fallen.
- **`left_footprint`** (failure): total COM outside the footprint. Statically unstable state.
- **`time_out`** (not a failure): 5 s elapsed (= the trajectory duration). Episode limit; bootstrapped, not penalized.

---

## 🎲 Randomization at reset

- **Base mass:** 5.5 to 38 kg
- **Payload at the TCP:** 0 to 3 kg
- **Footprint:** 0.19 to 0.52 m per side, fixed per env at spawn
- **Reference trajectory:** new fixed point (20%), Lissajous path at 0.1 to 0.45 Hz with an orientation oscillating up to 20° per axis about a down-pointing centre (40%), or quintic pose path spawn → A → B with the TCP pointing down (40%)

Sampling is biased toward the hard end: small, light platform with a heavy payload.
