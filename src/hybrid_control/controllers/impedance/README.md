# Task-Space Impedance Controller (Inverse-Dynamics QP)

`impedance_control.py` implements `ImpedanceController`, a torque-level controller for the Panda arm. Every control step (2 ms, once per MuJoCo physics step) it solves one quadratic program (QP) with OSQP. The QP's solution is the joint torques sent to the robot.

## 🧠 Theoretical Background

### Notation

| Symbol | Meaning | Shape |
|---|---|---|
| $n$ | number of controlled joints (7 for the Panda) | — |
| $q,\ \dot q,\ \ddot q$ | joint positions, velocities, accelerations | $n$ |
| $\tau$ | joint torques | $n$ |
| $M(q)$ | joint-space inertia matrix (includes armature) | $n \times n$ |
| $h(q,\dot q)$ | Coriolis, centrifugal and gravity forces, minus passive forces (joint damping) | $n$ |
| $J(q)$ | world-frame Jacobian of the hand frame, rows [linear; angular] | $6 \times n$ |
| $\dot J(q,\dot q)$ | time derivative of $J$ | $6 \times n$ |
| $x,\ \dot x$ | hand pose and twist (linear and angular velocity) | — , $6$ |
| $x_{ref},\ \dot x_{ref},\ \ddot x_{ref}$ | reference pose, twist and twist acceleration from the trajectory | — , $6$, $6$ |

The robot dynamics are

$$M(q)\,\ddot q + h(q,\dot q) = \tau$$

and the hand acceleration is

$$\ddot x = J\,\ddot q + \dot J\,\dot q.$$

### Cartesian impedance law

The desired hand acceleration is a PD law around the reference trajectory, plus the trajectory's own acceleration as feedforward:

$$\ddot x_{des} = \ddot x_{ref} + K_p\, e + K_d\,(\dot x_{ref} - \dot x), \qquad e = \begin{bmatrix} p_{ref} - p \\ \mathrm{Log}\!\left(R_{ref} R^\top\right) \end{bmatrix}$$

- $p$ and $R$ are the hand position and rotation matrix.
- $\mathrm{Log}(\cdot)$ is the rotation vector (axis × angle) in the world frame.
- $K_p = \mathrm{diag}(k_{p,pos} I_3,\ k_{p,ori} I_3)$, and $K_d$ likewise.
- By default $K_d = 2\sqrt{K_p}$ (critical damping).

The references come from `QuinticTrajectory` (rest-to-rest $s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$) or `SinusoidalTrajectory` in `utils/trajectories.py`.

### The QP

**Decision variable:** joint accelerations, joint torques and one slack variable,

$$z = \begin{bmatrix} \ddot q \\ \tau \\ s \end{bmatrix} \in \mathbb{R}^{2n+1}.$$

**Cost:**

$$\min_{\ddot q,\ \tau,\ s} \quad \underbrace{\left\lVert \ddot x_{des} - \left(J\ddot q + \dot J \dot q\right) \right\rVert_W^2}_{\text{task tracking}} \;+\; \underbrace{\lambda_{\ddot q} \left\lVert \ddot q - \ddot q_{ref} \right\rVert^2 + \lambda_\tau \lVert \tau \rVert^2}_{\text{Tikhonov regularization}} \;+\; \underbrace{\rho\, s^2}_{\text{slack penalty}}$$

with the regularization reference

$$\ddot q_{ref} = \underbrace{k_w\, \nabla_q w(q)}_{\text{option 3}} \;+\; K_{p,post}\,(q_{post} - q) - K_{d,post}\,\dot q.$$

**Equality constraint (rigid-body dynamics):**

$$\begin{bmatrix} M & -I \end{bmatrix} \begin{bmatrix} \ddot q \\ \tau \end{bmatrix} = -h$$

**Inequality constraints:**

$$\ddot q_{min} \le \ddot q \le \ddot q_{max} \qquad \text{(joint acceleration limits)}$$

$$\tau_{min} \le \tau \le \tau_{max} \qquad \text{(joint torque limits)}$$

$$\nabla_q w^\top \ddot q + s \;\ge\; -\dot{(\nabla_q w)}^\top \dot q \;-\; 2\alpha_w\, \nabla_q w^\top \dot q \;-\; \alpha_w^2\,(w - w_{min}) \qquad \text{(manipulability floor, option 3)}$$

$$s \ge 0$$

If joint position limits are enabled, the acceleration bounds are tightened per joint (but never beyond the physical acceleration limits):

$$-2\alpha_q \dot q_i - \alpha_q^2\,\big(q_i - (q_{i,min} + m)\big) \;\le\; \ddot q_i \;\le\; -2\alpha_q \dot q_i + \alpha_q^2\,\big((q_{i,max} - m) - q_i\big)$$

**Output.** The commanded torque is the QP's $\tau$, plus the optional null-space torque (option 2), clipped to $[\tau_{min}, \tau_{max}]$.

**Remarks**

- Gravity compensation comes from the equality constraint. When the reference is a fixed pose, the solution is $\ddot q = 0$, $\tau = h$.
- $\tau$ is fully determined by $\ddot q$ through the dynamics. Keeping it as a variable makes the torque limits plain linear bounds.
- $\lambda_{\ddot q} > 0$ makes the solution unique. The task weight $W$ (default $I$) is much larger than $\lambda_{\ddot q}$ (default $10^{-4}$), so the regularization mostly acts on the arm's redundant motion: motion that doesn't affect the hand.
- If the acceleration and torque limits together make the dynamics constraint infeasible (e.g. they can't hold the arm against gravity), OSQP fails and `compute` raises `RuntimeError`.
- Coupling to the gripper fingers is ignored in $M$ and $h$.
- $\dot J$ is computed by finite differences. MuJoCo 3.2.7's `mj_jacDot` returns wrong translational rows.

## ⚙️ Options

All options are fields of `ImpedanceQPConfig`.

### 1. Torque-level impedance control (always on)

This is the QP above, without options 2 and 3. The Panda's position servos are replaced by torque motors (`ctrl` is joint torque in N·m, bounded by each actuator's force range):

```python
import mujoco
from hybrid_control.controllers.impedance.impedance_control import (
    ImpedanceController, ImpedanceQPConfig, TaskSpaceTarget,
)
from hybrid_control.sim_interfaces.mujoco.adapter import MujocoTorqueRobotModel
from hybrid_control.sim_interfaces.mujoco.modifier import MujocoModifier

spec = MujocoModifier.build_torque_spec(xml_file, [f"actuator{i}" for i in range(1, 8)])
model = spec.compile()
robot = MujocoTorqueRobotModel(model, mujoco.MjData(model), joint_actuator_pairs)

controller = ImpedanceController(robot, ImpedanceQPConfig(
    acceleration_lower=-ddq_max, acceleration_upper=ddq_max,
    torque_lower=-tau_max, torque_upper=tau_max,   # intersected with the actuator limits
    kp_pos=400.0, kp_ori=200.0,                    # 1/s²; kd_* = None gives 2*sqrt(kp)
    acceleration_regularization=1e-4, torque_regularization=1e-6,
    posture_kp=10.0, posture=q_init,               # QP posture term in ddq_ref
))
command = controller.compute(TaskSpaceTarget.from_sample("grasp", trajectory.sample(t)), dt)
robot.apply_command(command)                       # JointTorqueCommand
```

| Parameter | Symbol | Default |
|---|---|---|
| `kp_pos`, `kp_ori` | $k_{p,pos}$, $k_{p,ori}$ | 400, 200 (1/s²) |
| `kd_pos`, `kd_ori` | $k_{d,pos}$, $k_{d,ori}$ | $2\sqrt{k_p}$ |
| `position_weight`, `orientation_weight` | $W$ | 1, 1 |
| `acceleration_regularization` | $\lambda_{\ddot q}$ | $10^{-4}$ |
| `torque_regularization` | $\lambda_\tau$ | $10^{-6}$ |
| `acceleration_lower/upper` | $\ddot q_{min}$, $\ddot q_{max}$ | required |
| `torque_lower/upper` | $\tau_{min}$, $\tau_{max}$ | actuator limits |
| `posture_kp`, `posture_kd`, `posture` | $K_{p,post}$, $K_{d,post}$, $q_{post}$ | 10, $2\sqrt{K_{p,post}}$, first measured $q$ |
| `joint_limit_margin`, `joint_limit_rate` | $m$, $\alpha_q$ | off, 10 (1/s) |

Run it with `python scripts/mujoco/impedance_tracking.py`. The `TRAJECTORY` setting picks `"quintic"` or `"sine"`.

### 2. Null-space projection to hold the joints near a center posture

After the QP, a joint-space PD torque towards a center posture $q_{mid}$ is projected into the null space of the task and added:

$$\Lambda = \left(J M^{-1} J^\top\right)^{-1}, \qquad N = I - J^\top \Lambda\, J M^{-1}$$

$$\tau = \tau_{QP} + N \left( P_{null}\,(q_{mid} - q) - D_{null}\,\dot q \right)$$

$\Lambda$ is the task-space inertia, and $N$ is the dynamically consistent null-space projector. $J M^{-1} N = 0$, so the added torque causes no hand acceleration. It only drives the arm's redundant motion: for the 7-DoF Panda with a 6-D task, that is a single DoF, the elbow swinging around the shoulder–wrist line. $\Lambda$ uses a pseudo-inverse so $N$ stays defined at singularities.

```python
ImpedanceQPConfig(
    ...,
    posture_kp=0.0, posture_kd=0.0,          # turn off the QP posture term so the two don't compete
    nullspace_posture=np.array([0, 0, 0, -1.5, 0, 1.885, 0]),
    nullspace_stiffness=10.0,                # P_null, N·m/rad (scalar or one per joint)
    nullspace_damping=1.0,                   # D_null, N·m·s/rad
)
```

In `scripts/mujoco/impedance_tracking.py` this is on by default (`USE_NULLSPACE`, `Q_MID`, `P_NULLSPACE`, `D_NULLSPACE`). The applied null-space torque is in `controller.last_solution.nullspace_torques`.

Only the component of $q_{mid} - q$ along the elbow swing has any effect. With the hand pose fixed, the Panda's start pose and $q_{mid} = [0, 0, 0, -1.5, 0, 1.885, 0]$ differ only in directions the task fixes, so almost no torque is produced. A center posture that differs in the swing direction, e.g. $[0.4, 0, -0.6, -1.5, 0, 1.885, 0]$, makes joints 1 and 3 rotate while the hand stays within 0.02 mm.

### 3. Maximizing manipulability to avoid singularities

The Yoshikawa manipulability of the chosen Jacobian rows $J_r$ (`manipulability_rows`: `"position"`, `"orientation"` or `"full"`) is

$$w(q) = \sqrt{\det\!\left(J_r J_r^\top\right)}$$

It is zero exactly at a singularity. Its gradient is computed analytically from the Jacobian's derivative with respect to each joint, $H_i = \partial J / \partial q_i$:

$$\frac{\partial w}{\partial q_i} = w \cdot \mathrm{tr}\!\left( \left(J_r J_r^\top\right)^{-1} J_r H_{r,i}^\top \right)$$

For a serial chain of revolute joints, with joint axes $z_k$ and linear columns $v_k$ of $J$, the derivatives are

$$\frac{\partial z_j}{\partial q_i} = z_i \times z_j \ \ (i < j,\ \text{else } 0), \qquad \frac{\partial v_j}{\partial q_i} = z_a \times v_b,\quad a = \min(i,j),\ b = \max(i,j)$$

(Haviland & Corke, *Manipulator Differential Kinematics, Part II*). This is in `utils/manipulability.py`.

There are two ways to use it, separately or together:

- **Secondary objective (`manipulability_gain` $= k_w$).** Adds $k_w \nabla_q w$ to $\ddot q_{ref}$, so the redundant motion climbs the manipulability gradient (gradient projection, as a soft priority in the QP).
- **Manipulability floor (`manipulability_min` $= w_{min} > 0$).** Adds the barrier inequality above, which keeps $h = w - w_{min} \ge 0$ by enforcing $\ddot h + 2\alpha_w \dot h + \alpha_w^2 h \ge 0$ (`manipulability_barrier_rate` $= \alpha_w$). $\dot{(\nabla_q w)}$ is a backward difference over one control step. The slack $s$, penalized by `manipulability_slack_weight` $= \rho$ (default $10^3$), keeps the QP feasible when the floor can't be met.

```python
ImpedanceQPConfig(
    ...,
    manipulability_rows="full",
    manipulability_gain=20.0,       # k_w, secondary objective
    manipulability_min=0.03,        # w_min, barrier floor (0 disables)
    manipulability_barrier_rate=10.0,
    joint_limit_margin=0.05,        # recommended together, see below
)
```

**Status and limitations**

- These options exist in the controller but are **not** wired into `impedance_tracking.py` yet. The gradient math is verified against finite differences, but the options themselves have no dedicated tests.
- The redundant motion can't avoid the Panda's **stretched-arm singularity**. Joints 1–3 form a spherical shoulder and joints 5–6 share a wrist point, so the shoulder–wrist distance depends only on $q_4$. A fixed hand pose leaves just an 88 mm circle for the wrist point, so $q_4$ is essentially set by the task. Near the workspace boundary only the floor helps, by letting the tracking error grow instead of becoming singular.
- The gradient vanishes at symmetric postures such as the home pose ($q_1 = q_3 = q_5 = 0$), which are already local maxima along the redundant motion.
- Because of the 82.5 mm elbow offsets, the Panda is geometrically straight at $q_4 \approx -0.467$ rad, not at $0$. `scripts/mujoco/impedance_singularity.py` drives the arm through that singularity. Without joint-limit barriers, the arm can end on the flipped side of the elbow, pushing joint 4 into its limit; `joint_limit_margin` prevents this.
