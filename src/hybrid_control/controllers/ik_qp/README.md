## 🧠 Theoretical Background

### Differential Inverse Kinematics (QP)

We solve the instantaneous kinematics problem by formulating it as a **Quadratic Program (QP)**. This allows us to handle joint limits and velocity bounds strictly.

**The Objective:**
Minimize the error between the desired task-space velocity ($p_d$) and the current robot velocity ($\dot{q}$), while regularizing joint velocities:
$$\min_{\dot{q}} \quad \frac{1}{2} || (\dot p_d - J(q)\dot{q}) +  (p_d-p_{current}) ||^2_W + \frac{1}{2} \lambda ||\dot{q}||^2$$
Subject to:
$$\quad q_{\min} \leq q \leq q_{\max}, \quad$$
$$\dot{q}_{\min} \leq \dot{q} \leq \dot{q}_{\max}$$

Where:

* $p_d$  Is the desired position (velocity here \dot p_d assumed as zero in steady-state).
* $p_{current}$  Is current robot position (forward kinematics).
* $J(q)$  is the robot Jacobian.
* $Kp$  is the CLIK gain.
* $\lambda$ is a regulating factor (Levy-Levenberg) to regulate the constrained solution.
