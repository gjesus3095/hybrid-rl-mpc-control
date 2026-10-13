import numpy as np

from hybrid_control.controllers.ik_qp.controller import IKQPConfig, InverseKinematicsController, PoseTarget
from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import CenterOfMassState, JointState, Pose
from hybrid_control.rl.qp_pool import QPPool

N_ENVS, N_DOF = 7, 7
LIMITS = (-np.full(N_DOF, 2.9), np.full(N_DOF, 2.9))
CONFIG = IKQPConfig(velocity_lower=-np.full(N_DOF, 2.0), velocity_upper=np.full(N_DOF, 2.0), kp_pos=10.0, damping=1e5)
DT = 1.0 / 60.0


class FixedModel(RobotModel):
    def __init__(self, q, pos, quat, jac):
        self.q, self.pos, self.quat, self.jac = q, pos, quat, jac

    joint_names = tuple(f"j{i}" for i in range(N_DOF))
    position_limits = LIMITS

    def get_state(self):
        return JointState(0.0, self.q, np.zeros(N_DOF))

    def get_com_state(self):
        return CenterOfMassState(0.0, np.zeros(3))

    def get_pose(self, frame_name):
        return Pose(self.pos, self.quat)

    def get_jacobian(self, frame_name):
        return self.jac

    def apply_command(self, command):
        raise NotImplementedError


def test_pool_matches_serial_controller_and_respects_active_mask():
    rng = np.random.default_rng(0)
    q = rng.uniform(-1, 1, (N_ENVS, N_DOF))
    pos = rng.uniform(0.3, 0.8, (N_ENVS, 3))
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (N_ENVS, 1))
    jac = rng.normal(size=(N_ENVS, 6, N_DOF))
    target = pos + rng.uniform(-0.2, 0.2, (N_ENVS, 3))
    active = np.array([1, 1, 0, 1, 1, 1, 0], dtype=float)

    pool = QPPool(N_ENVS, FixedModel.joint_names, LIMITS, CONFIG, DT, "hand", workers=3)
    try:
        q_cmd, failures = pool.solve(q, pos, quat, jac, target, quat, active)
        q_cmd_again, _ = pool.solve(q, pos, quat, jac, target, quat, active)  # a second request works too
    finally:
        pool.close()

    assert failures == 0
    assert np.allclose(q_cmd, q_cmd_again)
    for i in range(N_ENVS):
        if not active[i]:
            assert np.array_equal(q_cmd[i], q[i])
            continue
        serial = InverseKinematicsController(FixedModel(q[i], pos[i], quat[i], jac[i]), CONFIG)
        expected = serial.compute(PoseTarget("hand", Pose(target[i], quat[i])), DT).positions
        assert np.allclose(q_cmd[i], expected, atol=1e-6)
