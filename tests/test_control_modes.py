"""Control-mode contracts, compiled actuator laws, and controller compatibility."""

import mujoco
import numpy as np
import pytest

from hybrid_control.controllers.base import Controller
from hybrid_control.controllers.ik_qp.ik_qp_mujoco import InverseKinematicsController
from hybrid_control.utils.control_utils import IKQPConfig
from hybrid_control.controllers.ik_qp.parallel import ArrayRobotModel, ParallelPinkIK
from hybrid_control.controllers.impedance.impedance_control import ImpedanceController
from hybrid_control.utils.control_utils import ImpedanceQPConfig
from hybrid_control.robots.state import JointPositionCommand, JointVelocityCommand, JointTorqueCommand
from hybrid_control.sim_interfaces.mujoco.adapter import MujocoRobotModel
from test_model_controller import MODEL_XML, PAIRS, SliderModel


COMMANDS = {"position": JointPositionCommand, "velocity": JointVelocityCommand, "torque": JointTorqueCommand}
LAWS = {
    "position": 'gaintype="fixed" gainprm="100" biastype="affine" biasprm="0 -100 -2"',
    "velocity": 'gaintype="fixed" gainprm="100" biastype="affine" biasprm="0 0 -100"',
    "torque": 'gaintype="fixed" gainprm="1" biastype="none"',
}


def xml_for(mode, general=False):
    xml = MODEL_XML
    for joint, actuator in PAIRS:
        if general:
            replacement = f'<general name="{actuator}" joint="{joint}" {LAWS[mode]}/>'
        else:
            tag = "motor" if mode == "torque" else mode
            gain = 'kp="100"' if mode == "position" else 'kv="100"' if mode == "velocity" else ""
            replacement = f'<{tag} name="{actuator}" joint="{joint}" {gain}/>'
        xml = xml.replace(f'<position name="{actuator}" joint="{joint}" kp="100"/>', replacement)
    return xml


def make_robot(mode, general=False):
    model = mujoco.MjModel.from_xml_string(xml_for(mode, general))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MujocoRobotModel(model, data, PAIRS, control_mode=mode, base_actuator_names=["unused_servo"])


@pytest.mark.parametrize("mode", COMMANDS)
@pytest.mark.parametrize("general", [False, True])
def test_modes_initialize_and_write_only_selected_references(mode, general):
    robot = make_robot(mode, general)
    robot.data.time = 1.25
    robot.data.qvel[:] = 0.1
    robot.initialize([0.1, -0.2], extra_actuator_controls={"unused_servo": 0.3})
    np.testing.assert_array_equal(robot.get_state().positions, [0.1, -0.2])
    np.testing.assert_array_equal(robot.get_state().velocities, 0)
    expected = [0.1, -0.2] if mode == "position" else [0, 0]
    np.testing.assert_array_equal(robot.data.ctrl[robot._actuator_ids], expected)
    assert robot.data.qvel[robot.model.joint("unselected").dofadr[0]] == 0.1
    before = robot.data.qpos.copy(), robot.data.qvel.copy()
    # Values outside joint limits are written without clipping in every mode.
    robot.apply_command(COMMANDS[mode](robot.joint_names, [2, -3]))
    np.testing.assert_array_equal(robot.data.ctrl, [2, 0.3, -3])
    robot.apply_base_command([0.4])
    np.testing.assert_array_equal(robot.data.ctrl, [2, 0.4, -3])
    np.testing.assert_array_equal(robot.data.qpos, before[0])
    np.testing.assert_array_equal(robot.data.qvel, before[1])
    assert robot.data.time == 1.25
    with pytest.raises(AttributeError):
        robot.control_mode = "torque"


@pytest.mark.parametrize("actual", COMMANDS)
@pytest.mark.parametrize("expected", COMMANDS)
def test_constructor_and_command_mode_matrix(actual, expected):
    robot = make_robot(actual)
    if actual == expected:
        robot.validate_control_mode(expected)
        return
    before = robot.data.ctrl.copy()
    with pytest.raises(ValueError, match="Actuator.*joint.*control_mode"):
        MujocoRobotModel(robot.model, robot.data, PAIRS, control_mode=expected)
    with pytest.raises(TypeError, match="Expected.*control_mode"):
        robot.apply_command(COMMANDS[expected](robot.joint_names, [0, 0]))
    np.testing.assert_array_equal(robot.data.ctrl, before)


@pytest.mark.parametrize("mode", COMMANDS)
@pytest.mark.parametrize("fault", ["order", "shape", "nan"])
def test_command_rejection_is_atomic(mode, fault):
    robot = make_robot(mode)
    robot.data.ctrl[:] = 0.123
    command = COMMANDS[mode](robot.joint_names[::-1] if fault == "order" else robot.joint_names, [0, 0])
    # Exercise validation at application time even if a command's frozen data was bypassed.
    if fault != "order":
        field = {"position": "positions", "velocity": "velocities", "torque": "torques"}[mode]
        object.__setattr__(command, field, np.array([0]) if fault == "shape" else np.array([0, np.nan]))
    with pytest.raises(ValueError):
        robot.apply_command(command)
    np.testing.assert_array_equal(robot.data.ctrl, 0.123)


@pytest.mark.parametrize("mode", COMMANDS)
@pytest.mark.parametrize("fault", ["gear", "transmission", "stateful", "gain", "bias"])
def test_invalid_compiled_actuators_are_rejected(mode, fault):
    robot = make_robot(mode, general=True)
    model, actuator = robot.model, robot._actuator_ids[0]
    if fault == "gear":
        model.actuator_gear[actuator, 0] = 2
    elif fault == "transmission":
        model.actuator_trnid[actuator, 0] = model.joint("unselected").id
    elif fault == "stateful":
        model.actuator_dyntype[actuator] = mujoco.mjtDyn.mjDYN_FILTER
    elif fault == "gain":
        model.actuator_gainprm[actuator, 0] = np.nan
    elif mode == "torque":
        model.actuator_biastype[actuator] = mujoco.mjtBias.mjBIAS_AFFINE
    else:
        model.actuator_biasprm[actuator, 0] = 1
    with pytest.raises(ValueError):
        MujocoRobotModel(model, robot.data, PAIRS, control_mode=mode)


@pytest.mark.parametrize("mode", COMMANDS)
def test_xml_factory_and_required_mode(tmp_path, mode):
    path = tmp_path / "scene.xml"
    path.write_text(xml_for(mode))
    robot = MujocoRobotModel.from_xml_path(path, PAIRS, control_mode=mode)
    assert robot.control_mode == mode
    with pytest.raises(TypeError, match="control_mode"):
        MujocoRobotModel(robot.model, robot.data, PAIRS)
    with pytest.raises(TypeError, match="control_mode"):
        MujocoRobotModel.from_xml_path(path, PAIRS)
    for invalid in (None, "effort", "POSITION"):
        with pytest.raises(ValueError, match="Unknown control mode"):
            MujocoRobotModel(robot.model, robot.data, PAIRS, control_mode=invalid)
        with pytest.raises(ValueError, match="Unknown control mode"):
            robot.validate_control_mode(invalid)


@pytest.mark.parametrize("mode", COMMANDS)
def test_dynamics_are_mode_independent_and_reads_preserve_state(mode):
    robot = make_robot(mode)
    reference = make_robot("torque")
    before = robot.data.qpos.copy(), robot.data.qvel.copy(), robot.data.ctrl.copy()
    for method, args in [("get_mass_matrix", ()), ("get_nonlinear_forces", ()), ("get_jacobian_derivative", ("tool",))]:
        np.testing.assert_allclose(getattr(robot, method)(*args), getattr(reference, method)(*args))
    for field, values in zip(("qpos", "qvel", "ctrl"), before):
        np.testing.assert_array_equal(getattr(robot.data, field), values)
    if mode != "torque":
        with pytest.raises(NotImplementedError, match="torque"):
            _ = robot.torque_limits
    else:
        ids = robot._actuator_ids
        robot.model.actuator_ctrllimited[ids] = 1
        robot.model.actuator_ctrlrange[ids] = [-5, 8]
        robot.model.actuator_forcelimited[ids] = 1
        robot.model.actuator_forcerange[ids] = [-7, 6]
        lower, upper = robot.torque_limits
        np.testing.assert_array_equal(lower, [-5, -5])
        np.testing.assert_array_equal(upper, [6, 6])


class VelocityController(Controller):
    @property
    def control_mode(self):
        return "velocity"

    def compute(self, target, dt):
        return JointVelocityCommand(self.robot.joint_names, target)


@pytest.mark.parametrize("mode", COMMANDS)
@pytest.mark.parametrize("controller_type,expected", [
    (InverseKinematicsController, "position"), (ImpedanceController, "torque"), (VelocityController, "velocity"),
])
def test_controller_constructor_checks_modes_before_setup(mode, controller_type, expected):
    robot = make_robot(mode)
    before = robot.data.ctrl.copy()
    if mode != expected:
        # None would fail during concrete config access if mode checking were too late.
        args = () if expected == "velocity" else (None,)
        with pytest.raises(ValueError, match=f"requires control_mode='{expected}'.*control_mode='{mode}'"):
            controller_type(robot, *args)
    else:
        args = ((IKQPConfig([-1, -1], [1, 1]),) if mode == "position" else
                (ImpedanceQPConfig([-1, -1], [1, 1]),) if mode == "torque" else ())
        controller = controller_type(robot, *args)
        assert controller.robot is robot
        assert controller.control_mode == mode
        with pytest.raises(AttributeError):
            controller.control_mode = "velocity"
        if mode == "velocity":
            robot.apply_command(controller.compute([0, 0], 0.01))
    np.testing.assert_array_equal(robot.data.ctrl, before)


def test_optional_dynamics_fail_explicitly_and_batch_modes_match():
    robot = SliderModel()
    for operation in [lambda: robot.torque_limits, robot.get_mass_matrix,
                      robot.get_nonlinear_forces, lambda: robot.get_jacobian_derivative("tool")]:
        with pytest.raises(NotImplementedError, match="SliderModel"):
            operation()
    with ParallelPinkIK(0) as pool:
        array_robot = ArrayRobotModel(("j",), "root")
        assert pool.control_mode == array_robot.control_mode == "position"
        array_robot.validate_control_mode(pool.control_mode)


@pytest.mark.parametrize("mode", COMMANDS)
def test_initialize_rejects_invalid_extras_without_partial_writes(mode):
    robot = make_robot(mode)
    robot.data.ctrl[:] = 0.123
    before = {field: getattr(robot.data, field).copy() for field in ("qpos", "qvel", "ctrl")}
    with pytest.raises(ValueError, match="finite"):
        robot.initialize([0.1, -0.2], extra_actuator_controls={"unused_servo": np.nan})
    for field, values in before.items():
        np.testing.assert_array_equal(getattr(robot.data, field), values)
