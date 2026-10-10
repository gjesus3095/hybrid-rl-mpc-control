"""MuJoCo scene and viewer modifications: model spec helpers and viewer overlay."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from hybrid_control.robots.model import RobotModel
from hybrid_control.robots.state import Pose

_ARROW_ALPHA = 0.65
_ANCHOR_COLOR = np.array([0.95, 0.95, 0.95, 0.8], dtype=np.float32)


def _draw_frame(scene, geom_start: int, origin: np.ndarray, rot: np.ndarray,
                sphere_r: float, shaft_r: float, length: float, label: str = "") -> None:
    """Anchor sphere + X (red), Y (green), Z (blue) arrow geoms.  Uses 4 geoms from geom_start."""
    mujoco.mjv_initGeom(
        scene.geoms[geom_start], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, sphere_r), origin, np.eye(3).ravel(),
        _ANCHOR_COLOR,
    )
    scene.geoms[geom_start].label = label
    colors = [
        (1, 0, 0, _ARROW_ALPHA),
        (0, 1, 0, _ARROW_ALPHA),
        (0, 0, 1, _ARROW_ALPHA),
    ]
    for i, color in enumerate(colors):
        tip = origin + length * rot[:, i]
        geom = scene.geoms[geom_start + 1 + i]
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, shaft_r,
            origin.astype(np.float64),
            tip.astype(np.float64),
        )
        geom.rgba[:] = color
        geom.label = ""


class MujocoModifier:
    """MuJoCo scene modifiers: viewer overlay and model spec builders.

    Viewer overlay geom layout (9 total):
      0      world-origin anchor sphere (white), label "world"
      1-3    world frame X/Y/Z arrows,  shaft_r=8 mm, length=15 cm
      4      EE target anchor sphere    (white), label "ref"
      5-7    EE target X/Y/Z arrows,   shaft_r=5 mm, length=8 cm
      8      robot CoM sphere           (green-tinted), label "CoM"

    Usage::

        modifier = MujocoModifier(com_marker_diameter=0.05)
        modifier.update_overlay(handle, robot, target_pose)

        spec = MujocoModifier.build_mobile_base_spec(xml_file, base_range=2.0, servo_kp=5000.0)
    """

    def __init__(self, com_marker_diameter: float = 0.05):
        self._com_r = com_marker_diameter / 2

    def update_overlay(self, handle, robot: RobotModel, target_pose: Pose) -> None:
        """Refresh all overlay geoms.  No-op when handle is None (headless run)."""
        if handle is None:
            return
        com = robot.get_com_state()
        target_rot = Rotation.from_quat(target_pose.quaternion).as_matrix()
        with handle.lock():
            handle.user_scn.ngeom = 9
            _draw_frame(handle.user_scn, 0, np.zeros(3), np.eye(3),
                        sphere_r=0.012, shaft_r=0.008, length=0.15, label="world")
            _draw_frame(handle.user_scn, 4, target_pose.position, target_rot,
                        sphere_r=0.008, shaft_r=0.005, length=0.08, label="ref")
            mujoco.mjv_initGeom(
                handle.user_scn.geoms[8], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, self._com_r), com.position, np.eye(3).ravel(),
                np.array([0.2, 1, 0.3, 0.7], dtype=np.float32),
            )
            handle.user_scn.geoms[8].label = "CoM"

    @staticmethod
    def build_mobile_base_spec(xml_file: str, base_range: float = 2.0,
                               servo_kp: float = 5000.0) -> mujoco.MjSpec:
        """Load a scene XML and add X/Y slide joints to link0 with position servo actuators.

        The added joints (base_x, base_y) and actuators (base_x_servo, base_y_servo) sit
        outside any IK joint list — they act as a base disturbance controllable via
        MujocoRobotModel.apply_base_command.
        """
        spec = mujoco.MjSpec.from_file(xml_file)
        link0 = spec.find_body("link0")

        for name, axis in (("base_x", [1, 0, 0]), ("base_y", [0, 1, 0])):
            j = link0.add_joint()
            j.name = name
            j.type = mujoco.mjtJoint.mjJNT_SLIDE
            j.axis = axis
            j.range = [-base_range, base_range]
            j.limited = True

        for joint_name, servo_name in (("base_x", "base_x_servo"), ("base_y", "base_y_servo")):
            a = spec.add_actuator()
            a.name = servo_name
            a.target = joint_name
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = servo_kp
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[0] = 0
            a.biasprm[1] = -servo_kp
            a.ctrllimited = True
            a.ctrlrange = [-base_range, base_range]

        return spec
