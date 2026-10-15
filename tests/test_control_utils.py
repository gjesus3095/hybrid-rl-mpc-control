"""Shared targets and extracted math retain frame conventions and independent data."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hybrid_control.robots.state import Pose
from hybrid_control.utils.control_utils import TaskSpaceTarget
from hybrid_control.utils.math_helpers import (
    base_command_world, compose_poses, finite_vector, pose_error, pose_to_matrix,
    pose_to_se3, pose_yaw, poses_to_matrices, predict_after_base_motion,
)
from hybrid_control.utils.trajectories import QuinticTrajectory


def test_target_from_trajectory_owns_derivatives_and_defaults_to_regulation():
    start = Pose([0, 0, 0], [0, 0, 0, 1])
    goal = Pose([1, 2, 3], Rotation.from_euler("z", 0.8).as_quat())
    sample = QuinticTrajectory(start, goal, 2).sample(0.7)
    target = TaskSpaceTarget.from_sample("tool", sample)
    np.testing.assert_allclose(target.velocity, sample.velocity)
    np.testing.assert_allclose(target.acceleration, sample.acceleration)
    sample.velocity[:] = 0
    assert np.any(target.velocity)
    assert not target.velocity.flags.writeable
    assert not target.acceleration.flags.writeable
    regulation = TaskSpaceTarget("tool", start)
    np.testing.assert_array_equal(regulation.velocity, np.zeros(6))
    np.testing.assert_array_equal(regulation.acceleration, np.zeros(6))


@pytest.mark.parametrize("value,size", [([1, 2], 3), ([1, np.nan, 3], 3), ([np.inf], 1)])
def test_finite_vector_rejects_bad_shape_or_values(value, size):
    with pytest.raises(ValueError, match="finite vector"):
        finite_vector(value, size, "test")


@pytest.mark.parametrize("field,value,error", [
    ("frame_name", "", ValueError), ("pose", None, TypeError),
    ("velocity", np.zeros(3), ValueError), ("acceleration", [np.nan] * 6, ValueError),
])
def test_shared_target_validation(field, value, error):
    args = dict(frame_name="tool", pose=Pose([0, 0, 0], [0, 0, 0, 1]))
    args[field] = value
    with pytest.raises(error):
        TaskSpaceTarget(**args)


def test_composition_and_scalar_batch_helpers_agree():
    parent = Pose([1, -2, 0.4], Rotation.from_euler("z", np.pi / 2).as_quat())
    offset = Pose([0.3, 0.2, -0.1], Rotation.from_euler("x", 0.4).as_quat())
    parents = np.array([np.r_[parent.position, parent.quaternion]])
    composed = compose_poses(parents, offset)
    np.testing.assert_allclose(poses_to_matrices(composed)[0], pose_to_matrix(parent) @ pose_to_matrix(offset), atol=1e-15)
    np.testing.assert_allclose(pose_yaw(parents), [pose_yaw(parent)])
    command = np.array([1.0, 0.0, 0.3])
    np.testing.assert_allclose(base_command_world(command, pose_yaw(parent)), [0, 1, 0.3], atol=1e-15)
    np.testing.assert_allclose(base_command_world(command, pose_yaw(parents))[0],
                               base_command_world(command, pose_yaw(parent)))
    pose = Pose(composed[0, :3], composed[0, 3:])
    np.testing.assert_allclose(pose_error(parents, composed)[0], pose_error(parent, pose))


def test_base_prediction_rotates_about_displaced_base_and_uses_dt():
    base = np.array([[2, 3, 0, 0, 0, 0, 1]], dtype=float)
    frame = np.array([[3, 3, 0, 0, 0, 0, 1]], dtype=float)
    velocity = np.array([[0.4, -0.2, np.pi]])
    predicted = predict_after_base_motion(base, frame, velocity, 0.5)
    np.testing.assert_allclose(predicted[0, :3], [2.2, 3.9, 0], atol=1e-15)
    np.testing.assert_allclose(pose_yaw(predicted), [np.pi / 2])
    np.testing.assert_allclose(predict_after_base_motion(base, frame, velocity, 0), frame)


def test_pinocchio_conversion_matches_homogeneous_transform():
    pytest.importorskip("pinocchio")
    pose = Pose([1, 2, 3], Rotation.from_euler("xyz", [0.3, -0.2, 1.2]).as_quat())
    np.testing.assert_allclose(pose_to_se3(pose).homogeneous, pose_to_matrix(pose))
