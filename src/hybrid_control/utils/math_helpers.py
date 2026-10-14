"""Reusable vector and rigid-transform math; quaternions are ordered xyzw."""

import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation

from hybrid_control.robots.state import Pose

class MathHelpers:
    # ----------------------------------------------------------------------------------------------
    # HELPER METHODS (MATH & QUATERNIONS)
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def extract_position_quat_from_pose(pose) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parses pose data into position and quaternion arrays.
        Expected format: (pos(3,), quat(4,))
        """
        if isinstance(pose, (tuple, list)) and len(pose) == 2:
            p = np.asarray(pose[0], dtype=float).reshape(3)
            q = np.asarray(pose[1], dtype=float).reshape(4)
            return p, q
        raise ValueError(f"Unsupported pose format, expected (pos, quat), got {type(pose)}")

    @staticmethod
    def quat_normalize(q: np.ndarray) -> np.ndarray:
        """ Safely normalizes a quaternion. Returns identity if norm is near zero. """
        q = np.asarray(q, dtype=float).reshape(4)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / n

    @staticmethod
    def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
        """
        Enforces a positive scalar component (w >= 0) to ensure uniqueness
        in 'double cover' representation.
        Assumes input format [x, y, z, w].
        """
        if quat[3] < 0:
            return -quat
        return quat

    @staticmethod
    def change_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
        """
        Permutes quaternion elements.
        NOTE: Based on array logic: [0,1,2,3] -> [1,2,3,0].
        If input is [w, x, y, z] (MuJoCo), output is [x, y, z, w] (SciPy).
        """
        return np.array([quat[1], quat[2], quat[3], quat[0]])

    @staticmethod
    def align_quaternion_sign(reference_quat, candidate_quat):
        """
        Align candidate_quat to the same quaternion hemisphere as reference_quat
        to avoid discontinuities caused by the q and -q representation.
        """
        if np.dot(reference_quat, candidate_quat) < 0:
            return -candidate_quat
        return candidate_quat


def finite_vector(value, size: int, name: str) -> np.ndarray:
    """Return an independent read-only finite vector of the requested length."""
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    result.setflags(write=False)
    return result


def pose_to_se3(pose: Pose):
    """Convert a pose to Pinocchio SE3; Pinocchio is needed only for this operation."""
    import pinocchio as pin

    return pin.SE3(pin.Quaternion(pose.quaternion).toRotationMatrix(), pose.position)


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Return a (4, 4) homogeneous transform from a pose in metres and xyzw."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_quat(np.array(pose.quaternion)).as_matrix(), pose.position
    return T


def poses_to_matrices(poses: np.ndarray) -> np.ndarray:
    """(N, 7) poses (position, xyzw) -> (N, 4, 4) homogeneous transforms."""
    T = np.tile(np.eye(4), (len(poses), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = Rotation.from_quat(poses[:, 3:]).as_matrix(), poses[:, :3]
    return T


def compose_poses(parent: np.ndarray, offset: Pose) -> np.ndarray:
    """World poses (N, 7) of ``offset`` expressed in each of the ``parent`` poses (N, 7)."""
    rotation = Rotation.from_quat(parent[:, 3:])
    return np.c_[parent[:, :3] + rotation.apply(np.array(offset.position)),
                 (rotation * Rotation.from_quat(np.array(offset.quaternion))).as_quat()]


def predict_after_base_motion(base: np.ndarray, frame: np.ndarray, base_vel: np.ndarray, dt: float) -> np.ndarray:
    """World poses (N, 7) of ``frame`` after the base joints (world x, y, then yaw) move by base_vel*dt.

    T' = Trans(dx, dy) @ B @ Rz(dyaw) @ B^-1 @ T, with B the current base_link pose; base, frame
    (N, 7) world poses (xyzw), base_vel (N, 3).
    """
    B = poses_to_matrices(base)
    delta = np.tile(np.eye(4), (len(base), 1, 1))
    delta[:, :3, :3] = Rotation.from_euler("z", base_vel[:, 2:3] * dt).as_matrix()
    T = B @ delta @ np.linalg.inv(B) @ poses_to_matrices(frame)
    T[:, :2, 3] += base_vel[:, :2] * dt
    return np.c_[T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()]


def pose_yaw(pose: Pose | np.ndarray):
    """Return yaw (rad) for a Pose or an (N, 7) position/xyzw array."""
    quaternion = np.array(pose.quaternion) if isinstance(pose, Pose) else pose[:, 3:]
    rotation = Rotation.from_quat(quaternion).as_matrix()
    return np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])


def base_command_world(command: np.ndarray, base_yaw: float | np.ndarray) -> np.ndarray:
    """Rotate base-frame (vx, vy, wz) to world axes; scalar/batched yaw returns (3,)/(N, 3)."""
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    return np.stack([c * command[0] - s * command[1], s * command[0] + c * command[1],
                     np.full_like(c, command[2])], axis=-1)


def pose_error(target: Pose | np.ndarray, pose: Pose | np.ndarray):
    """Position (m) and rotation-angle (rad) errors; two Poses or two (N, 7) arrays.

    Returns a pair of floats for Poses, or an (N, 2) array for batched inputs.
    """
    if isinstance(target, Pose) and isinstance(pose, Pose):
        rotation = Rotation.from_quat(np.array(target.quaternion)) * Rotation.from_quat(np.array(pose.quaternion)).inv()
        return float(np.linalg.norm(target.position - pose.position)), float(rotation.magnitude())
    rotation = Rotation.from_quat(target[:, 3:]) * Rotation.from_quat(pose[:, 3:]).inv()
    return np.c_[np.linalg.norm(target[:, :3] - pose[:, :3], axis=1), rotation.magnitude()]
