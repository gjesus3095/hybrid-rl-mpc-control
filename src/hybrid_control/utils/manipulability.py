"""Yoshikawa manipulability and its joint-space gradient from a geometric Jacobian.

All functions take the world-frame Jacobian (6, n) of ``RobotModel.get_jacobian``,
with rows [linear, angular] and columns ordered from the base to the tip. The
analytic derivatives assume every column belongs to a revolute joint of one
serial chain and that the Jacobian is evaluated at the controlled frame's origin;
see Haviland & Corke, "Manipulator Differential Kinematics: Part II" (2023).
"""

import numpy as np

ROWS = {"position": slice(0, 3), "orientation": slice(3, 6), "full": slice(0, 6)}


def _rows(rows: str) -> slice:
    if rows not in ROWS:
        raise ValueError(f"rows must be one of {sorted(ROWS)}")
    return ROWS[rows]


def jacobian_hessian(jacobian: np.ndarray) -> np.ndarray:
    """Return H (n, 6, n) with H[i] = dJ/dq_i for a serial chain of revolute joints.

    With z_k the joint axes (angular rows) and v_k the linear columns:
    d z_j / d q_i = z_i x z_j for i < j (else 0), and
    d v_j / d q_i = z_a x v_b with a = min(i, j), b = max(i, j).
    """
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != 6:
        raise ValueError("jacobian must have shape (6, n)")
    linear, angular = jacobian[:3].T, jacobian[3:].T  # (n, 3) each
    index = np.arange(jacobian.shape[1])
    # All pairwise cross products at once (vectorized: called every control step)
    angular_linear = np.cross(angular[:, None], linear[None, :])  # [a, b] = z_a x v_b
    angular_angular = np.cross(angular[:, None], angular[None, :])  # [i, j] = z_i x z_j
    upper = (index[:, None] < index[None, :])[..., None]
    pairs = np.concatenate([
        angular_linear[np.minimum.outer(index, index), np.maximum.outer(index, index)],
        np.where(upper, angular_angular, 0.0),
    ], axis=2)  # [i, j, :] = dJ[:, j] / dq_i

    # n = jacobian.shape[1]
    # hessian = np.zeros((n, 6, n))
    # for i in range(n):
    #     for j in range(n):
    #         a, b = min(i, j), max(i, j)
    #         hessian[i, :3, j] = np.cross(angular[a], linear[b])
    #         if i < j:
    #             hessian[i, 3:, j] = np.cross(angular[i], angular[j])
    # return hessian

    return pairs.transpose(0, 2, 1)


def manipulability(jacobian: np.ndarray, rows: str = "position") -> float:
    """Return w = sqrt(det(J_r J_rᵀ)) for the selected Jacobian rows."""
    J = np.asarray(jacobian, dtype=float)[_rows(rows)]
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def manipulability_gradient(jacobian: np.ndarray, rows: str = "position",
                            hessian: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Return (w, dw/dq) with dw/dq_i = w · tr((J Jᵀ)⁻¹ J H_iᵀ) on the selected rows."""
    selected = _rows(rows)
    J = np.asarray(jacobian, dtype=float)
    if hessian is None:
        hessian = jacobian_hessian(J)
    Jr = J[selected]
    w = manipulability(J, rows)
    if w < 1e-12:
        return w, np.zeros(J.shape[1])  # Exactly singular: the gradient is undefined.
    weighted = np.linalg.solve(Jr @ Jr.T, Jr)  # (J Jᵀ)⁻¹ J
    gradient = w * np.einsum("rc,irc->i", weighted, hessian[:, selected, :])
    return w, gradient
