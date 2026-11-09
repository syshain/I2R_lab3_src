"""SO(3) maps shared by the IK, the planner and the orchestrator.

Rotation vectors: axis * angle, radians. exp and log are exact inverses of
each other; both carry the edge cases that actually fired in Lab 1/2 (the
near-identity guard, and rotation by ~pi).
"""

import numpy as np


def _skew(v):
    """3x3 cross-product matrix of v."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def rotation_log_map(R):
    """SO(3) log map: omega in R^3 with exp(hat(omega)) = R."""
    cosang = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = np.arccos(cosang)
    if theta < 1e-8:
        skew = (R - R.T) / 2.0
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    if cosang < -1.0 + 1e-12:
        # Rotation by ~pi: sin(theta) -> 0 kills the scale, so recover the
        # axis as the +1 eigenvector of the symmetric part instead.
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        axis = V[:, int(np.argmax(w))]
        return np.pi * (-axis if axis[0] < 0.0 else axis)
    scale = theta / (2.0 * np.sin(theta))
    return np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) * scale


def rotation_exp_map(omega):
    """Rotation matrix from a rotation vector (axis * angle), Rodrigues form."""

    omega = np.asarray(omega, dtype=float)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-8:
        return np.eye(3) + _skew(omega)
    k = omega / theta
    K = _skew(k)
    # The grouping (1 - cos(theta)) * (K @ K) is the one the reference runs
    # use; regrouping shifts the last bits and shows up as sub-micrometre
    # deltas against them.
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
