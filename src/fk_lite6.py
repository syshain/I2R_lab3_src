"""Forward kinematics for the UFACTORY Lite 6."""


import numpy as np
from scipy.spatial.transform import Rotation as R


def modified_dh_transform(a, alpha, d, theta):
    """One-joint modified (Craig) D-H transform; a,d in metres, angles in radians."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0.0, 0.0, 0.0, 1.0],
    ])

def pose_from_xyz_rpy(xyz_rpy):
    """Pose 4x4 from [x, y, z (m), roll, pitch, yaw (rad)]."""

    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', xyz_rpy[3:6]).as_matrix()
    T[:3, 3] = xyz_rpy[:3]/1000.00
    return T


def fk_from_zeta(q, zeta, n=6):
    """Chain n modified-DH transforms -> base->flange 4x4."""

    q = np.asarray(q, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    a, alpha, d = zeta[0:n]/1000.00, zeta[n:2 * n], zeta[2 * n:3 * n]/1000.00
    theta_off = zeta[3 * n:4 * n]
    T = np.eye(4)
    for i in range(n):
        T = T @ modified_dh_transform(a[i], alpha[i], d[i], theta_off[i] + q[i])
    return T
