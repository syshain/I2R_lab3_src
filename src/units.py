"""All millimetre and degree crossings, in one place.

Disk and the xArm SDK speak millimetres; the controller displays degrees.
Every calculation in Lab 3 runs in metres and radians (Lab 1 convention,
frame definitions unchanged). Each function below is the one crossing it
owns; none of them call each other.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_from_pose_mm_rad(pose):
    """Pose [x, y, z mm; roll, pitch, yaw rad] -> T (4x4, metres).

    The on-disk format of t6c_stage2 in calibration_results.npy: lengths in
    mm, orientation as a rotation vector in radians.
    """
    p = np.asarray(pose, dtype=float).reshape(-1)
    T = np.eye(4)
    T[:3, 3] = p[:3] / 1000.0
    T[:3, :3] = R.from_euler('xyz', p[3:6]).as_matrix()
    return T


def transform_from_pose_mm_deg(pose):
    """Pose [x, y, z mm; roll, pitch, yaw deg] -> T (4x4, metres).

    The controller's displayed pose format; x-y-z Euler, degrees.
    """
    p = np.asarray(pose, dtype=float).reshape(-1)
    T = np.eye(4)
    T[:3, 3] = p[:3] / 1000.0
    T[:3, :3] = R.from_euler('xyz', p[3:6], degrees=True).as_matrix()
    return T


def euler_deg_from_matrix(mat):
    """Rotation matrix -> [roll, pitch, yaw] in degrees (x-y-z intrinsic)."""
    return np.degrees(R.from_matrix(np.asarray(mat, dtype=float)).as_euler('xyz'))
