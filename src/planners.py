"""Via-point path planning between Cartesian poses.

plan_via_path() turns a start pose, any number of via poses and an end
pose into a dense list of poses that the executors in key_control.py
walk through. Between consecutive key poses the position moves along a
straight line and the orientation along the geodesic; a *profile*
function shapes how the arm distributes its motion along that line:

    linear   constant speed: it never slows at a via, so velocity is
             discontinuous there and the arm jolts under the dead-man
             gate -- but the path is the shortest and the tracking is
             the most even.
    cubic    eases in and out of every key pose (zero velocity at both
             segment ends), so the arm comes to a smooth stop at each
             via.
    quintic  eases yet more gently (zero velocity AND zero acceleration
             at the segment ends).

The lab exercise: run this file, compare the three profiles' total
length and largest step, then watch the same paths on the real arm in
each executor mode and connect what you see to the profiles.
"""

import numpy as np

from reporting import banner, field, progress
from so3 import rotation_log_map


def _profile_linear(s):
    """Constant-speed profile: velocity continuous through the segment
    ends (the arm never stops at a via point), acceleration zero
    everywhere. This is the profile for smooth g-gated tracking --
    cubic and quintic ease to a full stop at every segment end and
    crowd the waypoints there, which shows up as jolting under the
    dead-man executor.

    All profiles map the normalised segment parameter s in [0, 1]
    (0 = start pose, 1 = end pose) to the distance fraction already
    travelled along the straight line.
    """
    return s


def _profile_cubic(s):
    """Hermite profile: zero velocity at both segment ends, so the arm
    comes to a smooth stop at each via. (s in [0, 1]: see _profile_linear.)"""
    return s * s * (3.0 - 2.0 * s)


def _profile_quintic(s):
    """Quintic profile: zero velocity AND zero acceleration at both
    segment ends -- the gentlest start/stop of the three.
    (s in [0, 1]: see _profile_linear.)"""
    return <TO_FILL>


INTERPOLATORS = {'linear': _profile_linear, 'cubic': _profile_cubic, 'quintic': _profile_quintic}


def _skew(v):
    """3x3 cross-product matrix of v."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def rotation_exp_map(omega):
    """Rotation matrix from a rotation vector (axis * angle), Rodrigues form."""
    omega = np.asarray(omega, dtype=float)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-8:
        return np.eye(3) + _skew(omega)
    k = omega / theta
    K = _skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def interpolate_pose(T0, T1, frac):
    """Pose a fraction `frac` of the way (in [0, 1]) from T0 to T1.

    Orientation follows the geodesic (log/exp of the relative rotation),
    position is linear in the frame of T0. `frac` has usually already
    been shaped by one of the _profile_* functions, so the arm's SPEED
    along the straight line varies with the profile even though the
    geometric path does not.
    """
    T = np.eye(4)
    d_ori = rotation_log_map(T0[:3, :3].T @ T1[:3, :3])
    T[:3, :3] = T0[:3, :3] @ rotation_exp_map(frac * d_ori)
    T[:3, 3] = (1.0 - frac) * T0[:3, 3] + frac * T1[:3, 3]
    return T


def segment_poses(T0, T1, mode='cubic', n=25):
    """Sample one segment into n+1 poses (start included, end included).

    `n` sets the sampling DENSITY only: the profile decides where along
    the segment those n+1 samples sit, i.e. how the arm speeds up and
    slows down, not where the path goes.
    """
    return [interpolate_pose(T0, T1, INTERPOLATORS[mode](s))
            for s in np.linspace(0.0, 1.0, n + 1)]


def plan_via_path(T_start, vias, T_end, mode='cubic', n_per_seg=25):
    """Chain segments start -> via1 -> ... -> viaN -> end, sampling each
    with the chosen profile. Consecutive segments share their key pose
    exactly, so the arm passes through every via (or stops at it, for
    cubic/quintic). Returns the pose list for an executor."""
    keys = [T_start] + list(vias) + [T_end]
    poses = []
    for A, B in zip(keys[:-1], keys[1:]):
        poses += segment_poses(A, B, mode=mode, n=n_per_seg)[1:]
    return poses


def path_report(poses):
    """(total_length_mm, max_step_mm) along a pose list: the numbers
    pick_and_place.py prints when planning, for comparing profiles."""
    steps = [np.linalg.norm(poses[i + 1][:3, 3] - poses[i][:3, 3]) * 1000.0
             for i in range(len(poses) - 1)]
    return float(np.sum(steps)), float(np.max(steps))


if __name__ == '__main__':
    from ik_lite6 import pose_to_matrix
    T_start = pose_to_matrix([0.0, 0.0, 0.3], np.deg2rad([0, 180, 0]))
    T_via = pose_to_matrix([0.05, 0.1, 0.35], np.deg2rad([0, 180, 0]))
    T_end = pose_to_matrix([0.15, 0.05, 0.25], np.deg2rad([0, 180, 0]))
    banner('VIA-POINT PLANNER PROFILE COMPARISON')
    progress('2 segments, 25 waypoints each, same start/via/end poses.')
    field('profile', '   total_mm  max_step_mm')
    for mode in ('linear', 'cubic', 'quintic'):
        poses = plan_via_path(T_start, [T_via], T_end, mode=mode, n_per_seg=25)
        total, mx = path_report(poses)
        field(mode, '%9.2f %12.2f' % (total, mx))
    print()
    progress('linear has the shortest path but a velocity discontinuity at')
    progress('the via; cubic/quintic crowd waypoints near segment ends')
    progress('(smooth stops).')
