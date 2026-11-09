"""Numerical inverse kinematics for the UFACTORY Lite 6, on top of fk_lite6.

joint angles in radians, positions in metres.
Targets are 4x4 flange transforms; the command line takes the
controller's millimetres and degrees and converts them in units.py.

    python3 ik_lite6.py                  self-test 
    python3 ik_lite6.py X Y Z R P Y      solve one pose (mm, degrees) from rest
    python3 ik_lite6.py X Y Z R P Y q1 ... q6
                        solve from a starting guess, inside the joint box
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from fk_lite6 import fk_from_zeta
from lab_config import IK_FD_EPS_RAD, JOINT_LIMITS_DEG, RESULTS_NPY_NAME, data_path
from reporting import banner, field, heading, progress
from so3 import rotation_log_map
from units import transform_from_pose_mm_deg

DH_NOMINAL_RAW = [
    [0.0, 243.3, 0.0, 0.0],      
    [-90.0, 0.0, -90.0, 0.0],    
    [-90.0, 0.0, 180.0, 200.0],  
    [0.0, 227.6, 90.0, 87.0],    
    [0.0, 0.0, 90.0, 0.0],       
    [0.0, 61.5, -90.0, 0.0],     
]

RESIDUAL_TOL = 1e-4   # ~0.1 mm

JOINT_LIMITS_RAD = np.deg2rad(np.asarray(JOINT_LIMITS_DEG, dtype=float))

def get_nominal_zeta():
    """Factory Lite 6 modified (Craig) D-H params as a packed zeta
    [a|alpha|d|theta_off], lengths in mm + angles in rad"""

    theta_off = np.deg2rad([r[0] for r in DH_NOMINAL_RAW])
    d = np.array([r[1] for r in DH_NOMINAL_RAW])
    alpha = np.deg2rad([r[2] for r in DH_NOMINAL_RAW])
    a = np.array([r[3] for r in DH_NOMINAL_RAW])
    return np.concatenate([a, alpha, d, theta_off])


def load_calibrated_zeta():
    """zeta_stage2 from data/calibration_results.npy, or None when the file
    is missing."""

    path = data_path(RESULTS_NPY_NAME)
    if not os.path.exists(path):
        return None
    cal = np.load(path, allow_pickle=True).item()
    return np.asarray(cal['zeta_stage2'], dtype=float)


def default_zeta():
    """The zeta every IK solve runs on: the Lab 2 calibrated table when the
    calibration file is present, the nominal handbook table otherwise."""

    zeta = load_calibrated_zeta()
    if zeta is not None:
        return zeta
    return get_nominal_zeta()


def ik_residual(q_rad, T_target, zeta):
    """The 6-row residual e(q) = [position error (m); orientation error (rad)]"""

    T_cur = fk_from_zeta(q_rad, zeta)
    e_pos = T_cur[:3, 3] - T_target[:3, 3]
    e_ori = rotation_log_map(T_cur[:3, :3] @ T_target[:3, :3].T)
    return np.concatenate([e_pos, e_ori])


def _project_limits(q, lb, ub):
    """Clip into the box, then nudge joints sitting exactly on a bound inward
    so the finite-difference stencil never steps outside it."""

    qc = np.clip(q, lb, ub)
    at_lb = qc == lb
    at_ub = qc == ub
    if at_lb.any():
        qc[at_lb] = lb[at_lb] + 1e-6 * (ub[at_lb] - lb[at_lb])
    if at_ub.any():
        qc[at_ub] = ub[at_ub] - 1e-6 * (ub[at_ub] - lb[at_ub])
    return qc


def _lm_run(T_target, q, zeta, lb, ub, *, max_iter=200, tol=1e-8,
            lam0=1e-3, verbose=False):
    """One damped-LM run from a single seed q (radians), every step projected
    into the joint box [lb, ub] (pass lb=None for an unconstrained run).
    Returns (q_rad, converged, e_norm): the endpoint, the POSE-residual flag
    (not a step-size test), and the final residual norm."""

    q = np.asarray(q, dtype=float).copy()
    if lb is not None:
        q = _project_limits(q, lb, ub)

    def cost(x):
        e = ik_residual(x, T_target, zeta)
        return 0.5 * float(e @ e), e

    def fd_jacobian(x):
        J = np.empty((6, 6))
        for j in range(6):
            xp = x.copy()
            xm = x.copy()
            xp[j] += IK_FD_EPS_RAD
            xm[j] -= IK_FD_EPS_RAD
            J[:, j] = (ik_residual(xp, T_target, zeta)
                       - ik_residual(xm, T_target, zeta)) / (2.0 * IK_FD_EPS_RAD)
        return J

    c_old, e_old = cost(q)
    lam = lam0
    converged = False
    prev_c = c_old
    it = 0
    for it in range(1, max_iter + 1):
        # Damped least squares through the SVD: dx = -V (s/(s^2+lam)) U^T e.
        U, s, Vt = np.linalg.svd(fd_jacobian(q), full_matrices=False)
        dx = -Vt.T @ ((s / (s ** 2 + lam)) * (U.T @ e_old))
        qt = _project_limits(q + dx, lb, ub) if lb is not None else q + dx

        c_new, e_new = cost(qt)
        if c_new < c_old:
            q, e_old, c_old = qt, e_new, c_new
            # The same damping schedule as Lab 2's solver: the bigger the cut
            # in cost, the harder the damping is released.
            drop = (prev_c - c_new) / (abs(prev_c) + 1e-15)
            lam_factor = 0.1 if drop > 0.1 else (0.3 if drop > 0.01 else 0.5)
            lam = max(lam * lam_factor, 1e-12)
        else:
            lam *= 3.0
            if lam > 1e14:
                if verbose:
                    progress('lambda saturated at %.0e: the residual cannot be'
                             ' reduced further from here' % lam)
                break

        if verbose:
            progress('%4d  lam %12.3e  |dx| %12.3e  pos %9.3f mm  ori %8.3f deg'
                     % (it, lam, np.linalg.norm(dx),
                        np.linalg.norm(e_old[:3]) * 1000.0,
                        np.degrees(np.linalg.norm(e_old[3:]))))

        # The residual, not the step, is the convergence test: a tiny damped
        # step far from an unreachable target must not read as success.
        if np.linalg.norm(e_old) < RESIDUAL_TOL:
            converged = True
            break
        if np.linalg.norm(dx) < tol:
            if verbose:
                progress('step stalled below %.0e with residual %.3g still open'
                         % (tol, np.linalg.norm(e_old)))
            break
        prev_c = c_old

    return q, converged, float(np.linalg.norm(e_old))

# When the primary run fails, solve_ik retries from a set of standard poses
_RESTART_SEEDS_DEG = [
    [0.0, -30.0, 60.0, 0.0, -60.0, 0.0],
    [0.0, 30.0, 90.0, 0.0, -60.0, 0.0],
    [180.0, -20.0, 45.0, 0.0, -45.0, 0.0],
    [-180.0, -20.0, 45.0, 0.0, -45.0, 0.0],
    [90.0, 0.0, 120.0, 90.0, -30.0, 0.0],
    [-90.0, 0.0, 120.0, -90.0, -30.0, 0.0],
]
_RESTART_SEEDS_RAD = np.deg2rad(np.asarray(_RESTART_SEEDS_DEG, dtype=float))


def solve_ik(T_target, q0_rad, zeta=None, joint_limits_rad=JOINT_LIMITS_RAD,
             *, max_iter=200, tol=1e-8, lam0=1e-3, verbose=False):
    """Solve IK for the 4x4 flange target T_target (metres), starting from
    q0_rad (radians; seed it from the arm's current joints).
    One bounded LM run
    """

    if zeta is None:
        zeta = default_zeta()
    q = np.asarray(q0_rad, dtype=float).copy()
    if q.shape != (6,):
        raise ValueError('solve_ik expects 6 joint angles, got shape %s' % (q.shape,))

    use_limits = joint_limits_rad is not None
    lb = ub = None
    if use_limits:
        lim = np.asarray(joint_limits_rad, dtype=float)
        if lim.shape != (6, 2):
            raise ValueError('joint_limits_rad must be shape (6, 2), got %s' % (lim.shape,))
        lb, ub = lim[:, 0], lim[:, 1]

    q, converged, err = _lm_run(T_target, q, zeta, lb, ub, max_iter=max_iter,
                                tol=tol, lam0=lam0, verbose=verbose)
    if use_limits and not converged:
        best_q, best_e = q, err
        if verbose:
            progress('primary run did not converge; retrying standard postures')
        for seed in _RESTART_SEEDS_RAD:
            q_s, ok_s, e_s = _lm_run(T_target, seed, zeta, lb, ub,
                                     max_iter=max_iter, tol=tol, lam0=lam0,
                                     verbose=False)
            if e_s < best_e:
                best_q, best_e = q_s, e_s
            if ok_s:
                q, converged, err = q_s, ok_s, e_s
                field('restart', 'converged from posture [%s]'
                      % ', '.join('%d' % round(np.degrees(v)) for v in seed))
                break
        if not converged:
            q, err = best_q, best_e

    if verbose:
        e_f = ik_residual(q, T_target, zeta)
        field('result', 'q(deg) ' + ' '.join('%7.2f' % v for v in np.degrees(q)))
        field('error', 'pos %.4f mm  ori %.4f deg  |e| %.4g'
              % (np.linalg.norm(e_f[:3]) * 1000.0,
                 np.degrees(np.linalg.norm(e_f[3:])), err))
        field('status', 'CONVERGED' if converged else 'NOT CONVERGED')

    return q, converged, err


def pose_to_matrix(pos_xyz, euler_xyz):
    """4x4 transform from pos_xyz (m) + euler_xyz [r, p, y] (rad, XYZ)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', np.asarray(euler_xyz, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=float)
    return T


if __name__ == '__main__':
    zeta = default_zeta()
    zeta_label = ('calibrated (zeta_stage2)' if load_calibrated_zeta() is not None
                  else 'nominal (no calibration file)')

    if len(sys.argv) == 1:
        banner('LITE 6 IK SELF-TEST (%s)' % zeta_label)
        tests = [('Home', np.deg2rad([-180, 0.0, 4.0, 0.0, -45.0, 0.0]),
                  np.array([0.1, -1.2, 0.1, 0.1, 0.1, 0.1])),
                 ('ElbowDown', np.deg2rad([-180, -18.8, 4.0, 0.0, -45.0, 0.0]),
                  np.array([0.1, -0.8, 0.3, 0.1, 0.1, 0.1])),
                 ('Complex', np.deg2rad([-302.2, -19.9, 37.2, -207.4, -108.7, -39.9]),
                  np.array([0.3, -0.8, 0.5, 0.2, 0.3, 0.1]))]

        all_ok = True
        for name, q_true, q0 in tests:
            T_t = fk_from_zeta(q_true, zeta)
            q_sol, ok, err = solve_ik(T_t, q0, zeta)
            T_v = fk_from_zeta(q_sol, zeta)
            pos_err_mm = np.linalg.norm(T_v[:3, 3] - T_t[:3, 3]) * 1000.0
            ang_err_deg = np.degrees(np.linalg.norm(
                rotation_log_map(T_v[:3, :3] @ T_t[:3, :3].T)))
            all_ok &= bool(ok and pos_err_mm < 0.01)
            heading(name)
            field('converged', 'yes' if ok else 'NO')
            field('pos err', '%.4f mm' % pos_err_mm)
            field('ori err', '%.4f deg' % ang_err_deg)
            field('sol (deg)', ' '.join('%7.2f' % v for v in np.degrees(q_sol)))

        heading('Unreachable')
        T_far = np.eye(4)
        T_far[:3, 3] = np.array([1.5, 0.0, 1.5])   # ~2.1 m out; the Lite 6 reaches ~0.6 m
        q_sol, ok, err = solve_ik(T_far, np.zeros(6), zeta)
        T_v = fk_from_zeta(q_sol, zeta)
        all_ok &= bool(not ok)
        field('converged', 'NO (as required)' if not ok else 'WRONG -- flagged CONVERGED')
        field('best it got', '%.1f %.1f %.1f mm from base'
              % tuple(T_v[:3, 3] * 1000.0))
        field('residual', '%.4g' % err)

        print()
        print('  All checks OK' if all_ok
              else '  Check FAILED: see the blocks above')
        sys.exit(0 if all_ok else 1)

    elif len(sys.argv) in (7, 13):
        argv = [float(v) for v in sys.argv[1:]]
        pose_mm_deg = np.array(argv[:6])
        q0 = (np.deg2rad(np.array(argv[6:])) if len(argv) > 6 else np.zeros(6))
        banner('LITE 6 NUMERICAL INVERSE KINEMATICS (%s)' % zeta_label)
        field('target', 'pos %8.3f %8.3f %8.3f mm   rpy %7.3f %7.3f %7.3f deg'
              % (pose_mm_deg[0], pose_mm_deg[1], pose_mm_deg[2],
                 pose_mm_deg[3], pose_mm_deg[4], pose_mm_deg[5]))
        progress('from rest, inside the joint box' if len(argv) == 6
                 else 'from the given start, inside the joint box')
        T_target = transform_from_pose_mm_deg(pose_mm_deg)
        q_sol, ok, err = solve_ik(T_target, q0, verbose=True)
        T_v = fk_from_zeta(q_sol, zeta)
        field('round trip', '%.4f mm from the target'
              % (np.linalg.norm(T_v[:3, 3] - T_target[:3, 3]) * 1000.0))

    else:
        print(__doc__)
        sys.exit(2)
