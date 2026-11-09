"""Camera, OpenCV UI, calibration loading, and operator-input helpers."""

import os
import time
import numpy as np
import cv2
from lab_config import (AXES_LENGTH_MM, CAMERA_INDEX, CAMERA_MATRIX_NAME,
                        COLOR_THRESHOLDS, DIST_COEFFS_NAME,
                        GRIPPER_CALIBRATION_NAME, MASK_BANDS_NAME,
                        RESULTS_NPY_NAME, TOOL_OFFSET_MM, _clone_defaults,
                        data_path, save_mask_bands)
from motion import flange_pose_base
from so3 import rotation_exp_map
from reporting import progress
from vision import rgb_color_mask, compute_detection, \
    project_base_to_uv
import robot
WINDOW = 'Lab 3 : Pick & Place'
MASK1_WINDOW = 'Lab 3 : mask 1'
MASK2_WINDOW = 'Lab 3 : mask 2'
MASK_WINDOWS = [MASK1_WINDOW, MASK2_WINDOW]
_PREVIEW_PX = (960, 540)          # half-screen at 1920x1080


def load_camera_intrinsics():
    """Load camera matrix and distortion coefficients."""
    K = np.load(data_path(CAMERA_MATRIX_NAME))
    dist = np.load(data_path(DIST_COEFFS_NAME))
    return K, dist


def load_calibration_results():
    """Calibrated DH zeta (Stage 2) + the hand-eye T_6_C fitted with it.

    LAB3_ZETA=nominal loads the nominal DH instead (calibration_results.npy
    stores both). t6c_stage2's rotation is a ROTATION VECTOR (Rodrigues), as
    the calibration pipeline packs it — not Euler rpy.
    """

    cal = np.load(data_path(RESULTS_NPY_NAME), allow_pickle=True).item()
    use_nominal = os.environ.get('LAB3_ZETA', '').lower() == 'nominal'
    zeta = np.asarray(cal['zeta_nominal' if use_nominal else 'zeta_stage2'],
                      dtype=float)
    v = np.asarray(cal['t6c_stage2'], dtype=float)
    T_6_C = np.eye(4)
    T_6_C[:3, :3] = rotation_exp_map(v[3:])
    T_6_C[:3, 3] = v[:3] / 1000.0
    progress('zeta: %s   T_6_C: [%s] mm'
             % ('nominal DH' if use_nominal else 'stage-2 fit',
                ', '.join('%.1f' % x for x in v[:3])))
    return (zeta, T_6_C)


def load_gripper_calibration():
    """Stage-1 result (two-capture touch calibration) -> (tool_offset_m, source)."""

    path = data_path(GRIPPER_CALIBRATION_NAME)
    if not os.path.exists(path):
        return (TOOL_OFFSET_MM / 1000.0,
                'nominal TOOL_OFFSET_MM — run handeye_gripper.py to measure')
    cal = np.load(path, allow_pickle=True).item()
    d_mm = float(cal['tool_offset_mm'])
    try:
        src = 'two-capture touch calibration (d_std %.2f mm)' \
              % float(cal['d_std_mm'])
    except KeyError:
        src = 'two-capture touch calibration'
    return d_mm / 1000.0, src


def open_camera(cam_index=CAMERA_INDEX):
    ok = False
    cap = cv2.VideoCapture(cam_index)
    # K was calibrated at 1920x1080: any other resolution rescales the image
    # and silently inflates every PnP depth by the resolution ratio.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(10):                       # some drivers need a warm-up
        ok, _frame = cap.read()
        if ok:
            break
        time.sleep(0.2)
    if not ok:
        cap.release()
        raise RuntimeError('camera %d did not open' % cam_index)
    return cap


def size_windows():
    """Create (if needed) + resize the main feed and both mask windows."""

    for name in [WINDOW] + MASK_WINDOWS:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, *_PREVIEW_PX)


_TB_CHANNELS = ('r', 'g', 'b')
_TB_SUFFIXES = ('lo', 'hi')


def _win_for(color):
    return MASK1_WINDOW if color == '1' else MASK2_WINDOW


def _tb_name(color, band_idx, ch, suffix):
    return '%s_%s' % (ch.upper(), suffix)


def _iter_bounds():
    """Yield (color, band_idx, channel, suffix, value) for every live bound."""
    for color, bands in COLOR_THRESHOLDS.items():
        for i, band in enumerate(bands):
            for ch in _TB_CHANNELS:
                for sfx in _TB_SUFFIXES:
                    yield color, i, ch, sfx, band['%s_%s' % (ch, sfx)]


def create_threshold_trackbars():
    """Create one trackbar per live RGB bound (each channel 0..255)."""
    for color, i, ch, sfx, val in _iter_bounds():
        cv2.createTrackbar(_tb_name(color, i, ch, sfx), _win_for(color),
                           int(val), 255, lambda *_: None)


def sync_threshold_trackbars():
    """Push live threshold values back into every slider (per-frame)."""
    for color, i, ch, sfx, val in _iter_bounds():
        cv2.setTrackbarPos(_tb_name(color, i, ch, sfx), _win_for(color),
                           int(val))


def read_threshold_trackbars():
    """Pull slider values back into COLOR_THRESHOLDS; clamp hi>=lo per channel."""
    for color, i, ch, sfx, _ in _iter_bounds():
        val = cv2.getTrackbarPos(_tb_name(color, i, ch, sfx), _win_for(color))
        if val >= 0:
            COLOR_THRESHOLDS[color][i]['%s_%s' % (ch, sfx)] = int(val)
    for bands in COLOR_THRESHOLDS.values():
        for band in bands:
            for ch in _TB_CHANNELS:
                lo, hi = ch + '_lo', ch + '_hi'
                band[hi] = max(band[hi], band[lo])


def reset_thresholds():
    """Restore both colours to their default RGB bands."""
    COLOR_THRESHOLDS.clear()
    COLOR_THRESHOLDS.update(_clone_defaults())


def draw_debug(mask, label, accent_bgr=(0, 0, 255)):
    """Render a colour-segmentation mask as a labelled BGR debug image."""
    dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(dbg, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                accent_bgr, 2, cv2.LINE_AA)
    return dbg


def _both_masks(frame, K, dist_coeffs):
    """Both colour masks from one shared undistorted frame."""
    frame_u = cv2.undistort(frame, K, dist_coeffs)
    return {'1': rgb_color_mask(frame_u, '1'),
            '2': rgb_color_mask(frame_u, '2')}


def draw_artifact_axes(frame, rvec, tvec, K, dist_coeffs, length_mm=None):
    """Draw the artifact frame W's own axes on the image (Lab 2's draw_axes)."""

    if length_mm is None:
        length_mm = AXES_LENGTH_MM
    L = length_mm / 1000.0
    pts_w = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0],
                      [0.0, L, 0.0], [0.0, 0.0, L]])
    uv, _ = cv2.projectPoints(pts_w.reshape(-1, 1, 3).astype(np.float64),
                              np.asarray(rvec, dtype=float),
                              np.asarray(tvec, dtype=float).ravel(),
                              K, dist_coeffs)
    pts = uv.reshape(-1, 2).astype(int)
    h, w = frame.shape[:2]
    o = pts[0]
    if not (0 <= o[0] < w and 0 <= o[1] < h):
        cv2.circle(frame, (int(np.clip(o[0], 0, w - 1)),
                           int(np.clip(o[1], 0, h - 1))), 6, (0, 255, 0), -1)
        return frame
    for end, bgr in ((pts[1], (255, 0, 0)), (pts[2], (0, 255, 0)),
                     (pts[3], (0, 0, 255))):
        cv2.line(frame, (o[0], o[1]), (end[0], end[1]), bgr, 2, cv2.LINE_AA)
    return frame


def show_frame(cap, status_line, info_lines=None, det=None, color='1', hint=None, extra_marks=None, K=None, dist_coeffs=None, axes=None):
    """Draw overlays + detection box on a fresh frame, display, return key."""

    ok, frame = cap.read()
    if not ok:
        return 0
    disp = frame.copy()
    lines = [status_line]
    if info_lines:
        lines += list(info_lines)
    if hint:
        lines.append(hint)
    for i, line in enumerate(lines):
        cv2.putText(disp, line, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if det is not None and 'center_px' in det:
        x, y, w, h = det['bbox_px']
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 255), 2)
        c = (int(round(det['center_px'][0])), int(round(det['center_px'][1])))
        cv2.circle(disp, c, 4, (0, 255, 255), -1)
    if extra_marks:
        h, w = disp.shape[:2]
        for u, v, label, bgr in extra_marks:
            u, v = float(u), float(v)
            if not (np.isfinite(u) and np.isfinite(v)):
                continue        # a point on the camera plane projects to junk
            # Clip before drawing: a mark near the camera plane projects far
            # outside the frame, and an unclipped coordinate crashes drawMarker.
            p = (int(round(np.clip(u, -1e4, w + 1e4))),
                 int(round(np.clip(v, -1e4, h + 1e4))))
            cv2.drawMarker(disp, p, bgr, cv2.MARKER_CROSS, 14, 2)
            cv2.putText(disp, label, (p[0] + 8, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
    if axes is not None:
        draw_artifact_axes(disp, *axes)
    cv2.imshow(WINDOW, disp)
    if K is not None:
        masks = _both_masks(frame, K, dist_coeffs)
        cv2.imshow(MASK1_WINDOW, draw_debug(masks['1'], 'MASK 1'))
        cv2.imshow(MASK2_WINDOW, draw_debug(masks['2'], 'MASK 2'))
    # Drain the key queue so a HELD key cannot keep feeding events to the
    # caller after it is released — the g-gated motion must stop at once.
    key = cv2.waitKey(1) & 0xFF
    while True:
        k_next = cv2.waitKey(1) & 0xFF
        if k_next == 255:
            break
        key = k_next
    return key


def tune_masks(cap, K, dist_coeffs):
    """Pre-flight slider calibration: drag until each mask isolates its block."""

    size_windows()
    create_threshold_trackbars()
    progress('Mask tuning: drag the sliders until each mask isolates only its')
    progress('block. [Enter]=accept  [r]=reset defaults  [q]=quit')
    while True:
        ok, frame = cap.read()
        if not ok:
            progress('! camera frame grab failed')
            return False
        read_threshold_trackbars()
        masks = _both_masks(frame, K, dist_coeffs)
        cv2.imshow(WINDOW, frame)
        cv2.imshow(MASK1_WINDOW, draw_debug(masks['1'], 'MASK 1'))
        cv2.imshow(MASK2_WINDOW, draw_debug(masks['2'], 'MASK 2'))
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            save_mask_bands()
            progress('bands accepted and saved to data/%s; future runs start '
                     'from them' % MASK_BANDS_NAME)
            return True
        if key == ord('r'):
            reset_thresholds()
            sync_threshold_trackbars()
        if key == ord('q'):
            return False


def wait_next(status_line, cap, arm, zeta, K, dist, T_ee_cam, color, info_lines=None, abort_key='a'):
    """Block on the live feed until [Enter]=advance or [abort_key]/[q]=abort."""

    while True:
        ok, frame = cap.read()
        det = None
        lines = list(info_lines) if info_lines else []
        marks = []
        if ok:
            q_rad = _read_joints_safe(arm)
            if q_rad is not None:
                T_flange = flange_pose_base(q_rad, zeta)
                det = compute_detection(frame, color, K, dist, T_flange,
                                        T_ee_cam)
                if det is not None:
                    lines += [
                        'block : x %+.1f  y %+.1f  z %+.1f mm (top-face centre)'
                        % tuple(det['p_block_base'] * 1000.0),
                        'flange->block : %.1f mm  (xy gap %.1f mm)'
                        % (det['dist_flange_mm'], det['xy_gap_mm'])]
                    uv = project_base_to_uv(det['p_block_base'],
                                            T_flange @ T_ee_cam, K, dist)
                    if uv is not None:
                        marks.append((uv[0], uv[1], 'est', (0, 255, 255)))
        key = show_frame(cap, status_line, info_lines=lines, det=det,
                         color=color,
                         hint='[Enter]=advance  [%s/q]=abort' % abort_key,
                         extra_marks=marks, K=K, dist_coeffs=dist)
        if key in (13, 10):
            return True, False
        if key == ord(abort_key) or key == ord('q'):
            return False, True


def _read_joints_safe(arm):
    try:
        return robot.read_joints_rad(arm)
    except Exception:
        return None
