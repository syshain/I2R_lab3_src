"""Colour-block detection + one-shot pose estimation."""

import cv2
import numpy as np
from scipy.optimize import minimize

from lab_config import (BLOCK_SIDE_MM, COLOR_THRESHOLDS, MAX_AREA_FRAC,
                        MIN_AREA_PX, MIN_CONFIDENCE)

_FIT_LOSS_MAX = 0.05           # above this the silhouette fit is rejected
_YAW_SCAN_DEG = np.arange(0.0, 90.0, 15.0)   # coarse seeds; square face -> yaw mod 90


def rgb_color_mask(frame_bgr, color):
    """Binary mask (0/255, uint8) of pixels matching one colour.

    The frame is BGR: unpack channels in b, g, r order.  A pixel is
    white when all THREE channel values lie inside one band of
    COLOR_THRESHOLDS[color] (with several bands, any band counts).
    Hint: write each channel test as a boolean array and combine
    them with & 
    Make sure you give every test its own brackets to ensure correct operation priority 
    """

    mask = <TO_FILL>  # Initialize mask as an empty, all-black frame of the right size
    band = COLOR_THRESHOLDS[color][0]  # Each of the BGR is given only one box from the sliders/config
    b, g, r =  # Extract b g r arrays from frame_bgr
    # OpenCV frames are BGR, not RGB (historical carry over: Windows BMP byte order)
    """
    Use the color boxes in band to threshold the colors. HINT: See lab_config for the keys defining the boxes
    Example: band['b_lo'] retrieves the lower bound for blue defined from config/live sliders
    CAUTION: Python loops are extremely slow and unsuitable for live feeds. In practice, opencv is commonly used since it wraps around C++
    Here, use bitwise operators (>=, <= for comparison, & to combine) 
    """
    in_band = <TO_FILL> # Use the extracted b g r arrays and color bands to build the thresholded frame  
    mask |= in_band.astype(np.uint8) * 255 # in_band being a logical array, we convert to a uint8 BW frame for findContours``

    return mask


def _best_blob(mask):
    """The best square-ish blob in a mask -> tuple or None."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(mask.shape[0] * mask.shape[1]) 
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX or area > MAX_AREA_FRAC * img_area:
            continue
        rect = cv2.minAreaRect(cnt)
        rect_area = max(rect[1][0] * rect[1][1], 1e-9)
        fill = area / rect_area
        per = max(cv2.arcLength(cnt, True), 1e-9)
        circ = min(4.0 * np.pi * area / (per * per), 1.0)
        conf = 0.6 * fill + 0.4 * circ
        if conf < MIN_CONFIDENCE:
            continue
        if best is None or conf > best[3]:
            M = cv2.moments(cnt)
            centroid = (M['m10'] / M['m00'], M['m01'] / M['m00'])
            best = (centroid, cv2.boundingRect(cnt), area, conf, rect, cnt)
    return best


def detect_color_block(frame, color, K, dist_coeffs):
    """One frame -> detection dict for one colour, or {'mask': ...}."""

    frame_u = cv2.undistort(frame, K, dist_coeffs)
    mask = rgb_color_mask(frame_u, color)
    blob = _best_blob(mask)
    if blob is None:
        return {'mask': mask}
    centroid, bbox, area, conf, rect, cnt = blob
    s_px = 0.5 * (rect[1][0] + rect[1][1])
    depth_m = float(K[0, 0]) * (BLOCK_SIDE_MM / 1000.0) / max(s_px, 1e-9)
    return {'center_px': np.asarray(centroid, dtype=float),
            'bbox_px': bbox, 'rect': rect, 'contour': cnt,
            'depth_m': depth_m, 'confidence': float(conf), 'mask': mask}


def unproject_to_base(u, v, depth_m, K, T_cam):
    """Pixel (u, v) at depth_m along its ray -> point in the BASE frame (m)."""

    d_cam = depth_m * np.array([(u - K[0, 2]) / K[0, 0],
                                (v - K[1, 2]) / K[1, 1], 1.0])
    return T_cam[:3, :3] @ d_cam + T_cam[:3, 3]


def project_base_to_uv(p_base, T_cam, K, dist_coeffs):
    """Base-frame point -> (u, v) pixels on the RAW feed, or None if behind."""

    p_cam = T_cam[:3, :3].T @ (np.asarray(p_base, dtype=float) - T_cam[:3, 3])
    if p_cam[2] <= 0:
        return None
    uv, _ = cv2.projectPoints(p_cam.reshape(1, 1, 3).astype(np.float64),
                              np.zeros(3), np.zeros(3), K, dist_coeffs)
    return float(uv[0, 0, 0]), float(uv[0, 0, 1])


def compute_detection(frame, color, K, dist_coeffs, T_flange, T_ee_cam):
    """detect_color_block + chain into the base frame. None if not seen."""

    det = detect_color_block(frame, color, K, dist_coeffs)
    if 'center_px' not in det:
        return None
    T_cam = T_flange @ T_ee_cam
    p_block = unproject_to_base(det['center_px'][0], det['center_px'][1],
                                det['depth_m'], K, T_cam)
    p_flange = T_flange[:3, 3].copy()
    out = dict(det)
    out.update({'p_block_base': p_block, 'p_flange_base': p_flange,
                'dist_flange_mm': float(np.linalg.norm(p_block - p_flange) * 1000.0),
                'xy_gap_mm': float(np.linalg.norm(p_block[:2] - p_flange[:2]) * 1000.0),
                'depth_mm': float(det['depth_m'] * 1000.0)})
    return out


def estimate_block_yaw(det, T_cam_base):
    """Block yaw from the blob's edge angle + the camera's own yaw. Naive."""

    rect = det['rect']
    box = cv2.boxPoints(rect)
    theta = np.arctan2(box[1][1] - box[0][1], box[1][0] - box[0][0])
    R = T_cam_base[:3, :3]
    alpha = np.arctan2(R[1, 0], R[0, 0])
    psi = alpha - theta
    return float((psi + np.pi) % (2.0 * np.pi) - np.pi)


def _block_corners(p_top_m, yaw):
    """The 8 corners of the hypothesised block, BASE frame (m). Top face."""

    S = BLOCK_SIDE_MM / 1000.0
    half = S / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    local = np.array([[-half, -half, 0.0], [half, -half, 0.0],
                      [half, half, 0.0], [-half, half, 0.0],
                      [-half, -half, -S], [half, -half, -S],
                      [half, half, -S], [-half, half, -S]])
    return (Rz @ local.T).T + np.asarray(p_top_m, dtype=float)


def _rect_measures(rect):
    """(centre, sorted extents, in-image edge angle folded to [0, pi/2))."""

    box = cv2.boxPoints(rect)
    theta = np.mod(np.arctan2(box[1][1] - box[0][1],
                              box[1][0] - box[0][0]), np.pi / 2)
    return (np.asarray(rect[0], dtype=float),
            np.sort(np.asarray(rect[1], dtype=float)), theta)


def _project_base(pts_base, T_cam, K):
    """Base-frame points (N, 3) -> (N, 2) pixels through K, NO distortion."""

    R, t = T_cam[:3, :3], T_cam[:3, 3]
    p_cam = (np.asarray(pts_base, dtype=float) - t) @ R   # rows of R^T p_cam
    uv, _ = cv2.projectPoints(p_cam.reshape(-1, 1, 3).astype(np.float64),
                              np.zeros(3), np.zeros(3), K, np.zeros(5))
    return uv.reshape(-1, 2)


def _fit_loss(params, rect_real, K, T_cam):
    """Silhouette mismatch of a hypothesised (x, y, z, yaw) against the measured."""

    x, y, z, yaw = params
    uv = _project_base(_block_corners(np.array([x, y, z]), yaw),
                       T_cam, K)
    c_f, e_f, th_f = _rect_measures(cv2.minAreaRect(uv.astype(np.float32)))
    c_r, e_r, th_r = rect_real
    dth = np.mod(th_f - th_r + np.pi / 4, np.pi / 2) - np.pi / 4
    return float((((e_f - e_r) / e_r) ** 2).sum()
                 + (np.linalg.norm(c_f - c_r) / e_r.mean()) ** 2
                 + (dth / (np.pi / 4)) ** 2)


def fit_block_pose(det, K, T_cam):
    """One detection -> (p_top_base (m, 3), yaw_rad, info dict)."""

    u, v = det['center_px']
    d_cam = det['depth_m'] * np.array([(u - K[0, 2]) / K[0, 0],
                                       (v - K[1, 2]) / K[1, 1], 1.0])
    ray_base = T_cam[:3, :3] @ (d_cam / np.linalg.norm(d_cam))
    tilt_deg = float(np.degrees(np.arccos(np.clip(-ray_base[2], -1.0, 1.0))))
    p_naive = T_cam[:3, :3] @ d_cam + T_cam[:3, 3]

    rect_real = _rect_measures(det['rect'])
    seeds = [(p_naive[0], p_naive[1], p_naive[2], np.deg2rad(a))
             for a in _YAW_SCAN_DEG]
    x0 = min(seeds, key=lambda s: _fit_loss(s, rect_real, K, T_cam))
    res = minimize(_fit_loss, x0, args=(rect_real, K, T_cam),
                   method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 600})
    info = {'tilt_deg': tilt_deg, 'fit_loss': float(res.fun),
            'naive_used': False,
            'z_naive_mm': float(p_naive[2] * 1000.0)}
    if res.fun > _FIT_LOSS_MAX:
        yaw = estimate_block_yaw(det, T_cam)
        info['naive_used'] = True
        return p_naive, yaw, info
    x, y, z, yaw = res.x
    yaw = float((yaw + np.pi) % (2.0 * np.pi) - np.pi)
    return np.array([x, y, z]), yaw, info


def one_shot_estimate(frame, color, K, dist_coeffs, T_flange, T_ee_cam):
    """One frame -> the block's top-face pose in the BASE frame, or None."""

    det = detect_color_block(frame, color, K, dist_coeffs)
    if 'center_px' not in det:
        return None
    T_cam = T_flange @ T_ee_cam
    p_top, yaw, info = fit_block_pose(det, K, T_cam)
    return {'p_block_base': p_top, 'yaw_rad': yaw,
            'depth_mm': float(np.linalg.norm(p_top - T_cam[:3, 3]) * 1000.0),
            'tilt_deg': info['tilt_deg'], 'fit_loss': info['fit_loss'],
            'naive_used': info['naive_used'],
            'z_naive_mm': info['z_naive_mm'],
            'confidence': det['confidence'], 'center_px': det['center_px'],
            'bbox_px': det['bbox_px'], 'rect': det['rect'],
            'mask': det['mask']}
