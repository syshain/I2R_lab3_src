"""Stage 1 of Lab 3: calibrate T_6_G (flange->cup tip) and T_G_C (gripper->camera)"""


import os

import cv2
import json
import numpy as np

import aruco_artifact_detector
import pnp_probe
import robot
from lab_config import (GRIPPER_CALIBRATION_NAME, TABLE_Z_MM, TOOL_OFFSET_MM,
                        data_path)
from motion import flange_pose_base
from reporting import banner, field, heading, progress
from ui_display import (WINDOW, draw_artifact_axes, load_camera_intrinsics,
                        load_calibration_results, open_camera, show_frame)
from vision import project_base_to_uv


OUT_FILE = data_path(GRIPPER_CALIBRATION_NAME)
CAPTURE_LOG = data_path('capture_log.json')


def _log_capture(entry):
    """Append one CAPTURE 1 measurement to ../data/capture_log.json."""

    try:
        with open(CAPTURE_LOG) as f:
            log = json.load(f)
    except (OSError, ValueError):
        log = []
    log.append(entry)
    with open(CAPTURE_LOG, 'w') as f:
        json.dump(log, f, indent=1)

TOP_MARKER_ID = 0        # the flat face this marker sits on is the artifact's top face
MIN_TOUCHES = 3          # the consistency check needs several touches; about 6 is a good number
CONSISTENCY_WARN_MM = 5.0  # per-touch d spread above this means the touches disagree
GRAZE_COS = 0.5          # |n.z| below this (touch >60 deg from the face normal) is ill-conditioned
CAPTURE_AGREE_WARN_MM = 10.0  # CAPTURE 1 repeats disagreeing more: one is a bad PnP solution
INPLANE_WARN_MM = 20.0   # contacts farther than this from the face centre: CAPTURE 1 is suspect
MIN_CAPTURE_MARKERS = 2  # PnP with fewer markers is ill-conditioned (tilt comes
                         # from foreshortening alone) - refuse such captures
MEDOID_TIE_MM = 5.0      # captures this close in total disagreement are tied;
                         # the tiebreaker prefers the most markers seen

KEY_CAPTURE = ord(' ')
KEY_SAVE = 27            # ESC


def top_face_centre(marker_world, marker_id=TOP_MARKER_ID):
    """Centre of the artifact's top face: centroid of the face marker's corners (m)."""
    return marker_world[marker_id].mean(axis=0)


def artifact_pose_in_base(q_rad, zeta, T_6_C, T_C_W):
    """CAPTURE 1 maths: T_0_A = T_0_6 @ T_6_C @ T_C_W (all 4x4, metres)."""
    return flange_pose_base(q_rad, zeta) @ T_6_C @ T_C_W


def top_face_plane(T_0_A, marker_world, marker_id=TOP_MARKER_ID):
    """The top face as a PLANE in the base frame: (n, p_face), both (m)."""

    corners_A = marker_world[marker_id]
    p_face_A = corners_A.mean(axis=0)   # [0, 0, 0]: frame A's origin is the face centre
    d1 = corners_A[2] - corners_A[0]
    d2 = corners_A[3] - corners_A[1]
    n_A = np.cross(d1, d2)
    n_A /= np.linalg.norm(n_A)
    # n's sign is arbitrary: the touch solve uses the plane equation n.x = n.p_face
    # and the graze check uses |n.z|, both invariant to flipping n.
    R, t = T_0_A[:3, :3], T_0_A[:3, 3]
    return R @ n_A, R @ p_face_A + t


def touch_d_estimate(T_0_6, n, p_face):
    """One touch on the top face -> (d_i, n_z): solve n.(t + d z) = n.p_face for d."""

    z = T_0_6[:3, :3] @ np.array([0.0, 0.0, 1.0])
    n_z = float(n @ z)
    d = float(n @ (p_face - T_0_6[:3, 3])) / n_z
    return d, n_z


def wrist_tilt_deg(T_0_6):
    """Angle between the cup axis (flange +z) and base -z, i.e. 0 = vertical."""
    flange_z = T_0_6[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(-flange_z[2], -1.0, 1.0))))


def _show_frozen(frame, status_line, info_lines, hint, marks, axes=None):
    """Show one FROZEN frame with the same overlay style as show_frame."""

    img = frame.copy()
    if axes is not None:
        draw_artifact_axes(img, *axes)
    h, w = img.shape[:2]
    for u, v, lbl, colr in marks:
        u, v = float(u), float(v)
        if not (np.isfinite(u) and np.isfinite(v)):
            continue        # a point on the camera plane projects to junk
        # Clip before drawing: a mark near the camera plane projects far
        # outside the frame, and an unclipped coordinate crashes drawMarker.
        p = (int(round(np.clip(u, -1e4, w + 1e4))),
             int(round(np.clip(v, -1e4, h + 1e4))))
        cv2.drawMarker(img, p, colr, cv2.MARKER_TILTED_CROSS, 18, 3)
        cv2.putText(img, lbl, (p[0] + 10, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colr, 2)
    bar_h = 90 + len(info_lines) * 28
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], bar_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    lines = [status_line, hint]
    for i, ln in enumerate(lines):
        cc = (0, 200, 255) if i == 0 else (200, 200, 200)
        cv2.putText(img, ln, (10, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cc, 2)
    for j, ln in enumerate(info_lines):
        cv2.putText(img, ln, (10, 28 + 2 * 28 + j * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow(WINDOW, img)
    return cv2.waitKey(0) & 0xFF


def capture_artifact_pose(arm, cap, detector, zeta, T_6_C, p_top_A):
    """CAPTURE 1: camera looks at the stationary artifact -> (T_0_A, capture).

    Several captures can be taken ([SPACE] each; move the wrist between them
    so the views differ). The kept result is the capture whose top-face-centre
    claim agrees best with the others, because one near-planar view can PnP to
    a wildly wrong pose at a tiny reprojection error.
    """

    caps = []          # accepted captures: {'T_0_A', 'capture', 'p_claim'}
    T_0_A = None       # capture awaiting confirmation
    capture = None
    frame_cap = None
    while True:
        try:
            q_rad = robot.read_joints_rad(arm)
        except Exception as e:
            progress("! joint read failed: %s" % e)
            return None
        ret, frame = cap.read()
        det = detector.detect(frame) if ret else None

        info = ["captures accepted: %d" % len(caps)]
        if len(caps) >= 2:
            cl = np.array([c['p_claim'] for c in caps]) * 1000.0
            info.append("spread so far: %.1f mm"
                        % float(np.linalg.norm(cl - cl.mean(axis=0), axis=1).max()))
        info.append("artifact pose: %s"
                    % ("captured - confirm it" if T_0_A is not None
                       else "not captured"))
        if det is None:
            info.append("artifact NOT in view")
        else:
            info.append("reproj: %.1f px   markers: %s" % (det['reproj_px'], det['ids']))
        if T_0_A is not None:
            info.append("T_0_A position [mm]: %s"
                        % np.array2string(T_0_A[:3, 3] * 1000.0, precision=1))

        axes = None
        if det is not None:
            axes = (det['rvec'], det['tvec'], detector.K, detector.dist)
        key = show_frame(
            cap, "CAPTURE 1 - artifact pose in the base frame", info_lines=info,
            axes=axes,
            hint="[SPACE]=capture   [ESC]=finish (>=1 capture)   [r]=clear   [q]=abort")

        if key == KEY_CAPTURE:
            if T_0_A is not None:
                continue           # a capture is pending: confirm it first
            if det is None:
                progress("! nothing to capture: artifact not detected")
                continue
            if len(det['ids']) < MIN_CAPTURE_MARKERS:
                progress("! only %d marker(s) in view - PnP with so few points"
                         % len(det['ids']))
                progress("  is ill-conditioned: move the wrist until %d or more"
                         % MIN_CAPTURE_MARKERS)
                progress("  markers are detected, then capture.")
                continue
            capture = (q_rad, det['T_C_W'])
            frame_cap = frame.copy()
            T_0_A = artifact_pose_in_base(q_rad, zeta, T_6_C, det['T_C_W'])
            _log_capture({'q_rad': np.asarray(q_rad).tolist(),
                          'T_C_W': det['T_C_W'].tolist(),
                          'T_0_A': T_0_A.tolist(),
                          'marker_ids': det['ids'],
                          'corners_px': det['image_points'].tolist(),
                          'reproj_px': float(det['reproj_px'])})
            if os.environ.get('PNP_PROBE') == '1':
                pnp_probe.run_probe(det, detector.K, detector.dist)
            field("captured", "T_0_A position [mm] = %s"
                  % np.array2string(T_0_A[:3, 3] * 1000.0, precision=1))
        elif key == ord('r'):
            if T_0_A is not None:
                T_0_A = capture = frame_cap = None
                progress("capture cleared - press [SPACE] to retake.")
            else:
                progress("nothing to clear.")
            continue
        elif key == KEY_SAVE:
            if T_0_A is not None:
                continue           # confirm or clear the pending capture first
            if not caps:
                progress("! nothing captured yet - press [SPACE] first.")
                continue
            claims = np.array([c['p_claim'] for c in caps])
            D = np.linalg.norm(claims[:, None] - claims[None, :], axis=2)
            # Medoid of the face-centre claims; among captures within
            # MEDOID_TIE_MM of the best, keep the most markers seen.
            Dsum = D.sum(axis=1)
            near = np.where(Dsum <= Dsum.min() + MEDOID_TIE_MM)[0]
            k = int(near[int(np.argmax([caps[i]['n_markers'] for i in near]))])
            spread_mm = float(D.max()) * 1000.0
            field("kept capture", "%d of %d (%d markers); face-centre spread %.1f mm"
                  % (k + 1, len(caps), caps[k]['n_markers'], spread_mm))
            if len(caps) == 1:
                progress("only one capture - no cross-check was possible.")
            elif spread_mm > CAPTURE_AGREE_WARN_MM:
                progress("WARNING: captures disagree by %.1f mm > %.1f mm -"
                         % (spread_mm, CAPTURE_AGREE_WARN_MM))
                progress("         a bad PnP solution is likely among them;")
                progress("         [q] aborts so nothing bad is saved.")
            return caps[k]['T_0_A'], caps[k]['capture']
        elif key == ord('q'):
            return None
        else:
            continue

        # Frozen confirmation: the captured view, the pose printed, the
        # artifact's axes, and the touch point (top-face centre) marked
        # where it sat in that image.
        T_0_6 = flange_pose_base(capture[0], zeta)
        rv_cap, _ = cv2.Rodrigues(capture[1][:3, :3])
        axes_cap = (rv_cap, capture[1][:3, 3].reshape(3, 1),
                    detector.K, detector.dist)
        uv = project_base_to_uv((T_0_A @ np.hstack([p_top_A, 1.0]))[:3],
                                T_0_6 @ T_6_C, detector.K, detector.dist)
        marks = []
        if uv is not None:
            marks.append((uv[0], uv[1], "touch here", (255, 0, 255)))
        frozen_info = ["T_0_A position [mm]: %s"
                       % np.array2string(T_0_A[:3, 3] * 1000.0, precision=1),
                       "cross = the point the cup tip must touch"]
        # Plausibility guard: a bad PnP depth (wrong resolution, oblique
        # view) pushes the face along the camera ray. Warn before accept
        # so a bad capture is retaken, never saved.
        p_A = T_0_A[:3, 3]
        d_flange_mm = np.linalg.norm(p_A - T_0_6[:3, 3]) * 1000.0
        if p_A[2] * 1000.0 < TABLE_Z_MM:
            frozen_info.insert(1, "WARNING: face BELOW the table (z %.1f mm) -"
                                  % (p_A[2] * 1000.0))
            frozen_info.insert(2, "capture is bad: [r] retake, camera closer/frontal")
        elif d_flange_mm > 1500.0:
            frozen_info.insert(1, "WARNING: face %.0f mm from the flange -"
                                  % d_flange_mm)
            frozen_info.insert(2, "implausible: [r] retake, camera closer/frontal")
        while True:
            k = _show_frozen(
                frame_cap, "CAPTURE 1 captured - check the pose, then accept",
                frozen_info, "[SPACE]=accept   [r]=retake   [q]=abort", marks,
                axes=axes_cap)
            if k in (KEY_CAPTURE, KEY_SAVE):
                p_claim = (T_0_A @ np.hstack([p_top_A, 1.0]))[:3]
                caps.append({'T_0_A': T_0_A, 'capture': capture,
                             'p_claim': p_claim, 'n_markers': len(det['ids'])})
                if len(caps) == 1:
                    progress("capture 1 accepted - [SPACE] for another (move"
                             " the wrist between captures), [ESC] to finish.")
                else:
                    d_mm = float(np.linalg.norm(p_claim - caps[-2]['p_claim'])
                                 * 1000.0)
                    field("capture %d accepted" % len(caps),
                          "%.1f mm from capture %d's face centre"
                          % (d_mm, len(caps) - 1))
                    if d_mm > CAPTURE_AGREE_WARN_MM:
                        progress("  WARNING: above %.1f mm - at least one of"
                                 " these two captures is a bad"
                                 % CAPTURE_AGREE_WARN_MM)
                        progress("  PnP solution; retake them ([q] aborts,"
                                 " the log keeps everything).")
                T_0_A = capture = frame_cap = None
                break
            if k == ord('r'):
                progress("capture cleared - press [SPACE] to retake.")
                T_0_A = capture = frame_cap = None
                break
            if k == ord('q'):
                return None


def _touch_panel(info_lines, hint):
    """Status panel for CAPTURE 2: no camera feed, nothing drawn from it."""

    img = np.zeros((120 + max(len(info_lines), 1) * 28, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, "CAPTURE 2 - touches on the top face (any tilt is fine)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(img, hint, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)
    for j, ln in enumerate(info_lines):
        cv2.putText(img, ln, (10, 84 + j * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
    cv2.imshow(WINDOW, img)
    return cv2.waitKey(30) & 0xFF


def capture_touch(arm, zeta, T_0_A, p_top_A, n, p_face):
    """CAPTURE 2: several touches on the top face -> list of touch records or None."""

    touches = []
    while True:
        try:
            q_rad = robot.read_joints_rad(arm)
        except Exception as e:
            progress("! joint read failed: %s" % e)
            return None

        T_0_6 = flange_pose_base(q_rad, zeta)
        tilt = wrist_tilt_deg(T_0_6)

        info = ["touches recorded: %d%s" % (len(touches),
                 "  (need at least %d)" % MIN_TOUCHES if len(touches) < MIN_TOUCHES else "")]
        if touches:
            d_all = np.array([t['d_mm'] for t in touches])
            info.append("d: mean %.2f mm   std %.2f mm   spread %.2f mm"
                        % (d_all.mean(), d_all.std(), d_all.max() - d_all.min()))
        info.append("wrist tilt now: %.1f deg - any tilt is fine, avoid grazing" % tilt)
        info.append("flange-to-face-centre distance [mm]: %.0f"
                    % (np.linalg.norm(p_face - T_0_6[:3, 3]) * 1000.0))

        key = _touch_panel(
            info,
            "[SPACE]=record touch   [r]=remove last   [c]=clear all   "
            "[ESC]=save   [q]=abort")

        if key == KEY_CAPTURE:
            d_i, n_z = touch_d_estimate(T_0_6, n, p_face)
            z = T_0_6[:3, :3] @ np.array([0.0, 0.0, 1.0])
            p_contact_A = (np.linalg.inv(T_0_A) @
                           np.hstack([T_0_6[:3, 3] + d_i * z, 1.0]))[:3]
            touch = {
                'q_rad': q_rad.copy(), 'T_0_6': T_0_6,
                'd_mm': d_i * 1000.0, 'tilt_deg': tilt,
                'inplane_mm': float(np.linalg.norm(p_contact_A - p_top_A) * 1000.0),
                'graze': abs(n_z) < GRAZE_COS,
            }
            touches.append(touch)
            field("touch %d" % len(touches),
                  "d = %+.2f mm   tilt = %.1f deg   contact %.1f mm from the face centre"
                  % (touch['d_mm'], tilt, touch['inplane_mm']))
            if touch['graze']:
                progress("WARNING: nearly parallel to the face - this touch's d is"
                         " unreliable, consider [r]")
        elif key == ord('r'):
            if touches:
                touches.pop()
                progress("last touch removed (%d left)." % len(touches))
            else:
                progress("nothing to remove.")
        elif key == ord('c'):
            touches = []
            progress("all touches cleared.")
        elif key == KEY_SAVE:
            if len(touches) < MIN_TOUCHES:
                progress("! %d touches recorded, but at least %d are needed for "
                         "the consistency check." % (len(touches), MIN_TOUCHES))
                continue
            progress("touches accepted.")
            return touches
        elif key == ord('q'):
            return None


def report_and_save(T_0_A, touches, p_top_A, T_6_C, capture):
    """Solve d from the touches, build T_6_G and T_G_C, print the student-record."""

    d_mm = np.array([t['d_mm'] for t in touches])
    d = float(d_mm.mean()) / 1000.0
    d_std_mm = float(d_mm.std())
    d_spread_mm = float(d_mm.max() - d_mm.min())

    T_6_G = np.eye(4)
    T_6_G[2, 3] = d                      # cup tip lies d along the flange +z axis
    T_G_C = np.linalg.inv(T_6_G) @ T_6_C

    # T_G_A is pose-dependent (each touch lands at its own in-plane point), so
    # the touch that landed closest to the face centre is kept.
    best = min(touches, key=lambda t: t['inplane_mm'])
    T_G_A = np.linalg.inv(best['T_0_6'] @ T_6_G) @ T_0_A

    print()
    banner("GRIPPER CALIBRATION RESULT (multi-touch)")
    field("touches used", "%d" % len(touches))
    heading("per-touch touches")
    for k, t in enumerate(touches, 1):
        flags = "GRAZE (ill-conditioned)" if t['graze'] else ""
        field("touch %d" % k,
              "d %10.2f mm   tilt %6.1f deg   contact %8.1f mm   %s"
              % (t['d_mm'], t['tilt_deg'], t['inplane_mm'], flags))
    field("d stats", "mean %.2f / std %.2f / spread %.2f mm"
          % (d * 1000.0, d_std_mm, d_spread_mm))
    field("nominal", "TOOL_OFFSET_MM %.2f mm for reference" % TOOL_OFFSET_MM)
    if abs(d * 1000.0 - TOOL_OFFSET_MM) > 20.0:
        progress("WARNING: fitted d differs from the nominal %.1f mm by %.1f mm"
                 % (TOOL_OFFSET_MM, abs(d * 1000.0 - TOOL_OFFSET_MM)))
        progress("         the nominal is a rough tape measure, but a gap this")
        progress("         large usually means a bad CAPTURE 1 - retake it.")
    if d <= 0:
        progress("WARNING: d is not positive - the tip should sit in FRONT of")
        progress("         the flange along +z. Check the touches and retake.")
    if d_spread_mm > CONSISTENCY_WARN_MM:
        progress("WARNING: per-touch d spread %.2f mm > %.2f mm - the touches"
                 % (d_spread_mm, CONSISTENCY_WARN_MM))
        progress("         disagree. Check the GRAZE flags and retake outliers.")
    inplane_mm = np.array([t['inplane_mm'] for t in touches])
    if inplane_mm.max() > INPLANE_WARN_MM:
        progress("WARNING: contacts up to %.1f mm from the face centre"
                 % inplane_mm.max())
        progress("         (> %.1f mm): the touches are landing away from where"
                 % INPLANE_WARN_MM)
        progress("         CAPTURE 1 puts the face centre - CAPTURE 1 is suspect.")
    field("T_G_C", "translation [mm] %s"
          % np.array2string(T_G_C[:3, 3] * 1000.0, precision=1))
    field("T_G_A", "translation [mm] %s   (artifact relative to the tool at the"
          % np.array2string(T_G_A[:3, 3] * 1000.0, precision=1))
    progress("           touch closest to the face centre)")

    data = {
        'T_6_G': T_6_G,
        'T_G_C': T_G_C,
        'T_G_A': T_G_A,
        'tool_offset_mm': d * 1000.0,
        'T_0_A': T_0_A,
        'top_point_artifact_mm': p_top_A * 1000.0,
        'touch_q_rad': np.array([t['q_rad'] for t in touches]),
        'touch_d_mm': d_mm,
        'touch_tilt_deg': np.array([t['tilt_deg'] for t in touches]),
        'touch_inplane_mm': np.array([t['inplane_mm'] for t in touches]),
        'd_std_mm': d_std_mm,
        'd_spread_mm': d_spread_mm,
        'capture_q_rad': capture[0],
        'T_C_W_capture': capture[1],
        'top_marker_id': TOP_MARKER_ID,
    }
    np.save(OUT_FILE, data, allow_pickle=True)
    field("saved", OUT_FILE)
    progress("Record the per-touch d values, their spread, and T_G_A in your lab book.")


def main():
    banner("STAGE 1 - GRIPPER CALIBRATION (several camera captures + several touches)")
    progress("CAPTURE 1: [SPACE] captures the artifact's pose and freezes the frame")
    progress("           so you can check it before accepting; take several captures,")
    progress("           moving the wrist between them - the best-agreeing one is kept.")
    progress("CAPTURE 2: move the arm BY HAND and touch the artifact's TOP face (the")
    progress("           flat face carrying marker %d) with the cup tip several times:"
             % TOP_MARKER_ID)
    progress("           any point and any tilt are fine, at least %d touches. [SPACE]"
             % MIN_TOUCHES)
    progress("           records, [r] removes the last, [c] clears all, [ESC] saves.")
    print()

    arm = robot.connect_arm()
    cap = open_camera()
    try:
        K, dist_coeffs = load_camera_intrinsics()
        zeta, T_6_C = load_calibration_results()
        detector = aruco_artifact_detector.ArucoArtifactDetector(K, dist_coeffs)
        p_top_A = top_face_centre(detector.marker_world)

        capture = capture_artifact_pose(arm, cap, detector, zeta, T_6_C, p_top_A)
        if capture is None:
            progress("Aborted - nothing saved.")
            return
        T_0_A, capture = capture

        n, p_face = top_face_plane(T_0_A, detector.marker_world)
        print()
        progress("Now move the arm manually and touch the artifact's TOP face")
        progress("several times - any point on the face, any wrist tilt.")
        print()
        touches = capture_touch(arm, zeta, T_0_A, p_top_A, n, p_face)
        if touches is None:
            progress("Aborted - nothing saved.")
            return

        report_and_save(T_0_A, touches, p_top_A, T_6_C, capture)
    finally:
        cap.release()
        arm.disconnect()


if __name__ == '__main__':
    main()
