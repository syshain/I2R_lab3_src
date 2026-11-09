"""pnp_probe - throw every PnP method at one capture's corner data.

Reprojection error barely distinguishes poses for this artifact (a very flat
error valley), so solvePnP can settle on a different valley point from every
different view - tens of degrees and millimetres apart at nearly the same
reprojection. This helper collects all the solutions the standard methods can
find for one capture and reports whether several reproject almost equally
well.

From the main script:   PNP_PROBE=1 python handeye_gripper.py
Offline on the log:     python pnp_probe.py [entry_index]   (default: last)
"""

import json
import sys

import cv2
import numpy as np

from lab_config import data_path
from reporting import field, heading, progress

REPROJ_TIE_PX = 0.3     # candidates this close in reprojection are 'equally good'
AMBIG_ROT_DEG = 2.0     # ...yet this far apart in pose is the flat-valley signature
AMBIG_TRANS_MM = 10.0


def _reproj(cand, obj, img, K, dist):
    """Mean pixel distance between the projected model points and the measured."""

    proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), cand['rvec'],
                                cand['tvec'], K, dist)
    return float(np.mean(np.linalg.norm(
        proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)))


def _rot_deg(a, b):
    """Angle between two candidates' rotations [deg]."""

    Ra, _ = cv2.Rodrigues(a['rvec'])
    Rb, _ = cv2.Rodrigues(b['rvec'])
    cos = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _trans_mm(a, b):
    """Translation distance between two candidates [mm]."""

    return float(np.linalg.norm(a['tvec'] - b['tvec']) * 1000.0)


def solutions(object_points, image_points, K, dist):
    """One capture's points -> PnP candidates, sorted by reprojection error.

    Every candidate carries tag, rvec, tvec and reproj_px. Duplicates (same
    pose within 0.5 deg / 0.5 mm) are merged, keeping the first tag; methods
    that fail or reproject to NaN are dropped.
    """

    obj = np.asarray(object_points, dtype=np.float64)
    img = np.asarray(image_points, dtype=np.float64)
    found = []

    def try_add(tag, flags=None, guess=None):
        try:
            if guess is None:
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flags)
            else:
                ok, rvec, tvec = cv2.solvePnP(
                    obj, img, K, dist, guess[0].copy(), guess[1].copy(),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
            except cv2.error:
                pass
            found.append({'tag': tag,
                          'rvec': np.asarray(rvec, dtype=float).ravel(),
                          'tvec': np.asarray(tvec, dtype=float).ravel()})
        except cv2.error:
            pass

    methods = [('EPNP', cv2.SOLVEPNP_EPNP),
               ('ITERATIVE', cv2.SOLVEPNP_ITERATIVE),
               ('DLS', cv2.SOLVEPNP_DLS),
               ('UPNP', cv2.SOLVEPNP_UPNP)]
    if getattr(cv2, 'SOLVEPNP_SQP', None) is not None:
        methods.append(('SQP', cv2.SOLVEPNP_SQP))
    for tag, flags in methods:
        try_add(tag, flags)
    # ITERATIVE is seed-dependent: reseed it from every solution found so far
    for c in list(found):
        try_add('ITERATIVE<-%s' % c['tag'],
                guess=(c['rvec'].reshape(3, 1), c['tvec'].reshape(3, 1)))
    # IPPE on each single marker: the two planar folds of that marker
    for g in range(len(obj) // 4):
        try:
            out = cv2.solvePnPGeneric(obj[4 * g:4 * g + 4],
                                      img[4 * g:4 * g + 4],
                                      K, dist, flags=cv2.SOLVEPNP_IPPE)
            # version-dependent arity: (retval, rvecs, tvecs[, reprojErr])
            for j, (rvec, tvec) in enumerate(zip(out[1], out[2])):
                found.append({'tag': 'IPPE marker %d fold %d' % (g, j),
                              'rvec': np.asarray(rvec, dtype=float).ravel(),
                              'tvec': np.asarray(tvec, dtype=float).ravel()})
        except cv2.error:
            pass

    kept = []
    for c in found:
        c['reproj_px'] = _reproj(c, obj, img, K, dist)
        if not np.isfinite(c['reproj_px']):
            continue
        if not any(_rot_deg(c, k) < 0.5 and _trans_mm(c, k) < 0.5
                   for k in kept):
            kept.append(c)
    kept.sort(key=lambda c: c['reproj_px'])
    return kept


def report(cands, where=""):
    """Print the candidate table and the verdict; return the #near-equal rivals."""

    if not cands:
        progress("PnP probe: no candidate solutions - nothing to compare.")
        return 0
    best = cands[0]
    heading("PnP probe - %d candidate solutions%s"
            % (len(cands), " " + where if where else ""))
    for c in cands:
        field("%-24s reproj %6.2f px" % (c['tag'], c['reproj_px']),
              "%6.2f deg   %7.1f mm from the best"
              % (_rot_deg(c, best), _trans_mm(c, best)))
    rivals = [c for c in cands[1:]
              if c['reproj_px'] <= best['reproj_px'] + REPROJ_TIE_PX
              and (_rot_deg(c, best) > AMBIG_ROT_DEG
                   or _trans_mm(c, best) > AMBIG_TRANS_MM)]
    if rivals:
        progress("WARNING: the flat valley is real - %d rival solution(s) reproject"
                 % len(rivals))
        progress("  within %.1f px of the best yet sit far away in pose:"
                 % REPROJ_TIE_PX)
        for c in rivals:
            progress("  %s: reproj %.2f px, %.1f deg / %.1f mm from the best"
                     % (c['tag'], c['reproj_px'],
                        _rot_deg(c, best), _trans_mm(c, best)))
        progress("  solvePnP lands on whichever of these the view steers it to;")
        progress("  pick the candidate closest to the FK-predicted camera pose.")
    else:
        field("verdict", "no near-equal rival: worst candidate %.2f deg / %.1f mm"
              % (max(_rot_deg(c, best) for c in cands),
                 max(_trans_mm(c, best) for c in cands)))
    return len(rivals)


def run_probe(det, K, dist):
    """Probe one detector result (needs object_points/image_points); -> #rivals."""

    return report(solutions(det['object_points'], det['image_points'], K, dist))


def main():
    """Probe one capture-log entry offline: python pnp_probe.py [entry_index]."""

    import aruco_artifact_detector

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    with open(data_path('capture_log.json')) as f:
        log = json.load(f)
    label = idx if idx >= 0 else len(log) + idx
    entry = log[label]
    if 'corners_px' not in entry:
        progress("log entry %d predates corner logging (no corners_px) -" % label)
        progress("  take a capture with PNP_PROBE=1 to probe it.")
        return
    _, corners_m = aruco_artifact_detector.load_artifact_config()
    obj = np.vstack([corners_m[int(mid)] for mid in entry['marker_ids']])
    img = np.asarray(entry['corners_px'], dtype=np.float64)
    K = np.load(data_path('camera_matrix.npy'))
    dist = np.load(data_path('dist_coeffs.npy'))
    report(solutions(obj, img, K, dist), "(capture-log entry %d)" % label)


if __name__ == '__main__':
    main()
