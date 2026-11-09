"""ArUco artifact: model from board_config.json + monocular pose T_C_W."""

import json
import cv2
import numpy as np

from lab_config import BOARD_CONFIG_NAME, data_path

ARUCO_DICT_NAME = 'DICT_6X6_250'
REPROJ_REJECT_PX = 15.0
MIN_PNP_POINTS = 4               # one marker's corners suffice


def load_artifact_config(config_path=None):
    """board_config.json -> (marker_ids, {id: (4,3) corner coords in METRES})."""

    if config_path is None:
        config_path = data_path(BOARD_CONFIG_NAME)
    with open(config_path) as f:
        cfg = json.load(f)
    artifact = next((t for t in cfg['toolList'] if t.get('id') == 'ArucoBoard'),
                    cfg['toolList'][0])
    marker_ids = list(artifact['marker_ids'])
    corners_m = {int(mid): np.asarray(raw, dtype=np.float64) / 1000.0
                 for mid, raw in zip(marker_ids,
                                     artifact['marker_corners_mm'])}
    # Frame A is the config's own frame: marker 0 (the top face) is centred at
    # [0, 0, 0], so T_0_A's origin IS the touch point. No re-centring.
    return marker_ids, corners_m


class ArucoArtifactDetector:
    """Detects the artifact in one camera frame and solves its pose T_C_W (m)."""

    def __init__(self, K, dist_coeffs, config_path=None,
                 dict_name=ARUCO_DICT_NAME, reproj_reject_px=REPROJ_REJECT_PX):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist_coeffs, dtype=np.float64)
        self.reproj_reject_px = float(reproj_reject_px)
        self.marker_ids, self.marker_world = load_artifact_config(config_path)

        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    def _matched_points(self, corners, ids):
        """Pooled model and image points for every detected marker we know."""
        object_points, image_points, matched = [], [], []
        for i, mid in enumerate(ids):
            model = self.marker_world.get(int(mid))
            if model is None:
                continue
            object_points.append(model)
            image_points.append(np.asarray(corners[i][0], dtype=np.float32))
            matched.append(int(mid))
        if not matched:
            return None, None, []
        return (np.vstack(object_points).astype(np.float32),
                np.vstack(image_points), matched)

    def detect(self, frame_bgr):
        """One camera frame -> dict(T_C_W, rvec, tvec, reproj_px, ids) or None."""

        if frame_bgr is None:
            return None
        corners, ids, _ = self.detector.detectMarkers(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        if ids is None or len(ids) == 0:
            return None
        object_points, image_points, matched = self._matched_points(
            corners, ids.flatten())
        if object_points is None or len(object_points) < MIN_PNP_POINTS:
            return None
        rvec, tvec = self._solve(object_points, image_points)
        if rvec is None:
            return None
        projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                         self.K, self.dist)
        reproj_px = float(np.mean(np.linalg.norm(
            projected.reshape(-1, 2) - image_points, axis=1)))
        if reproj_px > self.reproj_reject_px:
            return None
        R, _ = cv2.Rodrigues(rvec)
        T_C_W = np.eye(4)
        T_C_W[:3, :3] = R
        T_C_W[:3, 3] = np.asarray(tvec, dtype=float).ravel()
        return {'T_C_W': T_C_W, 'rvec': np.asarray(rvec, dtype=float),
                'tvec': np.asarray(tvec, dtype=float).ravel(),
                'reproj_px': reproj_px, 'ids': matched,
                'object_points': object_points, 'image_points': image_points}

    def _solve(self, object_points, image_points):
        """PnP on the pooled points -> (rvec, tvec); (None, None) on failure."""
        for flags in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP):
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                          self.K, self.dist, flags=flags)
            if ok:
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        object_points, image_points, self.K, self.dist,
                        rvec, tvec)
                except cv2.error:
                    pass
                return np.asarray(rvec, dtype=float), np.asarray(tvec, dtype=float)
        return None, None
