"""lab_config — every tunable for Lab 3 (colour-block stacking) in one place."""

import copy
import json
import os

import numpy as np

# Hardware (env-overridable, same mechanism as Labs 1-2)
ROBOT_IP = os.environ.get('ROBOT_IP', '192.168.1.158')
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '4'))   # wrist USB camera

# Solver / joint box (degrees). These are the bench/controller limits this
# rig actually enforces, which are narrower than the Lite 6 manual's; Lab 2
# solved against them too. ik_lite6.py and robot.py both read these.
JOINT_LIMITS_DEG = np.array([[80.0, 270.0],
                             [-150.0, 150.0],
                             [-3.5, 300.0],
                             [-360.0, 360.0],
                             [-124.0, 124.0],
                             [-180.0, 180.0]])


IK_FD_EPS_RAD = 1e-5        # centred-difference step for the IK Jacobian, radians

# Block geometry / workflow / detection
BLOCK_SIDE_MM = 25.0
AXES_LENGTH_MM = 20.0       # axis arrows on the live preview when the artifact
                            # is detected (ui_display.py); frame W's own axes
TOOL_OFFSET_MM = 75.0       # flange -> cup-tip distance along flange +z (MM);
                            # measured by the two-capture touch calibration in
                            # handeye_gripper.py -> ../data/gripper_calibration.npy
DOWN_RPY_DEG = (0.0, 180.0, 0.0)
HOVER_ABOVE_MM = 50.0
LIFT_CLEAR_MM = 80.0
TABLE_Z_MM = 12.5
KEY_STEP_MM = 3.0
KEY_SPEED_PCT = 10
GRIPPER_SETTLE_S = 2.0      # vacuum build-up wait after [SPACE] engages it,
                            # before any further motion (pick_and_place.py)
STREAM_TICK_S = 0.02        # servo-stream tick period (s); mode-1 executor
STREAM_SPEED_MM_S = 40.0    # commanded cup speed while streaming (MM/s)
STREAM_DESCENT_MM_S = 20.0  # streamed cup speed for contact phases (MM/s)
MIN_CONFIDENCE = 0.35
MIN_AREA_PX = 300
MAX_AREA_FRAC = 0.5
# Masks are numbered, not coloured: block 1 / block 2 — any colour of block
# can be handled by re-tuning the bands at runtime (ui_display.tune_masks).
# The shipped defaults happen to suit mask 1 = green-ish block, mask 2 = red-ish.
_RGB_DEFAULTS = {'1': [{'r_lo': 0, 'r_hi': 110, 'g_lo': 120, 'g_hi': 255, 'b_lo': 0, 'b_hi': 110}], '2': [{'r_lo': 150, 'r_hi': 255, 'g_lo': 0, 'g_hi': 110, 'b_lo': 0, 'b_hi': 110}]}

def _clone_defaults():
    """Deep-copy the default RGB bands (so reset restores an independent copy)."""
    return copy.deepcopy(_RGB_DEFAULTS)
COLOR_THRESHOLDS = _clone_defaults()

# Data files (all under ../data/, reached via data_path())
DATA_FOLDER = 'data'
BOARD_CONFIG_NAME = 'aruco_config.json'   # the artifact config file on disk
CAMERA_MATRIX_NAME = 'camera_matrix.npy'
DIST_COEFFS_NAME = 'dist_coeffs.npy'
CALIBRATION_DATA_NAME = 'calibration_data.npy'
RESULTS_NPY_NAME = 'calibration_results.npy'
GRIPPER_CALIBRATION_NAME = 'gripper_calibration.npy'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_FOLDER)

def data_path(filename):
    """Absolute path to a file inside the data folder."""
    return os.path.join(DATA_DIR, filename)

TRAJECTORY_FOLDER = 'trajectories'  # executed-path plots + CSVs land here
TRAJECTORY_DIR = os.path.join(DATA_DIR, TRAJECTORY_FOLDER)   # (path_recorder.py)

MASK_BANDS_NAME = 'mask_bands.json'   # accepted colour bands (tune_masks)


def load_mask_bands():
    """Load bands accepted by tune_masks in a previous run into
    COLOR_THRESHOLDS; returns 'saved' or 'defaults' for the status line."""
    p = data_path(MASK_BANDS_NAME)
    if os.path.exists(p):
        try:
            with open(p) as f:
                saved = json.load(f)
            if set(saved) == set(COLOR_THRESHOLDS):
                for colour, bands in saved.items():
                    COLOR_THRESHOLDS[colour] = [dict(b) for b in bands]
                return 'saved'
        except (ValueError, OSError):
            pass
    return 'defaults'


def save_mask_bands():
    """Write the current COLOR_THRESHOLDS to data/mask_bands.json."""
    with open(data_path(MASK_BANDS_NAME), 'w') as f:
        json.dump(COLOR_THRESHOLDS, f, indent=2)
