# Lab 3 — Repository Structure

Lab 3 is **vision-guided pick and place** in Python: the wrist camera locates a red and a
green 25 mm block, the Lite 6 navigates its vacuum gripper over the green block using the
Lab 1/2 calibration, picks it up under a keygate, and places it on the red block. Every
autonomous motion is small and keygated; a human stays in the safety loop throughout.

The module layout and the reporting grammar match Labs 1 and 2: `units.py`, `so3.py` and
`reporting.py` are copies of the Lab 2 files, and `ik_lite6.py` / `fk_lite6.py` are the
Lab 2 solvers ported to this lab's calibrated parameters. If you did Lab 2, every file
here should look like a small delta from code you have already read.

## Frames and units

| Symbol | Meaning |
|--------|---------|
| `0`    | Robot base frame |
| `6`    | End-effector (flange) frame |
| `G`    | Vacuum cup tip frame (built by the Stage 1 gripper calibration) |
| `C`    | Wrist camera frame |
| `W`    | ArUco artifact frame (calibration captures only) |

- `T_0_6` from forward kinematics on joint angles; `T_6_C` is the hand-eye transform from
  Lab 1/2 (`calibration_results.npy`, fitted jointly with the calibrated DH `zeta_stage2`);
  `T_6_G` (the tool offset along the flange +z) from `gripper_calibration.npy`.
- Internal maths runs in **metres + radians**. The xArm SDK speaks **millimetres +
  degrees**. The conversion happens in exactly two places: `units.py` (pose 4x4 to/from
  the SDK's mm + rpy forms) and `robot.py` at the hardware boundary. `fk_from_zeta` takes
  `zeta` lengths in mm and divides by 1000.

## Top-level layout

```
Lab_3/
├── src/          # Active Python code. One flat directory of single-responsibility modules.
├── data/         # Calibration artifacts (.npy/.json) produced by Labs 1/2 + this lab's runs.
├── docs/         # This structure guide + lab instructions.
└── commit-log.md # Human-readable version history
```

`data/` is produced by earlier labs and by this lab's own scripts; paths are resolved
relative to the repo via `data_path()` in `lab_config.py`, never hardcoded absolute paths.

## Module roles

The orchestrator imports downward only — no cycles: `pick_and_place.py` imports from
`motion`, `vision`, `planners`, `key_control` and `ui_display`; those import the leaf
modules. Any leaf module can be read in isolation.

| Module | Owns |
|--------|------|
| `lab_config.py` | Every tunable in one place: block size, tool offset, hover height, thresholds (`_RGB_DEFAULTS`, live-mutable), hardware (`ROBOT_IP`, `CAMERA_INDEX` — the only two values readable from environment variables), every `data/` filename, and `data_path()`. Imports nothing local. |
| `reporting.py` | The one output grammar shared by every script: `banner`, `heading`, `field`, `progress`. Same as Lab 2. |
| `units.py` | The single SDK↔maths crossing: `transform_from_pose_mm_rad`, `transform_from_pose_mm_deg`, `euler_deg_from_matrix`. |
| `fk_lite6.py` | Modified (Craig) D-H forward kinematics for the Lite 6: `fk_from_zeta(q, zeta)` (zeta packed `[a\|alpha\|d\|theta_off]`, lengths in mm, angles in rad) and `pose_from_xyz_rpy` for cross-checking the controller's own pose readout. |
| `ik_lite6.py` | Numerical inverse kinematics: damped Levenberg–Marquardt on the 6-row pose residual (position in m, orientation in rad) through an SVD, every step projected into the joint box. Runs on the **calibrated** zeta from `calibration_results.npy` (nominal only as a fallback). Convergence is flagged on the POSE residual (`RESIDUAL_TOL`), never on step size alone — the Lab 1 lesson. When the primary bounded run stalls along a joint-limit face, `solve_ik` restarts from a handful of standard postures and the first converged run wins. CLI: `python ik_lite6.py X Y Z R P Y [q1..q6]` solves one pose (mm, degrees). Runnable directly for the residual self-check. |
| `so3.py` | Rotation log/exp maps on SO(3), with the ~π branch handled through `eigh` — the orientation error of the IK residual and of the pose interpolators. Same as Lab 2. |
| `robot.py` | The only module that touches the xArm SDK: connect, read joints (returned in rad), servo moves, joint-limit check, and the vacuum gripper wrappers. `open_gripper` engages suction, `close_gripper` releases, `stop_gripper` is a best-effort stop. |
| `motion.py` | Motion primitives: `flange_pose_base` (FK at the current joints) and `move_to_pose` (solve_ik + servo through the SDK). |
| `vision.py` | Colour detection → base-frame geometry: RGB masking, best-blob selection, monocular depth from the block's known 25 mm size, and `fit_block_pose` — a 4-parameter (x, y, z, yaw) silhouette fit that measures the block's full pose from the image with no plane/height assumption. |
| `planners.py` | Via-point interpolation between Cartesian poses (linear / cubic / quintic), with orientation interpolated on SO(3). Runnable directly for a self-test that compares the profiles. |
| `ui_display.py` | Camera, OpenCV windows, calibration/data loaders, the RGB threshold trackbars, and the keygated display/wait loops. No geometry, no planning. |
| `aruco_artifact_detector.py` | One camera frame → `T_C_W` by monocular PnP on the Lab 1/2 ArUco artifact (used during calibration capture). |
| `handeye_gripper.py` | Stage 1 gripper calibration: one camera look at the artifact (`CAPTURE 1`) plus several touches on its top face (`CAPTURE 2`) measure the flange→cup distance `d`, from which the cup-tip frame G is built (`T_6_G`, `T_G_C = inv(T_6_G) @ T_6_C`). Writes `gripper_calibration.npy`. |
| `key_control.py` | Key-gated final approach: the last millimetres from the hover point down to contact are driven by hand (`keygate_approach`), plus the cup-tip ↔ flange geometry helpers `flange_for_tip` / `tip_position`. |
| `pick_and_place.py` | **Orchestrator only** — the keygated phase sequence (tune masks → choose profile → one-shot estimate → planned transit → keygated descent → vacuum pick → place on the red block) and top-level `run()`. Owns no geometry or hardware detail. |

## data/ contents

| File | Meaning |
|------|---------|
| `camera_matrix.npy` / `dist_coeffs.npy` | Camera intrinsics and distortion (Lab 1 chessboard calibration). K is valid at 1920×1080; the camera is pinned to that resolution. |
| `calibration_results.npy` | Calibrated DH `zeta_stage2` + hand-eye `T_6_C` fitted WITH it (Lab 2). The pair must go into FK/IK together. |
| `gripper_calibration.npy` | Stage-1 tool offset (flange→cup distance) from `handeye_gripper.py`. |
| `aruco_config.json` | ArUco artifact definition for `aruco_artifact_detector.py`. |
| `capture_log.json` | Log of the stored calibration captures. |

## How to run

Create and activate a virtual environment at the `I2R_lab_development` root, then install
the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy opencv-python xarm-python-sdk pillow
```

Open `src/lab_config.py` and check the hardware parameters; `ROBOT_IP` and `CAMERA_INDEX`
can also be exported as environment variables on the bench (they have safe defaults).

Entry points (run from `Lab_3/src`):

```bash
python pick_and_place.py        # the full keygated pick-and-place (robot + camera)
python handeye_gripper.py       # Stage 1 tool-offset calibration (robot + camera)
python ik_lite6.py              # IK residual self-check (no hardware)
python ik_lite6.py 150 0 300 0 180 0     # solve one pose (mm, degrees), from rest
python ik_lite6.py 150 0 300 0 180 0 0 -30 60 0 -60 0   # ...from a joint guess, inside the box
python planners.py              # interpolator comparison self-test (no hardware)
```

Only `pick_and_place.py` and `handeye_gripper.py` need the arm and camera connected; all
other modules import without hardware.

Requires: Python 3.12+, NumPy 2.4.4, SciPy 1.17.1, OpenCV 4.13.0.92, xArm Python SDK 1.17.3,
Pillow 12.2.0, and a reachable UFACTORY Lite 6 with wrist camera and vacuum gripper.
