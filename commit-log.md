# v1 - First commit
21 Aug 2026
Working lab 3 code

# v2 - Streamed mode fixes and robot serial number recording
- Streamed mode fixes added to reduce jolting at start and end
- All robot-interfacing programs now record the serial number for hardware uniqueness

# v2.1 - Skeleton updates
- Skeletons changed. 
- new skeletons live in 
-- vision.py
-- handeye_gripper.py
-- key_control.py

# v3 - Streamed fix re-sync, NotImplementedError stubs, loader guards
## 12 Nov 2026
- key_control.py re-synced to Lab_3/src: priming tick, drain, settle poll and live q_rad refresh restored after the 11 Nov revert; streamed executors code-identical 
- <TO_FILL> in planners._profile_quintic replaced with a NotImplementedError stub; key_control imports again
- Camera/calibration loads guarded with a "copy your Lab 2 files into data/" message (ui_display, pnp_probe)

# v3.1 - Planners stub removed
## 12 Nov 2026
- Removed quintic stub from planners.py

# v4 - Wrist flip guard, cup tip protection and STRUCTURE.md update
## 12 Nov 2026
- Wrist-flip guard: _fine_plan rejects the whole move when any joint jumps more than IK_JUMP_DEG (10 deg) between consecutive fine steps
- Descent/seating z-floor now protects the cup tip, not the flange
- Seating readout measures against the destination block (dest_block), not the carried block
- PathRecorder buffers vias prompted before their segment opens, so they reach the saved plot
- CAMERA_INDEX default 4 to 0 (JOINT_LIMITS_DEG already defaults to the bench limits 80..270)
- docs/STRUCTURE.md synced with the reference copy's updated guide

# v4.1 - Minor fixes
## 13 Nov 2026
- PathRecorder: vias always held until the next transit starts, so later flights label their own vias (not "lift"/"retreat")
- Wrist-flip guard: first fine step compared against the arm's current joints; jump measured modulo 2pi so the +/-180 deg wrap is not read as a flip
- handeye_gripper: touch spread uses sample std (ddof=1), matching the reference copy
