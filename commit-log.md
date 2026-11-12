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
