"""Lab 3 stages 2-3: one-shot estimate -> planned transit -> g-gated motion.

SAFETY GATE: nothing on the arm moves unless [g] is HELD DOWN. Each frame
the key is held advances the arm one small step (KEY_STEP_MM); releasing it
stops the motion immediately. [SPACE] starts (and stops) the vacuum pump
while the arm is stood down during the descent/seating phases, [q] aborts.
Two liftoffs run on their own once the vacuum is toggled: the post-pick
lift starts automatically GRIPPER_SETTLE_S after [SPACE] engages the
vacuum, and the post-seating retreat starts automatically after [SPACE]
releases it, after which the arm is disabled. Both still abort on [q].

One run of this script is the whole task: the flights before the descent
end at an offer ([Enter] to carry on to the descent, [p] to plan another
path with a fresh profile, executor and vias, [q] to finish the run), and
the one-shot estimate offers another try ([e]) before it is accepted. [q]
DURING a pre-descent flight aborts only that flight and returns to the
offer; once the vacuum has engaged, [q] ends the run (the block is
released, never dragged).
"""


import numpy as np
from scipy.spatial.transform import Rotation

import robot

# Diagnostic cross-check of the FK chain against UFactory Studio at the
# one-shot estimate. Set False once verified.
ONE_SHOT_DIAG = False

from lab_config import (BLOCK_SIDE_MM, DOWN_RPY_DEG, HOVER_ABOVE_MM,
                        KEY_SPEED_PCT, KEY_STEP_MM, LIFT_CLEAR_MM, MASK_BANDS_NAME,
                        STREAM_DESCENT_MM_S, STREAM_SPEED_MM_S, TABLE_Z_MM,
                        load_mask_bands)
from key_control import flange_for_tip, gated_path, streamed_path, tip_position
from path_recorder import PathRecorder, save_path_plots
from motion import flange_pose_base
from planners import INTERPOLATORS, path_report, plan_via_path
from reporting import banner, field, heading, progress
from ui_display import (load_camera_intrinsics, load_calibration_results,
                        load_gripper_calibration, open_camera, tune_masks,
                        wait_next)
from vision import one_shot_estimate


N_PER_SEG = 10              # planner samples per segment
TRANSIT_SPEED = 30          # % speed for planned transits
LIFT_SPEED = 20             # % speed for the post-pick lift and retreat


# ===========================================================================
# Student input
# ===========================================================================
def choose_profile():
    """Ask which interpolator to use; default linear. Returns the key.

    Linear keeps the velocity continuous through every segment end, so the
    g-gated steps stay even; cubic/quintic ease to a full stop at each via
    and crowd the waypoints there — the motion jolts under the dead-man
    executor.
    """
    names = sorted(INTERPOLATORS)
    print()
    progress("Interpolation profiles available:")
    for name in names:
        progress(name)
    while True:
        raw = input("Profile [Enter = linear]: ").strip().lower()
        if not raw:
            return 'linear'
        if raw in INTERPOLATORS:
            return raw
        progress("! pick one of: %s" % ", ".join(names))


def choose_executor():
    """Ask which path executor to run the planned poses with."""

    print()
    progress("Path executor (the dead-man gate is the same for both):")
    progress("  gated    - position mode, one %g mm step per held-[g] frame"
             % KEY_STEP_MM)
    progress("  streamed - mode 1 servo stream, smooth continuous motion")
    while True:
        raw = input("Executor [Enter = gated]: ").strip().lower()
        if not raw:
            return 'gated'
        if raw in ('gated', 'streamed'):
            return raw
        progress("! pick one of: gated, streamed")


def prompt_vias(p_target_m, tool_d_m, recorder=None):
    """Read via points from stdin, as cup-TIP waypoints in mm (base frame)."""

    print()
    progress("Via points for the transit (cup-TIP x y z in mm, base frame).")
    progress("Target tip is at (%.0f, %.0f, %.0f) mm."
             % tuple(np.asarray(p_target_m) * 1000.0))
    progress("Leave the arm head-room: route AROUND obstacles, then come")
    progress("down onto the hover point. Blank line when done.")

    R_down = Rotation.from_euler('xyz', np.deg2rad(DOWN_RPY_DEG)).as_matrix()
    vias = []
    while True:
        line = input("  via %d (x y z, blank to finish): "
                     % (len(vias) + 1)).strip()
        if not line:
            break
        parts = line.replace(',', ' ').split()
        if len(parts) != 3:
            progress("! need exactly three numbers")
            continue
        try:
            xyz = np.asarray([float(p) for p in parts]) / 1000.0
        except ValueError:
            progress("! could not parse numbers")
            continue
        vias.append(flange_for_tip(xyz, R_down, tool_d_m))
        if recorder is not None:
            recorder.add_via(xyz)
    return vias


# ===========================================================================
# Path execution
# ===========================================================================
def transit_to_hover(arm, cap, K, dist, zeta, T_6_C, tool_d_m, q_rad,
                     p_tip_target_m, profile, recorder=None):
    """Plan the transit, stopping HOVER_ABOVE_MM above the target tip.

    Planning only — execution is gated or streamed in do_transfer.
    """

    R_down = Rotation.from_euler('xyz', np.deg2rad(DOWN_RPY_DEG)).as_matrix()
    p_hover = np.asarray(p_tip_target_m, dtype=float) \
        + np.array([0.0, 0.0, HOVER_ABOVE_MM / 1000.0])
    T_hover = flange_for_tip(p_hover, R_down, tool_d_m)

    vias = prompt_vias(p_hover, tool_d_m, recorder=recorder)
    T_start = flange_pose_base(q_rad, zeta)
    poses = plan_via_path(T_start, vias, T_hover, mode=profile,
                          n_per_seg=N_PER_SEG)
    length_mm, max_step_mm = path_report(poses)
    field("transit", "profile '%s': %d via(s), %d poses, path %.0f mm,"
          " max step %.2f mm"
          % (profile, len(vias), len(poses), length_mm, max_step_mm))

    return poses, vias


# ===========================================================================
# One transfer: estimate -> transit -> g-gated descent -> actuate
# ===========================================================================
def one_shot_gate(arm, cap, K, dist, zeta, T_6_C, tool_d_m, color):
    """Wait for [Enter], take the one-shot estimate. [e] takes another
    estimate in the same run (aim the camera again first); [Enter] on the
    summary accepts it and the descent starts. Retry until seen."""

    while True:
        adv, abrt = wait_next(
            "block %s: aim the camera, [Enter] takes the one-shot estimate"
            % color, cap, arm, zeta, K, dist, T_6_C, color)
        if abrt:
            return None, True
        ret, frame = cap.read()
        q_rad = robot.read_joints_rad(arm)
        T_flange = flange_pose_base(q_rad, zeta)
        if ONE_SHOT_DIAG:
            progress("[diag] joints read (deg): %s"
                     % np.array2string(np.degrees(q_rad), precision=2,
                                       suppress_small=True))
            progress("[diag] FK flange   : x %+7.1f  y %+7.1f  z %+7.1f mm"
                     % tuple(T_flange[:3, 3] * 1000.0))
            p_tip = tip_position(T_flange, tool_d_m)
            progress("[diag] FK tool tip : x %+7.1f  y %+7.1f  z %+7.1f mm"
                     % tuple(p_tip * 1000.0))
            progress("[diag] cross-check the tip against UFactory Studio's "
                     "live readout at the same pose")
        est = one_shot_estimate(frame, color, K, dist, T_flange, T_6_C)
        if est is None:
            progress("! block not detected — check the mask windows / lighting")
            continue
        p = est['p_block_base']
        d_g = T_flange[:3, :3].T @ (p - tip_position(T_flange, tool_d_m))
        print()
        heading("ONE-SHOT ESTIMATE (block %s) — record for the report" % color)
        field("position", "x %+7.1f   y %+7.1f   z %+7.1f  mm (top-face centre)"
              % tuple(p * 1000.0))
        field("z before fit", "%+7.1f mm   (camera ray, before the silhouette fit)"
              % est['z_naive_mm'])
        field("fit loss", "%.4f" % est['fit_loss'])
        field("gripper", "lateral (%+.1f, %+.1f) mm, %+7.1f mm along the cup axis"
              % (d_g[0] * 1000.0, d_g[1] * 1000.0, d_g[2] * 1000.0))
        field("yaw", "%+.1f deg   (mod 90 deg for a square face)"
              % np.degrees(est['yaw_rad']))
        field("depth", "%.1f mm   confidence %.2f"
              % (est['depth_mm'], est['confidence']))
        field("view", "%.1f deg onto the block%s"
              % (est['tilt_deg'],
                 "   [tilt fit failed, naive depth kept]"
                 if est['naive_used'] else ""))
        field("image", "centre (%.0f, %.0f) px"
              % tuple(est['center_px']))
        print()
        while True:
            choice = input('[Enter]=descent on this estimate   '
                           '[e]=estimate again   [q]=abort: ').strip().lower()
            if choice == '':
                return est, False
            if choice == 'e':
                break
            if choice == 'q':
                return None, True
        continue


def do_transfer(arm, cap, K, dist, zeta, T_6_C, tool_d_m, color,
                dest_block, profile, executor, recorder=None):
    """Full pipeline: pick block `color`, place it on top of `dest_block`.

    executor is 'gated' or 'streamed' (choose_executor). Every pre-descent
    flight ends at an offer (see the module docstring): [Enter] descends,
    [p] re-asks the profile and executor and flies another path (vias are
    asked again either way), [q] finishes the run — one run covers Task 3's
    several estimates and flights. With recorder
    (path_recorder.PathRecorder), each executed segment's tip path is
    recorded and the transit vias are attached to their segment for the
    data/trajectories/ plots.
    """

    def execute(q_in, poses, label, speed, allow_gripper=False,
                gripping=False, p_tip_target_m=None, color=None,
                auto=False):
        """One gated phase, run by the chosen executor.

        streamed: descent/seating slow to STREAM_DESCENT_MM_S so the cup
        approach stays gentle; everything else streams at
        STREAM_SPEED_MM_S. gated keeps the caller's % speed. auto=True
        passes through: the phase runs un-gated (post-vacuum liftoff).
        """
        if executor == 'streamed':
            s_speed = (STREAM_DESCENT_MM_S
                       if label in ('descent', 'seating')
                       else STREAM_SPEED_MM_S)
            return streamed_path(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                 q_in, poses, label, speed=s_speed,
                                 allow_gripper=allow_gripper,
                                 gripping=gripping,
                                 p_tip_target_m=p_tip_target_m, color=color,
                                 recorder=recorder, auto=auto)
        return gated_path(arm, cap, K, dist, zeta, T_6_C, tool_d_m, q_in,
                          poses, label, speed=speed,
                          allow_gripper=allow_gripper, gripping=gripping,
                          p_tip_target_m=p_tip_target_m, color=color,
                          recorder=recorder, auto=auto)

    est, abrt = one_shot_gate(arm, cap, K, dist, zeta, T_6_C, tool_d_m, color)
    if abrt:
        return None, True
    p_target = est['p_block_base']

    # Flight loop: Task 3 asks for several estimates and flights inside ONE
    # run, so every pre-descent transit ends at an offer — [Enter] carries
    # on to the descent, [p] re-asks the profile and executor and flies
    # another path, [q] finishes the run. [q] DURING a flight aborts only
    # that flight and comes back to the offer.
    while True:
        # The arm was jogged in Manual Mode to aim the camera after the
        # script read its joints, so the transit MUST be planned from a
        # fresh read: a stale start pose sends the first planned poses back
        # toward the old pose — the arm moves away from the block before
        # arcing to the hover. T_pre is the pose before the descent: the
        # post-pick lift returns here.
        q_rad = robot.read_joints_rad(arm)
        T_pre = flange_pose_base(q_rad, zeta)

        field("transit", "hover %d mm above the estimate tip, profile '%s'"
              % (HOVER_ABOVE_MM, profile))
        poses, _ = transit_to_hover(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                    q_rad, p_target, profile,
                                    recorder=recorder)
        q_rad, abrt, _ = execute(q_rad, poses, 'transit', TRANSIT_SPEED)
        # abrt is deliberately not acted on here: [q] mid-flight aborts
        # only this flight, and the offer below decides what happens next.

        choice = input('Transit flown. [Enter]=descend onto the block   '
                       '[p]=plan another path   [q]=finish run: ').strip().lower()
        if choice == '':
            break
        if choice == 'p':
            profile = choose_profile()
            executor = choose_executor()
            continue
        return None, True        # [q] at the offer finishes the run

    # Descent onto the block, still under the [g] dead-man switch: the cup
    # tip stops at the estimated top-face centre. [SPACE] engages the
    # vacuum once the cup rests on the block.
    R_flange = flange_pose_base(q_rad, zeta)[:3, :3]
    T_touch = flange_for_tip(p_target, R_flange, tool_d_m)
    T_touch[2, 3] = max(T_touch[2, 3], (TABLE_Z_MM + 2.0) / 1000.0)
    print()
    heading("DESCENT — hold [g] until the cup rests on the block, then [SPACE]")
    q_rad, abrt, gripping = execute(q_rad, [T_touch], 'descent',
                                    KEY_SPEED_PCT, allow_gripper=True,
                                    gripping=False, p_tip_target_m=p_target,
                                    color=color)
    if abrt or not gripping:
        robot.close_gripper(arm)      # better a drop than a drag
        if not gripping:
            progress("! vacuum never engaged — aborting this transfer")
        return None, True

    # Lift back to the descent's start pose, on its own (auto=True): the
    # settle wait already ran at [SPACE]; the dead-man gate resumes below.
    q_rad, abrt, _ = execute(q_rad, [T_pre], 'lift', LIFT_SPEED,
                             allow_gripper=False, gripping=gripping,
                             auto=True)
    if abrt:
        robot.close_gripper(arm)
        return None, True

    # ---- place: destination is the top face of the destination block,
    # estimated (one-shot) from wherever the arm is jogged to aim ----
    est_d, abrt = one_shot_gate(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                dest_block)
    if abrt:
        robot.close_gripper(arm)
        return None, True
    # The cup tip rests on the placed block's top face, one block height
    # above the destination block's top-face centre.
    p_dest_tip_m = (est_d['p_block_base']
                    + np.array([0.0, 0.0, BLOCK_SIDE_MM / 1000.0]))
    field("destination", "on top of block %s — tip (%.0f, %.0f, %.0f) mm"
          % (dest_block, p_dest_tip_m[0] * 1000.0,
             p_dest_tip_m[1] * 1000.0, p_dest_tip_m[2] * 1000.0))

    # The arm may have been jogged to aim the camera at the destination
    # block: plan the transit from a fresh read, as after the pick estimate.
    q_rad = robot.read_joints_rad(arm)

    field("transit", "to the destination, again stopping %d mm above"
          % HOVER_ABOVE_MM)
    poses, _ = transit_to_hover(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                q_rad, p_dest_tip_m, profile,
                                recorder=recorder)
    q_rad, abrt, _ = execute(q_rad, poses, 'transit 2', TRANSIT_SPEED,
                             allow_gripper=False, gripping=gripping)
    if abrt:
        robot.close_gripper(arm)
        return None, True

    # Seating descent, still gated: [SPACE] releases the vacuum once the
    # block rests on what is below it.
    R_flange = flange_pose_base(q_rad, zeta)[:3, :3]
    T_seat = flange_for_tip(p_dest_tip_m, R_flange, tool_d_m)
    T_seat[2, 3] = max(T_seat[2, 3], (TABLE_Z_MM + 2.0) / 1000.0)
    print()
    heading("SEATING — hold [g] until the block rests on what is below it,"
            " then [SPACE]")
    q_rad, abrt, gripping = execute(q_rad, [T_seat], 'seating',
                                    KEY_SPEED_PCT, allow_gripper=True,
                                    gripping=gripping,
                                    p_tip_target_m=p_dest_tip_m, color=color)
    if abrt:
        robot.close_gripper(arm)
        return None, True
    # Defensive second release: guarantees the block is not dragged along
    # by the retreat (no-op if [SPACE] already released the vacuum).
    if robot.close_gripper(arm) != 1:
        progress("! vacuum failed to release")
    gripping = False

    seat_z_mm = float(tip_position(flange_pose_base(q_rad, zeta),
                                   tool_d_m)[2] * 1000.0)
    field("seat height", "%.1f mm (record this: the stack builds on it)"
          % seat_z_mm)

    # Retreat so the next transit starts clear of the stack (auto=True,
    # un-gated); the arm is disabled once clear.
    T_flange = flange_pose_base(q_rad, zeta)
    T_up = T_flange.copy()
    T_up[2, 3] += LIFT_CLEAR_MM / 1000.0
    q_rad, _, _ = execute(q_rad, [T_up], 'retreat', LIFT_SPEED,
                          allow_gripper=False, gripping=gripping, auto=True)
    robot.disable_arm(arm)
    field("robot", "disabled (controller stopped) — re-enable in UFactory "
          "Studio if needed")
    return {'seat_z_mm': seat_z_mm}, False


# ===========================================================================
# Orchestrator
# ===========================================================================
def run():
    K, dist = load_camera_intrinsics()
    zeta, T_6_C = load_calibration_results()
    tool_d_m, tool_src = load_gripper_calibration()
    field("tool offset", "%.2f mm  [%s]" % (tool_d_m * 1000.0, tool_src))

    arm = robot.connect_arm()
    cap = open_camera()
    recorder = PathRecorder()   # executed tip paths -> data/trajectories/
    try:
        bands = load_mask_bands()
        if bands == 'saved':
            progress('colour bands loaded from data/%s (accepted in a '
                     'previous run); tune_masks [r] still resets to defaults'
                     % MASK_BANDS_NAME)
        if not tune_masks(cap, K, dist):
            progress("! mask tuning aborted — quitting.")
            return
        profile = choose_profile()
        executor = choose_executor()

        banner("TRANSFER — block 1 picked, placed on top of block 2")
        state, abrt = do_transfer(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                  '1', '2', profile, executor,
                                  recorder=recorder)
        if abrt:
            # [q] at an offer, or an abort mid-transfer: the estimates and
            # path reports printed so far are still the pair's to record.
            print()
            progress("Run ended. The one-shot estimates and path reports")
            progress("printed so far are still yours to record.")
            return

        print()
        progress("Transfer complete. Record both one-shot estimates, the")
        progress("seat height and the path reports for the handout.")
    finally:
        save_path_plots(recorder)
        robot.stop_gripper(arm)
        robot.disable_arm(arm)
        cap.release()
        arm.disconnect()


if __name__ == '__main__':
    run()
