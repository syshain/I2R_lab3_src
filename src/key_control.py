"""The dead-man (g-gated) path executors used by pick_and_place.py.

The arm advances only while [g] is held and stops the moment it is
released; both executors pre-solve the whole path's IK before anything
moves. Tip<->flange helpers live here too.
"""

import threading
import time

import numpy as np

from ik_lite6 import rotation_log_map, solve_ik
from lab_config import (GRIPPER_SETTLE_S, IK_JUMP_DEG, KEY_SPEED_PCT,
                        KEY_STEP_MM, STREAM_DRAIN_S, STREAM_SETTLE_TIMEOUT_S,
                        STREAM_TICK_S, STREAM_SPEED_MM_S)
from motion import flange_pose_base
from planners import interpolate_pose
from vision import compute_detection, project_base_to_uv
from ui_display import show_frame
from reporting import progress
import robot

HOLD_KEY = ord('g')
HOLD_LEGEND = '[hold g]=move (release to stop) [SPACE]=gripper on/off [q]=abort'
HOLD_LEGEND_NO_GRIPPER = '[hold g]=move (release to stop) [q]=abort'

def flange_for_tip(p_tip_base, R_flange, tool_d_m):
    """Flange pose (4x4) that puts the cup tip at p_tip_base (m)."""

    T = np.eye(4)
    T[:3, :3] = R_flange
    T[:3, 3] = (np.asarray(p_tip_base, dtype=float)
                - R_flange @ np.array([0.0, 0.0, tool_d_m]))
    return T

def tip_position(T_flange, tool_d_m):
    """Current cup-tip position in the base frame (m), from the flange pose."""
    raise NotImplementedError(
        "Task 1, step 6.1(h): return where the cup tip is, given the flange "
        "pose and d. You have just measured d; this is where it is used. "
        "One line. flange_for_tip() above does the reverse and is given.")

def _fine_plan(poses, q_rad, zeta, label, step_mm):
    """Subdivide planned poses into ~step_mm steps and pre-solve their IK.

    The whole path is tracked before anything moves, so an unfollowable plan
    aborts with the arm untouched. IK failure, or a joint jump beyond
    IK_JUMP_DEG between consecutive steps (a wrist flip), prints why and
    returns (None, None); otherwise (fine_poses, plan), plan[i] for
    fine_poses[i].
    """

    fine = []
    prev_T = flange_pose_base(q_rad, zeta)
    for T in poses:
        d_mm = np.linalg.norm(T[:3, 3] - prev_T[:3, 3]) * 1000.0
        n = max(1, int(np.ceil(d_mm / step_mm)))
        for s in np.linspace(1.0 / n, 1.0, n):
            fine.append(interpolate_pose(prev_T, T, s))
        prev_T = T

    q_chain = np.asarray(q_rad, dtype=float).copy()
    plan = []
    for i, T in enumerate(fine):
        q_sol, ok, err = solve_ik(T, q_chain, zeta)
        if not ok:
            T_end = flange_pose_base(q_sol, zeta)
            progress("! IK failed at %s step %d/%d — best it reached was"
                     " %.1f mm, %.2f deg off the pose (resid %.2e) — nothing"
                     " moved; %d fine pose(s) rejected by IK"
                     % (label, i + 1, len(fine),
                        np.linalg.norm(T_end[:3, 3] - T[:3, 3]) * 1000.0,
                        np.degrees(np.linalg.norm(
                            rotation_log_map(T_end[:3, :3] @ T[:3, :3].T))),
                        err, len(fine) - i))
            return None, None
        # Joint angles wrap at +/-180 deg: compare modulo 2*pi so an
        # equivalent solution on the other side of the wrap is not read
        # as a flip.
        jump_deg = float(np.max(np.abs(
            np.degrees((q_sol - q_chain + np.pi) % (2.0 * np.pi) - np.pi))))
        if jump_deg > IK_JUMP_DEG:
            progress("! joint jump %.1f deg at %s step %d/%d (wrist flip)"
                     " — move rejected, nothing moved; %d fine pose(s)"
                     " rejected"
                     % (jump_deg, label, i + 1, len(fine), len(fine) - i))
            return None, None
        plan.append(q_sol)
        q_chain = q_sol
    return fine, plan

def gated_path(arm, cap, K, dist_coeffs, zeta, T_ee_cam, tool_d_m, q_rad,
               poses, label, speed=KEY_SPEED_PCT, allow_gripper=False,
               gripping=False, p_tip_target_m=None, color=None,
               recorder=None, auto=False):
    """G-gated path execution: the arm moves ONLY while [g] is held down.

    Each held frame advances one KEY_STEP_MM step (IK pre-solved); releasing
    [g] stops the motion at once. With allow_gripper, [SPACE] toggles the
    vacuum and a successful toggle ends the phase so the automatic
    liftoff/retreat runs. auto=True runs un-gated. Returns (q_rad, aborted,
    gripping).
    """

    q_rad = np.asarray(q_rad, dtype=float).copy()
    if recorder is not None:
        recorder.start_segment(label)
    fine, plan = _fine_plan(poses, q_rad, zeta, label, KEY_STEP_MM)
    if plan is None:
        return q_rad, True, gripping

    aborted = done = False
    i = 0
    # Like the streamed executor, the phase stays open after the last
    # step when [SPACE] is live, so the toggle is never raced by the end
    # of the plan.
    while not aborted and not done and (i < len(plan) or allow_gripper):
        T_flange = flange_pose_base(q_rad, zeta)
        p_tip = tip_position(T_flange, tool_d_m)
        T_cam = T_flange @ T_ee_cam
        lines, marks = [], []
        det = None
        ok_f, frame = cap.read()
        if ok_f:
            if color is not None:
                det = compute_detection(frame, color, K, dist_coeffs,
                                        T_flange, T_ee_cam)
                if det is not None:
                    d_g = T_flange[:3, :3].T @ (det['p_block_base'] - p_tip)
                    lines.append('tip->block: lateral (%+.1f, %+.1f) mm, '
                                 '%+.1f mm along cup axis'
                                 % (d_g[0] * 1000.0, d_g[1] * 1000.0,
                                    d_g[2] * 1000.0))
            if p_tip_target_m is not None:
                for p_mark, lbl, bgr in ((p_tip, 'tip', (255, 0, 255)),
                                         (p_tip_target_m, 'target',
                                          (0, 255, 255))):
                    uv = project_base_to_uv(p_mark, T_cam, K, dist_coeffs)
                    if uv is not None:
                        marks.append((uv[0], uv[1], lbl, bgr))
        if auto:
            title = '%s: auto liftoff — %d/%d — [q] to abort' % (label, i,
                                                                 len(plan))
            hint = '[q]=abort'
        else:
            title = '%s: %d/%d — hold [g] to move' % (label, i, len(plan))
            hint = HOLD_LEGEND if allow_gripper else HOLD_LEGEND_NO_GRIPPER
        key = show_frame(
            cap, title,
            info_lines=lines, det=det, color=color, hint=hint,
            extra_marks=marks, K=K, dist_coeffs=dist_coeffs)
        if key == ord('q'):
            aborted = True
        elif allow_gripper and key == 32:
            # A successful vacuum toggle ends the phase so the automatic
            # liftoff/retreat can run; a failed toggle keeps it open for retry.
            if gripping:
                if robot.close_gripper(arm) == 1:
                    gripping = False
                    progress('vacuum released.')
                    done = True
                else:
                    progress('! vacuum failed to release')
            else:
                if robot.open_gripper(arm) == 1:
                    gripping = True
                    progress('vacuum engaged — settling %.0f s before motion.'
                             % GRIPPER_SETTLE_S)
                    time.sleep(GRIPPER_SETTLE_S)
                    done = True
                else:
                    progress('! vacuum failed to engage')
        elif auto or key == HOLD_KEY:      # auto phase, or held down -> step
            ok_mv, msg = robot.move_to_joints(arm, plan[i], speed=speed,
                                              wait=True)
            if ok_mv:
                try:
                    q_rad = robot.read_joints_rad(arm)
                except Exception:
                    q_rad = plan[i]
                i += 1
                if recorder is not None:
                    recorder.add_tip(tip_position(
                        flange_pose_base(q_rad, zeta), tool_d_m))
            else:
                progress('! move rejected: %s' % msg)
    return q_rad, aborted, gripping

def streamed_path(arm, cap, K, dist_coeffs, zeta, T_ee_cam, tool_d_m, q_rad,
                  poses, label, speed=STREAM_SPEED_MM_S, allow_gripper=False,
                  gripping=False, p_tip_target_m=None, color=None,
                  recorder=None, auto=False):
    """Mode-1 servo-streaming executor: same dead-man gate, smooth motion.

    Like gated_path but in servo mode: pose ticks stream while [g] is held
    and the controller blends between them, jolt-free; releasing [g] pauses
    the stream (halt within one tick). [SPACE] is handled while paused, so
    only one thread talks to the arm; a successful toggle ends the phase as
    in gated_path. Mode 0 restored on exit; auto=True streams un-gated.
    Returns (q_rad, aborted, gripping).
    """

    q_rad = np.asarray(q_rad, dtype=float).copy()
    if recorder is not None:
        recorder.start_segment(label)
    fine, plan = _fine_plan(poses, q_rad, zeta, label,
                            speed * STREAM_TICK_S)   # mm/s * s = mm per tick
    if plan is None:
        return q_rad, True, gripping

    if robot.set_control_mode(arm, robot.SERVO_MOTION_MODE) != 0:
        progress('! could not enter servo motion mode (mode 1) — arm untouched')
        return q_rad, True, gripping
    progress('%s: streaming %d ticks at %.0f mm/s (mode 1)%s'
             % (label, len(plan), speed,
                ' — auto liftoff, [q] to abort' if auto
                else ' — hold [g] to stream'))

    arm_lock = threading.Lock()
    state = {'i': 0, 'held': auto, 'run': True, 'error': None, 'done': False,
             'drained': False}

    def _mvpose(T):
        """Wire format for set_servo_cartesian_aa: [x y z] mm + axis-angle rad."""
        return [T[0, 3] * 1000.0, T[1, 3] * 1000.0, T[2, 3] * 1000.0] \
            + list(rotation_log_map(T[:3, :3]))

    # Prime the stream: send the pose the CONTROLLER is actually at before
    # the first planned tick. Servo commands have no speed ramp (the SDK's
    # speed argument is reserved; the controller dashes to each new target
    # at full speed), so without this the first tick is a max-speed dash
    # from wherever the controller sits relative to our forward
    # kinematics — the jolt at the start of every streamed phase.
    try:
        prime = robot.read_flange_pose_aa(arm)
    except Exception as exc:
        progress('! could not read the flange pose to prime the stream (%s)'
                 ' — leaving servo mode' % exc)
        robot.set_control_mode(arm, robot.POSITION_MODE)
        return q_rad, True, gripping
    if robot.stream_flange_pose(arm, prime, speed=speed) != 0:
        progress('! priming tick rejected — leaving servo mode')
        robot.set_control_mode(arm, robot.POSITION_MODE)
        return q_rad, True, gripping

    def _stream():
        next_t = time.perf_counter()
        drain_deadline = None
        while state['run']:
            next_t += STREAM_TICK_S
            if state['held'] and state['i'] < len(plan):
                with arm_lock:
                    if not state['run']:
                        break
                    code = robot.stream_flange_pose(
                        arm, _mvpose(fine[state['i']]), speed=speed)
                if code == 0:
                    if recorder is not None:
                        recorder.add_tip(tip_position(fine[state['i']],
                                                      tool_d_m))
                    state['i'] += 1
                else:
                    state['error'] = ('stream rejected at tick %d/%d '
                                      '(SDK code %d)'
                                      % (state['i'] + 1, len(plan), code))
                    state['run'] = False
                    break
            elif state['i'] >= len(plan):
                # Plan exhausted. The controller does NOT queue servo
                # commands: it chases only the last pose received and
                # halts the moment the stream stops, so going silent here
                # would strand the arm short of the final pose. Keep
                # re-sending the final pose for STREAM_DRAIN_S so the
                # controller actually arrives. The drain runs with [g]
                # released too — the motion was already fully commanded,
                # and the dead-man gate still kills any NEW motion
                # instantly — but never past the deadline.
                if drain_deadline is None:
                    drain_deadline = time.perf_counter() + STREAM_DRAIN_S
                with arm_lock:
                    if not state['run']:
                        break
                    code = robot.stream_flange_pose(
                        arm, _mvpose(fine[-1]), speed=speed)
                if code != 0:
                    state['error'] = ('drain tick rejected at %d/%d '
                                      '(SDK code %d)'
                                      % (state['i'], len(plan), code))
                    state['run'] = False
                    break
                if time.perf_counter() >= drain_deadline:
                    state['drained'] = True
                    break
            delay = next_t - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter() + STREAM_TICK_S

    thread = threading.Thread(target=_stream, name='servo-stream', daemon=True)
    thread.start()

    aborted = False
    # The phase stays open after the last tick when [SPACE] is live: the
    # cup reaches the block on the final tick, and ending the loop on
    # plan completion would race the [SPACE] keypress and lose it. It
    # also stays open until the drain has finished re-sending the final
    # pose (state['drained']), so the controller is actually at the
    # final pose before mode 0 is restored.
    while state['run'] and not aborted and not state['done'] \
            and (state['i'] < len(plan) or not state['drained']
                 or allow_gripper):
        # Refresh from the controller every frame: the streamed plan runs
        # ahead of q_rad otherwise, and the tip marker and the tip->block
        # numbers would sit at the phase's starting pose for the whole
        # flight. Under arm_lock so the read never races the stream thread.
        with arm_lock:
            try:
                q_rad = robot.read_joints_rad(arm)
            except Exception:
                pass
        T_flange = flange_pose_base(q_rad, zeta)
        p_tip = tip_position(T_flange, tool_d_m)
        T_cam = T_flange @ T_ee_cam
        lines, marks = [], []
        det = None
        ok_f, frame = cap.read()
        if ok_f:
            if color is not None:
                det = compute_detection(frame, color, K, dist_coeffs,
                                        T_flange, T_ee_cam)
                if det is not None:
                    d_g = T_flange[:3, :3].T @ (det['p_block_base'] - p_tip)
                    lines.append('tip->block: lateral (%+.1f, %+.1f) mm, '
                                 '%+.1f mm along cup axis'
                                 % (d_g[0] * 1000.0, d_g[1] * 1000.0,
                                    d_g[2] * 1000.0))
            if p_tip_target_m is not None:
                for p_mark, lbl, bgr in ((p_tip, 'tip', (255, 0, 255)),
                                         (p_tip_target_m, 'target',
                                          (0, 255, 255))):
                    uv = project_base_to_uv(p_mark, T_cam, K, dist_coeffs)
                    if uv is not None:
                        marks.append((uv[0], uv[1], lbl, bgr))
        if auto:
            title = '%s: auto liftoff — %d/%d — [q] to abort' % (
                label, state['i'], len(plan))
            hint = '[q]=abort'
        else:
            title = '%s: %d/%d — hold [g] to stream' % (label, state['i'],
                                                        len(plan))
            hint = HOLD_LEGEND if allow_gripper else HOLD_LEGEND_NO_GRIPPER
        key = show_frame(
            cap, title,
            info_lines=lines, det=det, color=color, hint=hint,
            extra_marks=marks, K=K, dist_coeffs=dist_coeffs)
        if key == ord('q'):
            aborted = True
        elif allow_gripper and key == 32:
            state['held'] = False               # pause before touching the arm
            with arm_lock:
                if gripping:
                    if robot.close_gripper(arm) == 1:
                        gripping = False
                        progress('vacuum released.')
                        state['done'] = True    # toggle ends the phase
                    else:
                        progress('! vacuum failed to release')
                else:
                    if robot.open_gripper(arm) == 1:
                        gripping = True
                        progress('vacuum engaged — settling %.0f s before'
                                 ' motion.' % GRIPPER_SETTLE_S)
                        time.sleep(GRIPPER_SETTLE_S)
                        state['done'] = True    # toggle ends the phase
                    else:
                        progress('! vacuum failed to engage')
        else:
            state['held'] = auto or (key == HOLD_KEY)   # auto -> always stream

    state['held'] = False
    state['run'] = False
    thread.join()
    # Let the last ticks reach the controller, then wait for the arm to
    # actually STOP before the joints are read: the controller keeps
    # blending after the final tick, so joints read mid-motion would plan
    # the next phase from a pose the arm has already left (the jolt and
    # the offset). Bounded by STREAM_SETTLE_TIMEOUT_S so a controller that
    # never reports idle cannot hang the phase. Same moving test as
    # connect_arm: is_moving, or controller state 1 (in motion).
    time.sleep(0.2)
    settle_deadline = time.perf_counter() + STREAM_SETTLE_TIMEOUT_S
    while time.perf_counter() < settle_deadline:
        try:
            stopped = not arm.get_is_moving() and arm.get_state() != 1
        except Exception:
            stopped = True
        if stopped:
            break
        time.sleep(0.05)
    if state['error']:
        progress('! %s' % state['error'])
        try:
            progress('controller: state %s, mode %s, error %s, warn %s'
                     % (arm.state, arm.mode, arm.error_code,
                        arm.warn_code))
        except Exception:
            progress('controller status unreadable')
        aborted = True
    try:
        q_rad = robot.read_joints_rad(arm)
    except Exception:
        pass
    robot.set_control_mode(arm, robot.POSITION_MODE)
    return q_rad, aborted, gripping


