"""The only module that touches the xArm SDK: connect, joints, servo moves,
and the vacuum gripper wrappers.

Vacuum gripper semantics, since the names read backwards: `open_gripper`
ENGAGES suction (open = suck the block onto the cup), `close_gripper`
RELEASES it, and `stop_gripper` is a best-effort pump stop.
"""

import time

import numpy as np
from xarm.wrapper import XArmAPI

from lab_config import JOINT_LIMITS_DEG, ROBOT_IP, STREAM_SPEED_MM_S
from reporting import progress

SERVO_SPEED = 30
POSITION_MODE = 0
SERVO_MOTION_MODE = 1


def connect_arm(ip=ROBOT_IP):
    """Connect + enable motion; returns an XArmAPI. No reset() so we don't wipe."""

    arm = XArmAPI(ip, do_not_open=True)
    arm.connect()
    if not arm.connected:
        raise RuntimeError('could not connect to %s' % ip)
    time.sleep(0.5)
    if arm.get_is_moving() or arm.get_state() == 1:
        progress('warning: arm is in motion at connect - holding')
        arm.emergency_stop(30)
        time.sleep(1.0)
    # Clear the stopped state (4) a previous run leaves behind: without
    # this, every motion command of the NEXT run is refused.
    arm.set_state(0)
    return arm


def read_joints_rad(arm):
    """Current joint angles as a (6,) array in radians."""
    code, angles_deg = arm.get_servo_angle()
    if code != 0:
        raise RuntimeError('joint read failed (code %d)' % code)
    return np.deg2rad(np.asarray(angles_deg[:6], dtype=float))


def within_joint_limits(q_deg):
    """True if a (6,) degree vector is inside the Lite 6 joint limits."""
    q_deg = np.asarray(q_deg, dtype=float)
    return bool(np.all(q_deg >= JOINT_LIMITS_DEG[:, 0])
                and np.all(q_deg <= JOINT_LIMITS_DEG[:, 1]))


def set_control_mode(arm, mode):
    """Switch the control mode (0 position, 1 servo motion) and leave the
    controller ready. The arm must be stationary. Returns the SDK code."""

    code = arm.set_mode(mode)
    if code != 0:
        return code
    return arm.set_state(0)


def stream_flange_pose(arm, mvpose, speed=STREAM_SPEED_MM_S):
    """Send one servo-cartesian pose in servo motion mode (mode 1).

    mvpose is [x, y, z, wx, wy, wz]: position in MILLIMETRES, orientation as
    an axis-angle vector in radians. The controller executes only the LAST
    instruction received, so this must be called at a fixed rate
    (key_control.streamed_path owns the pacing). speed is in mm/s. Returns
    the SDK code.
    """

    return arm.set_servo_cartesian_aa(mvpose, speed=speed, is_radian=True)


def move_to_joints(arm, q_target_rad, speed=SERVO_SPEED, wait=True):
    """Drive servos to q_target_rad (radians); wait=True blocks on arrival."""

    q_deg = np.rad2deg(np.asarray(q_target_rad, dtype=float))
    if not within_joint_limits(q_deg):
        return False, 'target outside joint limits: %s' % np.round(q_deg, 1)
    code = arm.set_servo_angle(angle=q_deg.tolist(), speed=speed,
                               is_radian=False, wait=wait)
    if code != 0:
        return False, 'servo move failed (code %d)' % code
    return True, 'ok'


HOME_JOINTS_DEG = [0, 0, 20, 0, 0, 0]


def go_home(arm):
    """Move the arm back to its home position (blocking). Returns (ok, msg)."""
    return move_to_joints(arm, np.deg2rad(HOME_JOINTS_DEG))


def disable_arm(arm):
    """Put the controller into the stopped state (state 4): every motion
    command is refused until re-enabled (UFactory Studio, or this script's
    next set_control_mode call). motion_enable(False) is deliberately NOT
    used: recovering from it knocks the arm out of Manual Mode mid-lab.
    Returns the SDK code.
    """

    return arm.set_state(4)


def open_gripper(arm):
    """Engage suction. Returns 1 on success, 0 otherwise."""
    try:
        return 1 if arm.open_lite6_gripper() == 0 else 0
    except Exception as e:
        progress('! open_gripper: %s' % e)
        return 0


def close_gripper(arm):
    """Release suction. Returns 1 on success, 0 otherwise."""
    try:
        return 1 if arm.close_lite6_gripper() == 0 else 0
    except Exception as e:
        progress('! close_gripper: %s' % e)
        return 0


def stop_gripper(arm):
    """Best-effort stop of the vacuum pump (disable it). Returns True on success."""
    try:
        return arm.stop_lite6_gripper()
    except Exception as e:
        progress('! stop_gripper: %s' % e)
        return False
