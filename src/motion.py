"""Robot-facing move primitives + flange pose helpers."""


from fk_lite6 import fk_from_zeta
from ik_lite6 import solve_ik
import robot


def flange_pose_base(q_rad, zeta):
    """Base->flange transform from current joints (metres/radians)."""
    return fk_from_zeta(q_rad, zeta)


def move_to_pose(arm, T_target, zeta, q_seed_rad, speed=None):
    """Seed IK with q_seed_rad, drive servos to T_target. Optional speed %."""

    q_sol, ok, err = solve_ik(T_target, q_seed_rad, zeta)
    if not ok:
        return False, "IK did not converge (residual %.2e)" % err, None
    servo_speed = robot.SERVO_SPEED if speed is None else speed
    ok2, msg = robot.move_to_joints(arm, q_sol, speed=servo_speed)
    return ok2, msg, (q_sol if ok2 else None)
