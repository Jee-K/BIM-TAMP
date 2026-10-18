"""Reporting metrics for saved arm trajectories; no simulation mutations."""
import math


def joint_path_length(path):
    """Euclidean joint-space arc length in radians for this revolute robot arm.

    Use the saved, bounded-joint trajectory values, including intermediate
    waypoints. Do not replace a commanded rotation by a wrapped shortcut.
    """
    values = [tuple(getattr(conf, 'values', conf)) for conf in path]
    if any(not all(math.isfinite(v) for v in q) for q in values):
        raise ValueError('Arm trajectory contains nonfinite joint values')
    if values and any(len(q) != len(values[0]) for q in values):
        raise ValueError('Inconsistent arm trajectory dimensions')
    return sum(math.dist(a, b) for a, b in zip(values, values[1:]))


def arm_motion_metrics(plan, weight=1.):
    """Count approach and reversed retreat for every pick/place/disconnect.

    The current command lowering executes each stored arm trajectory in both
    directions. Gripper opening/closing is separate from the six arm joints.
    The weight is score units per radian; the combined score is not metres.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Arm cost weight must be finite and nonnegative')
    approach = 0.
    for name, args in plan:
        if name in ('pick', 'place'):
            commands = args[5]
        elif name == 'disconnect':
            commands = args[4]
        else:
            continue
        for trajectory in commands.commands:
            approach += joint_path_length(trajectory.path)
    return dict(arm_approach_distance=approach, arm_retreat_distance=approach,
        arm_joint_distance=2*approach, arm_cost_weight=weight,
        arm_motion_cost=weight*2*approach)
