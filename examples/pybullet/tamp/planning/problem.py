"""Instantiate symbolic initial facts and ordered placement slots from a scene."""
import numpy as np
from pddlstream.language.constants import Equal
from examples.pybullet.utils.pybullet_tools.pr2_primitives import Conf, Pose
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    ARM_NAMES, get_arm_joints, get_group_joints, get_group_conf)
from examples.pybullet.utils.pybullet_tools.utils import (
    get_joint_positions, get_pose, get_aabb, is_placement)
from robot_geometry import object_dimensions
from placement_layout import ordered_layout, original_resting_orientation


def build_initial_facts(problem):
    robot = problem.robot
    constant_map = {'@sink': 'sink', '@stove': 'stove'}

    initial_bq = Conf(robot, get_group_joints(robot, 'base'), get_group_conf(robot, 'base'))
    init = [
        ('CanMove',),
        ('BConf', initial_bq),
        ('AtBConf', initial_bq),
        Equal(('PickCost',), 1),
        Equal(('PlaceCost',), 1),
        Equal(('DisconnectCost',), 1),
    ]

    for arm in ARM_NAMES:
        joints = get_arm_joints(robot, arm)
        conf = Conf(robot, joints, get_joint_positions(robot, joints))
        init += [('Arm', arm), ('AConf', arm, conf), ('HandEmpty', arm), ('AtAConf', arm, conf)]
        if arm in problem.arms:
            init += [('Controllable', arm)]

    group_map = getattr(problem, 'object_groups', {})


    group_order = getattr(problem, 'group_order', None)
    if group_order is None:
        group_order = sorted(set(group_map.values()))
    def grp_obj(g):
        return 'grp_{}'.format(g)


    for i in range(len(group_order)):
        for j in range(i + 1, len(group_order)):
            init += [('GroupBefore', grp_obj(group_order[i]), grp_obj(group_order[j]))]
    default_group = group_order[0] if group_order else 'A'

    goal_surface_map = {obj: surf for obj, surf in problem.goal_on}
    for body in problem.movable:
        pose = Pose(body, get_pose(body), init=True)
        init += [('Graspable', body), ('Pose', body, pose), ('AtPose', body, pose)]
        group = group_map.get(body, default_group)
        init += [('InGroup', body, grp_obj(group))]

        target_surface = goal_surface_map.get(body)
        for surface in problem.surfaces:
            if surface == target_surface:
                init += [('Stackable', body, surface)]
            if is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    # Slots belong to removal ranks within a group, independently of member identity.
    placement_slots = {}
    for group in group_order:
        members = [b for b in problem.movable if group_map.get(b, default_group) == group]
        if not members:
            continue
        surfaces = {goal_surface_map[b] for b in members}
        if len(surfaces) != 1:
            raise ValueError('Ordered placement requires one target surface per group')
        surface = next(iter(surfaces))
        member_dimensions = [object_dimensions(b) for b in members]
        upright = group in getattr(problem, 'upright_placement_groups', set())
        layout = ordered_layout(member_dimensions, upright=upright)
        original_face = getattr(problem, 'original_resting_face', False)
        if original_face:
            envelope = tuple(np.max([original_resting_orientation(d)[0]
                                     for d in member_dimensions], axis=0))
        else:
            envelope = tuple(layout['cell_size']) + (max(d[2] if upright else min(d) for d in member_dimensions),)
        bounds = get_aabb(surface)
        available = np.asarray(bounds[1][:2]) - np.asarray(bounds[0][:2])
        if np.any(available + 1e-6 < np.asarray(layout['required_size'])):
            raise ValueError('Target surface is too small for laid-down placement slots')
        center = (np.asarray(bounds[0]) + np.asarray(bounds[1]))/2
        tokens = ['placement_{}_{}'.format(grp_obj(group), i+1)
                  for i in range(len(members)+1)]
        init += [('NextPlacementSlot', grp_obj(group), tokens[0])]
        for rank, (dx, dy) in enumerate(layout['centers']):
            slot = tokens[rank]
            placement_slots[slot] = dict(group=group, rank=rank+1, surface=surface,
                center=(float(center[0]+dx), float(center[1]+dy)), top=float(bounds[1][2]),
                envelope=envelope)
            init += [('PlacementSlot', slot, surface), ('SlotNext', slot, tokens[rank+1])]
        print('[PLACEMENT] {}: {} ordered slots, gap {:.3f} m, orientation {}'.format(
            group, len(members), layout['gap'], 'original positive-Z resting face' if original_face
            else ('upright' if upright else 'largest-face down')))


    for conn in getattr(problem, 'connections', []):
        init += [('Connection', conn), ('Connected', conn)]
        for body in conn.bodies:
            init += [('ConnectionInvolves', conn, body)]

    return constant_map, init, placement_slots, group_map, default_group
