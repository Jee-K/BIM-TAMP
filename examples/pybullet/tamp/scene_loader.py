"""Construct simulator scenes from IFC files or JSON manifests."""
import json
import os
import numpy as np
import pybullet as p
from scene_manifest import load_ifc_scene, load_scene_manifest
from placement_layout import ordered_layout
from robot_geometry import detect_connections
from examples.pybullet.utils.pybullet_tools.pr2_problems import Problem, create_pr2
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    set_arm_conf, arm_conf, REST_LEFT_ARM, open_arm)
from examples.pybullet.utils.pybullet_tools.utils import (
    add_data_path, load_pybullet, create_box, set_point, Point, stable_z,
    add_body_name, add_line, pairwise_collision, is_placement, get_aabb)


def make_ground_clutter(n, rng_seed=12345, x_range=(-0.5, 0.5), y_range=(-2.0, 2.0),
                        reject=None):
    """N static obstacle boxes scattered over [x_range] x [y_range]. The layout is
    drawn from a SEPARATE RandomState keyed on `rng_seed` (= the run seed), so it
    varies across seeds but does NOT consume the global RNG — keeping the planner's
    own sampling identical with vs without clutter (a clean paired comparison).
    `reject(x, y) -> True` excludes a position (used to keep clutter off the robot
    start, the structure, and the plates). Static obstacles the robot avoids but
    never manipulates.
    """
    rng = np.random.RandomState(rng_seed)
    bodies = []
    tries = 0
    while len(bodies) < n and tries < n * 400:
        tries += 1
        cx = float(rng.uniform(*x_range))
        cy = float(rng.uniform(*y_range))
        if reject is not None and reject(cx, cy):
            continue
        box = create_box(0.2, 0.2, 0.4, color=(0.4, 0.4, 0.4, 1.0))
        set_point(box, Point(x=cx, y=cy, z=0.2))
        p.changeDynamics(box, -1, contactProcessingThreshold=0.02, collisionMargin=0.0)
        bodies.append(box)
    return bodies


def create_ifc_problem(ifc_path, scale=1, clutter=0, clutter_seed=None,
                       require_ifc_ranks=False, selected_guids=None, ifc_scene=None,
                       exclude_groups=None):
    """Instantiate IFC-derived boxes, targets and geometrically detected contacts."""
    if ifc_scene is None:
        ifc_scene = load_ifc_scene(ifc_path, require_explicit_ranks=require_ifc_ranks,
                                   selected_guids=selected_guids,
                                   exclude_groups=exclude_groups)
    print('[IFC2SIM] Loaded {} members from {}'.format(len(ifc_scene.members), ifc_scene.path))
    print('[IFC2SIM] Removal order: {}'.format(' -> '.join(ifc_scene.group_order)))


    # Translate the whole structure without changing relative member geometry.
    center = (ifc_scene.aabb_min + ifc_scene.aabb_max) / 2.0
    offset = np.array([1.5 - center[0], 0.6 - center[1], -ifc_scene.aabb_min[2]])
    model_min, model_max = ifc_scene.aabb_min + offset, ifc_scene.aabb_max + offset
    copy_dy = max(2.0, float(model_max[1] - model_min[1]) + 1.0)
    objects_info = []
    for copy_index in range(scale):
        for member in ifc_scene.members:
            name = member.name if copy_index == 0 else '{}_c{}'.format(member.name, copy_index)
            position = member.center + offset + np.array([0., -copy_dy * copy_index, 0.])
            objects_info.append((name, tuple(np.maximum(member.dimensions, .005)),
                                 tuple(position), member.group))

    add_data_path()
    floor = load_pybullet('plane.urdf')
    robot = create_pr2(fixed_base=False)
    robot_start = (float(model_max[0]) + 1.4, float(model_max[1]) + 1.4)
    set_point(robot, Point(x=robot_start[0], y=robot_start[1], z=.15))
    p.changeDynamics(robot, -1, contactProcessingThreshold=.02, collisionMargin=0.)
    for joint in range(p.getNumJoints(robot)):
        p.changeDynamics(robot, joint, contactProcessingThreshold=.02, collisionMargin=0.)
    arm = 'left'
    set_arm_conf(robot, arm, arm_conf(arm, REST_LEFT_ARM))
    open_arm(robot, arm)

    group_order = list(ifc_scene.group_order)
    upright_groups = {member.group for member in ifc_scene.members
        if 'stud' in (member.source_type + ' ' + member.name + ' ' + member.role).lower()
        or member.ifc_type == 'IfcColumn'}
    groups = {group: [item for item in objects_info if item[3] == group] for group in group_order}
    colors = [(0.90, 0.10, 0.10, 1.), (0.10, 0.25, 0.90, 1.),
              (1.00, 0.50, 0.00, 1.), (0.20, 0.75, 0.25, 1.),
              (0.65, 0.25, 0.80, 1.)]
    group_colors = {group: colors[i % len(colors)] for i, group in enumerate(group_order)}


    plate_specs = []
    lane_height = 2.5
    for group in group_order:
        items = groups[group]
        layout = ordered_layout([size for _, size, _, _ in items], upright=group in upright_groups)
        width, height = layout['plate_size']
        plate_specs.append((group, width, height))
        lane_height = max(lane_height, height)
    max_plate_width = max(width for _, width, _ in plate_specs)
    plate_x = float(model_min[0]) - 1.4 - max_plate_width / 2.0
    plates, plate_centers = {}, {}
    for i, (group, width, height) in enumerate(plate_specs):
        plate_y = float((i - (len(plate_specs) - 1) / 2.0) * (lane_height + .6))
        plate = create_box(width, height, .001, color=group_colors[group][:-1] + (.5,))
        set_point(plate, Point(x=plate_x, y=plate_y, z=stable_z(plate, floor)))
        plates[group], plate_centers[group] = plate, (plate_x, plate_y)

    objs, object_groups, body_names = [], {}, {}
    for name, size, position, group in objects_info:
        obj = create_box(*size, color=group_colors[group])
        add_body_name(obj, name)
        set_point(obj, Point(x=position[0], y=position[1], z=position[2]))
        p.changeDynamics(obj, -1, contactProcessingThreshold=.02, collisionMargin=0.)
        objs.append(obj)
        object_groups[obj], body_names[obj] = group, name

    min_y = min(float(model_min[1] - copy_dy * (scale - 1)),
                -.5 * (len(plate_specs) - 1) * (lane_height + .6) - lane_height / 2.0)
    max_y = max(float(model_max[1]),
                .5 * (len(plate_specs) - 1) * (lane_height + .6) + lane_height / 2.0)
    base_limits = (np.array([plate_x - max_plate_width / 2.0 - 1., min_y - 1.]),
                   np.array([max(float(model_max[0]), robot_start[0]) + 1., max_y + 1.]))
    (xmin, ymin), (xmax, ymax) = base_limits
    for a, b in zip([(xmin, ymin, .01), (xmax, ymin, .01), (xmax, ymax, .01), (xmin, ymax, .01)],
                    [(xmax, ymin, .01), (xmax, ymax, .01), (xmin, ymax, .01), (xmin, ymin, .01)]):
        add_line(a, b, color=(1, 1, 0), width=2)

    connections = detect_connections(objs, body_names)
    all_plates = list(plates.values())
    problem = Problem(robot=robot, movable=objs, arms=[arm], grasp_types=['top'],
                      surfaces=[floor] + all_plates,
                      goal_on=[(body, plates[object_groups[body]]) for body in objs],
                      base_limits=base_limits)
    problem.object_groups = object_groups
    problem.group_order = group_order
    problem.upright_placement_groups = upright_groups
    problem.body_names = body_names
    problem.connections = connections
    problem.ifc_path = ifc_scene.path
    problem.ifc_selected_guids = sorted(selected_guids) if selected_guids is not None else None
    problem.scale, problem.clutter = scale, clutter
    problem.fixed = [body for body in problem.fixed if body != floor and body not in all_plates]

    if clutter > 0:
        def reject(x, y):
            if any((x - ox) ** 2 + (y - oy) ** 2 < (.5 * float(np.hypot(size[0], size[1])) + .3) ** 2
                   for _, size, (ox, oy, _), _ in objects_info):
                return True
            for group, width, height in plate_specs:
                px, py = plate_centers[group]
                if abs(x - px) < width / 2. + .3 and abs(y - py) < height / 2. + .3:
                    return True
            return (x - robot_start[0]) ** 2 + (y - robot_start[1]) ** 2 < .9 ** 2
        clutter_bodies = make_ground_clutter(
            clutter, rng_seed=clutter_seed if clutter_seed is not None else 12345,
            x_range=(float(xmin) + .4, float(xmax) - .4),
            y_range=(float(ymin) + .4, float(ymax) - .4), reject=reject)
        problem.fixed = list(problem.fixed) + clutter_bodies
    print('[IFC2SIM] Scene: {} groups, {} target plates, {} connections'.format(
        len(group_order), len(all_plates), len(connections)))
    return problem


def create_scene_problem(scene_path, scale=1, clutter=0, clutter_seed=None, exclude_groups=None):
    """Build DeconTAMP from an IFC2SIM JSON manifest; no IFC parsing occurs."""
    with open(scene_path) as stream:
        config = json.load(stream)
    scene = load_scene_manifest(scene_path)
    if 'simulation' in config:
        return create_configured_scene_problem(scene_path, config, scale=scale, clutter=clutter,
                                               clutter_seed=clutter_seed, exclude_groups=exclude_groups)
    if exclude_groups:
        from dataclasses import replace
        members = tuple(m for m in scene.members if m.group not in exclude_groups)
        if not members:
            raise ValueError('Group filter excludes every scene member')
        scene = replace(scene, members=members,
                        group_order=tuple(g for g in scene.group_order if g not in exclude_groups))
    problem = create_ifc_problem(None, scale=scale, clutter=clutter,
                                 clutter_seed=clutter_seed, ifc_scene=scene)
    problem.scene_path = os.path.abspath(scene_path)
    return problem


def create_configured_scene_problem(scene_path, config, scale=1, clutter=0, clutter_seed=None, exclude_groups=None):
    """Build a manifest instance with supplied workspace and target settings."""
    if scale < 1:
        raise ValueError('Scene scale must be positive')
    if not config.get('members'):
        raise ValueError('Scene manifest has no members')
    settings = config['simulation']
    add_data_path()
    floor = load_pybullet("plane.urdf")

    robot = create_pr2(fixed_base=False)
    set_point(robot, tuple(settings["robot_start"]))


    p.changeDynamics(robot, -1, contactProcessingThreshold=0.02,
                    collisionMargin=0.0)

    num_joints = p.getNumJoints(robot)
    for i in range(num_joints):
        p.changeDynamics(robot, i, contactProcessingThreshold=0.02,
                        collisionMargin=0.0)

    arm = 'left'
    set_arm_conf(robot, arm, arm_conf(arm, REST_LEFT_ARM))
    open_arm(robot, arm)

    plates = {}
    for target in settings['targets']:
        size = np.asarray(target['size'], dtype=float)
        size += (scale - 1) * np.asarray(target.get('size_growth', [0., 0., 0.]))
        size = np.minimum(size, target.get('max_size', size))
        plate = create_box(*size, color=target['color'])
        position = list(target['position'])
        if position[2] is None:
            position[2] = stable_z(plate, floor)
        set_point(plate, tuple(position))
        plates[target['group']] = plate
    all_plates = list(plates.values())
    members = [(m['name'], m['dimensions'], m['center'], m['group'])
                      for m in config['members']]
    copy_offset = np.asarray(settings.get('copy_offset', [0., 0., 0.]))
    objects_info = []
    for i in range(scale):
        for name, size, (px, py, pz), group in members:
            if group in (exclude_groups or ()):
                continue
            nm = name if i == 0 else "{}_c{}".format(name, i)
            objects_info.append((nm, size, tuple(np.asarray((px, py, pz)) + copy_offset * i), group))
    print("[SCALE] scale={} -> {} objects".format(scale, len(objects_info)))

    objs = []
    object_groups = {}
    body_names = {}
    for name, size, pos, group in objects_info:
        color = settings['colors'][group]
        obj = create_box(size[0], size[1], size[2], color=color)
        add_body_name(obj, name)
        set_point(obj, Point(x=pos[0], y=pos[1], z=pos[2]))


        p.changeDynamics(obj, -1, contactProcessingThreshold=0.02,
                        collisionMargin=0.0)

        objs.append(obj)
        object_groups[obj] = group
        body_names[obj] = name


    print("[COLLISION CHECK] Verifying robot initial position...")
    robot_collides = False
    for obj in objs:
        if pairwise_collision(robot, obj):
            obj_name = body_names.get(obj, "unknown")
            print(f"[COLLISION CHECK] WARNING: Robot base collides with {obj_name} at initial position!")
            robot_collides = True
    if not robot_collides:
        print("[COLLISION CHECK] Robot initial position is collision-free")

    def debug_initial_geometry(bodies, surfaces):
        print("[GEOM DEBUG] Initial placement/collision checks")
        for body in bodies:
            for surface in surfaces:
                placement_ok = is_placement(body, surface)
                print("[GEOM DEBUG] body {} on surface {} placement_ok={}".format(
                    body, surface, placement_ok))
            for other in bodies:
                if other == body:
                    continue
                if pairwise_collision(body, other):
                    print("[GEOM DEBUG] collision body {} with {}".format(body, other))

    base_limits = tuple(np.asarray(v, dtype=float) for v in settings['base_limits'])
    xmin, ymin = base_limits[0]
    xmax, ymax = base_limits[1]
    z = 0.01
    corners = [(xmin, ymin, z), (xmax, ymin, z), (xmax, ymax, z), (xmin, ymax, z)]
    for i in range(4):
        add_line(corners[i], corners[(i + 1) % 4], color=(1, 1, 0), width=2)


    connections = detect_connections(objs, body_names)

    goals = [(b, plates[object_groups[b]]) for b in objs]
    problem = Problem(robot=robot, movable=objs,
                      arms=[arm], grasp_types=['top'],
                      surfaces=[floor] + all_plates, goal_on=goals,
                      base_limits=base_limits)
    problem.object_groups = object_groups
    problem.group_order = [g for g in config['group_order'] if g in object_groups.values()]
    problem.body_names = body_names
    problem.connections = connections
    problem.scale = scale
    problem.clutter = clutter
    policy = settings.get('placement', {})
    problem.original_resting_face = policy.get('original_resting_face', False) and (
        not exclude_groups or policy.get('preserve_face_with_group_filter', True))
    problem.upright_placement_groups = set(policy.get('upright_groups', []))
    problem.scene_path = os.path.abspath(scene_path)
    problem.fixed = [b for b in problem.fixed if b != floor and b not in all_plates]


    if clutter > 0:
        rng_seed = clutter_seed if clutter_seed is not None else 12345


        (xmn, ymn), (xmx, ymx) = base_limits
        keepout = settings.get('clutter_keepout', {})
        boundary = keepout.get('target_boundary')
        if boundary:
            axis = boundary['axis']
            target_edge = max(get_aabb(b)[1][axis] for b in all_plates) if boundary['side'] == 'lower' else min(get_aabb(b)[0][axis] for b in all_plates)
        keepouts = [(*settings['robot_start'][:2], keepout.get('robot_radius', .9))]
        for _nm, _sz, (_ox, _oy, _oz), _g in objects_info:
            keepouts.append((_ox, _oy, 0.5 * float(np.hypot(_sz[0], _sz[1])) + keepout.get('member_margin', .3)))

        def _reject(x, y):
            if boundary:
                value = (x, y)[axis]
                margin = boundary.get('margin', .3)
                if (boundary['side'] == 'lower' and value < target_edge + margin) or (boundary['side'] == 'upper' and value > target_edge - margin):
                    return True
            return any((x - kx) ** 2 + (y - ky) ** 2 < kr ** 2 for kx, ky, kr in keepouts)

        clutter_bodies = make_ground_clutter(
            clutter, rng_seed=rng_seed,
            x_range=(float(xmn) + keepout.get('workspace_margin', .4), float(xmx) - keepout.get('workspace_margin', .4)),
            y_range=(float(ymn) + keepout.get('workspace_margin', .4), float(ymx) - keepout.get('workspace_margin', .4)),
            reject=_reject)
        problem.fixed = list(problem.fixed) + clutter_bodies
        print("[CLUTTER] placed {}/{} obstacles over the workspace (layout seed {})".format(
            len(clutter_bodies), clutter, rng_seed))
    debug_initial_geometry(problem.movable, [floor] + all_plates)
    return problem


def create_problem(scale=1, clutter=0, clutter_seed=None, ifc_path=None,
                   require_ifc_ranks=False, selected_guids=None, scene_path=None,
                   exclude_groups=None):
    """Build a planning instance from an explicit JSON scene or IFC input."""
    if scene_path is not None:
        return create_scene_problem(scene_path, scale=scale, clutter=clutter,
                                    clutter_seed=clutter_seed, exclude_groups=exclude_groups)
    if ifc_path is None:
        raise ValueError('Provide a JSON scene (--scene) or IFC file (--ifc)')
    return create_ifc_problem(ifc_path, scale=scale, clutter=clutter,
                             clutter_seed=clutter_seed,
                             require_ifc_ranks=require_ifc_ranks,
                             selected_guids=selected_guids,
                             exclude_groups=exclude_groups)
