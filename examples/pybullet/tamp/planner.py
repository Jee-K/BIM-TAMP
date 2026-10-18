#!/usr/bin/env python3
from __future__ import print_function

import os
import copy
import sys
import time
import datetime
import numpy as np
import pybullet as p
import random


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.pybullet.utils.pybullet_tools.utils import get_body_name, add_body_name


from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import Equal, PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test
from examples.pybullet.utils.pybullet_tools.pr2_primitives import iterate_approach_path
from mppi_motion_planner import mppi_ik_gen
from scene_manifest import load_ifc_scene, load_scene_manifest, prompt_for_types
from placement_layout import ordered_layout, flat_orientations, original_resting_orientation
from run_statistics import get_run_statistics, track_current, with_run_statistics
from motion_metrics import arm_motion_metrics


from examples.pybullet.utils.pybullet_tools.utils import (
    connect, disconnect, add_data_path, load_pybullet, create_box, set_point,
    Point, stable_z, get_pose, is_placement, HideOutput, LockRenderer, has_gui,
    WorldSaver, get_joint_positions, multiply, invert, BodySaver, set_base_values,
    set_joint_positions, get_collision_fn, get_moving_links, link_from_name,
    pairwise_link_collision, pairwise_collision, get_point, simulate_for_duration,
    get_aabb, get_min_limit, get_max_limit, set_pose, get_distance,
    get_joint_name, get_joint_names, get_movable_joints, add_line, set_random_seed, set_numpy_seed,
    create_sphere, set_color, wait_for_user, get_links, get_link_name, get_link_pose,
    create_attachment, PoseSaver, unit_pose, quat_from_matrix, matrix_from_quat,
    joint_from_name, get_extend_fn, remove_body, get_self_link_pairs, get_limits_fn
)
from examples.pybullet.utils.pybullet_tools.pr2_problems import Problem, create_pr2
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
    Pose, Conf, get_motion_gen, get_stable_gen, get_grasp_gen, get_ir_sampler,
    apply_commands, State, Commands, create_trajectory, Command, GripperCommand,
    Attach, Detach
)
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    get_arm_joints, ARM_NAMES, get_group_joints, get_group_conf,
    set_arm_conf, open_arm, close_arm, arm_conf, REST_LEFT_ARM, get_gripper_link
)
from examples.pybullet.tamp.streams import (
    get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test,
    distance_fn
)
from examples.pybullet.namo.stream import get_custom_limits
from examples.pybullet.pr2.run import post_process


from robot_geometry import *
from robot_geometry import _unit
from planning.definitions import DOMAIN_PDDL, STREAM_PDDL
from planning.stream_bindings import pddlstream_from_problem


from scene_loader import create_problem, create_scene_problem, create_ifc_problem


class TeeLogger:
    """Duplicates stdout/stderr to both terminal and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def fileno(self):
        return self.terminal.fileno()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def setup_logging():
    """Create ./log/ directory and redirect stdout/stderr to a timestamped log file."""
    log_dir = os.environ.get('BIM_TAMP_LOG_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'run_{timestamp}.log')
    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee
    print(f"[LOG] Logging to {log_path}")
    return tee


def save_plan_json(problem, plan, log_path, cost, planner, seed=None):
    """Save the plan as structured JSON for sim replay and ROS conversion.

    Format per action:
      move_base  -> base_trajectory: list of [x, y, theta] waypoints
      pick       -> arm_approach_trajectory, arm_retreat_trajectory (joint arrays),
                    gripper_close_position, object_pose (w_T_obj at pick)
      place      -> arm_approach_trajectory, arm_retreat_trajectory,
                    gripper_open_position, object_target_pose
    """
    import json

    robot = problem.robot
    body_names = getattr(problem, 'body_names', {})

    def traj_waypoints(traj_commands):
        traj = traj_commands.commands[0]
        return [list(conf.values) for conf in traj.path]

    def joint_names_for(traj_commands):
        traj = traj_commands.commands[0]
        if traj.path:
            joints = traj.path[0].joints
            if joints:
                return get_joint_names(robot, joints)
        return []

    def pose_to_dict(pose_val):
        pos, quat = pose_val
        return {'position': list(pos), 'orientation': list(quat)}

    actions = []
    for i, (name, args) in enumerate(plan):
        entry = {'index': i, 'type': name}

        if name == 'move_base':
            bq1, bq2, traj_commands = args[0], args[1], args[2]
            entry['start_base_conf'] = list(bq1.values)
            entry['end_base_conf'] = list(bq2.values)
            entry['base_trajectory'] = traj_waypoints(traj_commands)
            entry['transit_arm_confs'] = getattr(traj_commands, 'transit_arm_confs', {})

        elif name == 'pick':
            arm, body, pose, grasp, bq, atraj_commands = args[:6]
            wp = traj_waypoints(atraj_commands)
            entry['object_name'] = body_names.get(body, str(body))
            entry['arm'] = arm
            entry['arm_joint_names'] = joint_names_for(atraj_commands)
            entry['base_conf'] = list(bq.values)
            entry['arm_approach_trajectory'] = wp
            entry['arm_retreat_trajectory'] = list(reversed(wp))
            entry['gripper_close_position'] = float(grasp.grasp_width)
            entry['object_pose'] = pose_to_dict(pose.value)
            entry['grasp_transform'] = pose_to_dict(grasp.value)

        elif name == 'place':
            arm, body, pose, grasp, bq, atraj_commands = args[:6]
            wp = traj_waypoints(atraj_commands)
            from examples.pybullet.utils.pybullet_tools.pr2_primitives import get_gripper_joints
            gripper_joint = get_gripper_joints(robot, arm)[0]
            entry['object_name'] = body_names.get(body, str(body))
            entry['arm'] = arm
            entry['arm_joint_names'] = joint_names_for(atraj_commands)
            entry['base_conf'] = list(bq.values)
            entry['arm_approach_trajectory'] = wp
            entry['arm_retreat_trajectory'] = list(reversed(wp))
            entry['gripper_open_position'] = float(get_max_limit(robot, gripper_joint))
            entry['object_target_pose'] = pose_to_dict(pose.value)
            entry['placement_rank'] = pose.placement_rank
            entry['placement_group'] = pose.placement_group

        elif name == 'disconnect':
            arm, conn, extract_obj, bq, atraj_commands = args
            wp = traj_waypoints(atraj_commands)
            entry['connection_name'] = getattr(conn, 'name', str(conn))
            entry['connection_point'] = list(getattr(conn, 'point', ()))
            entry['connection_bodies'] = [body_names.get(b, str(b))
                                          for b in getattr(conn, 'bodies', ())]
            entry['extract_object'] = body_names.get(extract_obj, str(extract_obj))
            entry['arm'] = arm
            entry['arm_joint_names'] = joint_names_for(atraj_commands)
            entry['base_conf'] = list(bq.values)
            entry['arm_approach_trajectory'] = wp
            entry['arm_retreat_trajectory'] = list(reversed(wp))

        actions.append(entry)


    objects = []
    for body, name in body_names.items():
        objects.append({
            'name': name,
            'body_id': body,
            'pose': pose_to_dict(get_pose(body))
        })

    output = {
        'metrics': getattr(problem, 'plan_metrics', None),
        'metadata': {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'planner': planner,
            'arm_cost_weight': getattr(problem, 'arm_cost_weight', 1.),
            'seed': seed,
            'scale': getattr(problem, 'scale', 1),
            'clutter': getattr(problem, 'clutter', 0),
            'ifc_file': getattr(problem, 'ifc_path', None),
            'ifc_selected_guids': getattr(problem, 'ifc_selected_guids', None),
            'scene_file': getattr(problem, 'scene_path', None),
            'exclude_groups': getattr(problem, 'exclude_groups', []),
            'placement_policy': ('group-removal-order-original-z-up-v1'
                if getattr(problem, 'original_resting_face', False)
                else 'group-removal-order-upright-studs-v1'),
            'upright_placement_groups': sorted(getattr(problem, 'upright_placement_groups', set())),
            'cost': float(cost),
            'num_actions': len(plan),
            'log_file': os.path.basename(log_path) if log_path else None,
        },
        'initial_robot_state': getattr(problem, 'initial_robot_state', None),
        'motion_protocol': 'checked-transit-v1',
        'scene': {
            'objects': objects,
        },
        'actions': actions,
    }

    json_path = log_path.replace('.log', '_plan.json') if log_path else 'plan.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"[PLAN] Saved replayable plan to {json_path}")
    return json_path


class LiveAttach(Attach):
    """Attach that captures the grasp from the ACTUAL poses at pick time.

    The stock Attach.assign() first teleports the body to
    ``tool_pose * grasp.value`` using a precomputed grasp offset. For these side
    grasps that offset does not match the pose the IK actually reached, so the
    object was welded away from the gripper (held outside the fingers).

    By pick time the executed trajectory has already placed the gripper at the
    real grasp configuration and the (static) target is still at its pick pose,
    so the true gripper->object relationship is exactly the current one. This
    variant therefore does NOT teleport the body; it leaves it where it is and
    lets ``create_attachment`` capture the correct relative transform, so the
    held object stays exactly where the fingers are.
    """
    def assign(self):

        return

    def apply(self, state, **kwargs):

        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        state.grasps[self.body] = self.grasp
        if self.body in state.poses:
            del state.poses[self.body]
        yield


@track_current('execution_checks', success=lambda result: True)
def apply_checked_commands(problem, state, commands, time_step=0.0, pause=False):
    """Reject discontinuous commands or penetrations during actual execution."""
    from examples.pybullet.utils.pybullet_tools.utils import get_difference_fn, get_base_values
    robot = problem.robot
    fingers = get_gripper_finger_links(robot)
    obstacles = list(dict.fromkeys(problem.fixed + problem.movable))
    self_pairs = get_self_link_pairs(robot, get_movable_joints(robot))
    for i, command in enumerate(commands):
        allowed = set(getattr(command, 'allowed_contacts', ()))
        if hasattr(command, 'path') and command.path:
            first = command.path[0]
            if not first.joints and len(first.values) == 3:
                current = np.asarray(get_base_values(robot))
                delta = np.asarray(first.values) - current
                delta[2] = (delta[2] + np.pi) % (2*np.pi) - np.pi
                if np.max(np.abs(delta)) > 0.02:
                    raise RuntimeError('Unchecked base jump at command {}'.format(i))
            if first.joints:
                difference = get_difference_fn(robot, first.joints)
                current = get_joint_positions(robot, first.joints)
                if max(abs(v) for v in difference(first.values, current)) > 0.05:
                    raise RuntimeError('Unchecked arm jump at command {}'.format(i))
        for _ in command.apply(state):
            state.assign()
            held = set(state.attachments)
            for link1, link2 in self_pairs:
                if pairwise_link_collision(robot, link1, robot, link2, max_distance=0.):
                    raise RuntimeError('Execution self-collision: command {} {} vs {}'.format(
                        i, get_link_name(robot, link1), get_link_name(robot, link2)))
            for body in obstacles:
                bad = [v for v in p.getClosestPoints(robot, body, distance=0.)
                       if v[8] < -0.001 and not
                       (v[3] in fingers and body in (allowed | held))]
                if bad:
                    raise RuntimeError('Execution collision: command {} robot link {} vs {}; penetration {} pose {}'.format(
                        i, bad[0][3], problem.body_names.get(body, body), -bad[0][8], get_pose(robot)))
                for carried in held:
                    if body != carried and any(v[8] < -0.001
                            for v in p.getClosestPoints(carried, body, distance=0.)):
                        raise RuntimeError('Execution carried-object collision at command {}'.format(i))
            if time_step:
                time.sleep(time_step)
        if pause:
            wait_for_user('Continue execution?')


def post_process_with_disconnect(problem, plan, teleport=False):
    """Like pr2.run.post_process but also lowers `disconnect` actions to commands.

    A disconnect plays the touch trajectory, flips the connection to disconnected
    (DisconnectMarker), then retreats along the reverse trajectory.
    """
    if plan is None:
        return None
    commands = []
    for i, (name, args) in enumerate(plan):
        if name == 'disconnect':
            a, c, o, q, cmds = args
            [t] = cmds.commands
            new_commands = [t, DisconnectMarker(c), t.reverse()]
            print(i, name, args, new_commands)
            for command in new_commands:
                command.allowed_contacts = set(c.bodies)
            commands += new_commands
        elif name == 'pick':


            a, b, p, g, _, c = args[:6]
            [t] = c.commands
            open_gripper = HandEGripperCommand(problem.robot, a, HANDE_OPEN_Q, teleport=teleport)
            close_gripper = HandEGripperCommand(problem.robot, a, g.grasp_width, teleport=teleport)
            attach = LiveAttach(problem.robot, a, g, b)
            new_commands = [open_gripper, t, close_gripper, attach, t.reverse()]
            print(i, name, args, new_commands)
            for command in new_commands:
                command.allowed_contacts = {b}
            commands += new_commands
        elif name == 'place':
            a, b, p, g, _, c = args[:6]
            [t] = c.commands
            open_gripper = HandEGripperCommand(problem.robot, a, HANDE_OPEN_Q, teleport=teleport)
            new_commands = [t, Detach(problem.robot, a, b), open_gripper, t.reverse()]
            print(i, name, args, new_commands)
            for command in new_commands:
                command.allowed_contacts = {b}
            commands += new_commands
        else:
            commands += post_process(problem, [(name, args)], teleport=teleport)
    return commands


def stage_problem_view(problem, group):
    """A lightweight copy of `problem` restricted to one removal group.

    - movable      = members of this group only
    - fixed        = original fixed obstacles + ALL members of LOWER-ranked
                     groups (removed later), so they are present in every
                     collision check (IK, base motion, cfree tests) but never
                     manipulated. Already placed members from earlier groups
                     also remain fixed obstacles at their actual placements.
    - goal_on      = this group's members -> their target plates
    - connections  = those that involve at least one member of this group
    The robot, surfaces, plates, body_names and object_groups are shared.
    """
    order = list(problem.group_order)
    gi = order.index(group)
    lower_groups = set(order[gi + 1:])
    og = problem.object_groups
    this_group = [b for b in problem.movable if og.get(b) == group]
    lower = [b for b in problem.movable if og.get(b) in lower_groups]

    view = copy.copy(problem)
    view.movable = this_group


    view.fixed = list(problem.fixed) + [b for b in problem.movable if b not in this_group]
    view.goal_on = [(b, s) for (b, s) in problem.goal_on if og.get(b) == group]
    view.group_order = (group,)


    present = set(this_group) | set(lower)
    members = set(this_group)
    # Keep only connections whose members are still present in this stage.
    view.connections = [c for c in problem.connections
                        if any(b in members for b in c.bodies)
                        and all(b in present for b in c.bodies)]

    view.ik_stats = {'attempts': 0, 'success': 0}
    return view


@with_run_statistics
def main(batch=False, seed=None, base_planner='birrt', motion_fallback=False,
         ik_solver='pybullet', task_planner='ff-astar2', scale=1, max_time=600,
         use_mppi=True, clutter=0, ifc_path=None, require_ifc_ranks=False,
         selected_guids=None, select_ifc_types=False,
         scene_path=None, view_scene=False, exclude_groups=None,
         staged=False, collision_check_order='early', arm_cost_weight=1.,
         search_mode='first', min_iterations=0):
    logger = setup_logging()
    run_stats = get_run_statistics()
    run_stats.log_path = logger.log_file.name
    print('[STATS DEFINITIONS] Run-local counts, no historical stream totals. '
          'Rates = successes/(successes+failures); errors and unfinished calls reported separately. '
          'base_queries include retries; base_planner_attempts count underlying calls. '
          'ik_solver_calls success means finite output, NOT reachability; *_ik_endpoints '
          'test approach/contact pairs before arm-path search. Timings are nested; do not sum categories.')


    planner_name = base_planner
    if seed is not None:
        set_random_seed(seed)
        set_numpy_seed(seed)
        print(f"[SEED] Random seed set to {seed}")

    connect(use_gui=not batch)


    if scene_path is None and ifc_path is None:
        raise ValueError('Provide a JSON scene (--scene) or IFC file (--ifc)')
    if select_ifc_types and scene_path is not None:
        raise ValueError('IFC element selection belongs to IFC2SIM; export the selected manifest first.')
    if select_ifc_types:
        if selected_guids is not None:
            raise ValueError('IFC type selection cannot be combined with a preselected member set.')
        selection_scene = load_ifc_scene(ifc_path, require_explicit_ranks=require_ifc_ranks)
        selected_guids = prompt_for_types(selection_scene)
    print(f"[IFC2SIM] Scene input: {scene_path or ifc_path}")
    problem = create_problem(scale=scale, clutter=clutter, clutter_seed=seed,
                                  ifc_path=ifc_path,
                                  require_ifc_ranks=require_ifc_ranks,
                                  selected_guids=selected_guids, scene_path=scene_path,
                                  exclude_groups=exclude_groups)
    if not np.isfinite(arm_cost_weight) or arm_cost_weight < 0:
        raise ValueError('Arm cost weight must be finite and nonnegative')
    problem.arm_cost_weight = arm_cost_weight
    problem.use_mppi = use_mppi
    problem.exclude_groups = sorted(exclude_groups or ())


    if view_scene:
        body_names = getattr(problem, 'body_names', {})
        object_groups = getattr(problem, 'object_groups', {})
        print('\n[VIEW] Scene members ({} total):'.format(len(problem.movable)))
        print('[VIEW] {:<28} {:<6} {:>24} {:>22}'.format(
            'name', 'group', 'position (x,y,z)', 'size (dx,dy,dz)'))
        rows = []
        for b in problem.movable:
            name = body_names.get(b, str(b))
            grp = object_groups.get(b, '?')
            pos = get_point(b)
            lo, up = get_aabb(b)
            size = (up[0] - lo[0], up[1] - lo[1], up[2] - lo[2])
            rows.append((name, grp, pos, size))
        for name, grp, pos, size in sorted(rows, key=lambda r: r[0]):
            print('[VIEW] {:<28} {:<6} ({:7.3f},{:7.3f},{:7.3f}) ({:7.3f},{:7.3f},{:7.3f})'.format(
                name, str(grp), pos[0], pos[1], pos[2], size[0], size[1], size[2]))
        print('[VIEW] Scene preview open (planning skipped). '
              'Press Enter in this terminal to exit.')
        if has_gui():
            wait_for_user('[VIEW] Press Enter to close the preview...')
        disconnect()
        if isinstance(sys.stdout, TeeLogger):
            log_path = sys.stdout.log_file.name
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal
            print(f"[LOG] Member list saved to {log_path}")
        return


    planner_config = PLANNER_CONFIGS.get(base_planner, PLANNER_CONFIGS['birrt'])
    problem.planner_config = planner_config
    problem.base_planner = base_planner
    problem.motion_fallback = motion_fallback
    problem.ik_solver = ik_solver
    print(f"[PLANNER] Primary base motion planner: {base_planner} "
          f"(fallback={'on' if motion_fallback else 'off'}) | config: {planner_config}")
    print(f"[IK] Solver: {ik_solver}")
    print(f"[TASK] FastDownward search config: {task_planner}")
    print(f"[SCALE] structure copies: {scale}")
    print(f"[MPPI] arm-trajectory optimization: {'on' if use_mppi else 'off'}")
    print(f"[CLUTTER] ground obstacles: {clutter}")

    def emit_result(status, cost=None, runtime=None, metrics=None):


        parts = [f"planner={planner_name}", f"ik={ik_solver}", f"task={task_planner}",
                 f"scale={scale}", f"mppi={'on' if use_mppi else 'off'}", f"clutter={clutter}",
                 f"seed={seed}", f"status={status}"]
        if runtime is not None:
            parts.append(f"runtime={runtime:.3f}")
        if cost is not None:
            parts.append(f"cost={float(cost):.3f}")
        if metrics is not None:
            parts.append(f"moves={metrics['moves']}")
            parts.append(f"picks={metrics['picks']}")
            parts.append(f"places={metrics['places']}")
            parts.append(f"move_cost={metrics['move_cost']:.3f}")
            parts.append(f"actual_xy={metrics['actual_xy']:.3f}")
            parts.append(f"actual_full={metrics['actual_full']:.3f}")
            parts.append(f"actual_cost={metrics['actual_cost']:.3f}")
            parts.append(f"disconnects={metrics['disconnects']}")
            for key in ('arm_approach_distance', 'arm_retreat_distance', 'arm_joint_distance',
                        'arm_cost_weight', 'arm_motion_cost', 'actual_cost_with_arm'):
                parts.append('{}={:.6f}'.format(key, metrics[key]))
        run_stats.status = status
        print("[RESULT] " + " ".join(parts))


    saver = WorldSaver()

    def solve_one(prob):
        pp = pddlstream_from_problem(
            prob, collisions=True, pose_collisions=True, motion_collisions=True,
            teleport=False, pre_sample_poses=True, max_poses_per_obj=int(os.environ.get("EXP_POSES", "4")),
            max_attempts_per_obj=500, fallback_pose_sampling=True,
            fallback_max_poses=10, fallback_max_attempts=200)
        print('[COLLISION CHECK ORDER] {}'.format(collision_check_order))
        # Anytime bounds use symbolic cost; trajectory metrics are reporting only.
        success_cost = 0. if search_mode == 'anytime' else float('inf')
        print('[SEARCH CONFIG] unit_costs={} algorithm={} success_cost={} '
              'min_iterations={} improvement_bound=best_symbolic_plan_cost '
              'actual_path_cost_used_as_bound=False'.format(
                  os.environ.get('EXP_UNIT', '0') == '1',
                  os.environ.get('EXP_ALGORITHM', 'adaptive'), success_cost, min_iterations))
        tracked_solve = run_stats.track('tamp_search', success=lambda result: result[0] is not None)(solve)
        sol = tracked_solve(pp, max_time=max_time, success_cost=success_cost, verbose=True, unit_costs=(os.environ.get("EXP_UNIT", "0") == "1"),
                    algorithm=os.environ.get("EXP_ALGORITHM", "adaptive"),
                    effort_weight=(None if os.environ.get("EXP_EFFORT", "1") == "none" else float(os.environ.get("EXP_EFFORT", "1"))), planner=task_planner,
                    reorder=('early-collision' if collision_check_order == 'early' else True),


                    search_sample_ratio=float(os.environ.get("EXP_RATIO", "0.5")), max_planner_time=10, max_skeletons=int(os.environ.get("EXP_SKELETONS", "8")), min_iterations=min_iterations)
        return sol


    robot = problem.robot
    problem.initial_robot_state = {
        'joint_names': get_joint_names(robot, get_movable_joints(robot)),
        'joint_positions': list(get_joint_positions(robot, get_movable_joints(robot))),
        'base_pose': {'position': list(get_pose(robot)[0]), 'orientation': list(get_pose(robot)[1])}}
    solve_start = time.time()
    staged_executed = False
    if staged and len(getattr(problem, 'group_order', ())) > 1:
        plan = []
        cost = 0.0
        stage_commands = []
        for gi, group in enumerate(problem.group_order):
            run_stats.set_stage(group)
            view = stage_problem_view(problem, group)
            print(f"[STAGE {gi + 1}/{len(problem.group_order)}] group '{group}': "
                  f"{len(view.movable)} movable, {len(view.fixed)} fixed obstacles, "
                  f"{len(view.connections)} connections")


            stage_saver = WorldSaver()


            with LockRenderer():
                try:
                    sol = solve_one(view)
                finally:
                    stage_saver.restore()
                    run_stats.emit(stage=group)
            print_solution(sol)
            sub_plan, sub_cost, _ = sol
            if sub_plan is None:
                print(f"[STAGE {gi + 1}] No plan found for group '{group}'")
                plan = None
                break
            plan += sub_plan
            if sub_cost is not None:
                cost += float(sub_cost)


            cmds = post_process_with_disconnect(view, sub_plan, teleport=False)
            try:
                apply_checked_commands(problem, State(), cmds, time_step=(0.02 if (has_gui() and not batch) else 0.0),
                                       pause=False)
            except RuntimeError:

                rejected_log = sys.stdout.log_file.name.replace('.log', '_rejected_stage_{}.log'.format(gi + 1))
                save_plan_json(problem, plan, rejected_log, cost, planner_name, seed=seed)
                raise
            for action, args in sub_plan:
                if action == 'place':
                    _, body, target_pose, _, _, _ = args[:6]
                    actual = get_pose(body)
                    error = float(np.linalg.norm(np.asarray(actual[0]) - np.asarray(target_pose.value[0])))
                    dot = abs(float(np.dot(actual[1], target_pose.value[1])))
                    angle = 2.0 * float(np.arccos(np.clip(dot, 0.0, 1.0)))
                    print('[EXECUTION CHECK] {} position_error={:.6f} orientation_error={:.6f}'.format(
                        problem.body_names.get(body, body), error, angle))
                    if error > 0.02 or angle > 0.3:
                        raise RuntimeError('Executed placement does not match the planned pose')
            stage_commands += cmds


            stage_log = sys.stdout.log_file.name.replace('.log', '_partial_stage_{}.log'.format(gi + 1))
            save_plan_json(problem, plan, stage_log, cost, planner_name, seed=seed)
        staged_executed = True
    else:
        solution = solve_one(problem)
        print_solution(solution)
        plan, cost, _ = solution
    solve_runtime = time.time() - solve_start
    if cost is not None:
        print("[SEARCH] current cost={:.3f}".format(float(cost)))

    if plan is None:
        print("No plan found")
        emit_result('fail', runtime=solve_runtime)
        disconnect()
        if isinstance(sys.stdout, TeeLogger):
            log_path = sys.stdout.log_file.name
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal
            print(f"[LOG] Results saved to {log_path}")
        return

    def print_plan_analytics(plan):
        num_moves = 0
        num_picks = 0
        num_places = 0
        num_disconnects = 0
        move_cost = 0.0
        actual_xy_total = 0.0
        actual_full_total = 0.0
        for action in plan:
            if not action:
                continue
            if len(action) == 2 and isinstance(action[1], (tuple, list)):
                name, args = action[0], action[1]
            else:
                name, args = action[0], action[1:]
            if name == 'move_base' and len(args) >= 3:
                num_moves += 1
                move_cost += float(distance_fn(args[0], args[1]))
                traj = args[2].commands[0]
                actual_xy_total += traj.distance(distance_fn=lambda q1, q2: get_distance(q1[:2], q2[:2]))
                actual_full_total += traj.distance()
            elif name == 'pick':
                num_picks += 1
            elif name == 'place':
                num_places += 1
            elif name == 'disconnect':
                num_disconnects += 1
        pick_cost = float(num_picks)
        place_cost = float(num_places)


        unit_actions = float(num_picks + num_places + num_disconnects)
        actual_cost = actual_xy_total + unit_actions
        arm_metrics = arm_motion_metrics(plan, weight=arm_cost_weight)
        actual_cost_with_arm = actual_cost + arm_metrics['arm_motion_cost']
        avg_move = 0.0 if num_moves == 0 else move_cost / float(num_moves)
        avg_actual_xy = 0.0 if num_moves == 0 else actual_xy_total / float(num_moves)
        avg_actual_full = 0.0 if num_moves == 0 else actual_full_total / float(num_moves)
        print("[PLAN ANALYTICS] moves={} picks={} places={} disconnects={}".format(
            num_moves, num_picks, num_places, num_disconnects))
        print("[PLAN ANALYTICS] move_cost(proxy)={:.3f} actual_xy={:.3f}m actual_full={:.3f} actual_cost={:.3f}".format(
            move_cost, actual_xy_total, actual_full_total, actual_cost))
        print("[PLAN ANALYTICS] avg_move_cost={:.3f} avg_actual_xy={:.3f}m avg_actual_full={:.3f}".format(
            avg_move, avg_actual_xy, avg_actual_full))
        print('[ARM COST] joint_space=euclidean units=rad includes=approach+retreat '
              'gripper_motion_included=False base_heading_included=False '
              'arm_joint_distance={:.6f} arm_cost_weight={:.6f} '
              'actual_cost_with_arm={:.6f} reporting_only=True'.format(
                  arm_metrics['arm_joint_distance'], arm_cost_weight, actual_cost_with_arm))
        return {**arm_metrics, 'actual_cost_with_arm': actual_cost_with_arm,
                'moves': num_moves, 'picks': num_picks, 'places': num_places,
                'disconnects': num_disconnects, 'move_cost': move_cost,
                'actual_xy': actual_xy_total, 'actual_full': actual_full_total,
                'actual_cost': actual_cost}


    if batch and not staged_executed:
        saver.restore()
        commands = post_process_with_disconnect(problem, plan, teleport=False)
        try:
            apply_checked_commands(problem, State(), commands, time_step=0., pause=False)
        except RuntimeError:
            rejected_log = sys.stdout.log_file.name.replace('.log', '_rejected_execution.log')
            save_plan_json(problem, plan, rejected_log, cost, planner_name, seed=seed)
            raise
    metrics = print_plan_analytics(plan)
    problem.plan_metrics = metrics
    emit_result('success', cost=cost, runtime=solve_runtime, metrics=metrics)

    log_path = sys.stdout.log_file.name if isinstance(sys.stdout, TeeLogger) else None
    save_plan_json(problem, plan, log_path, cost, planner_name, seed=seed)

    if batch:
        print("[BATCH] SUCCESS")
        disconnect()
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal
            print(f"[LOG] Results saved to {log_path}")
        return

    if staged_executed:

        disconnect()
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal
        return

    commands = post_process_with_disconnect(problem, plan, teleport=False)
    saver.restore()

    if not has_gui():
        disconnect()
        return

    step_time = 0.02
    command_pause = True
    if command_pause:
        print("[DEMO] Press the GUI 'continue' prompt to advance each command.")
    apply_checked_commands(problem, State(), commands, time_step=step_time, pause=command_pause)
    if hasattr(problem, 'ik_stats'):
        attempts = problem.ik_stats.get('attempts', 0)
        success = problem.ik_stats.get('success', 0)
        rate = 0.0 if attempts == 0 else float(success) / float(attempts)
        print("[IK STATS] attempts: {} success: {} rate: {:.3f}".format(attempts, success, rate))


    if hasattr(problem, 'collision_test_stats'):
        print("[COLLISION STATS] traj-pose tests: {} failures: {} rate: {:.3f}".format(
            problem.collision_test_stats['traj_pose_tests'],
            problem.collision_test_stats['traj_pose_failures'],
            problem.collision_test_stats['traj_pose_failures'] / max(1, problem.collision_test_stats['traj_pose_tests'])
        ))

    time.sleep(2.0)
    disconnect()
    if isinstance(sys.stdout, TeeLogger):
        log_path = sys.stdout.log_file.name
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal
        print(f"[LOG] Results saved to {log_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: exit immediately after plan analytics without running the demo')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible runs (seeds both random and numpy)')
    base_choices = list(PLANNER_CONFIGS.keys()) + ['reuse_prm']
    parser.add_argument('--base-planner', dest='base_planner', type=str, default=None,
                        choices=base_choices,
                        help='Primary base motion planner, used STANDALONE (independent). '
                             'birrt/rrt_star/prm/astar = library planners; '
                             "'reuse_prm' = opt-in reusable roadmap. (default: birrt)")
    parser.add_argument('--planner', type=str, default=None, choices=list(PLANNER_CONFIGS.keys()),
                        help='Alias for --base-planner (kept for old scripts).')
    parser.add_argument('--motion-fallback', dest='motion_fallback', action='store_true',
                        help='Only for reuse_prm: fall back to BiRRT on unconnectable queries. '
                             'Leave OFF for a clean comparison (default: off).')
    parser.add_argument('--ik-solver', dest='ik_solver', type=str, default='pybullet',
                        choices=['pybullet', 'pybullet_nullspace'],
                        help='IK calculator for arm configs (default: pybullet). '
                             "'pybullet_nullspace' adds joint limits + rest-pose bias.")
    parser.add_argument('--task-planner', dest='task_planner', type=str, default='ff-astar2',
                        choices=['ff-astar2', 'ff-lazy', 'ff-eager', 'cea-wastar2', 'dijkstra'],
                        help='FastDownward search config for task planning '
                             '(default: ff-astar2). Satisficing: ff-astar2/ff-lazy/ff-eager/'
                             'cea-wastar2; optimal (uniform-cost): dijkstra. '
                             '(lmcut/hmax dropped: they do not support this domain\'s axioms.)')
    parser.add_argument('--scale', type=int, default=1,
                        help='Number of copies of the imported IFC scene for a scaling '
                             'experiment. 1 = the IFC model once (default).')
    parser.add_argument('--search-mode', choices=['first', 'anytime'], default='first',
                        help='Stop at first solution, or retain and seek improvements within the search budget.')
    parser.add_argument('--min-iterations', type=int, default=0,
                        help='Minimum total search iterations; prefer anytime mode for budgeted improvement.')
    parser.add_argument('--arm-cost-weight', type=float, default=1.,
                        help='Reporting only: score units per radian of arm joint-space '
                             'path length, including approach and retreat (default 1).')
    parser.add_argument('--max-time', dest='max_time', type=int, default=600,
                        help="Planner's own time budget in seconds (default 600). "
                             'Raise for larger scales; keep the OS timeout above it.')
    parser.add_argument('--mppi', type=str, default='on', choices=['on', 'off'],
                        help='MPPI arm-trajectory optimization (default on).')
    parser.add_argument('--clutter', type=int, default=0,
                        help='Number of static ground obstacles in the navigation '
                             'corridor (default 0). Same fixed layout across seeds.')
    parser.add_argument('--ifc', dest='ifc_path', type=str,
                        default=None,
                        help='IFC model consumed by IFC2SIM. Provide --ifc or --scene.')
    parser.add_argument('--scene', dest='scene_path', type=str, default=None,
                        help='IFC2SIM JSON manifest. When supplied, DeconTAMP does not parse IFC.')
    parser.add_argument('--require-ifc-ranks', action='store_true',
                        help='Require Pset_DeconTAMP.RemovalGroup and RemovalRank on every '
                             'structural IFC member; do not fall back to name/type inference.')
    parser.add_argument('--select-ifc-types', action='store_true',
                        help='Interactively choose whole IFC/Revit component types before building the scene.')
    parser.add_argument('--view-scene', dest='view_scene', action='store_true',
                        help='Show the built scene in the GUI and print a member '
                             'table (name/group/position/size), then skip planning. '
                             'Press Enter to close. Needs a display.')
    parser.add_argument('--exclude-groups', dest='exclude_groups', type=str, default=None,
                        help='Comma-separated member groups to drop from the IFC scene '
                             '(e.g. "blocking"). Groups: blocking, joist, beam, column.')
    parser.add_argument('--staged', dest='staged', action='store_true',
                        help='Plan each removal group in order as its own small '
                             'problem (lower groups kept as fixed obstacles). Splits '
                             'a large disassembly into fast per-group sub-plans.')
    parser.add_argument('--collision-check-order', choices=('early', 'auto'), default='early',
                        help='Run collision tests as soon as dependencies are ready (early, default), '
                             'or use the previous automatic cost-based stream ordering (auto).')
    args = parser.parse_args()

    base_planner = args.base_planner or args.planner or 'birrt'
    selected_guids = None
    main(batch=args.batch, seed=args.seed, base_planner=base_planner,
         motion_fallback=args.motion_fallback, ik_solver=args.ik_solver,
         task_planner=args.task_planner, scale=args.scale, max_time=args.max_time,
         use_mppi=(args.mppi == 'on'), clutter=args.clutter, ifc_path=args.ifc_path,
         require_ifc_ranks=args.require_ifc_ranks,
         selected_guids=selected_guids, select_ifc_types=args.select_ifc_types,
         scene_path=args.scene_path,
         view_scene=args.view_scene,
         exclude_groups=(set(g.strip() for g in args.exclude_groups.split(','))
                         if args.exclude_groups else None),
         staged=args.staged, collision_check_order=args.collision_check_order,
         arm_cost_weight=args.arm_cost_weight, search_mode=args.search_mode,
         min_iterations=args.min_iterations)
