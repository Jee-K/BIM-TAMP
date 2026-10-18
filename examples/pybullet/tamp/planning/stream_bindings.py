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


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
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
from planning.problem import build_initial_facts

def pddlstream_from_problem(problem, base_limits=None, collisions=False, teleport=False,
                            pose_collisions=None, motion_collisions=None,
                            pre_sample_poses=True, max_poses_per_obj=6, max_attempts_per_obj=200,
                            fallback_pose_sampling=True, fallback_max_poses=3, fallback_max_attempts=50):


    if pose_collisions is None:
        pose_collisions = collisions
    if motion_collisions is None:
        motion_collisions = collisions
    robot = problem.robot
    constant_map, init, placement_slots, group_map, default_group = build_initial_facts(problem)


    # Stream evaluation mutates poses; restore the expected state between calls.
    initial_obj_poses = {obj: get_pose(obj) for obj in problem.movable}


    run_stats = get_run_statistics()
    rest_arm_confs = {a: get_joint_positions(robot, get_arm_joints(robot, a)) for a in ARM_NAMES}

    def restore_movable_poses():
        for obj in problem.movable:
            set_pose(obj, initial_obj_poses[obj])

    custom_limits = {}
    if base_limits is not None:
        custom_limits.update(get_custom_limits(robot, problem.base_limits))

    def simple_grasp_gen(body):


        dims = object_dimensions(body)
        base_grasp = get_grasp_gen(problem, collisions=False)(body)[0][0]
        grasps = []
        for kind, o_T_g, o_T_a, width in build_member_grasps(dims):
            g = copy.copy(base_grasp)
            g.grasp_type = kind
            g.value = invert(o_T_g)
            g.approach = invert(o_T_a)
            g.grasp_width = hande_q_for_width(width)
            grasps.append((g,))
        if not grasps:
            print('[GRASP] WARNING: no grasp fits {} (dims {}); widest gap is {:.3f} m'.format(
                getattr(problem, 'body_names', {}).get(body, body), np.round(dims, 3), HANDE_MAX_GAP))
        return grasps

    @run_stats.track(lambda *args, **kw: 'arm_path_' + kw.get('phase', 'unknown'))
    def checked_arm_path(arm, collision_fn, approach_conf, contact_conf,
                         target=None, grasp=None, target_pose=None, extra_obstacles=(), phase='unknown'):
        """Plan transit -> approach -> contact, checking empty and carried phases.

        The reversed path returns to the SAME transit posture before base travel.
        Other movable poses are certified separately at their action-time poses.
        """
        from motion_planners.rrt_connect import birrt
        from examples.pybullet.utils.pybullet_tools.utils import get_sample_fn, get_distance_fn
        joints = get_arm_joints(robot, arm)
        start = rest_arm_confs[arm]
        extend = get_extend_fn(robot, joints, resolutions=[0.035] * len(joints))
        base_link = link_from_name(robot, 'base_link')
        moving = get_moving_links(robot, joints)
        fingers = get_gripper_finger_links(robot)
        obstacles = list(problem.fixed) + list(extra_obstacles)

        def collides(q):
            if target is not None:
                set_pose(target, target_pose)
            if collision_fn(q):
                return True
            if any(pairwise_link_collision(robot, link, robot, base_link) for link in moving):
                return True
            if any(pairwise_collision(robot, b) for b in extra_obstacles):
                return True
            if target is not None:
                try:
                    tool = get_link_pose(robot, get_gripper_link(robot, arm))
                    set_pose(target, multiply(tool, grasp.value))
                    if any(pairwise_collision(target, b) for b in obstacles if b != target):
                        return True
                    if any(pairwise_link_collision(robot, link, target)
                           for link in [-1] + get_links(robot) if link not in fingers):
                        return True
                finally:
                    set_pose(target, target_pose)
            return False


        local = [tuple(approach_conf)] + list(extend(approach_conf, contact_conf))
        if any(collides(q) for q in local):
            checked_arm_path.failure_reason = 'local_approach_collision'
            return None
        transit = birrt(tuple(start), tuple(approach_conf), get_distance_fn(robot, joints),
            get_sample_fn(robot, joints), extend, collides,
            max_iterations=60, max_time=0.15, restarts=0, smooth=0)
        if transit is None:
            checked_arm_path.failure_reason = 'transit_search_failed'
            return None

        path = [transit[0]]
        for q in transit[1:] + local[1:]:
            path.extend(extend(path[-1], q))
        if any(collides(q) for q in path):
            checked_arm_path.failure_reason = 'dense_path_collision'
            return None
        set_joint_positions(robot, joints, contact_conf)
        return path

    def minimal_ik_gen(problem):
        debug_state = {'count': 0}
        if not hasattr(problem, 'ik_stats'):
            problem.ik_stats = {'attempts': 0, 'success': 0}

        ik_solver = getattr(problem, 'ik_solver', 'pybullet')
        ns_args = nullspace_ik_args(problem.robot) if ik_solver == 'pybullet_nullspace' else None
        base_limits = problem.base_limits
        x_limits = (float(base_limits[0][0]), float(base_limits[1][0]))
        y_limits = (float(base_limits[0][1]), float(base_limits[1][1]))


        SHOULDER_FORWARD = 0.298
        REACH_MIN, REACH_MAX = 0.20, 0.80

        def iter_base_candidates(gripper_pose):
            """Base poses sampled around the GRIPPER TARGET (as upstream
            PDDLStream's IR sampler does), not around the object centre.

            The shoulder is placed within arm reach of arm_tool0's target and
            the base faces it. For side grasps the fingers point along the
            approach axis, so the arm must come from behind the gripper: the
            shoulder is sampled within +-70 deg of the reverse approach direction.
            Top grasps (vertical approach) sample the full circle.
            """
            target_xy = np.array(gripper_pose[0][:2])
            fingers = np.array(matrix_from_quat(gripper_pose[1])) @ HANDE_APPROACH_AXIS
            horiz = fingers[:2]
            if np.linalg.norm(horiz) > 0.5:
                back = float(np.arctan2(-horiz[1], -horiz[0]))
                spread = np.radians(70)
            else:
                back, spread = 0.0, np.pi
            while True:
                d = float(np.random.uniform(REACH_MIN, REACH_MAX))
                phi = back + float(np.random.uniform(-spread, spread))
                shoulder = target_xy + d * np.array([np.cos(phi), np.sin(phi)])
                yaw = float(np.arctan2(target_xy[1] - shoulder[1], target_xy[0] - shoulder[0]))
                yaw += float(np.random.uniform(-0.3, 0.3))
                base_xy = shoulder - SHOULDER_FORWARD * np.array([np.cos(yaw), np.sin(yaw)])
                if not (x_limits[0] <= base_xy[0] <= x_limits[1]):
                    continue
                if not (y_limits[0] <= base_xy[1] <= y_limits[1]):
                    continue
                yield base_xy, yaw

        def gen(arm, obj, pose, grasp):
            robot = problem.robot
            saver = BodySaver(robot)
            obj_name = getattr(problem, 'body_names', {}).get(obj, get_body_name(obj))

            if debug_state['count'] < 5:
                print(f"[IK DEBUG] generator called for obj {obj} ({obj_name})")

            arm_link = get_gripper_link(robot, arm)
            arm_joints = get_arm_joints(robot, arm)
            base_joints = get_group_joints(robot, 'base')


            restore_movable_poses()


            other_movable = []
            is_place = (obj in initial_obj_poses and
                        np.linalg.norm(np.array(pose.value[0]) - np.array(initial_obj_poses[obj][0])) > 0.01)


            base_obstacles = list(problem.fixed) + [obj]
            print(f"[IK] {'PLACE' if is_place else 'PICK'} mode for {obj_name}: "
                  f"checking fixed obstacles + target; other members checked per plan step")


            stats = {
                'total_attempts': 0,
                'base_collision': 0,
                'ik_nan': 0,
                'arm_collision': 0,
                'arm_base_collision': 0,
                'ee_error': 0,
                'success': 0
            }
            collision_fn = make_grasp_collision_fn(
                robot, arm_joints, fixed_obstacles=problem.fixed,
                other_movable=other_movable, target=obj, self_collisions=True)
            base_link = link_from_name(robot, 'base_link')
            arm_links = get_moving_links(robot, arm_joints)
            pose.assign()
            obj_pos = np.array(pose.value[0])
            obj_xy = obj_pos[:2]
            lower, upper = get_aabb(obj)
            obj_height = float(upper[2] - lower[2])
            obj_top_z = float(obj_pos[2]) + obj_height / 2.0
            grasp_type = getattr(grasp, 'grasp_type', 'top')


            gripper_pose = multiply(pose.value, invert(grasp.value))
            approach_pose = multiply(pose.value, invert(grasp.approach))
            gripper_rot = np.array(matrix_from_quat(gripper_pose[1]))


            relinquish_after = 150
            place_max_attempts = 900
            success_cap = int(os.environ.get("EXP_IK_CAP", "3"))
            since_yield = 0
            rest_conf = rest_arm_confs[arm]
            for base_xy, yaw in iter_base_candidates(gripper_pose):
                stats['total_attempts'] += 1
                since_yield += 1
                if stats['success'] == 0 and stats['total_attempts'] > place_max_attempts:
                    print(f"[IK] {obj_name}: {'PLACE' if is_place else 'PICK'} gave up after {place_max_attempts} tries "
                          f"(base_coll={stats['base_collision']} arm_coll={stats['arm_collision']} "
                          f"ee_err={stats['ee_error']})")
                    return
                if since_yield >= relinquish_after:
                    since_yield = 0
                    print(f"[IK] {obj_name}: {stats['total_attempts']} tries, "
                          f"{stats['success']} ok (base_coll={stats['base_collision']} "
                          f"arm_coll={stats['arm_collision']} arm_base={stats['arm_base_collision']} "
                          f"ee_err={stats['ee_error']}) -> relinquish")
                    yield None
                    saver.restore()
                    restore_movable_poses()

                phase = 'place' if is_place else 'pick'
                sample_ticket = run_stats.begin(phase + '_base_candidates')
                dist_from_base = np.linalg.norm(obj_xy - base_xy)

                saver.restore()
                pose.assign()
                bq = Conf(robot, base_joints, (base_xy[0], base_xy[1], yaw))
                if base_joints:
                    bq.assign()
                else:
                    set_base_values(robot, bq.values)


                set_joint_positions(robot, arm_joints, rest_conf)


                base_in_collision = False
                colliding_obj = None
                for obs in base_obstacles:
                    if pairwise_collision(robot, obs, max_distance=(-0.001 if obs in problem.surfaces else 0.005)):
                        stats['base_collision'] += 1
                        base_in_collision = True
                        colliding_obj = obs
                        break
                if base_in_collision:
                    run_stats.end(sample_ticket, 'environment_collision')
                    if stats['total_attempts'] <= 5 or stats['total_attempts'] % 50 == 0:

                        colliding_name = getattr(problem, 'body_names', {}).get(colliding_obj, get_body_name(colliding_obj))
                        print(f"[IK] Attempt {stats['total_attempts']} for {obj_name}: base at ({base_xy[0]:.2f}, {base_xy[1]:.2f}) collides with {colliding_name}")
                    continue

                run_stats.end(sample_ticket, 'accepted', True)
                endpoint_ticket = run_stats.begin(phase + '_ik_endpoints')
                ik_grasp = calc_ik(robot, arm_link, gripper_pose[0], gripper_pose[1],
                                   ik_solver, ns_args)
                ik_approach = calc_ik(robot, arm_link, approach_pose[0], approach_pose[1],
                                      ik_solver, ns_args)
                movable_joints = [j for j in range(p.getNumJoints(robot))
                                  if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
                if len(ik_grasp) == len(movable_joints):
                    movable_index = {j: i for i, j in enumerate(movable_joints)}
                    grasp_conf = [ik_grasp[movable_index[j]] for j in arm_joints]
                    approach_conf = [ik_approach[movable_index[j]] for j in arm_joints]
                else:
                    grasp_conf = [ik_grasp[j] for j in arm_joints]
                    approach_conf = [ik_approach[j] for j in arm_joints]

                problem.ik_stats['attempts'] += 1
                if any(np.isnan(q) for q in grasp_conf) or any(np.isnan(q) for q in approach_conf):
                    stats['ik_nan'] += 1
                    run_stats.end(endpoint_ticket, 'nonfinite_solution')
                    continue

                if collision_fn(approach_conf) or collision_fn(grasp_conf):
                    stats['arm_collision'] += 1
                    run_stats.end(endpoint_ticket, collision_fn.last_failure or 'collision')
                    continue

                if any(pairwise_link_collision(robot, link, robot, base_link) for link in arm_links):
                    stats['arm_base_collision'] += 1
                    run_stats.end(endpoint_ticket, 'arm_base_collision')
                    continue

                set_joint_positions(robot, arm_joints, grasp_conf)
                ee_pos, ee_quat = get_link_pose(robot, arm_link)
                ee_err = float(np.linalg.norm(np.array(ee_pos) - np.array(gripper_pose[0])))

                rel = gripper_rot.T @ np.array(matrix_from_quat(ee_quat))
                orn_err = float(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)))


                if ee_err > 0.002 or orn_err > GRASP_ORN_TOL:
                    stats['ee_error'] += 1
                    run_stats.end(endpoint_ticket, 'pose_error')
                    continue
                run_stats.end(endpoint_ticket, 'accepted', True)


                if not is_place and stats['total_attempts'] < 450:
                    blocked = False
                    for conf in (rest_conf, approach_conf, grasp_conf):
                        set_joint_positions(robot, arm_joints, conf)
                        if any(pairwise_collision(robot, b) for b in problem.movable if b != obj):
                            blocked = True
                            break
                    if blocked:
                        continue
                    set_joint_positions(robot, arm_joints, grasp_conf)


                if debug_state['count'] < 3:
                    print("[IK DEBUG] arm_link:", arm_link, "arm_joints:", arm_joints)
                    print("[IK DEBUG] obj_pos:", obj_pos)
                    print("[IK DEBUG] base:", bq.values, "ee_err:", ee_err)
                    print("[IK DEBUG] grasp_pose:", gripper_pose[0], gripper_pose[1])
                    print("[IK DEBUG] approach_pose:", approach_pose[0], approach_pose[1])
                    print("[IK DEBUG] ik_grasp_len:", len(ik_grasp), "ik_approach_len:", len(ik_approach))
                    print("[IK DEBUG] grasp_conf:", grasp_conf)
                    print("[IK DEBUG] approach_conf:", approach_conf)
                    debug_state['count'] += 1


                extra = []
                if not is_place and stats['total_attempts'] < 450:
                    extra = [b for b in problem.movable if b != obj]


                slot_obstacles = []
                try:
                    if is_place and stats['total_attempts'] < 450:
                        for cell in placement_slots.values():
                            if cell['group'] != pose.placement_group or cell['rank'] >= pose.placement_rank:
                                continue
                            obstacle = create_box(*cell['envelope'], color=(0., 0., 0., 0.))
                            slot_obstacles.append(obstacle)
                            p.changeDynamics(obstacle, -1, collisionMargin=0.)
                            set_pose(obstacle, ((cell['center'][0], cell['center'][1],
                                cell['top'] + cell['envelope'][2]/2), (0., 0., 0., 1.)))
                    path = checked_arm_path(arm, collision_fn, approach_conf, grasp_conf,
                        target=obj, grasp=grasp, target_pose=pose.value,
                        extra_obstacles=extra + slot_obstacles, phase=phase)
                finally:
                    for obstacle in slot_obstacles:
                        remove_body(obstacle)
                if path is None:
                    stats['arm_collision'] += 1
                    continue
                traj = create_trajectory(robot, arm_joints, path)
                cmd = Commands(State(), savers=[BodySaver(robot)], commands=[traj])
                cmd.target = obj
                cmd.grasp = grasp
                cmd.arm = arm
                problem.ik_stats['success'] += 1
                stats['success'] += 1
                print(f"[IK SUCCESS] obj {obj} ({obj_name}) [{grasp_type} grasp] - found after {stats['total_attempts']} attempts")
                print(f"[IK SUCCESS] Base position: ({base_xy[0]:.2f}, {base_xy[1]:.2f}, {yaw:.2f}), distance from obj: {dist_from_base:.2f}")
                since_yield = 0
                yield (bq, cmd)
                saver.restore()
                restore_movable_poses()


                if stats['success'] >= success_cap:
                    return
        return gen

    def touch_ik_gen(problem):
        """IK for the disconnect action: drive the EE to touch a connection point.

        Mirrors minimal_ik_gen but the target is a fixed world point (the joint)
        rather than a grasp on an object. The two bodies the connection joins are
        excluded from arm-collision checks, since touching them is the whole point;
        every other body (and the base) must still be collision-free.
        """
        ik_solver = getattr(problem, 'ik_solver', 'pybullet')
        ns_args = nullspace_ik_args(problem.robot) if ik_solver == 'pybullet_nullspace' else None
        base_limits = problem.base_limits
        x_limits = (float(base_limits[0][0]), float(base_limits[1][0]))
        y_limits = (float(base_limits[0][1]), float(base_limits[1][1]))

        def iter_base_candidates(target_xy):


            current = get_group_conf(problem.robot, 'base')
            yield np.array([current[0], current[1]]), float(current[2])
            radii = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            angles = np.linspace(-np.pi, np.pi, 36, endpoint=False)
            for r in radii:
                for ang in angles:
                    base_xy = target_xy + r * np.array([np.cos(ang), np.sin(ang)])
                    if not (x_limits[0] <= base_xy[0] <= x_limits[1]):
                        continue
                    if not (y_limits[0] <= base_xy[1] <= y_limits[1]):
                        continue
                    yaw = float(np.arctan2(target_xy[1] - base_xy[1],
                                           target_xy[0] - base_xy[0]))
                    yield base_xy, yaw
            while True:
                r = np.random.uniform(0.25, 1.6)
                ang = np.random.uniform(-np.pi, np.pi)
                base_xy = target_xy + r * np.array([np.cos(ang), np.sin(ang)])
                if not (x_limits[0] <= base_xy[0] <= x_limits[1]):
                    continue
                if not (y_limits[0] <= base_xy[1] <= y_limits[1]):
                    continue
                yaw = float(np.arctan2(target_xy[1] - base_xy[1],
                                       target_xy[0] - base_xy[0]))
                yaw += float(np.random.uniform(-0.3, 0.3))
                yield base_xy, yaw

        def gen(arm, conn):
            robot = problem.robot
            saver = BodySaver(robot)
            arm_link = get_gripper_link(robot, arm)
            arm_joints = get_arm_joints(robot, arm)
            base_joints = get_group_joints(robot, 'base')
            restore_movable_poses()


            contact_bodies = set(conn.bodies)
            all_bodies = list(problem.fixed) + list(conn.bodies)
            collision_fn = make_grasp_collision_fn(
                robot, arm_joints, fixed_obstacles=list(problem.fixed),
                other_movable=[], target=list(contact_bodies), self_collisions=True)
            base_link = link_from_name(robot, 'base_link')
            arm_links = get_moving_links(robot, arm_joints)
            movable_joints = [j for j in range(p.getNumJoints(robot))
                              if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
            midx = {j: i for i, j in enumerate(movable_joints)}


            target = np.array(conn.point, dtype=float)
            target_xy = target[:2]
            touch_candidates = build_touch_poses(target)


            relinquish_after = 150


            TOUCH_SUCCESS_CAP = int(os.environ.get("EXP_TOUCH_CAP", "2"))
            attempts = 0
            successes = 0
            since_yield = 0
            for base_xy, yaw in iter_base_candidates(target_xy):
                attempts += 1
                since_yield += 1
                if since_yield >= relinquish_after:
                    since_yield = 0
                    yield None
                    saver.restore()
                    restore_movable_poses()
                sample_ticket = run_stats.begin('disconnect_base_candidates')
                dist = float(np.linalg.norm(target_xy - base_xy))
                if dist > 1.85 or dist < 0.15:
                    run_stats.end(sample_ticket, 'reach_filter')
                    continue

                saver.restore()
                bq = Conf(robot, base_joints, (base_xy[0], base_xy[1], yaw))
                if base_joints:
                    bq.assign()
                else:
                    set_base_values(robot, bq.values)


                set_joint_positions(robot, arm_joints, rest_arm_confs[arm])
                if any(pairwise_collision(robot, obs, max_distance=(-0.001 if obs in problem.surfaces else 0.005)) for obs in all_bodies):
                    run_stats.end(sample_ticket, 'environment_collision')
                    continue
                run_stats.end(sample_ticket, 'accepted', True)


                for dir_name, w_T_g, w_T_a in touch_candidates:
                    endpoint_ticket = run_stats.begin('disconnect_ik_endpoints')
                    ik_touch = calc_ik(robot, arm_link, w_T_g[0], w_T_g[1], ik_solver, ns_args)
                    ik_appr = calc_ik(robot, arm_link, w_T_a[0], w_T_a[1], ik_solver, ns_args)
                    if len(ik_touch) == len(movable_joints):
                        touch_conf = [ik_touch[midx[j]] for j in arm_joints]
                        appr_conf = [ik_appr[midx[j]] for j in arm_joints]
                    else:
                        touch_conf = [ik_touch[j] for j in arm_joints]
                        appr_conf = [ik_appr[j] for j in arm_joints]
                    if any(np.isnan(q) for q in touch_conf) or any(np.isnan(q) for q in appr_conf):
                        run_stats.end(endpoint_ticket, 'nonfinite_solution')
                        continue
                    if collision_fn(appr_conf) or collision_fn(touch_conf):
                        run_stats.end(endpoint_ticket, collision_fn.last_failure or 'collision')
                        continue
                    if any(pairwise_link_collision(robot, link, robot, base_link) for link in arm_links):
                        run_stats.end(endpoint_ticket, 'arm_base_collision')
                        continue
                    set_joint_positions(robot, arm_joints, touch_conf)
                    ee_pos = p.getLinkState(robot, arm_link)[4]
                    if float(np.linalg.norm(np.array(ee_pos) - np.array(w_T_g[0]))) > 0.01:
                        run_stats.end(endpoint_ticket, 'pose_error')
                        continue
                    run_stats.end(endpoint_ticket, 'accepted', True)


                    if attempts < 900:
                        other_members = [b for b in problem.movable if b not in contact_bodies]
                        blocked = False
                        for conf in (rest_arm_confs[arm], appr_conf, touch_conf):
                            set_joint_positions(robot, arm_joints, conf)
                            if any(pairwise_collision(robot, b) for b in other_members):
                                blocked = True
                                break
                        if blocked:
                            continue
                        set_joint_positions(robot, arm_joints, touch_conf)
                    extra = ([b for b in problem.movable if b not in contact_bodies]
                             if attempts < 900 else [])
                    path = checked_arm_path(arm, collision_fn, appr_conf, touch_conf,
                                            extra_obstacles=extra, phase='disconnect')
                    if path is None:
                        continue
                    traj = create_trajectory(robot, arm_joints, path)
                    cmd = Commands(State(), savers=[BodySaver(robot)], commands=[traj])
                    cmd.target = conn
                    cmd.touch_point = tuple(float(v) for v in target)
                    successes += 1
                    since_yield = 0
                    print('[TOUCH IK SUCCESS] {} via {} after {} attempts; base ({:.2f}, {:.2f}, {:.2f})'.format(
                        conn.name, dir_name, attempts, base_xy[0], base_xy[1], yaw))
                    yield (bq, cmd)
                    saver.restore()
                    restore_movable_poses()
                    if successes >= TOUCH_SUCCESS_CAP:
                        return
                    break
        return gen


    problem.collision_test_stats = {'traj_pose_tests': 0, 'traj_pose_failures': 0}


    goal_surface_map = {obj: surf for obj, surf in problem.goal_on}

    def get_cfree_traj_pose_test_with_target(robot, collisions=True):
        def test(c, b2, p2):
            if not collisions:
                return True
            target = getattr(c, 'target', None)


            state = c.assign()
            p2.assign()
            fingers = get_gripper_finger_links(robot)
            contacts = set(target.bodies) if isinstance(target, Connection) else {target}
            for _ in c.apply(state):
                state.assign()
                if b2 in contacts:
                    if any(pairwise_link_collision(robot, link, b2)
                           for link in [-1] + get_links(robot) if link not in fingers):
                        return False
                elif pairwise_collision(robot, b2):
                    return False
                if hasattr(c, 'grasp') and b2 != target:
                    old_pose = get_pose(target)
                    try:
                        tool = get_link_pose(robot, get_gripper_link(robot, c.arm))
                        set_pose(target, multiply(tool, c.grasp.value))
                        if pairwise_collision(target, b2):
                            return False
                    finally:
                        set_pose(target, old_pose)
            return True
        return test

    def ordered_pose_gen(body, surface, slot):
        cell = placement_slots[slot]
        if surface != cell['surface'] or group_map.get(body, default_group) != cell['group']:
            return
        if getattr(problem, 'original_resting_face', False):
            footprint, rotation = original_resting_orientation(object_dimensions(body))
            rotations = [rotation]
        elif cell['group'] in getattr(problem, 'upright_placement_groups', set()):
            footprint, rotations = object_dimensions(body), [np.eye(3)]
        else:
            footprint, rotations = flat_orientations(object_dimensions(body))
        x, y = cell['center']
        for rotation in rotations:
            value = ((x, y, cell['top'] + footprint[2]/2), tuple(quat_from_matrix(rotation)))
            pose = Pose(body, value, surface)
            pose.placement_rank = cell['rank']
            pose.placement_group = cell['group']
            yield (pose,)

    sample_pose_stream = from_gen_fn(ordered_pose_gen)


    def lazy_grasp_gen(body):
        grasps = simple_grasp_gen(body)
        random.shuffle(grasps)
        if object_dimensions(body)[2] > 0.8:
            center = np.mean([np.asarray(v[0][:2]) for v in initial_obj_poses.values()], axis=0)
            source = np.asarray(initial_obj_poses[body][0][:2])
            outward = source - center
            def grasp_priority(item):
                g = item[0]
                tool = multiply(initial_obj_poses[body], invert(g.value))
                offset = np.asarray(tool[0][:2]) - source
                return (g.grasp_type == 'top', -float(np.dot(offset, outward)))
            grasps.sort(key=grasp_priority)
        for grasp_tuple in grasps:
            yield grasp_tuple


    def get_prm_motion_gen(num_samples=300, k=10, use_fallback=True):


        import heapq
        robot = problem.robot
        base_joints = get_group_joints(robot, 'base')
        (xmin, ymin), (xmax, ymax) = problem.base_limits
        res = np.array([0.1, 0.1, 0.2])
        roadmap = {}

        def set_base(q):


            if base_joints:
                set_joint_positions(robot, base_joints, q)
            else:
                set_base_values(robot, q)

        def diff(q2, q1):
            return (q2[0] - q1[0], q2[1] - q1[1],
                    (q2[2] - q1[2] + np.pi) % (2 * np.pi) - np.pi)

        def dist(q1, q2):
            d = diff(q2, q1)
            return float(np.hypot(d[0], d[1]) + 0.3 * abs(d[2]))

        def extend(q1, q2):
            d = diff(q2, q1)
            n = int(np.max(np.abs(np.array(d) / res))) + 1
            for i in range(1, n + 1):
                yield (q1[0] + d[0] * i / n, q1[1] + d[1] * i / n, q1[2] + d[2] * i / n)

        def coll_fn(obstacles):
            def c(q):
                set_base(q)
                return any(pairwise_collision(robot, o) for o in obstacles)
            return c

        def edge_free(a, b, c):
            return not any(c(q) for q in extend(a, b))

        def build():
            restore_movable_poses()
            obstacles = list(problem.fixed) + list(problem.movable)
            c = coll_fn(obstacles)
            saver = BodySaver(robot)
            nodes = []
            tries = 0
            while len(nodes) < num_samples and tries < num_samples * 30:
                tries += 1
                q = (float(np.random.uniform(xmin, xmax)),
                     float(np.random.uniform(ymin, ymax)),
                     float(np.random.uniform(-np.pi, np.pi)))
                if not c(q):
                    nodes.append(q)


            n_uniform = len(nodes)
            for obj in problem.movable:
                ox, oy = get_point(obj)[:2]
                for r in (0.4, 0.55, 0.7, 0.85, 1.0):
                    for ang in np.linspace(-np.pi, np.pi, 12, endpoint=False):
                        bx, by = ox + r * np.cos(ang), oy + r * np.sin(ang)
                        if not (xmin <= bx <= xmax and ymin <= by <= ymax):
                            continue
                        q = (float(bx), float(by), float(ang + np.pi))
                        if not c(q):
                            nodes.append(q)
            saver.restore()
            xy = np.array([(q[0], q[1]) for q in nodes]) if nodes else np.zeros((0, 2))
            nbrs = []
            for i in range(len(nodes)):
                d = np.linalg.norm(xy - xy[i], axis=1)
                nbrs.append([int(j) for j in np.argsort(d)[1:k + 1]])
            roadmap.update(nodes=nodes, nbrs=nbrs, xy=xy)
            print(f"[BASE PRM] roadmap built: {len(nodes)} nodes "
                  f"({n_uniform} uniform + {len(nodes) - n_uniform} object-ring), k={k}")

        def query(start, goal, obstacles):
            c = coll_fn(obstacles)
            start, goal = tuple(start), tuple(goal)
            if c(start) or c(goal):
                return None
            nodes, nbrs, xy = roadmap['nodes'], roadmap['nbrs'], roadmap['xy']
            N = len(nodes)
            pts = nodes + [start, goal]
            S, G = N, N + 1
            connect_radius = 2.5
            cache = {}

            def valid(i, j):
                key = (i, j) if i < j else (j, i)
                if key not in cache:
                    cache[key] = edge_free(pts[i], pts[j], c)
                return cache[key]

            def nbr_idx(u):
                if u < N:
                    return list(nbrs[u])
                p = np.array([pts[u][0], pts[u][1]])
                d = np.linalg.norm(xy - p, axis=1) if N else np.zeros(0)
                return [int(j) for j in np.argsort(d)[:k]]

            h = lambda i: dist(pts[i], goal)
            openq = [(h(S), 0.0, S)]
            gsc = {S: 0.0}
            came = {}
            closed = set()
            saver = BodySaver(robot)
            found = False
            while openq:
                _, gu, u = heapq.heappop(openq)
                if u == G:
                    found = True
                    break
                if u in closed:
                    continue
                closed.add(u)
                cand = nbr_idx(u)
                if dist(pts[u], goal) <= connect_radius:
                    cand = cand + [G]
                for v in cand:
                    if v in closed:
                        continue
                    if not valid(u, v):
                        continue
                    ng = gu + dist(pts[u], pts[v])
                    if ng < gsc.get(v, float('inf')):
                        gsc[v] = ng
                        came[v] = u
                        heapq.heappush(openq, (ng + h(v), ng, v))
            saver.restore()
            if not found:
                return None
            idxs = [G]
            while idxs[-1] != S:
                idxs.append(came[idxs[-1]])
            idxs.reverse()
            cfgs = [pts[i] for i in idxs]
            dense = [cfgs[0]]
            for a, b in zip(cfgs, cfgs[1:]):
                dense.extend(extend(a, b))
            return dense


        birrt_fallback = get_motion_gen(problem, custom_limits=custom_limits,
                                        collisions=motion_collisions, teleport=teleport)

        def fn(bq1, bq2, fluents=[]):
            if not roadmap:
                build()
            held = {f[2] for f in fluents if f and f[0].lower() == 'atgrasp'}
            obstacles = list(problem.fixed) + [o for o in problem.movable if o not in held]
            saver = BodySaver(robot)
            raw = query(bq1.values, bq2.values, obstacles)
            saver.restore()
            if raw is not None:
                bt = create_trajectory(robot, base_joints, raw)
                return (Commands(State(), savers=[BodySaver(robot)], commands=[bt]),)
            if use_fallback:
                print('[BASE PRM] no roadmap path — falling back to BiRRT')
                return birrt_fallback(bq1, bq2, fluents=fluents)
            print('[BASE PRM] no roadmap path (fallback disabled)')
            return None
        return fn


    base_planner = getattr(problem, 'base_planner', 'reuse_prm')
    if motion_collisions and base_planner == 'reuse_prm':
        _raw_motion_gen = get_prm_motion_gen(
            use_fallback=getattr(problem, 'motion_fallback', False))
    else:
        _raw_motion_gen = get_motion_gen(problem, custom_limits=custom_limits,
                                         collisions=motion_collisions, teleport=teleport)
    _raw_motion_gen = run_stats.track('base_planner_attempts')(_raw_motion_gen)

    @run_stats.track('base_queries')
    def wrapped_motion_gen(bq1, bq2, fluents=[]):

        print(f"[MOTION DEBUG] fluents received: {len(fluents)}")
        for f in fluents:
            name = f[0] if f else '?'
            body_names = getattr(problem, 'body_names', {})
            if name in ('atpose', 'AtPose'):
                obj_name = body_names.get(f[1], f[1])
                print(f"  {name}({obj_name}, {f[2]})")
            elif name in ('atgrasp', 'AtGrasp'):
                obj_name = body_names.get(f[2], f[2])
                print(f"  {name}({f[1]}, {obj_name}, {f[3]})")
            else:
                print(f"  UNKNOWN fluent: {f}")


        held_objects = set()
        mentioned = set()
        for fluent in fluents:
            name = fluent[0]
            if name in ('atpose', 'AtPose'):
                obj, pose_obj = fluent[1], fluent[2]
                pose_obj.assign()
                mentioned.add(obj)
            elif name in ('atgrasp', 'AtGrasp'):
                held_objects.add(fluent[2])
                mentioned.add(fluent[2])

        for obj in problem.movable:
            if obj not in mentioned:
                set_pose(obj, initial_obj_poses[obj])


        # Retry stochastic planners; repeating the same reusable-roadmap query adds no samples.
        max_retries = 1 if getattr(problem, 'base_planner', 'birrt') == 'reuse_prm' else 3
        for attempt in range(max_retries):
            result = _raw_motion_gen(bq1, bq2, fluents=fluents)
            if result is not None:
                result[0].transit_arm_confs = {
                    a: {'joint_names': get_joint_names(robot, get_arm_joints(robot, a)),
                        'joint_positions': list(rest_arm_confs[a])} for a in ARM_NAMES}
                if attempt > 0:
                    print(f"[MOTION RETRY] Succeeded on attempt {attempt + 1}/{max_retries}")
                return result
            if attempt < max_retries - 1:
                print(f"[MOTION RETRY] Attempt {attempt + 1}/{max_retries} failed, retrying...")
        print(f"[MOTION] no base path after {max_retries} attempt(s)")
        return None

    _raw_cfree_pose_pose = get_cfree_pose_pose_test(collisions=pose_collisions)
    def wrapped_cfree_pose_pose(b1, p1, b2, p2, **kwargs):


        result = _raw_cfree_pose_pose(b1, p1, b2, p2, **kwargs)
        restore_movable_poses()
        return result

    _raw_cfree_approach_pose = get_cfree_approach_pose_test(problem, collisions=collisions)
    def wrapped_cfree_approach_pose(b1, p1, g1, b2, p2, **kwargs):


        result = _raw_cfree_approach_pose(b1, p1, g1, b2, p2, **kwargs)
        restore_movable_poses()
        return result

    _raw_cfree_traj_pose = get_cfree_traj_pose_test_with_target(robot, collisions=collisions)
    def wrapped_cfree_traj_pose(*args, **kwargs):
        result = _raw_cfree_traj_pose(*args, **kwargs)
        restore_movable_poses()
        return result

    stream_map = {
        'sample-pose': sample_pose_stream,
        'sample-grasp': from_gen_fn(lazy_grasp_gen),

        'inverse-kinematics': from_gen_fn(
            mppi_ik_gen(problem, minimal_ik_gen,
                       use_mppi=getattr(problem, 'use_mppi', True),
                       num_samples=20, horizon=5, num_iterations=2)
        ),


        'inverse-kinematics-touch': from_gen_fn(touch_ik_gen(problem)),
        'plan-base-motion': from_fn(wrapped_motion_gen),
        'test-cfree-pose-pose': from_test(run_stats.track('cfree_pose_pose', success=bool)(wrapped_cfree_pose_pose)),
        'test-cfree-approach-pose': from_test(run_stats.track('cfree_approach_pose', success=bool)(wrapped_cfree_approach_pose)),
        'test-cfree-traj-pose': from_test(run_stats.track('cfree_traj_pose', success=bool)(wrapped_cfree_traj_pose)),
        'Distance': distance_fn,
    }

    goal = ('Done',)
    goal_literals = [('On', obj, surface) for obj, surface in problem.goal_on]
    if goal_literals:
        goal = ('and',) + tuple(goal_literals)

    return PDDLProblem(DOMAIN_PDDL, constant_map, STREAM_PDDL, stream_map, init, goal)

