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


class Connection(object):
    """Geometric contact between two members, with a name and world-space point."""
    _ids = {}

    def __init__(self, name, point, bodies, marker=None):
        self.name = name
        self.point = tuple(float(c) for c in point)
        self.bodies = tuple(bodies)
        self.marker = marker

    def __repr__(self):
        return self.name


class DisconnectMarker(Command):
    """Execution-time effect of a disconnect action: flag the connection broken.

    Recolors the connection marker to grey and floats a 'DISCONNECTED' label so
    the demo visibly shows the joint has been released after the EE touches it.
    """
    def __init__(self, connection):
        self.connection = connection

    def apply(self, state, **kwargs):
        conn = self.connection
        if conn.marker is not None:
            try:
                set_color(conn.marker, (0.5, 0.5, 0.5, 1.0))
            except Exception:
                pass
        if has_gui():
            p.addUserDebugText('DISCONNECTED', textPosition=conn.point,
                               textColorRGB=(1, 0, 0), lifeTime=0)
        print('[DISCONNECT] connection {} is now DISCONNECTED'.format(conn.name))
        yield

    def control(self, **kwargs):
        pass

    def __repr__(self):
        return 'DisconnectMarker({})'.format(self.connection.name)


def compute_connection_point(body_a, body_b):
    """Centre of the region where two members' AABBs meet.

    This is where the two members meet; the EE goes here to disconnect.
    """
    la, ua = get_aabb(body_a)
    lb, ub = get_aabb(body_b)


    lo = np.maximum(la, lb)
    hi = np.minimum(ua, ub)
    return tuple(float(v) for v in (lo + hi) / 2.0)


CONTACT_MIN_OVERLAP = 0.002


def aabb_face_contact(body_a, body_b, tol=0.02):
    """True if two members meet FACE TO FACE.

    Per axis, the signed gap is max(lower) - min(upper): > 0 separated, < 0
    overlapping. The boxes must not be separated by more than `tol` on any axis,
    and must genuinely overlap on at least two axes. Two members that only share
    an edge (overlap on one axis) or a corner (none) are NOT connected -- with
    the old 'within tol on all axes' test, a corner stud touching the edge of the
    other wall's plates counted as joined to them.
    """
    la, ua = get_aabb(body_a)
    lb, ub = get_aabb(body_b)
    gaps = [max(la[i], lb[i]) - min(ua[i], ub[i]) for i in range(3)]
    if any(g > tol for g in gaps):
        return False
    return sum(1 for g in gaps if g < -CONTACT_MIN_OVERLAP) >= 2


def is_vertical_member(body):
    """True if the member's longest dimension is vertical (stud, column, post)."""
    lower, upper = get_aabb(body)
    dims = np.array(upper) - np.array(lower)
    return int(np.argmax(dims)) == 2


def detect_connections(bodies, body_names, touch_tol=0.02):
    """Create a Connection for every pair of bodies whose AABBs touch/overlap.

    Replaces the old hardcoded beam1/joist1 special case: any two members that
    physically meet get a joint the planner must `disconnect` before picking
    either one. Each joint is marked with a green sphere in the GUI.
    """
    connections = []
    ordered = list(bodies)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            if not aabb_face_contact(a, b, tol=touch_tol):
                continue


            if is_vertical_member(a) and is_vertical_member(b):
                continue
            connections.append(make_connection(a, b, body_names))
    if not connections:
        print('[CONNECTION] no touching member pairs detected')
    return connections


def make_connection(a, b, body_names, source='detected'):
    """Create a geometric contact with a visual-only marker."""
    cpoint = compute_connection_point(a, b)
    name_a = body_names.get(a, str(a))
    name_b = body_names.get(b, str(b))
    marker = create_sphere(0.035, color=(0.1, 0.9, 0.1, 1.0), collision=False)
    set_point(marker, Point(x=cpoint[0], y=cpoint[1], z=cpoint[2]))
    conn = Connection('conn_{}_{}'.format(name_a, name_b), cpoint, (a, b), marker=marker)
    print('[CONNECTION] {} between {} and {} at ({:.3f}, {:.3f}, {:.3f}) [{}]'.format(
        conn.name, name_a, name_b, cpoint[0], cpoint[1], cpoint[2], source))
    return conn


# Only fingers may contact the target; the palm and arm remain collision obstacles.
GRIPPER_LINK_NAME_FRAGMENTS = ('finger',)

_FINGER_LINK_CACHE = {}

def get_gripper_finger_links(robot):
    """Set of finger link ids; palm, coupler, and tool are not exempt.

    Identified by name fragment (Robotiq Hand-E links + the tool flange). Cached
    per robot id. These are the only links permitted to contact a grasp target.
    """
    if robot in _FINGER_LINK_CACHE:
        return _FINGER_LINK_CACHE[robot]
    finger = set()
    for link in get_links(robot):
        name = get_link_name(robot, link).lower()
        if any(frag in name for frag in GRIPPER_LINK_NAME_FRAGMENTS):
            finger.add(link)
    _FINGER_LINK_CACHE[robot] = finger
    return finger


def make_grasp_collision_fn(robot, arm_joints, fixed_obstacles, other_movable,
                            target, self_collisions=True):
    """Collision fn where only gripper fingers may contact `target`.

    - `fixed_obstacles` + `other_movable` (everything except the target) are
      checked against ALL moving arm links (standard full collision).
    - `target` is checked against only the NON-finger moving links, so the
      forearm/wrist/base passing through it is still a collision, but the
      fingers closing on it is permitted (the intended contact).
    Returns a function conf -> bool (True == in collision).
    """
    base_obstacle_fn = get_collision_fn(robot, arm_joints,
                                        obstacles=list(fixed_obstacles) + list(other_movable),
                                        self_collisions=self_collisions)
    self_pairs = get_self_link_pairs(robot, arm_joints) if self_collisions else []
    limits_fn = get_limits_fn(robot, arm_joints)
    targets = [target] if isinstance(target, int) else list(target)
    finger_links = get_gripper_finger_links(robot)
    moving_links = list(get_moving_links(robot, arm_joints))
    non_finger_moving = [l for l in moving_links if l not in finger_links]

    def collision_fn(conf, verbose=False):
        collision_fn.last_failure = None
        set_joint_positions(robot, arm_joints, conf)

        if base_obstacle_fn(conf, verbose=verbose):
            if limits_fn(conf):
                collision_fn.last_failure = 'joint_limit'
            elif any(pairwise_link_collision(robot, a, robot, b, max_distance=0.)
                     for a, b in self_pairs):
                collision_fn.last_failure = 'self_collision'
            else:
                collision_fn.last_failure = 'environment_collision'
            return True

        for tgt in targets:
            for link in non_finger_moving:
                if pairwise_link_collision(robot, link, tgt):
                    collision_fn.last_failure = 'target_collision'
                    if verbose:
                        print('[GRASP COLLISION] non-finger link {} ({}) hits target {}'.format(
                            link, get_link_name(robot, link), tgt))
                    return True
        return False
    collision_fn.last_failure = None
    return collision_fn


# Hand-E dimensions are measured in arm_tool0 coordinates; lengths are metres.
HANDE_APPROACH_AXIS = np.array([0.0, 0.0, 1.0])
HANDE_CLOSING_AXIS = np.array([-1.0, 0.0, 0.0])
HANDE_TOOL_TO_FINGERTIP = 0.157
HANDE_OPEN_Q = 0.025
HANDE_MAX_GAP = 2 * HANDE_OPEN_Q - 0.002
HANDE_FINGER_JOINT_NAMES = ('hande_robotiq_hande_left_finger_joint',
                            'hande_robotiq_hande_right_finger_joint')
GRASP_APPROACH_DIST = 0.10
TOUCH_STANDOFF = 0.06
GRASP_TALL_MEMBER = 0.20
GRASP_SIDE_HEIGHTS = (0.7,)
GRASP_MAX_SIDE_EXTENT = 0.30
GRASP_ORN_TOL = 0.01


def hande_q_for_width(width):
    """Finger joint value whose inner gap equals `width` (fingers touch, not penetrate)."""
    return float(np.clip((width + 0.002) / 2.0, 0.0, HANDE_OPEN_Q))


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def tool_pose_in_object(approach, closing, fingertip):
    """arm_tool0 pose in the OBJECT frame such that the fingers point along
    `approach`, close along `closing`, and the fingertip sits at `fingertip`."""
    a = _unit(approach)
    c = np.asarray(closing, dtype=float)
    c = _unit(c - (c @ a) * a)
    local = np.column_stack([HANDE_APPROACH_AXIS, HANDE_CLOSING_AXIS,
                             np.cross(HANDE_APPROACH_AXIS, HANDE_CLOSING_AXIS)])
    target = np.column_stack([a, c, np.cross(a, c)])
    rot = target @ local.T
    pos = np.asarray(fingertip, dtype=float) - a * HANDE_TOOL_TO_FINGERTIP
    return (tuple(pos), tuple(quat_from_matrix(rot)))


def object_dimensions(body):
    """Box dimensions (dx, dy, dz) in the body's own frame, independent of its current pose."""
    with PoseSaver(body):
        set_pose(body, unit_pose())
        lower, upper = get_aabb(body)
    return np.array(upper) - np.array(lower)


def build_member_grasps(dims):
    """Candidate grasps for a box member, all expressed in the object frame.

    Returns a list of (grasp_type, tool_from_object_pose, approach_tool_pose, width)
    where tool poses are arm_tool0 poses in the object frame and `width` is the
    member extent the fingers close across.
    """
    half = dims / 2.0
    axes = np.eye(3)
    grasps = []

    def add(kind, approach, closing, fingertip, width):
        o_T_g = tool_pose_in_object(approach, closing, fingertip)
        backoff = np.asarray(o_T_g[0]) - _unit(approach) * GRASP_APPROACH_DIST
        o_T_a = (tuple(backoff), o_T_g[1])
        grasps.append((kind, o_T_g, o_T_a, float(width)))


    depth = min(0.03, 0.5 * dims[2])
    fitting = [k for k in (0, 1) if dims[k] <= HANDE_MAX_GAP]
    if fitting:
        k = min(fitting, key=lambda i: dims[i])
        o = 1 - k
        offsets = (-0.25, 0.0, 0.25) if dims[o] > GRASP_MAX_SIDE_EXTENT else (0.0,)
        for s in offsets:
            tip = np.array([0.0, 0.0, half[2] - depth]) + axes[o] * s * dims[o]
            add('top', -axes[2], axes[k], tip, dims[k])


    fractions = GRASP_SIDE_HEIGHTS if dims[2] > GRASP_TALL_MEMBER else (0.5,)
    for k in (0, 1):
        if dims[k] > GRASP_MAX_SIDE_EXTENT:
            continue
        other = 1 - k

        options = [(dims[other], axes[other]), (dims[2], axes[2])]
        fits = sorted([o for o in options if o[0] <= HANDE_MAX_GAP], key=lambda o: o[0])
        if not fits:
            continue
        width, closing = fits[0]
        grip = min(dims[k], 0.025)
        for sign in (1.0, -1.0):
            a = sign * axes[k]
            for f in fractions:
                z = -half[2] + f * dims[2]
                fingertip = -a * half[k] + a * grip + np.array([0.0, 0.0, z])
                add('side', a, closing, fingertip, width)
    return grasps


class HandEGripperCommand(GripperCommand):
    """GripperCommand that drives BOTH Hand-E fingers.

    The right finger is a URDF <mimic> joint, which PyBullet does not enforce, so
    the stock command (which only drives the left actuator) left it frozen.
    """
    def apply(self, state, **kwargs):
        joints = [joint_from_name(self.robot, name) for name in HANDE_FINGER_JOINT_NAMES]
        start = get_joint_positions(self.robot, joints)
        end = [float(self.position)] * len(joints)
        path = [start, end] if self.teleport else \
            [start] + list(get_extend_fn(self.robot, joints)(start, end))
        for positions in path:
            set_joint_positions(self.robot, joints, positions)
            yield positions


PLANNER = 'birrt'

PLANNER_CONFIGS = {
    'birrt':       {'algorithm': None, 'restarts': 16, 'max_iterations': 100, 'max_time': 15},
    'rrt_star':    {'algorithm': 'rrt_star', 'restarts': 16, 'max_iterations': 200, 'radius': 2.0, 'max_time': 15},
    'prm':         {'algorithm': 'prm', 'num_samples': 300, 'max_time': 15},


    'lazy_prm':    {'algorithm': 'lazy_prm', 'num_samples': 300, 'max_time': 15},

    'rrt_connect': {'algorithm': 'rrt_connect', 'max_time': 15},
    'astar':       {'algorithm': 'lattice', 'max_time': 15},
}

def get_cfree_approach_pose_test_debug(problem, collisions=True):
    arm = 'left'
    gripper = problem.get_gripper()
    test_count = {'count': 0}
    def test(b1, p1, g1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        test_count['count'] += 1
        p2.assign()
        for i, _ in enumerate(iterate_approach_path(problem.robot, arm, gripper, p1, g1, body=b1)):
            if pairwise_collision(b1, b2):
                if test_count['count'] < 5:
                    print(f"[CFREE_APPROACH DEBUG {test_count['count']}] FAILED: b1-b2 collision at step {i}")
                return False
            if pairwise_collision(gripper, b2):
                if test_count['count'] < 5:
                    print(f"[CFREE_APPROACH DEBUG {test_count['count']}] FAILED: gripper-b2 collision at step {i}")
                return False
        if test_count['count'] < 5:
            print(f"[CFREE_APPROACH DEBUG {test_count['count']}] PASSED")
        return True
    return test


def nullspace_ik_args(robot):
    """lower/upper/range/rest arrays for pybullet null-space IK over all movable joints.

    Captured once at the arm's rest config. Continuous (unlimited) joints get a
    [-pi, pi] surrogate range. restPoses = the robot's current joint positions, so
    IK is biased toward this neutral posture and away from the joint limits.
    """
    lower, upper, jrange, rest = [], [], [], []
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        if info[2] == p.JOINT_FIXED:
            continue
        lo, hi = info[8], info[9]
        if hi <= lo:
            lo, hi = -np.pi, np.pi
        cur = p.getJointState(robot, j)[0]
        lower.append(lo)
        upper.append(hi)
        jrange.append(hi - lo)
        rest.append(min(max(cur, lo), hi))
    return lower, upper, jrange, rest


@track_current('ik_solver_calls', success=lambda q: len(q) > 0 and np.all(np.isfinite(q)))
def calc_ik(robot, arm_link, pos, orn, ik_solver='pybullet', ns_args=None):
    """Numerical IK with optional joint limits and rest-pose bias."""
    IK_MAX_ITERS = 300
    IK_RESIDUAL = 1e-5
    if ik_solver == 'pybullet_nullspace' and ns_args is not None:
        ll, ul, jr, rp = ns_args
        return p.calculateInverseKinematics(robot, arm_link, pos, orn,
                                            lowerLimits=ll, upperLimits=ul,
                                            jointRanges=jr, restPoses=rp,
                                            maxNumIterations=IK_MAX_ITERS,
                                            residualThreshold=IK_RESIDUAL)
    return p.calculateInverseKinematics(robot, arm_link, pos, orn,
                                        maxNumIterations=IK_MAX_ITERS,
                                        residualThreshold=IK_RESIDUAL)


def build_touch_poses(point):
    """World tool_tool0 poses that bring the FINGERTIP to the joint `point`, one
    per approach direction (top-down + four sides), with a pre-touch pose backed
    off along the approach.

    A disconnect is a symbolic release: the EE only needs to get NEAR the joint,
    not thread the fingertip exactly onto the contact. TOUCH_STANDOFF stops the
    fingertip a few cm short along the approach direction, which is what makes
    joints at the base of a tall member (e.g. a stud rising straight out of a
    bottom-plate joint) reachable -- an exact-point touch there drives the wrist
    into the floor/plate, so only the open row-end studs could be reached.
    """
    point = np.asarray(point, dtype=float)
    axes = np.eye(3)
    directions = [('top', -axes[2], axes[0]),
                  ('side+x', axes[0], axes[1]), ('side-x', -axes[0], axes[1]),
                  ('side+y', axes[1], axes[0]), ('side-y', -axes[1], axes[0])]
    out = []
    for name, approach, closing in directions:

        tip = point - _unit(approach) * TOUCH_STANDOFF
        w_T_g = tool_pose_in_object(approach, closing, tip)
        backoff = (tuple(np.asarray(w_T_g[0]) - _unit(approach) * GRASP_APPROACH_DIST), w_T_g[1])
        out.append((name, w_T_g, backoff))
    return out


__all__ = ['Connection', 'DisconnectMarker', 'compute_connection_point', 'CONTACT_MIN_OVERLAP', 'aabb_face_contact', 'is_vertical_member', 'detect_connections', 'make_connection', 'GRIPPER_LINK_NAME_FRAGMENTS', '_FINGER_LINK_CACHE', 'get_gripper_finger_links', 'make_grasp_collision_fn', 'HANDE_APPROACH_AXIS', 'HANDE_CLOSING_AXIS', 'HANDE_TOOL_TO_FINGERTIP', 'HANDE_OPEN_Q', 'HANDE_MAX_GAP', 'HANDE_FINGER_JOINT_NAMES', 'GRASP_APPROACH_DIST', 'TOUCH_STANDOFF', 'GRASP_TALL_MEMBER', 'GRASP_SIDE_HEIGHTS', 'GRASP_MAX_SIDE_EXTENT', 'GRASP_ORN_TOL', 'hande_q_for_width', '_unit', 'tool_pose_in_object', 'object_dimensions', 'build_member_grasps', 'HandEGripperCommand', 'PLANNER', 'PLANNER_CONFIGS', 'get_cfree_approach_pose_test_debug', 'nullspace_ik_args', 'calc_ik', 'build_touch_poses']
