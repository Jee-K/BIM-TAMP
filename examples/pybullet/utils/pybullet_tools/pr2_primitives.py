from __future__ import print_function

import copy
import pybullet as p
import random
import time
from itertools import islice, count

import numpy as np

from .ikfast.pr2.ik import is_ik_compiled, pr2_inverse_kinematics
from .ikfast.utils import USE_CURRENT, USE_ALL
from .pr2_problems import get_fixed_bodies
from .pr2_utils import TOP_HOLDING_LEFT_ARM, SIDE_HOLDING_LEFT_ARM, GET_GRASPS, get_gripper_joints, \
    get_carry_conf, get_top_grasps, get_side_grasps, get_thin_grasps, open_arm, arm_conf, get_gripper_link, get_arm_joints, \
    learned_pose_generator, PR2_TOOL_FRAMES, get_x_presses, PR2_GROUPS, joints_from_names, \
    is_drake_pr2, get_group_joints, get_group_conf, compute_grasp_width, PR2_GRIPPER_ROOTS
from .utils import invert, multiply, get_name, set_pose, get_link_pose, is_placement, \
    pairwise_collision, set_joint_positions, get_joint_positions, sample_placement, get_pose, waypoints_from_path, \
    unit_quat, plan_base_motion, plan_joint_motion, base_values_from_pose, pose_from_base_values, \
    uniform_pose_generator, sub_inverse_kinematics, add_fixed_constraint, remove_debug, remove_fixed_constraint, \
    disable_real_time, enable_gravity, joint_controller_hold, get_distance, \
    get_min_limit, user_input, step_simulation, get_body_name, get_bodies, BASE_LINK, \
    add_segments, get_max_limit, link_from_name, BodySaver, get_aabb, Attachment, interpolate_poses, \
    plan_direct_joint_motion, has_gui, create_attachment, wait_for_duration, get_extend_fn, set_renderer, \
    get_custom_limits, all_between, get_unit_vector, wait_if_gui, \
    set_base_values, euler_from_quat, INF, elapsed_time, get_moving_links, flatten_links, get_relative_pose

BASE_EXTENT = 3.5 # 2.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False

##################################################

def get_base_limits(robot):
    if is_drake_pr2(robot):
        joints = get_group_joints(robot, 'base')[:2]
        lower = [get_min_limit(robot, j) for j in joints]
        upper = [get_max_limit(robot, j) for j in joints]
        return lower, upper
    return BASE_LIMITS

##################################################

class Pose(object):
    num = count()
    #def __init__(self, position, orientation):
    #    self.position = position
    #    self.orientation = orientation
    def __init__(self, body, value=None, support=None, init=False):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.support = support
        self.init = init
        self.index = next(self.num)
    @property
    def bodies(self):
        return flatten_links(self.body)
    def assign(self):
        set_pose(self.body, self.value)
    def iterate(self):
        yield self
    def to_base_conf(self):
        values = base_values_from_pose(self.value)
        return Conf(self.body, range(len(values)), values)
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'p{}'.format(index)

class Grasp(object):
    def __init__(self, grasp_type, body, value, approach, carry):
        self.grasp_type = grasp_type
        self.body = body
        self.value = tuple(value) # gripper_from_object
        self.approach = tuple(approach)
        self.carry = tuple(carry)
    def get_attachment(self, robot, arm):
        tool_link = link_from_name(robot, PR2_TOOL_FRAMES[arm])
        return Attachment(robot, tool_link, self.value, self.body)
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

class Conf(object):
    def __init__(self, body, joints, values=None, init=False):
        self.body = body
        self.joints = joints
        if values is None:
            values = get_joint_positions(self.body, self.joints)
        self.values = tuple(values)
        self.init = init
    @property
    def bodies(self): # TODO: misnomer
        return flatten_links(self.body, get_moving_links(self.body, self.joints))
    def assign(self):
        if not self.joints and (len(self.values) == 3):
            set_base_values(self.body, self.values)
            return
        set_joint_positions(self.body, self.joints, self.values)
    def iterate(self):
        yield self
    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)

#####################################

class Command(object):
    def control(self, dt=0):
        raise NotImplementedError()
    def apply(self, state, **kwargs):
        raise NotImplementedError()
    def iterate(self):
        raise NotImplementedError()

class Commands(object):
    def __init__(self, state, savers=[], commands=[]):
        self.state = state
        self.savers = tuple(savers)
        self.commands = tuple(commands)
    def assign(self):
        for saver in self.savers:
            saver.restore()
        return copy.copy(self.state)
    def apply(self, state, **kwargs):
        for command in self.commands:
            for result in command.apply(state, **kwargs):
                yield result
    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)

#####################################

class Trajectory(Command):
    _draw = False
    def __init__(self, path):
        self.path = tuple(path)
        # TODO: constructor that takes in this info
    def apply(self, state, sample=1):
        handles = add_segments(self.to_points()) if self._draw and has_gui() else []
        for conf in self.path[::sample]:
            conf.assign()
            yield
        end_conf = self.path[-1]
        if isinstance(end_conf, Pose):
            state.poses[end_conf.body] = end_conf
        for handle in handles:
            remove_debug(handle)
    def control(self, dt=0, **kwargs):
        # TODO: just waypoints
        for conf in self.path:
            if isinstance(conf, Pose):
                conf = conf.to_base_conf()
            for _ in joint_controller_hold(conf.body, conf.joints, conf.values):
                step_simulation()
                time.sleep(dt)
    def to_points(self, link=BASE_LINK):
        # TODO: this is computationally expensive
        points = []
        for conf in self.path:
            with BodySaver(conf.body):
                conf.assign()
                #point = np.array(point_from_pose(get_link_pose(conf.body, link)))
                point = np.array(get_group_conf(conf.body, 'base'))
                point[2] = 0
                point += 1e-2*np.array([0, 0, 1])
                if not (points and np.allclose(points[-1], point, atol=1e-3, rtol=0)):
                    points.append(point)
        points = get_target_path(self)
        return waypoints_from_path(points)
    def distance(self, distance_fn=get_distance):
        total = 0.
        for q1, q2 in zip(self.path, self.path[1:]):
            total += distance_fn(q1.values, q2.values)
        return total
    def iterate(self):
        for conf in self.path:
            yield conf
    def reverse(self):
        return Trajectory(reversed(self.path))
    #def __repr__(self):
    #    return 't{}'.format(id(self) % 1000)
    def __repr__(self):
        d = 0
        if self.path:
            conf = self.path[0]
            d = 3 if isinstance(conf, Pose) else len(conf.joints)
        return 't({},{})'.format(d, len(self.path))

def create_trajectory(robot, joints, path):
    return Trajectory(Conf(robot, joints, q) for q in path)

##################################################

class GripperCommand(Command):
    def __init__(self, robot, arm, position, teleport=False):
        self.robot = robot
        self.arm = arm
        self.position = position
        self.teleport = teleport
    def apply(self, state, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        start_conf = get_joint_positions(self.robot, joints)
        if isinstance(self.position, (list, tuple)):
            # If it's already a list or tuple, use it directly
            end_conf = self.position
        else:
            # If it's a single value (like a float), replicate it for all joints
            end_conf = [self.position] * len(joints)
        #end_conf = self.position #[self.position] * len(joints) #jk 1106
        print('start:',start_conf)
        print('end:',end_conf)
        if self.teleport:
            path = [start_conf, end_conf]
        else:
            extend_fn = get_extend_fn(self.robot, joints)
            path = [start_conf] + list(extend_fn(start_conf, end_conf))
        for positions in path:
            set_joint_positions(self.robot, joints, positions)
            yield positions
    def control(self, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        positions = [self.position]*len(joints)
        for _ in joint_controller_hold(self.robot, joints, positions):
            yield

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, self.position)
class Attach(Command):
    vacuum = True
    def __init__(self, robot, arm, grasp, body):
        self.robot = robot
        self.arm = arm
        self.grasp = grasp
        self.body = body
        self.link = link_from_name(self.robot, PR2_TOOL_FRAMES.get(self.arm, self.arm))
        #self.attachment = None
    def assign(self):
        gripper_pose = get_link_pose(self.robot, self.link)
        body_pose = multiply(gripper_pose, self.grasp.value)
        set_pose(self.body, body_pose)
    def apply(self, state, **kwargs):
        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        state.grasps[self.body] = self.grasp
        del state.poses[self.body]
        yield
    def control(self, dt=0, **kwargs):
        if self.vacuum:
            add_fixed_constraint(self.body, self.robot, self.link)
            #add_fixed_constraint(self.body, self.robot, self.link, max_force=1) # Less force makes it easier to pick
        else:
            # TODO: the gripper doesn't quite work yet
            gripper_name = '{}_gripper'.format(self.arm)
            joints = joints_from_names(self.robot, PR2_GROUPS[gripper_name])
            values = [get_min_limit(self.robot, joint) for joint in joints] # Closed
            for _ in joint_controller_hold(self.robot, joints, values):
                step_simulation()
                time.sleep(dt)
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

class Detach(Command):
    def __init__(self, robot, arm, body):
        self.robot = robot
        self.arm = arm
        self.body = body
        self.link = link_from_name(self.robot, PR2_TOOL_FRAMES.get(self.arm, self.arm))
        # TODO: pose argument to maintain same object
    def apply(self, state, **kwargs):
        del state.attachments[self.body]
        state.poses[self.body] = Pose(self.body, get_pose(self.body))
        del state.grasps[self.body]
        yield
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

##################################################

class Clean(Command):
    def __init__(self, body):
        self.body = body
    def apply(self, state, **kwargs):
        state.cleaned.add(self.body)
        self.control()
        yield
    def control(self, **kwargs):
        p.addUserDebugText('Cleaned', textPosition=(0, 0, .25), textColorRGB=(0,0,1), #textSize=1,
                           lifeTime=0, parentObjectUniqueId=self.body)
        #p.setDebugObjectColor(self.body, 0, objectDebugColorRGB=(0,0,1))
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class Cook(Command):
    # TODO: global state here?
    def __init__(self, body):
        self.body = body
    def apply(self, state, **kwargs):
        state.cleaned.remove(self.body)
        state.cooked.add(self.body)
        self.control()
        yield
    def control(self, **kwargs):
        # changeVisualShape
        # setDebugObjectColor
        #p.removeUserDebugItem # TODO: remove cleaned
        p.addUserDebugText('Cooked', textPosition=(0, 0, .5), textColorRGB=(1,0,0), #textSize=1,
                           lifeTime=0, parentObjectUniqueId=self.body)
        #p.setDebugObjectColor(self.body, 0, objectDebugColorRGB=(1,0,0))
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

##################################################

def get_grasp_gen(problem, collisions=False, randomize=True):
    for grasp_type in problem.grasp_types:
        if grasp_type not in GET_GRASPS:
            raise ValueError('Unexpected grasp type:', grasp_type)
    def fn(body):
        # TODO: max_grasps
        # TODO: return grasps one by one
        grasps = []
        arm = 'left'
        #carry_conf = get_carry_conf(arm, 'top')
        if 'top' in problem.grasp_types:
            approach_vector = APPROACH_DISTANCE*get_unit_vector([1, 0, 0])
            grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
                          for g in get_top_grasps(body, grasp_length=GRASP_LENGTH))
        if 'side' in problem.grasp_types:
            approach_vector = APPROACH_DISTANCE*get_unit_vector([2, 0, -1])
            grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
                          for g in get_side_grasps(body, grasp_length=GRASP_LENGTH))
        if 'thin' in problem.grasp_types:
            approach_vector = APPROACH_DISTANCE*get_unit_vector([2, 0, -1])
            grasps.extend(Grasp('thin', body, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
                          for g in get_thin_grasps(body, grasp_length=GRASP_LENGTH))
        filtered_grasps = []
        for grasp in grasps:
            grasp_width = compute_grasp_width(problem.robot, arm, body, grasp.value) if collisions else 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        if randomize:
            random.shuffle(filtered_grasps)
        return [(g,) for g in filtered_grasps]
        #for g in filtered_grasps:
        #    yield (g,)
    return fn

##################################################

def accelerate_gen_fn(gen_fn, max_attempts=1):
    def new_gen_fn(*inputs):
        generator = gen_fn(*inputs)
        while True:
            for i in range(max_attempts):
                try:
                    output = next(generator)
                except StopIteration:
                    return
                if output is not None:
                    print(gen_fn.__name__, i)
                    yield output
                    break
    return new_gen_fn

def get_stable_gen(problem, collisions=True, **kwargs):
    obstacles = (list(problem.fixed) + list(problem.movable)) if collisions else []
    if obstacles:
        obstacles = [o for o in obstacles if get_body_name(o) not in ('plane', 'plane.urdf')]
    def gen(body, surface):
        # TODO: surface poses are being sampled in pr2_belief
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]
        while True:
            surface = random.choice(surfaces) # TODO: weight by area
            body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)
    # TODO: apply the acceleration technique here
    return gen

##################################################

def get_tool_from_root(robot, arm):
    root_link = link_from_name(robot, PR2_GRIPPER_ROOTS[arm])
    tool_link = link_from_name(robot, PR2_TOOL_FRAMES[arm])
    return get_relative_pose(robot, root_link, tool_link)

def iterate_approach_path(robot, arm, gripper, pose, grasp, body=None):
    tool_from_root = get_tool_from_root(robot, arm)
    grasp_pose = multiply(pose.value, invert(grasp.value))
    approach_pose = multiply(pose.value, invert(grasp.approach))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(gripper, multiply(tool_pose, tool_from_root))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.value))
        yield

def get_ir_sampler(problem, custom_limits={}, max_attempts=25, collisions=True, learned=True):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    gripper = problem.get_gripper()

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
            if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
                return
        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = arm_conf(arm, grasp.carry)
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_group_joints(robot, 'base')
        if learned:
            base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)
        #lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        
        ##########
        # We must manually define the limits array here.
        if base_joints: 
            lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        else:
            # We are reading the pose (x, y, yaw) directly from the body's pose.
            # We must use the base limits defined in problem.base_limits (e.g., BASE_LIMITS)
            # The structure of problem.base_limits is usually ((x_min, x_max), (y_min, y_max), (yaw_min, yaw_max))
            x_limits, y_limits = zip(*problem.base_limits) # Unzip [(min, min), (max, max)] into [min, max]
            
            # NOTE: problem.base_limits is typically already in the format [(min, max), (min, max)] for x and y
            # Assuming problem.base_limits is a tuple of (x_limit, y_limit)
            x_limits, y_limits = problem.base_limits # Assuming this is the format
            
            # The base generator yields a 3-element tuple (x, y, yaw)
            # We define the limits based on the structure of the generator output (x, y, yaw)
            lower_limits = np.array([x_limits[0], y_limits[0], -np.pi])
            upper_limits = np.array([x_limits[1], y_limits[1], +np.pi])
        # --- END CRITICAL FIX BLOCK ---
        ##########
        while True:
            count = 0
            for base_conf in islice(base_generator, max_attempts):
                count += 1
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                if base_joints:
                    bq.assign()
                else:
                    set_base_values(robot, base_conf)
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                #print('IR attempts:', count)
                yield (bq,)
                break
            else:
                yield None
    return gen_fn

##################################################
'''
def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    print(f'Using PyBullet IK for UR5e robot')

    custom_limits = {
        'arm_shoulder_pan_joint': (-3.14159, 3.14159),
        'arm_shoulder_lift_joint': (-3.14159, 3.14159),
        'arm_elbow_joint': (-3.14159, 3.14159),
        'arm_wrist_1_joint': (-3.14159, 3.14159),
        'arm_wrist_2_joint': (-3.14159, 3.14159),
        'arm_wrist_3_joint': (-3.14159, 3.14159),
        }

    def fn(arm, obj, pose, grasp, base_conf):
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        #approach_pose = multiply(grasp.approach, gripper_pose)
        approach_pose = multiply(pose.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot, arm)
        arm_joints = get_arm_joints(robot, arm)

        default_conf = arm_conf(arm, grasp.carry)
        #sample_fn = get_sample_fn(robot, arm_joints)
        pose.assign()
        base_conf.assign()
        open_arm(robot, arm)
        set_joint_positions(robot, arm_joints, default_conf) # default_conf | sample_fn()
        '''
'''
        grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
                                            #nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles): # [obj]
            #print('Grasp IK failure', grasp_conf)
            #if grasp_conf is not None:
            #    print(grasp_conf)
            #    #wait_if_gui()
            return None
        #approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
        #                                       upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            #print('Approach IK failure', approach_conf)
            #wait_if_gui()
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.05**np.ones(len(arm_joints))
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  custom_limits=custom_limits, resolutions=resolutions/2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              restarts=2, max_iterations=25, smooth=25)
            if approach_path is None:
                print('Approach path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd,)
        '''
'''
        # --- CRITICAL IK OVERRIDE 1: GRASP POSE ---
        
        # PyBullet's IK needs the target pose (gripper_pose), the end-effector link (arm_link),
        # and usually a known initial configuration.
        print("[UR5e] Computing IK for grasp pose...")
        grasp_conf_raw = p.calculateInverseKinematics(
            robot, arm_link, gripper_pose[0], gripper_pose[1], restPoses=default_conf,
            # Optionally pass custom limits or rest pose here if needed
            # lowerLimits=..., upperLimits=..., restPoses=... 
            # nearby_conf=get_joint_positions(robot, arm_joints) # Use default conf as seed
        )
        
        if (grasp_conf_raw is None) or any(math.isnan(q) for q in grasp_conf_raw):
            print("[ERROR] Grasp IK returned NaN values / [ERROR] Grasp IK failed - no solution found")
            return None
            
        # The result from p.calculateInverseKinematics might include all robot joints.
        # We need to filter for the arm joints only (which are the first 6 movable joints).
        grasp_conf = grasp_conf_raw[:len(arm_joints)]
        print(f"[UR5e] Grasp config: {[f'{q:.3f}' for q in grasp_conf]}")
        
        # Test IK solution against obstacles
        set_joint_positions(robot, arm_joints, grasp_conf)
        if any(pairwise_collision(robot, b) for b in obstacles):
            print("[ERROR] Grasp pose collides with obstacles")
            return None

        print("[UR5e] ✓ Grasp pose valid (no collisions)")
        # --- CRITICAL IK OVERRIDE 2: APPROACH POSE ---
        approach_conf_raw = p.calculateInverseKinematics(
            robot, arm_link, approach_pose[0], approach_pose[1],
        )
        if (approach_conf_raw is None) or any(math.isnan(q) for q in approach_conf_raw):
            print("[ERROR] Approach IK failed - no solution found / [ERROR] Approach IK returned NaN values")
            return None
        
        approach_conf = approach_conf_raw[:len(arm_joints)]
        print(f"[UR5e] Approach config: {[f'{q:.3f}' for q in approach_conf]}")

        set_joint_positions(robot, arm_joints, approach_conf)
        if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            print("[ERROR] Approach pose collides with obstacles or object")
            return None
        print("[UR5e] ✓ Approach pose valid (no collisions)")
        # --- PATH PLANNING (Now uses the valid 6-DOF configs) ---
        
        # The rest of the path planning logic (lines 400+) can remain, 
        # but will now use these validated 6-DOF UR5e configurations.
        
        # Ensure the grasp path planning uses the correct 6-DOF setup:
        attachments = {grasp.body: grasp.get_attachment(problem.robot, arm)}
        
        resolutions = 0.05*np.ones(len(arm_joints))
        
        grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                              obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions/2.)
        if grasp_path is None:
            print('Grasp path failure')
            return None
        
        print(f"[UR5e] ✓ Grasp path found ({len(grasp_path)} waypoints)")

        set_joint_positions(robot, arm_joints, default_conf)
        approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                          obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                          custom_limits=custom_limits, resolutions=resolutions,
                                          restarts=2, max_iterations=25, smooth=25)
        if approach_path is None:
            print('Approach path failure')
            return None
            
        print(f"[UR5e] ✓ Approach path found ({len(approach_path)} waypoints)")

        path = approach_path + grasp_path
        print(f"[UR5e] ✓ Complete path: {len(path)} waypoints")
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd,)
    return fn
'''
##################################################
import numpy as np
import pybullet as p
import time
from scipy.spatial.transform import Rotation

# Deprecated: old Husky wheel IK implementation (kept for reference).
def get_ik_fn_old(problem, custom_limits=None, collisions=True, teleport=False):
    """
    IK solver that uses wheel control to move the Husky base and solves
    IK for a 6-DOF arm and gripper.
    """
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    
    # --- Helper to map names to indices ---
    def get_joints_by_names(robot, names):
        num_joints = p.getNumJoints(robot)
        name_to_id = {}
        for i in range(num_joints):
            info = p.getJointInfo(robot, i)
            name_to_id[info[1].decode('utf-8')] = i
        return [name_to_id[name] for name in names if name in name_to_id]

    # Configuration
    WHEEL_JOINTS = [3, 4, 5, 6] # front_left, front_right, rear_left, rear_right
    
    print("\n" + "="*70)
    print("IK SOLVER: HUSKY WHEELS + 6-DOF ARM + GRIPPER")
    print("="*70 + "\n")

    def move_base_to_position(target_base_pos, max_time=5.0):
        """
        Physically drives the Husky to the target by applying wheel torque.
        """
        # 1. Physics Fixes: Release brakes and set friction
        for wheel_id in WHEEL_JOINTS:
            p.changeDynamics(robot, wheel_id, lateralFriction=2.0, spinningFriction=0.1, rollingFriction=0.1)
            p.setJointMotorControl2(robot, wheel_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.changeDynamics(p.loadURDF("plane.urdf") if p.getNumBodies() < 2 else 0, -1, lateralFriction=2.0)

        start_time = time.time()
        while time.time() - start_time < max_time:
            current_pos, _ = p.getBasePositionAndOrientation(robot)
            diff = np.array(target_base_pos[:2]) - np.array(current_pos[:2])
            dist = np.linalg.norm(diff)
            
            if dist < 0.1: # Threshold to stop
                break
            
            # Simple Proportional Control
            target_vel = min(3.0, dist * 2.5)
            for wheel_id in WHEEL_JOINTS:
                p.setJointMotorControl2(
                    robot, wheel_id, p.VELOCITY_CONTROL,
                    targetVelocity=target_vel,
                    force=1000 # High torque for heavy TAMP model
                )
            
            p.stepSimulation()
            if not teleport:
                time.sleep(1./240.)

        # Stop wheels
        for wheel_id in WHEEL_JOINTS:
            p.setJointMotorControl2(robot, wheel_id, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
        return True

    def fn(arm, obj, pose, grasp, base_conf):
        from examples.pybullet.utils.pybullet_tools.utils import (
            multiply, invert, get_joint_positions, set_joint_positions,
            pairwise_collision, is_placement, get_group_conf, set_group_conf,
            BodySaver, get_link_pose
        )
        from examples.pybullet.utils.pybullet_tools.pr2_utils import (
            get_gripper_link, get_arm_joints, open_arm
        )
        from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
            Commands, State, create_trajectory, plan_direct_joint_motion, 
            plan_joint_motion
        )

        # 1. Identify joints from names provided in PR2_GROUPS
        arm_joints = get_arm_joints(robot, arm) # Should match your 6 names
        gripper_joints = get_joints_by_names(robot, ['hande_left_finger_joint', 'hande_right_finger_joint']) 
        arm_link = get_gripper_link(robot, arm)

        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose.value, invert(grasp.value))
        approach_pose = multiply(pose.value, invert(grasp.approach))

        # Reset Robot State
        pose.assign()
        base_joints = get_group_joints(robot, 'base')
        if base_joints:
            base_conf.assign()
        else:
            set_base_values(robot, base_conf.values)
        open_arm(robot, arm)
        
        # 2. Base Placement Calculation
        # The 6-DOF arm has a sweet spot. We place the base 0.5m away from the target.
        target_world_pos = np.array(gripper_pose[0])
        target_world_quat = gripper_pose[1]
        
        # Determine orientation toward target
        current_base_pos, _ = p.getBasePositionAndOrientation(robot)
        direction = target_world_pos[:2] - np.array(current_base_pos[:2])
        angle = np.arctan2(direction[1], direction[0])
        
        # Target position for base (0.5m away from object)
        required_base_pos = target_world_pos - 0.5 * np.array([np.cos(angle), np.sin(angle), 0])
        
        # Physically move the base
        print(f"[IK+Wheels] Driving base to: {required_base_pos[:2]}")
        move_base_to_position(required_base_pos)

        # 3. Solve Arm IK
        with BodySaver(robot):
            # Calculate IK (PyBullet handles base transform automatically)
            ik_result = p.calculateInverseKinematics(
                robot, arm_link, target_world_pos, target_world_quat,
                restPoses=[0, -1.57, 1.57, -1.57, -1.57, 0]
            )

            # Map IK result to specific arm joints. PyBullet returns values for movable joints.
            num_joints = p.getNumJoints(robot)
            movable_joints = [j for j in range(num_joints)
                              if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
            if len(ik_result) == len(movable_joints):
                movable_index = {j: i for i, j in enumerate(movable_joints)}
                grasp_arm_conf = [ik_result[movable_index[j]] for j in arm_joints]
            else:
                grasp_arm_conf = [ik_result[j] for j in arm_joints]

            if any(np.isnan(q) for q in grasp_arm_conf):
                print("✗ IK returned NaN")
                return None

            #set_joint_positions(robot, arm_joints, grasp_arm_conf)
            #if any(pairwise_collision(robot, b) for b in obstacles):
            #    print("✗ Grasp collision")
            #    return None

        # 4. Repeat for Approach
        with BodySaver(robot):
            approach_world_pos = approach_pose[0]
            approach_world_quat = approach_pose[1]
            ik_approach = p.calculateInverseKinematics(
                robot, arm_link, approach_world_pos, approach_world_quat,
                restPoses=grasp_arm_conf
            )
            if len(ik_approach) == len(movable_joints):
                approach_arm_conf = [ik_approach[movable_index[j]] for j in arm_joints]
            else:
                approach_arm_conf = [ik_approach[j] for j in arm_joints]
            
            #set_joint_positions(robot, arm_joints, approach_arm_conf)
            #if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            #    print("✗ Approach collision")
            #    return None

        # 5. Planning
        print("[IK+Wheels] Planning kinematic paths...")
        attachments = {grasp.body: grasp.get_attachment(problem.robot, arm)}
        resolutions = 0.05 * np.ones(len(arm_joints))

        grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_arm_conf,
                                              attachments=attachments.values(),
                                              obstacles=approach_obstacles,
                                              self_collisions=True,
                                              resolutions=resolutions/2.)
        if grasp_path is None: return None

        approach_path = plan_joint_motion(robot, arm_joints, approach_arm_conf,
                                          attachments=attachments.values(),
                                          obstacles=obstacles,
                                          self_collisions=True,
                                          resolutions=resolutions,
                                          restarts=2, max_iterations=25, smooth=25)
        if approach_path is None: return None

        path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        
        print(f"[IK+Wheels] ✓ Success! Paths created.")
        return (cmd,)

    return fn
############################

def get_ik_fn(problem, custom_limits=None, collisions=True, teleport=False):
    """
    UR5e IK stream for a rigid 4-wheel base. Base pose is applied explicitly when no base joints exist.
    """
    robot = problem.robot
    obstacles = problem.fixed if collisions else []

    def fn(arm, obj, pose, grasp, base_conf):
        arm_link = get_gripper_link(robot, arm)
        arm_joints = get_arm_joints(robot, arm)

        pose.assign()
        base_joints = get_group_joints(robot, 'base')
        if base_joints:
            base_conf.assign()
        else:
            set_base_values(robot, base_conf.values)

        gripper_pose = multiply(pose.value, invert(grasp.value))
        approach_pose = multiply(pose.value, invert(grasp.approach))

        ik_grasp = p.calculateInverseKinematics(robot, arm_link,
                                                gripper_pose[0], gripper_pose[1])
        ik_approach = p.calculateInverseKinematics(robot, arm_link,
                                                   approach_pose[0], approach_pose[1])

        movable_joints = [j for j in range(p.getNumJoints(robot))
                          if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
        if len(ik_grasp) == len(movable_joints):
            movable_index = {j: i for i, j in enumerate(movable_joints)}
            grasp_conf = [ik_grasp[movable_index[j]] for j in arm_joints]
            approach_conf = [ik_approach[movable_index[j]] for j in arm_joints]
        else:
            grasp_conf = [ik_grasp[j] for j in arm_joints]
            approach_conf = [ik_approach[j] for j in arm_joints]

        if any(np.isnan(q) for q in grasp_conf) or any(np.isnan(q) for q in approach_conf):
            return None

        path = [approach_conf, grasp_conf] if teleport else [approach_conf, grasp_conf]
        traj = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[traj])
        return (cmd,)

    return fn


def get_ik_ir_gen(problem, max_attempts=25, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=1, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)
    def gen(*inputs):
        b, a, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
            #if not p.init:
            #    return
    return gen

##################################################

def get_motion_gen(problem, custom_limits={}, collisions=True, teleport=False):
    # Include both fixed AND movable objects as obstacles for base motion planning
    robot = problem.robot
    saver = BodySaver(robot)

    def fn(bq1, bq2, fluents=[]):
        if collisions:
            # Identify held objects from fluents — they are not obstacles
            held_objects = set()
            for fluent in fluents:
                if fluent[0].lower() == 'atgrasp':
                    # ('atgrasp'/'AtGrasp', arm, body, grasp)
                    held_objects.add(fluent[2])

            # Obstacles = fixed + movable objects that are NOT held
            obstacles = list(problem.fixed) + [obj for obj in problem.movable if obj not in held_objects]
            print(f"[BASE MOTION] Planning with {len(obstacles)} obstacles "
                  f"({len(problem.fixed)} fixed + {len(obstacles)-len(problem.fixed)} movable, "
                  f"{len(held_objects)} held)")
        else:
            obstacles = []
        saver.restore()
        bq1.assign()
        attachments = []
        for fluent in fluents:
            if fluent[0].lower() == 'atgrasp':
                _, arm, obj, grasp = fluent
                attachments.append(grasp.get_attachment(robot, arm))
        for attachment in attachments:
            attachment.assign()
        if teleport:
            path = [bq1, bq2]
        elif is_drake_pr2(robot):
            raw_path = plan_joint_motion(robot, bq2.joints, bq2.values, attachments=[],
                                         obstacles=obstacles, custom_limits=custom_limits, self_collisions=SELF_COLLISIONS,
                                         restarts=4, max_iterations=50, smooth=50)
            if raw_path is None:
                print('Failed motion plan!')
                #set_renderer(True)
                #for bq in [bq1, bq2]:
                #    bq.assign()
                #    wait_if_gui()
                return None
            path = [Conf(robot, bq2.joints, q) for q in raw_path]
        else:
            if hasattr(bq2, 'joints') and (not bq2.joints):
                goal_conf = bq2.values
            else:
                goal_conf = base_values_from_pose(bq2.value)
            planner_kwargs = getattr(problem, 'planner_config', {'restarts': 16, 'max_iterations': 100})
            raw_path = plan_base_motion(robot, goal_conf, problem.base_limits, obstacles=obstacles,
                                                attachments=attachments, contact_obstacles=problem.surfaces, **planner_kwargs)
            if raw_path is None:
                print('Failed motion plan!')
                return None
            # Certify the returned/smoothed path independently at 5 mm / 0.005 rad.
            # Store exactly those waypoints; do not execute unchecked sparse edges.
            dense = [tuple(bq1.values)]
            for goal in raw_path:
                delta = np.asarray(goal) - np.asarray(dense[-1])
                delta[2] = (delta[2] + np.pi) % (2*np.pi) - np.pi
                start = np.asarray(dense[-1])
                count = max(1, int(np.ceil(np.max(np.abs(delta) / 0.005))))
                for k in range(1, count+1):
                    dense.append(tuple(start + delta*k/count))
            for q in dense:
                set_base_values(robot, q)
                for attachment in attachments:
                    attachment.assign()
                moving = [robot] + [a.child for a in attachments]
                if any(pairwise_collision(mover, obs,
                           max_distance=(-0.001 if mover == robot and obs in problem.surfaces else 0.005))
                       for mover in moving for obs in obstacles if mover != obs):
                    print('[BASE CERTIFICATION] Rejecting a collision in the returned path')
                    return None
            raw_path = dense
            if len(raw_path) > 1:
                actual_xy = sum(get_distance(raw_path[i][:2], raw_path[i+1][:2])
                                for i in range(len(raw_path) - 1))
                actual_full = sum(get_distance(raw_path[i], raw_path[i+1])
                                  for i in range(len(raw_path) - 1))
                print(f"[BASE MOTION] Actual path: XY={actual_xy:.3f}m, XYTheta={actual_full:.3f} ({len(raw_path)} waypoints)")
            if hasattr(bq1, 'joints') and (not bq1.joints):
                start_pose = pose_from_base_values(bq1.values, get_pose(robot))
            else:
                start_pose = bq1.value
            if hasattr(bq2, 'joints') and (not bq2.joints):
                path = [Conf(robot, bq2.joints, q) for q in raw_path]
            else:
                path = [Pose(robot, pose_from_base_values(q, start_pose)) for q in raw_path]
        bt = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
        return (cmd,)
    return fn

##################################################

def get_press_gen(problem, max_attempts=25, learned=True, teleport=False):
    robot = problem.robot
    fixed = get_fixed_bodies(problem)

    def gen(arm, button):
        fixed_wo_button = list(filter(lambda b: b != button, fixed))
        pose = get_pose(button)
        grasp_type = 'side'

        link = get_gripper_link(robot, arm)
        default_conf = get_carry_conf(arm, grasp_type)
        joints = get_arm_joints(robot, arm)

        presses = get_x_presses(button)
        approach = ((APPROACH_DISTANCE, 0, 0), unit_quat())
        while True:
            for _ in range(max_attempts):
                press_pose = random.choice(presses)
                gripper_pose = multiply(pose, invert(press_pose)) # w_f_g = w_f_o * (g_f_o)^-1
                #approach_pose = gripper_pose # w_f_g * g_f_o * o_f_a = w_f_a
                approach_pose = multiply(gripper_pose, invert(multiply(press_pose, approach)))

                if learned:
                    base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp_type)
                else:
                    base_generator = uniform_pose_generator(robot, gripper_pose)
                set_joint_positions(robot, joints, default_conf)
                set_pose(robot, next(base_generator))
                raise NotImplementedError('Need to change this')
                if any(pairwise_collision(robot, b) for b in fixed):
                    continue

                approach_movable_conf = sub_inverse_kinematics(robot, joints[0], link, approach_pose)
                #approach_movable_conf = inverse_kinematics(robot, link, approach_pose)
                if (approach_movable_conf is None) or any(pairwise_collision(robot, b) for b in fixed):
                    continue
                approach_conf = get_joint_positions(robot, joints)

                gripper_movable_conf = sub_inverse_kinematics(robot, joints[0], link, gripper_pose)
                #gripper_movable_conf = inverse_kinematics(robot, link, gripper_pose)
                if (gripper_movable_conf is None) or any(pairwise_collision(robot, b) for b in fixed_wo_button):
                    continue
                grasp_conf = get_joint_positions(robot, joints)
                bp = Pose(robot, get_pose(robot)) # TODO: don't use this

                if teleport:
                    path = [default_conf, approach_conf, grasp_conf]
                else:
                    control_path = plan_direct_joint_motion(robot, joints, approach_conf,
                                                     obstacles=fixed_wo_button, self_collisions=SELF_COLLISIONS)
                    if control_path is None: continue
                    set_joint_positions(robot, joints, approach_conf)
                    retreat_path = plan_joint_motion(robot, joints, default_conf,
                                                     obstacles=fixed, self_collisions=SELF_COLLISIONS)
                    if retreat_path is None: continue
                    path = retreat_path[::-1] + control_path[::-1]
                mt = Trajectory(Conf(robot, joints, q) for q in path)
                yield (bp, mt)
                break
            else:
                yield None
    return gen

#####################################

def control_commands(commands, **kwargs):
    wait_if_gui('Control?')
    disable_real_time()
    enable_gravity()
    for i, command in enumerate(commands):
        print(i, command)
        command.control(*kwargs)

class State(object):
    def __init__(self, attachments={}, cleaned=set(), cooked=set()):
        self.poses = {body: Pose(body, get_pose(body))
                      for body in get_bodies() if body not in attachments}
        self.grasps = {}
        self.attachments = attachments
        self.cleaned = cleaned
        self.cooked = cooked
    def assign(self):
        for attachment in self.attachments.values():
            #attach.attachment.assign()
            attachment.assign()

def apply_commands(state, commands, time_step=None, pause=False, **kwargs):
    #wait_if_gui('Apply?')
    for i, command in enumerate(commands):
        print(i, command)
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                wait_if_gui('Command {}, Step {}) Next?'.format(i, j))
            else:
                wait_for_duration(time_step)
        if pause:
            wait_if_gui()

#####################################

def get_target_point(conf):
    # TODO: use full body aabb
    robot = conf.body
    link = link_from_name(robot, 'torso_lift_link')
    #link = BASE_LINK
    # TODO: center of mass instead?
    # TODO: look such that cone bottom touches at bottom
    # TODO: the target isn't the center which causes it to drift
    with BodySaver(conf.body):
        conf.assign()
        lower, upper = get_aabb(robot, link)
        center = np.average([lower, upper], axis=0)
        point = np.array(get_group_conf(conf.body, 'base'))
        #point[2] = upper[2]
        point[2] = center[2]
        #center, _ = get_center_extent(conf.body)
        return point


def get_target_path(trajectory):
    # TODO: only do bounding boxes for moving links on the trajectory
    return [get_target_point(conf) for conf in trajectory.path]
