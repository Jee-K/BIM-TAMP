#!/usr/bin/env python3
"""Kinematic replay with dense checks, rigid attachments, and no placement snaps."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import pybullet as p
from planner import create_problem, get_gripper_finger_links, HANDE_FINGER_JOINT_NAMES, object_dimensions
from placement_layout import ordered_layout
from examples.pybullet.utils.pybullet_tools.utils import (
    connect, disconnect, get_pose, set_pose, get_link_pose, multiply, invert,
    joints_from_names, set_joint_positions, get_joint_positions, set_base_values,
    get_extend_fn, get_difference_fn, get_joint_name, get_body_name, get_links, get_link_name, wait_for_user, get_base_values, get_aabb, matrix_from_quat, get_self_link_pairs, get_movable_joints)
from examples.pybullet.utils.pybullet_tools.pr2_utils import get_gripper_link


def pose_error(actual, expected):
    distance = math.dist(actual[0], expected[0])
    angle = 2*math.acos(min(1., abs(float(np.dot(actual[1], expected[1])))))
    return distance, angle


def pose_value(data):
    return (data['position'], data['orientation'])


class Replay:
    def __init__(self, data, audit=False, speed=1., pause=False, headless=False):
        meta = data['metadata']
        scene_path = meta.get('scene_file')
        if not scene_path and meta.get('full_legacy'):
            scene_path = str(Path(__file__).resolve().parent/'scenes/floor8.json')
        self.problem = create_problem(scale=meta.get('scale', 1), clutter=meta.get('clutter', 0),
            clutter_seed=meta.get('seed'), ifc_path=meta.get('ifc_file'),
            selected_guids=set(meta['ifc_selected_guids']) if meta.get('ifc_selected_guids') else None,
            scene_path=scene_path, exclude_groups=set(meta.get('exclude_groups', [])))
        self.robot = self.problem.robot
        self.self_pairs = get_self_link_pairs(self.robot, get_movable_joints(self.robot))
        self.bodies = {name: body for body, name in self.problem.body_names.items()}
        self.obstacles = list(dict.fromkeys(self.problem.fixed + self.problem.movable))
        self.fingers = {link for link in get_links(self.robot) if 'finger' in get_link_name(self.robot,link).lower()}
        self.held = None
        self.audit, self.speed, self.pause, self.headless = audit, speed, pause, headless
        self.contacts = set()
        self.events = []
        self.counts = {}
        self.seen = set()
        self.action = -1
        self.samples = 0
        self.gaps = []
        self.placements = []
        self.original_resting_face = meta.get('placement_policy') == 'group-removal-order-original-z-up-v1'
        self.ordered = self.original_resting_face or meta.get('placement_policy') in ('group-removal-order-largest-face-v1', 'group-removal-order-upright-studs-v1')
        self.dimensions = {b: object_dimensions(b) for b in self.problem.movable}
        self.removal_counts = {}
        self.removal_ranks = {}
        first = next(a for a in data['actions'] if a['type']=='move_base')
        set_base_values(self.robot, first['base_trajectory'][0])
        initial = data.get('initial_robot_state')
        if initial:
            set_joint_positions(self.robot, joints_from_names(self.robot, initial['joint_names']), initial['joint_positions'])
            set_pose(self.robot, pose_value(initial['base_pose']))
        self.data = data

    def issue(self, kind, **details):
        self.counts[kind] = self.counts.get(kind, 0)+1
        key=(self.action, kind, details.get('body'), details.get('link'))
        if key not in self.seen:
            self.seen.add(key)
            event=dict(action=self.action, kind=kind, **details)
            self.events.append(event)
            print('[AUDIT]', event, flush=True)
        if not self.audit:
            raise RuntimeError('Replay stopped: '+kind+'; see the audit event above')

    def update_held(self):
        if self.held:
            body, arm, transform=self.held
            set_pose(body, multiply(get_link_pose(self.robot, get_gripper_link(self.robot, arm)), transform))

    def check(self):
        self.samples+=1
        held_body=self.held[0] if self.held else None
        for link1, link2 in self.self_pairs:
            points=p.getClosestPoints(self.robot, self.robot, distance=0.,
                linkIndexA=link1, linkIndexB=link2)
            if points:
                self.issue('robot_self_collision', link=link1, other_link=link2,
                    link_name=get_link_name(self.robot, link1),
                    other_link_name=get_link_name(self.robot, link2),
                    penetration_m=max(0., -min(v[8] for v in points)))
        for body in self.obstacles:
            points=p.getClosestPoints(self.robot, body, distance=0.)
            bad=[v for v in points if v[8]<-.001 and not
                 (v[3] in self.fingers and (body in self.contacts or body==held_body))]
            if bad:
                v=min(bad, key=lambda x:x[8])
                self.issue('robot_environment_penetration', body=self.problem.body_names.get(body, get_body_name(body)),
                    link=v[3], penetration_m=-v[8])
            if held_body is not None and body!=held_body:
                points=p.getClosestPoints(held_body, body, distance=0.)
                bad=[v for v in points if v[8]<-.001]
                if bad:
                    self.issue('held_object_penetration', body=self.problem.body_names.get(body,get_body_name(body)),
                        penetration_m=-min(v[8] for v in bad))

    def tick(self, delay=.02):
        self.update_held()
        self.check()


        if not self.headless:
            time.sleep(delay/max(self.speed,.01))

    def arm(self, action, key):
        joints=joints_from_names(self.robot, action['arm_joint_names'])
        waypoints=action[key]
        current=get_joint_positions(self.robot,joints)
        extend=get_extend_fn(self.robot,joints,resolutions=[.035]*len(joints))
        gap=max(abs(v) for v in get_difference_fn(self.robot,joints)(waypoints[0],current))
        if gap>.05:
            self.gaps.append(dict(action=self.action, segment=key, max_joint_jump_rad=gap))
        for goal in waypoints:
            for q in extend(current, goal):
                set_joint_positions(self.robot,joints,q)
                self.tick()
            current=goal
        if self.pause and not self.headless:
            wait_for_user('[PAUSE] '+key)

    def base(self, action):
        for arm, conf in action.get('transit_arm_confs', {}).items():
            joints=joints_from_names(self.robot,conf['joint_names'])
            current_arm=get_joint_positions(self.robot,joints)
            delta=get_difference_fn(self.robot,joints)(conf['joint_positions'],current_arm)
            if max(abs(v) for v in delta)>.05:
                self.issue('base_transit_posture_mismatch',arm=arm,max_joint_error_rad=max(abs(v) for v in delta))
        current=get_base_values(self.robot)
        for goal in action['base_trajectory']:
            delta=np.asarray(goal,dtype=float)-np.asarray(current)
            delta[2]=(delta[2]+math.pi)%(2*math.pi)-math.pi
            n=max(1,int(math.ceil(max(abs(delta[0])/.035,abs(delta[1])/.035,abs(delta[2])/.035))))
            for k in range(1,n+1):
                set_base_values(self.robot, np.asarray(current)+delta*k/n)
                self.tick()
            current=goal
        if self.pause and not self.headless:
            wait_for_user('[PAUSE] move_base')

    def gripper(self, position):
        joints=joints_from_names(self.robot,HANDE_FINGER_JOINT_NAMES)
        start=np.asarray(get_joint_positions(self.robot,joints))
        for k in range(1,11):
            set_joint_positions(self.robot,joints,start+(position-start)*k/10)
            self.tick()

    def run(self):
        for action in self.data['actions']:
            self.action=action['index']
            kind=action['type']
            print('[ACTION]',self.action,kind,action.get('object_name',''),flush=True)
            self.contacts=set()
            if self.held:
                self.contacts.add(self.held[0])
            if kind=='move_base':
                self.base(action)
            elif kind=='disconnect':
                self.contacts.update(self.bodies[name] for name in action['connection_bodies'])
                self.arm(action,'arm_approach_trajectory')
                for conn in self.problem.connections:
                    if conn.name==action['connection_name'] and conn.marker is not None:
                        p.changeVisualShape(conn.marker,-1,rgbaColor=[.5,.5,.5,1])
                self.arm(action,'arm_retreat_trajectory')
            elif kind=='pick':
                body=self.bodies[action['object_name']]
                if self.ordered:
                    group = self.problem.object_groups[body]
                    self.removal_counts[group] = self.removal_counts.get(group, 0)+1
                    self.removal_ranks[body] = self.removal_counts[group]
                self.contacts.add(body)
                self.gripper(.025)
                self.arm(action,'arm_approach_trajectory')
                expected=multiply(pose_value(action['object_pose']),invert(pose_value(action['grasp_transform'])))
                error,angle=pose_error(get_link_pose(self.robot,get_gripper_link(self.robot,action['arm'])),expected)
                if error>.01 or angle>.15:
                    self.issue('invalid_grasp_contact',body=action['object_name'],position_error_m=error,orientation_error_rad=angle)
                self.gripper(action['gripper_close_position'])
                tool=get_link_pose(self.robot,get_gripper_link(self.robot,action['arm']))
                self.held=(body,action['arm'],multiply(invert(tool),get_pose(body)))
                self.arm(action,'arm_retreat_trajectory')
            elif kind=='place':
                self.arm(action,'arm_approach_trajectory')
                body=self.bodies[action['object_name']]
                error,angle=pose_error(get_pose(body),pose_value(action['object_target_pose']))
                self.placements.append(dict(body=action['object_name'],position_error_m=error,orientation_error_rad=angle))
                if error>.02 or angle>.3:
                    self.issue('placement_mismatch',body=action['object_name'],position_error_m=error,orientation_error_rad=angle)
                if self.ordered:
                    group = self.problem.object_groups[body]
                    rank = self.removal_ranks[body]
                    if action.get('placement_group') != group or action.get('placement_rank') != rank:
                        self.issue('placement_order_mismatch', body=action['object_name'], expected_rank=rank)
                    members = [b for b in self.problem.movable if self.problem.object_groups[b] == group]
                    upright = group in self.data['metadata'].get('upright_placement_groups', [])
                    layout = ordered_layout([self.dimensions[b] for b in members], upright=upright)
                    plate = dict(self.problem.goal_on)[body]
                    bounds = get_aabb(plate)
                    center = (np.asarray(bounds[0])+np.asarray(bounds[1]))/2
                    expected_xy = center[:2]+layout['centers'][rank-1]
                    actual = get_pose(body)
                    extent = abs(np.asarray(matrix_from_quat(actual[1]))) @ self.dimensions[body]
                    if np.linalg.norm(np.asarray(actual[0][:2])-expected_xy) > .003:
                        self.issue('placement_slot_mismatch', body=action['object_name'], expected_rank=rank)
                    shortest = 2 if (upright or self.original_resting_face) else int(np.argmin(self.dimensions[body]))
                    normal = matrix_from_quat(actual[1])[2][shortest]
                    if not self.original_resting_face:
                        normal = abs(normal)
                    if normal < np.cos(.01):
                        self.issue('placement_orientation_mismatch', body=action['object_name'])
                    self.placements[-1].update(group=group, rank=rank, upright=upright, orientation_correct=bool(normal >= np.cos(.01)))
                self.held=None
                self.gripper(action['gripper_open_position'])
                self.arm(action,'arm_retreat_trajectory')
            else:
                raise ValueError('Unsupported action '+kind)
        return self.report()

    def report(self):
        return dict(valid=not self.counts, samples=self.samples, failure_counts=self.counts,
            events=self.events, missing_arm_transitions=self.gaps, placements=self.placements)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan_json')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--audit',action='store_true',help='Collect failures instead of stopping at the first one')
    parser.add_argument('--report')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--pause',action='store_true')
    args=parser.parse_args()
    connect(use_gui=not args.headless)
    engine=None
    try:
        engine=Replay(json.loads(Path(args.plan_json).read_text()),audit=args.audit,
            speed=args.speed,pause=args.pause,headless=args.headless)
        report=engine.run()
        print('[AUDIT SUMMARY]',json.dumps(report['failure_counts']), 'samples=',report['samples'])
        if not args.headless:
            wait_for_user('Press Enter to close')
        return 0 if report['valid'] else 2
    except RuntimeError as error:
        print('[REPLAY STOPPED]', error, flush=True)
        if not args.headless:
            wait_for_user('Motion stopped at the failure. Inspect the scene; press Enter to close.')
        return 2
    finally:
        if engine is not None and args.report:
            Path(args.report).write_text(json.dumps(engine.report(),indent=2)+'\n')
        disconnect()

if __name__=='__main__':
    raise SystemExit(main())
