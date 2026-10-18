#!/usr/bin/env python3
"""Run a planner trial in its isolated cwd, or inspect a scene without searching."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--script',required=True,type=Path)
    p.add_argument('--describe',type=Path)
    p.add_argument('--scene',type=Path)
    p.add_argument('--ifc',type=Path)
    p.add_argument('--exclude-groups',default='')
    p.add_argument('--scale',type=int,default=1)
    p.add_argument('--planner-args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    script=a.script.resolve()
    dependencies=Path(__file__).resolve().parent/'dependencies'
    if dependencies.is_dir():sys.path.insert(0,str(dependencies))
    sys.path.insert(0,str(script.parent))
    sys.path.insert(0,str(script.parents[3]))
    if a.describe:
        import planner as tamp
        tamp.connect(use_gui=False)
        try:
            problem=tamp.create_problem(scene_path=str(a.scene) if a.scene else None,
                ifc_path=str(a.ifc) if a.ifc else None,scale=a.scale,
                exclude_groups=set(filter(None,a.exclude_groups.split(','))))
            tamp.pddlstream_from_problem(problem, collisions=True, motion_collisions=True)
            info=dict(members=len(problem.movable),connections=len(problem.connections),
                group_order=list(problem.group_order),
                groups={g:[problem.body_names[b] for b in problem.movable if problem.object_groups[b]==g]
                    for g in problem.group_order})
            if not info['members']:raise ValueError('Scene has no members')
            import importlib.metadata
            info['packages']={name:importlib.metadata.version(name) for name in
                ('numpy','scipy','pybullet','ifcopenshell')}
            a.describe.write_text(json.dumps(info,indent=2)+'\n')
        finally:tamp.disconnect()
        return
    stdout,stderr=sys.stdout,sys.stderr
    sys.argv=[str(script)]+(a.planner_args or [])
    try:runpy.run_path(str(script),run_name='__main__')
    finally:sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':main()
