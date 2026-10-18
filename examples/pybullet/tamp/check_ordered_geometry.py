import json
from pathlib import Path
import itertools
import numpy as np
import pybullet as p
from planner import *
from placement_layout import ordered_layout, flat_orientations
connect(use_gui=False)
try:
    problem=create_scene_problem(str(Path(__file__).parent/'scenes/corner_2_7col.json'))
    print('Initial robot/surface penetrations:',[(s, min([v[8] for v in p.getClosestPoints(problem.robot,s,0)] or [0])) for s in problem.surfaces])
    for group in problem.group_order:
        members=[b for b in problem.movable if problem.object_groups[b]==group]
        plate=dict(problem.goal_on)[members[0]]
        bounds=get_aabb(plate)
        center=(np.asarray(bounds[0])+np.asarray(bounds[1]))/2
        upright=group in problem.upright_placement_groups
        layout=ordered_layout([object_dimensions(b) for b in members],upright=upright)
        original={b:get_pose(b) for b in members}
        for rank,b in enumerate(reversed(members)):
            dims=object_dimensions(b)
            extent, rotations=(dims,[np.eye(3)]) if upright else flat_orientations(dims)
            for rot in rotations:
                assert np.allclose(rot.T@rot,np.eye(3)) and np.isclose(np.linalg.det(rot),1)
                assert np.isclose((abs(rot)@dims)[2],dims[2] if upright else min(dims))
                xy=center[:2]+layout['centers'][rank]
                set_pose(b,((xy[0],xy[1],bounds[1][2]+extent[2]/2),quat_from_matrix(rot)))
                aabb=get_aabb(b)
                assert abs(aabb[0][2]-bounds[1][2])<1e-6
                assert np.all(np.asarray(aabb[0][:2])>=np.asarray(bounds[0][:2])+.15-1e-6)
                assert np.all(np.asarray(aabb[1][:2])<=np.asarray(bounds[1][:2])-.15+1e-6)
        assert not any(pairwise_collision(a,b) for a,b in itertools.combinations(members,2))
        print('PASS group',group,'reverse member order, support height, requested orientation, separation, plate bounds')
        for b,pose in original.items(): set_pose(b,pose)

    for dims in [[(2,.1,.05),(.2,1,.1),(.08,.06,3)],[(.1,.1,.1)]*40]:
        layout=ordered_layout(dims)
        sizes=[flat_orientations(d)[0] for d in dims]
        for i,j in itertools.combinations(range(len(dims)),2):
            delta=abs(np.array(layout['centers'][i])-layout['centers'][j])
            assert np.any(delta-(np.array(sizes[i][:2])+sizes[j][:2])/2>=.12-1e-9)
    print('PASS mixed sizes and 40-member wrapped layout')
finally: disconnect()
