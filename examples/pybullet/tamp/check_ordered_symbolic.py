from planner import DOMAIN_PDDL
from pddlstream.language.constants import PDDLProblem, Equal
from pddlstream.algorithms.meta import solve
init=[('Arm','arm'),('Controllable','arm'),('BConf','q'),('AtBConf','q'),('CanMove',),
      ('AtGrasp','arm','z_first','grasp'),('GroupBefore','ga','gb'),
      Equal(('PickCost',),1),Equal(('PlaceCost',),1),Equal(('DisconnectCost',),1)]
objects=['a_second','z_first','b_one','b_two']
allposes=[]
for group,members in [('ga',objects[:2]),('gb',objects[2:])]:
    init += [('NextPlacementSlot',group,group+'1'),('SlotNext',group+'1',group+'2'),('SlotNext',group+'2',group+'done')]
    for body in members:
        init += [('InGroup',body,group),('Grasp',body,'grasp')]
        for rank in [0,1,2]:
            pose=body+'_p'+str(rank)
            allposes.append((body,pose))
            init += [('Pose',body,pose),('Kin','arm',body,pose,'grasp','q','traj')]
            if rank:
                init += [('PoseSlot',body,pose,group+str(rank)),('Supported',body,pose,group+'_plate')]
            elif body!='z_first': init += [('AtPose',body,pose)]
for a,pa in allposes:
    for b,pb in allposes:
        init += [('CFreePosePose',a,pa,b,pb),('CFreeApproachPose',a,pa,'grasp',b,pb)]
problem=PDDLProblem(DOMAIN_PDDL,{'@sink':'sink','@stove':'stove'},'(define (stream pr2-tamp))',{},init,
    ('and',)+tuple(('On',b,('ga' if b in objects[:2] else 'gb')+'_plate') for b in objects))
plan,_,_=solve(problem,algorithm='adaptive',unit_costs=True,max_time=30,planner='ff-lazy')
assert plan is not None
places=[args for name,args in plan if name=='place']
assert places[0][1]=='z_first' and places[0][-2]=='ga1'
for group in ['ga','gb']:
    assert [a[-2] for a in places if a[-3]==group]==[group+'1',group+'2']
print('PASS actual domain: non-ID first member gets p1; p1,p2 consumed once; independent groups reset at p1')
print([(a[1],a[-3],a[-2]) for a in places])
