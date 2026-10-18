"""Regression tests for logging, costs, search retries, and self-collision checks."""
import contextlib
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import patch

from motion_metrics import arm_motion_metrics, joint_path_length
import run_statistics as rs
import run_100_experiments as runner
import planner as tamp

HERE=Path(__file__).resolve().parent


class MetricsTests(unittest.TestCase):
    def test_all_manipulation_approaches_and_retreats(self):
        path=[Obj(values=[0.,0.]),Obj(values=[3.,4.])]
        commands=Obj(commands=[Obj(path=path)])
        plan=[('pick',[None]*5+[commands]),('place',[None]*5+[commands]+[None]*3),
              ('disconnect',[None]*4+[commands])]
        metrics=arm_motion_metrics(plan,weight=.1)
        self.assertEqual(metrics['arm_joint_distance'],30.)
        self.assertEqual(metrics['arm_motion_cost'],3.)
        self.assertEqual(arm_motion_metrics(plan,weight=0)['arm_motion_cost'],0.)
        self.assertEqual(joint_path_length([[3.],[-3.]]),6.)
        self.assertEqual(joint_path_length([]),0.)
        for weight in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):arm_motion_metrics(plan,weight)

    def test_statistics_survive_error_and_reset(self):
        with TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            log=Path(d)/'run.log'
            @rs.with_run_statistics
            def fail():
                stats=rs.get_run_statistics();stats.log_path=str(log);stats.set_stage('grp_1')
                wrapped=stats.track('base_queries')(lambda:(_ for _ in ()).throw(RuntimeError('test')))
                wrapped()
            with self.assertRaises(RuntimeError):fail()
            data=json.loads(next(s[len('[RUN STATS] '):] for s in log.read_text().splitlines() if s.startswith('[RUN STATS] ')))
            self.assertEqual(data['status'],'error')
            self.assertEqual(data['metrics']['base_queries']['errors'],1)
            self.assertIsNone(data['metrics']['base_queries']['success_rate'])
            @rs.with_run_statistics
            def fresh():self.assertEqual(rs.get_run_statistics().snapshot()['base_queries']['attempts'],0)
            fresh()

    def test_experiment_counts_and_cost_columns(self):
        with TemporaryDirectory() as d:
            out=Path(d);plan=out/'plan.json';plan.write_text('{}')
            log=out/'console.log'
            metric=dict(attempts=3,successes=1,failures=2,errors=0,completed=3,unfinished=0,
                seconds=.1,success_rate=1/3,outcomes={'success':1,'failure':2})
            stage=dict(scope='stage',stage='grp_0',metrics={'base_queries':metric})
            full=dict(scope='run',stage=None,metrics={'base_queries':metric})
            log.write_text('[RESULT] status=success cost=10 actual_cost=12 actual_cost_with_arm=12.5\n'
                '[PLAN] Saved replayable plan to '+str(plan)+'\n'
                '[RUN STATS] '+json.dumps(stage)+'\n[RUN STATS] '+json.dumps(full)+'\n')
            result,stats,stages,_,saved=runner.parse_console(log)
            self.assertEqual(runner.classify(0,False,result,saved),'success')
            self.assertEqual(runner.classify(-9,True,result,saved),'timeout')
            self.assertEqual(runner.classify(-9,False,{},''),'killed')
            self.assertEqual(runner.classify(-11,False,{},''),'crash')
            rows=[dict(trial_id='seed0',example='corner',planner='birrt',seed=0,status='success',
                wall_seconds=1,runtime=1,actual_cost=12,actual_cost_with_arm=12.5,
                statistics=stats,stage_statistics=stages,plan_json=saved),
                dict(trial_id='seed1',example='corner',planner='birrt',seed=1,status='fail',
                wall_seconds=1,statistics=stats,stage_statistics=stages)]
            runner.save_tables(out,rows,dict(examples=['corner'],base_planners=['birrt'],runs=2))
            summary=json.loads((out/'summary.json').read_text())[0]
            self.assertEqual(summary['success_rate'],.5)
            self.assertEqual(summary['diagnostics_complete_runs_only']['base_queries']['attempts'],6)
            self.assertIn('actual_cost_with_arm',(out/'runs.csv').read_text())


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        tamp.connect(use_gui=False)
        rs._active=rs.RunStatistics()
        self.stats=rs.get_run_statistics();self.stats.set_stage('grp_0')
        with TemporaryFile(mode="w+") as sink, contextlib.redirect_stdout(sink):
            self.problem=tamp.create_problem(scene_path=str(HERE/'scenes/corner_2_7col.json'),clutter=0)
        self.problem.base_planner='birrt';self.problem.use_mppi=False

    def tearDown(self):
        tamp.disconnect();rs._active=None

    @unittest.skipUnless((HERE/'scenes/floor8.json').is_file(), 'Optional floor scene is not distributed')
    def test_floor8_keeps_original_resting_face(self):
        self.assertFalse(getattr(self.problem,'original_resting_face',False))
        tamp.p.resetSimulation()
        problem=tamp.create_problem(scene_path=str(HERE/'scenes/floor8.json'))
        self.assertTrue(problem.original_resting_face)
        self.assertEqual(len(problem.connections),12)
        pddl=tamp.pddlstream_from_problem(problem,collisions=True,motion_collisions=True)
        sample=pddl.stream_map['sample-pose']
        for body in problem.movable:
            group=problem.object_groups[body]
            surface=dict(problem.goal_on)[body]
            dimensions=tamp.object_dimensions(body)
            self.assertTrue(tamp.build_member_grasps(dimensions))
            self.assertEqual(dimensions[2],.1)
            for rank in (1,2):
                outputs=list(sample(body,surface,'placement_grp_{}_{}'.format(group,rank)))
                self.assertEqual(len(outputs),1)
                [(pose,)]=outputs[0]
                self.assertAlmostEqual(tamp.matrix_from_quat(pose.value[1])[2][2],1.)
                self.assertAlmostEqual(pose.value[0][2],tamp.get_aabb(surface)[1][2]+.05)
                extent,_=tamp.original_resting_orientation(dimensions)
                self.assertAlmostEqual(extent[2],dimensions[2])

    def test_query_and_retry_denominators(self):
        calls=[]
        def fake_motion(*args,**kwargs):
            calls.append(1)
            return (Obj(),) if len(calls)==3 else None
        with TemporaryFile(mode='w+') as sink,contextlib.redirect_stdout(sink),patch('planning.stream_bindings.get_motion_gen',return_value=fake_motion):
            pp=tamp.pddlstream_from_problem(self.problem,collisions=True,motion_collisions=True)
            q=tamp.Conf(self.problem.robot,tamp.get_group_joints(self.problem.robot,'base'),(0.,0.,0.))
            fn=pp.stream_map['plan-base-motion']
            self.assertTrue(next(fn(q,q)));self.assertFalse(next(fn(q,q)))
        rows=self.stats.snapshot()
        self.assertEqual(rows['base_queries']['attempts'],2)
        self.assertEqual(rows['base_queries']['successes'],1)
        self.assertEqual(rows['base_planner_attempts']['attempts'],6)
        self.assertAlmostEqual(rows['base_planner_attempts']['success_rate'],1/6)
        self.stats.set_stage('grp_1')
        self.assertEqual(self.stats.snapshot('grp_1')['base_queries']['attempts'],0)

    def test_submillimetre_self_collision_rejected_in_planning_and_execution(self):
        fixture=json.loads((HERE/'fixtures/forearm_coupler_collision.json').read_text())
        robot=self.problem.robot;joints=tamp.get_arm_joints(robot,'left')
        self.assertEqual(tamp.get_joint_names(robot,joints),fixture['joint_names'])
        fn=tamp.get_collision_fn(robot,joints,obstacles=[],self_collisions=True)
        self.assertFalse(fn(tamp.get_joint_positions(robot,joints)))
        q=fixture['joint_positions'];self.assertTrue(fn(q))
        target=tamp.create_box(.01,.01,.01);tamp.set_point(target,(100,100,100))
        collision=tamp.make_grasp_collision_fn(robot,joints,[],[],target)
        self.assertTrue(collision(q));self.assertEqual(collision.last_failure,'self_collision')
        class Command:
            def apply(self,state):
                tamp.set_joint_positions(robot,joints,q)
                yield
        with self.assertRaisesRegex(RuntimeError,'self-collision'):
            tamp.apply_checked_commands(Obj(robot=robot,fixed=[],movable=[],body_names={}),
                Obj(assign=lambda:None,attachments={}),[Command()])
        self.assertEqual(self.stats.snapshot()['execution_checks']['errors'],1)


if __name__=='__main__':unittest.main()
