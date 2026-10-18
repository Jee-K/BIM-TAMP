#!/usr/bin/env python3
"""Sequential, reproducible fixed-attempt experiments with per-trial diagnostics."""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
ENGINE=ROOT.parents[2]

sys.path.insert(0, str(ENGINE))
SCRIPT_REL=Path('examples/pybullet/tamp/planner.py')
COST_FIELDS=('cost','move_cost','actual_xy','actual_full','actual_cost',
    'arm_joint_distance','arm_approach_distance','arm_retreat_distance',
    'arm_cost_weight','arm_motion_cost','actual_cost_with_arm')
FIELDS=('trial_id','example','planner','seed','status','exit_code','wall_seconds','runtime',
    'moves','picks','places','disconnects')+COST_FIELDS+('stats_complete','console_log','planner_log','plan_json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    temp=Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def parse_console(path):
    result={}; full=None; snapshots={}; planner_log=''; plan=''
    with Path(path).open(errors='replace') as f:
        for line in f:
            if line.startswith('[RESULT] '):
                result=dict(token.split('=',1) for token in line.split()[1:] if '=' in token)
            elif line.startswith('[RUN STATS] '):
                try:d=json.loads(line[len('[RUN STATS] '):])
                except json.JSONDecodeError:continue
                if d.get('scope')=='run':full=d
                else:snapshots[d.get('stage')]=d
            elif line.startswith('[LOG] Logging to '):planner_log=line.split(' to ',1)[1].strip()
            elif line.startswith('[PLAN] Saved replayable plan to '):
                candidate=line.split(' to ',1)[1].strip()
                if '_partial_stage_' not in candidate and '_rejected_' not in candidate:plan=candidate
    return result,full,list(snapshots.values()),planner_log,plan


def classify(exit_code,timed_out,result,plan,interrupted=False):
    if interrupted:return 'interrupted'
    if timed_out:return 'timeout'
    if exit_code in (-9,137):return 'killed'
    if exit_code in (-11,139):return 'crash'
    if exit_code != 0:return 'error'
    if result.get('status')=='success' and plan and Path(plan).is_file():return 'success'
    if result.get('status')=='fail':return 'fail'
    return 'error'


def distribution(values):
    vals=sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not vals:return dict(n=0)
    def quantile(q):
        i=(len(vals)-1)*q;lo=int(i);hi=min(lo+1,len(vals)-1)
        return vals[lo]+(vals[hi]-vals[lo])*(i-lo)
    return dict(n=len(vals),mean=statistics.mean(vals),median=statistics.median(vals),
        std=statistics.stdev(vals) if len(vals)>1 else 0.,q25=quantile(.25),q75=quantile(.75))


def save_tables(out,records,config):

    latest={r['trial_id']:r for r in records}
    ordered=list(latest.values())
    with (out/'runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');w.writeheader();w.writerows(ordered)
    stat_fields=('trial_id','example','planner','seed','scope','stage','metric','attempts',
        'successes','failures','errors','unfinished','completed','success_rate','seconds','outcomes')
    with (out/'statistics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=stat_fields);w.writeheader()
        for r in ordered:
            summaries=([r['statistics']] if r.get('statistics') else [])+r.get('stage_statistics',[])
            for summary in summaries:
                for metric,row in summary['metrics'].items():
                    entry={k:r[k] for k in ('trial_id','example','planner','seed')}
                    entry.update(scope=summary['scope'],stage=summary.get('stage'),metric=metric)
                    entry.update({k:row.get(k) for k in stat_fields if k in row})
                    entry['outcomes']=json.dumps(row.get('outcomes',{}),sort_keys=True)
                    w.writerow(entry)
    groups=defaultdict(list)
    for r in ordered:groups[(r['example'],r['planner'])].append(r)
    summary=[]
    for example in config['examples']:
        for planner in config['base_planners']:
            rows=groups[(example,planner)]
            finished=[r for r in rows if r['status']!='interrupted']
            successes=[r for r in finished if r['status']=='success']
            counts=Counter(r['status'] for r in rows)
            entry=dict(example=example,planner=planner,planned_trials=config['runs'],
                completed_trials=len(finished),status_counts=dict(counts),successes=len(successes),
                success_rate=len(successes)/len(finished) if finished else None,
                wall_seconds_all=distribution([r['wall_seconds'] for r in finished]),
                successful_metrics={k:distribution([r.get(k) for r in successes]) for k in ('runtime',)+COST_FIELDS})
            best=min(successes,key=lambda r:float(r['actual_cost'])) if successes else None
            entry['best_by_actual_cost']=dict(trial_id=best['trial_id'],plan_json=best['plan_json'],
                actual_cost=best['actual_cost']) if best else None

            totals={}
            for r in finished:
                if not r.get('statistics'):continue
                for metric,row in r['statistics']['metrics'].items():
                    dst=totals.setdefault(metric,dict(attempts=0,successes=0,failures=0,errors=0,unfinished=0,seconds=0.))
                    for k in dst:dst[k]+=row.get(k,0)
            for row in totals.values():
                n=row['successes']+row['failures']
                row['success_rate']=row['successes']/n if n else None
            entry['diagnostic_runs_with_final_summary']=sum(bool(r.get('statistics')) for r in finished)
            entry['diagnostics_complete_runs_only']=totals
            summary.append(entry)
    write_json(out/'summary.json',summary)


def example_presets():
    with (ROOT/'scenes/examples.json').open() as stream:
        registry = json.load(stream)
    return {name: spec for name, spec in registry.items()
            if (ROOT/'scenes'/spec['scene']).is_file()}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--example','--examples',dest='examples',nargs='+',choices=list(example_presets())+['custom'])
    source=p.add_mutually_exclusive_group();source.add_argument('--scene',type=Path);source.add_argument('--ifc',type=Path)
    p.add_argument('--exclude-groups',default='',help='Comma-separated group names to omit from the input instance.')
    p.add_argument('--runs',type=int,default=100,help='Fixed attempts per example/base-planner condition, not successful plans.')
    p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--base-planners',nargs='+',choices=['birrt','rrt_connect','lazy_prm','rrt_star','prm','astar','reuse_prm'],default=['birrt'])
    p.add_argument('--task-planner',choices=['ff-astar2','ff-lazy','ff-eager','cea-wastar2','dijkstra'],default='ff-lazy')
    p.add_argument('--algorithm',choices=['adaptive','focused','binding','incremental'],default='adaptive')
    p.add_argument('--ik-solver',choices=['pybullet','pybullet_nullspace'],default='pybullet')
    p.add_argument('--planning',choices=['staged','whole'],default='staged')
    p.add_argument('--cost-mode',choices=['distance','unit'],default='distance')
    p.add_argument('--search-mode',choices=['first','anytime'],default='first')
    p.add_argument('--max-time',type=int,default=300,help='Search budget per group for staged planning, per problem otherwise.')
    p.add_argument('--timeout',type=float,help='Hard wall-clock limit per full trial; automatic allowance includes all groups.')
    p.add_argument('--scale',type=int,default=1);p.add_argument('--clutter',type=int,default=0)
    p.add_argument('--mppi',choices=['off','on'],default='off')
    p.add_argument('--arm-cost-weight',type=float,default=.1,help='Reporting only; raw arm distance is retained.')
    p.add_argument('--ik-cap',type=int,default=1);p.add_argument('--touch-cap',type=int,default=3)
    p.add_argument('--poses',type=int,default=3);p.add_argument('--skeletons',type=int,default=4)
    p.add_argument('--sample-ratio',type=float,default=3.)
    p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--prepare-only',action='store_true',help='Freeze code and inspect scenes, then stop before any planner trial.')
    p.add_argument('--dry-run',action='store_true',help='Print settings/scene selection without creating a campaign or running planning.')
    return p


def selected_examples(a):
    examples=a.examples or (['custom'] if a.scene or a.ifc else ['corner'])
    if len(set(examples))!=len(examples):raise ValueError('Example names must be unique')
    if (a.scene or a.ifc) and examples!=['custom']:raise ValueError('--scene/--ifc requires --example custom (or omit --example)')
    if 'custom' in examples and not (a.scene or a.ifc):raise ValueError('custom requires --scene or --ifc')
    items={}
    excluded=set(filter(None,a.exclude_groups.split(',')))
    for name in examples:
        item=dict(exclude_groups=sorted(excluded),scale=a.scale)
        if name != 'custom':
            preset = example_presets()[name]
            item['scene'] = str((ROOT/'scenes'/preset['scene']).resolve())
            item['exclude_groups'] = sorted(excluded | set(preset.get('exclude_groups', [])))
        elif a.scene:item['scene']=str(a.scene.expanduser().resolve())
        else:item['ifc']=str(a.ifc.expanduser().resolve())
        for key in ('scene','ifc'):
            if key in item and not Path(item[key]).is_file():raise ValueError('Missing input: '+item[key])
        items[name]=item
    return examples,items


def worker_scene_args(item):
    args=['--scale',str(item['scale']),'--exclude-groups',','.join(item['exclude_groups'])]
    for key in ('scene','ifc'):
        if key in item:args+=['--'+key,item[key]]
    return args


def setup_campaign(a):
    examples,items=selected_examples(a)
    config=vars(a).copy()
    for k in ('output','resume','dry_run','prepare_only','scene','ifc'):config.pop(k)
    config.update(examples=examples,inputs=items)
    if a.dry_run:
        print(json.dumps(config,indent=2));print('No planning started.');return None,None
    out=(a.output or ROOT/'results'/('repetitions_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))).expanduser().resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory is not empty; use --resume for an existing campaign')
    out.mkdir(parents=True,exist_ok=True)
    snapshot=out/'code'
    ignore=shutil.ignore_patterns('.git','__pycache__','temp','statistics','log','results','*.pkl','*.log','*_plan.json','*_audit.json')
    def snapshot_ignore(directory, names):
        skipped = set(ignore(directory, names))
        path = Path(directory)
        if path == ENGINE/'downward/builds':
            skipped.update(set(names)-{'release'})
        elif path == ENGINE/'downward/builds/release':
            skipped.update(set(names)-{'bin'})
        return skipped
    shutil.copytree(ENGINE,snapshot,ignore=snapshot_ignore,symlinks=True)
    spec=importlib.util.find_spec('motion_planners')
    dependency=Path(next(iter(spec.submodule_search_locations))) if spec and spec.submodule_search_locations else None
    if dependency is not None and dependency.is_dir():
        shutil.copytree(dependency,out/'dependencies/motion_planners',ignore=ignore)
    shutil.copy2(ROOT/'experiment_worker.py',out/'experiment_worker.py')
    shutil.copy2(Path(__file__),out/'run_100_experiments.py')
    (out/'inputs').mkdir()
    for name,item in items.items():
        for key in ('scene','ifc'):
            if key in item:
                src=Path(item[key]);dest=out/'inputs'/(name+src.suffix)
                shutil.copy2(src,dest)
                item['original_'+key]=str(src);item[key]=str(dest);item[key+'_sha256']=sha(dest)
    source_hashes={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file() and p.suffix in ('.py','.pddl','.urdf','.dae','.stl')}
    config.update(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,
        planner_script=str(snapshot/SCRIPT_REL),worker=str(out/'experiment_worker.py'),source_hashes=source_hashes)
    from xml.etree import ElementTree as ET
    urdf=snapshot/'examples/pybullet/tamp/full_robot.urdf'
    meshes={n.attrib['filename'] for n in ET.parse(urdf).findall('.//mesh')}
    config['external_mesh_hashes']={p:sha(p) for p in meshes if Path(p).is_absolute() and Path(p).is_file()}
    infos={}
    for name,item in items.items():
        wd=out/'preflight'/name;wd.mkdir(parents=True)
        dest=wd/'scene_info.json'
        command=[sys.executable,config['worker'],'--script',config['planner_script'],'--describe',str(dest)]+worker_scene_args(item)
        with (wd/'scene_check.log').open('w') as f:
            result=subprocess.run(command,cwd=wd,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        if result.returncode or not dest.exists():raise ValueError('Scene preflight failed; inspect '+str(wd/'scene_check.log'))
        infos[name]=json.loads(dest.read_text())
        print('[SCENE] {}: {} members, {} connections, groups={}'.format(name,infos[name]['members'],infos[name]['connections'],infos[name]['group_order']),flush=True)
    config['scene_info']=infos
    if a.algorithm=='incremental' and a.search_mode=='anytime':raise ValueError('anytime mode is supported here for adaptive/focused/binding; use first for incremental')
    config['dependency_hashes']={str(p.relative_to(out)):sha(p) for p in (out/'dependencies').rglob('*.py')}
    config['worker_sha256']=sha(out/'experiment_worker.py')
    write_json(out/'config.json',config)
    return out,config


def command_for(config,example,planner,seed):
    item=config['inputs'][example]
    args=['--batch','--seed',str(seed),'--base-planner',planner,'--task-planner',config['task_planner'],
        '--ik-solver',config['ik_solver'],'--scale',str(config['scale']),'--clutter',str(config['clutter']),
        '--mppi',config['mppi'],'--max-time',str(config['max_time']),'--collision-check-order','early',
        '--search-mode',config['search_mode'],'--arm-cost-weight',str(config['arm_cost_weight'])]
    if config['planning']=='staged':args+=['--staged']
    for key in ('scene','ifc'):
        if key in item:args+=['--'+key,item[key]]
    if item['exclude_groups']:args+=['--exclude-groups',','.join(item['exclude_groups'])]
    return [config['python'],config['worker'],'--script',config['planner_script'],'--planner-args']+args


def stop_process(proc):
    if proc.poll() is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()


def run_trial(out,config,example,planner,seed):
    trial_id='{}_{}_seed{:06d}'.format(example,planner,seed)
    parent=out/'trials'/trial_id;parent.mkdir(parents=True,exist_ok=True)
    number=len(list(parent.glob('attempt_*')))+1
    wd=parent/('attempt_{:02d}'.format(number));wd.mkdir()
    console=wd/'console.log';command=command_for(config,example,planner,seed)
    groups=len(config['scene_info'][example]['group_order']) if config['planning']=='staged' else 1
    timeout=config['timeout'] or (max(1,groups)*(config['max_time']+20)+120)
    env=os.environ.copy()
    env['PYTHONHASHSEED']=str(seed)
    env.update(BIM_TAMP_LOG_DIR=str(wd/'log'),EXP_ALGORITHM=config['algorithm'],EXP_UNIT='0' if config['cost_mode']=='distance' else '1',
        EXP_IK_CAP=str(config['ik_cap']),EXP_TOUCH_CAP=str(config['touch_cap']),EXP_POSES=str(config['poses']),
        EXP_RATIO=str(config['sample_ratio']),EXP_SKELETONS=str(config['skeletons']),EXP_EFFORT='1')
    write_json(wd/'invocation.json',dict(command=command,timeout=timeout,environment={k:v for k,v in env.items() if k.startswith('EXP_') or k=='BIM_TAMP_LOG_DIR'}))
    timed_out=False;interrupted=False;started=time.monotonic()
    with console.open('w') as f:
        proc=subprocess.Popen(command,cwd=wd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:timed_out=True;stop_process(proc)
        except KeyboardInterrupt:interrupted=True;stop_process(proc)
    wall=time.monotonic()-started
    result,full,stages,log,plan=parse_console(console)
    row=dict(trial_id=trial_id,example=example,planner=planner,seed=seed,exit_code=proc.returncode,
        wall_seconds=round(wall,6),status=classify(proc.returncode,timed_out,result,plan,interrupted),
        console_log=str(console),planner_log=log,plan_json=plan,stats_complete=bool(full),
        statistics=full,stage_statistics=stages)
    for k in ('runtime','moves','picks','places','disconnects')+COST_FIELDS:
        if k in result:
            try:
                val=float(result[k]);row[k]=val if math.isfinite(val) else None
            except ValueError:row[k]=None
    write_json(wd/'outcome.json',row)
    return row


def resume_campaign(path):
    out=path.expanduser().resolve();config=json.loads((out/'config.json').read_text())
    root=Path(config['planner_script']).parents[3]
    for rel,digest in config['source_hashes'].items():
        if sha(root/rel)!=digest:raise ValueError('Frozen code changed: '+rel)
    for item in config['inputs'].values():
        for key in ('scene','ifc'):
            if key in item and sha(item[key])!=item[key+'_sha256']:raise ValueError('Frozen input changed: '+item[key])
    if sha(config['worker'])!=config['worker_sha256']:raise ValueError('Frozen worker changed')
    for rel,digest in config.get('dependency_hashes',{}).items():
        if sha(out/rel)!=digest:raise ValueError('Frozen dependency changed: '+rel)
    for p,digest in config['external_mesh_hashes'].items():
        if sha(p)!=digest:raise ValueError('Robot mesh changed: '+p)
    return out,config


def main(argv=None):
    a=parser().parse_args(argv)
    if a.seed_start+a.runs-1>2**32-1:raise ValueError('Seeds must fit NumPy/Python 32-bit seed range')
    if a.runs<=0 or a.seed_start<0 or a.max_time<=0 or a.scale<=0 or a.clutter<0:raise ValueError('Invalid run/seed/time/scale/clutter value')
    if a.timeout is not None and (not math.isfinite(a.timeout) or a.timeout<=0):raise ValueError('timeout must be positive and finite')
    if not math.isfinite(a.arm_cost_weight) or a.arm_cost_weight<0:raise ValueError('arm weight must be nonnegative and finite')
    if min(a.ik_cap,a.touch_cap,a.poses,a.skeletons)<=0 or not math.isfinite(a.sample_ratio) or a.sample_ratio<=0:raise ValueError('Sampling settings must be positive')
    if len(set(a.base_planners))!=len(a.base_planners):raise ValueError('Base planners must be unique')
    if a.algorithm=='incremental' and a.search_mode=='anytime':raise ValueError('Use first mode with incremental')
    if a.resume:
        out,config=resume_campaign(a.resume)
        if a.dry_run:print(json.dumps(config,indent=2));return
        print('[RESUME] Using the saved configuration; new configuration flags do not alter this campaign.',flush=True)
    else:
        out,config=setup_campaign(a)
        if out is None:return
    if a.prepare_only:
        print('[PREPARED] No planning started. Resume with --resume '+str(out),flush=True)
        return
    lock=(out/'.campaign.lock').open('a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('Another process is running this campaign')
    journal=out/'runs.jsonl'
    records=[]
    if journal.exists():
        content=journal.read_text()
        if content and not content.endswith('\n'):

            complete,_,fragment=content.rpartition('\n')
            (out/'interrupted_journal_fragment.txt').write_text(fragment)
            content=complete+'\n' if complete else ''
            journal.write_text(content)
        records=[json.loads(line) for line in content.splitlines() if line.strip()]
    done={r['trial_id'] for r in records if r['status']!='interrupted'}
    total=len(config['examples'])*len(config['base_planners'])*config['runs']
    print('[EXPERIMENT] {} fixed trials; output={}; cost_mode={}; search_mode={}'.format(total,out,config['cost_mode'],config['search_mode']),flush=True)
    try:
        for example in config['examples']:
            for planner in config['base_planners']:
                for seed in range(config['seed_start'],config['seed_start']+config['runs']):
                    trial_id='{}_{}_seed{:06d}'.format(example,planner,seed)
                    if trial_id in done:continue
                    print('[START] '+trial_id,flush=True)
                    row=run_trial(out,config,example,planner,seed)
                    with journal.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                    records.append(row);save_tables(out,records,config)
                    print('[DONE] {} status={} wall={:.1f}s'.format(trial_id,row['status'],row['wall_seconds']),flush=True)
                    if row['status']=='interrupted':return
    finally:
        save_tables(out,records,config)
        lock.close()
    print('[COMPLETE] '+str(out/'summary.json'),flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,subprocess.TimeoutExpired) as exc:
        print('[ERROR] '+str(exc),file=sys.stderr);sys.exit(2)
