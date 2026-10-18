"""Run-local diagnostics; never load PDDLStream's historical statistics."""
from collections import Counter
from functools import wraps
import json
import sys
import time


class RunStatistics:
    METRICS = ('base_queries', 'base_planner_attempts', 'ik_solver_calls',
        'pick_base_candidates', 'place_base_candidates', 'disconnect_base_candidates',
        'pick_ik_endpoints', 'place_ik_endpoints', 'disconnect_ik_endpoints',
        'arm_path_pick', 'arm_path_place', 'arm_path_disconnect',
        'cfree_pose_pose', 'cfree_approach_pose', 'cfree_traj_pose',
        'execution_checks', 'tamp_search')

    def __init__(self):
        self.stage = 'whole_problem'
        self.rows = {}
        self.log_path = None
        self.status = 'incomplete'
        self.last_checkpoint = time.perf_counter()
        self.set_stage(self.stage)

    def set_stage(self, stage):
        self.stage = stage
        for metric in self.METRICS:
            self.rows.setdefault((stage, metric), dict(attempts=0, successes=0,
                failures=0, errors=0, seconds=0., outcomes=Counter()))

    def begin(self, metric):
        key = (self.stage, metric)
        row = self.rows.setdefault(key, dict(attempts=0, successes=0, failures=0,
            errors=0, seconds=0., outcomes=Counter()))
        row['attempts'] += 1
        return key, time.perf_counter()

    def end(self, ticket, outcome, successful=False):
        key, started = ticket
        row = self.rows[key]
        row['seconds'] += time.perf_counter() - started
        row['outcomes'][outcome] += 1
        row['errors' if outcome.startswith('error') else 'successes' if successful else 'failures'] += 1
        if self.log_path and time.perf_counter() - self.last_checkpoint >= 30:
            self.last_checkpoint = time.perf_counter()
            self.emit(stage=self.stage, scope='checkpoint')

    def track(self, metric, success=lambda result: result is not None):
        """Measure actual function calls, not generator yields or stream history."""
        def decorate(fn):
            @wraps(fn)
            def wrapped(*args, **kwargs):
                name = metric(*args, **kwargs) if callable(metric) else metric
                ticket = self.begin(name)
                wrapped.failure_reason = None
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:
                    reason = 'error:' + type(exc).__name__
                    if name == 'execution_checks':
                        reason += ':' + ('self_collision' if 'self-collision' in str(exc)
                            else 'collision' if 'collision' in str(exc)
                            else 'discontinuity' if 'jump' in str(exc) else 'other')
                    self.end(ticket, reason)
                    raise
                ok = bool(success(result))
                self.end(ticket, 'success' if ok else wrapped.failure_reason or 'failure', ok)
                return result
            return wrapped
        return decorate

    def snapshot(self, stage=None):
        merged = {}
        for (group, metric), row in self.rows.items():
            if stage is not None and group != stage:
                continue
            dest = merged.setdefault(metric, dict(attempts=0, successes=0, failures=0,
                errors=0, seconds=0., outcomes=Counter()))
            for field in ('attempts', 'successes', 'failures', 'errors', 'seconds'):
                dest[field] += row[field]
            dest['outcomes'].update(row['outcomes'])
        for row in merged.values():
            row['completed'] = row['successes'] + row['failures']
            row['unfinished'] = row['attempts'] - row['completed'] - row['errors']
            row['success_rate'] = (row['successes'] / row['completed']
                                   if row['completed'] else None)
            row['seconds'] = round(row['seconds'], 6)
        return merged

    def emit(self, stage=None, scope=None):
        payload = dict(scope=scope or ('run' if stage is None else 'stage'), stage=stage,
            status=self.status, metrics=self.snapshot(stage))
        lines = ['[RUN STATS] ' + json.dumps(payload, sort_keys=True)]
        if payload['scope'] == 'run':
            for metric, row in payload['metrics'].items():
                rate = 'n/a' if row['success_rate'] is None else '{:.2%}'.format(row['success_rate'])
                lines.append('[SUCCESS RATE] {} attempts={} successes={} failures={} '
                    'errors={} unfinished={} rate={} seconds={:.3f}'.format(
                        metric, row['attempts'], row['successes'], row['failures'],
                        row['errors'], row['unfinished'], rate, row['seconds']))

        stream = sys.stdout
        log_file = getattr(stream, 'log_file', None)
        if log_file is not None and log_file.closed:
            stream = stream.terminal
        for line in lines:
            print(line, file=stream, flush=True)
        if self.log_path and (log_file is None or log_file.closed):
            with open(self.log_path, 'a') as handle:
                handle.write('\n'.join(lines) + '\n')


_active = None


def get_run_statistics():
    global _active
    if _active is None:
        _active = RunStatistics()
    return _active


def track_current(metric, success=lambda result: result is not None):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            return get_run_statistics().track(metric, success)(fn)(*args, **kwargs)
        return wrapped
    return decorate


def with_run_statistics(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        global _active
        previous = _active
        stats = _active = RunStatistics()
        try:
            return fn(*args, **kwargs)
        except BaseException:
            stats.status = 'error'
            raise
        finally:
            try:
                stats.emit()
            finally:
                _active = previous
    return wrapped
