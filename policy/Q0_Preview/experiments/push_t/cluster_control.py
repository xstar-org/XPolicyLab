"""Trusted SSH deployment helper for this experiment's private Slurm queue."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from freeze_layouts import layout_manifest


def tail(path, limit):
    if not path.exists():
        return ''
    with path.open('rb') as file:
        file.seek(max(0, path.stat().st_size-limit))
        return file.read(limit).decode('utf-8', errors='replace')


def run(root, message):
    action = message['action']
    benchmark_name = message.get('benchmark_name', 'benchmark')
    if not isinstance(benchmark_name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', benchmark_name):
        raise ValueError('Invalid private benchmark snapshot name')
    bench = root/'code'/benchmark_name
    task = message.get('task', 'push_T')
    if not isinstance(task, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,64}', task):
        raise ValueError('Invalid task name')
    if action == 'deployment':
        files = {}
        for folder, directories, names in os.walk(bench, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in
                ('Assets', 'third_party', 'eval_result', '__pycache__', '.git'))
            for name in sorted(names):
                path = Path(folder)/name
                if path.suffix in ('.py', '.sh', '.json', '.yml', '.yaml'):
                    files[str(path.relative_to(bench))] = hashlib.sha256(path.read_bytes()).hexdigest()
        site = Path(message['site_setup']).resolve()
        if not site.is_relative_to((root/'code').resolve()):
            raise ValueError('Site setup must belong to this private deployment')
        files['site_setup'] = hashlib.sha256(site.read_bytes()).hexdigest()
        if message.get('include_overlay'):
            overlay = root/'code/overlay'
            if not overlay.is_dir():
                raise ValueError('Strict controller dependency overlay is missing')
            for path in sorted(overlay.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    files['overlay/'+str(path.relative_to(overlay))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(benchmark_name=benchmark_name, file_count=len(files),
            sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    if action == 'freeze':
        actual = layout_manifest(bench/'Assets/Eval_Layout/RoboDojo/arx_x5/1')
        path = root/'frozen-layouts.json'
        if path.exists():
            if json.loads(path.read_text()) != actual:
                raise ValueError('Layouts changed after freezing')
        else:
            with path.open('x') as file:
                json.dump(actual, file, indent=2)
        return actual
    name = message['name']
    if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,96}', name):
        raise ValueError('Invalid run name')
    service = root/'runs'/name
    if action == 'launch_overlap':
        job_id = message['job_id']
        if not isinstance(job_id, str) or not job_id.isdigit():
            raise ValueError('Invalid allocation identity')
        request = json.loads((root/'requests'/(name+'.json')).read_text())
        if request.get('validation_episodes') != 1:
            raise ValueError('Only explicit one-episode validation may overlap')
        marker = root/'claims'/(name+'.overlap-launch.json')
        try:
            with marker.open('x') as file:
                json.dump(dict(job_id=job_id, started_unix=time.time()), file)
        except FileExistsError:
            if json.loads(marker.read_text())['job_id'] != job_id:
                raise ValueError('Conflicting allocation under an existing validation name')
            return dict(launched=name, already_launched=True)
        with (root/'logs'/(name+'-scheduler.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('run_overlap_validation.py')),
                '--root', str(root), '--name', name, '--job-id', job_id], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return dict(launched=name, pid=child.pid)
    if action == 'stop':
        if service.is_dir():
            (service/'STOP').touch()
        return dict(stopped=name)
    if action == 'submit':
        mode, count = message['mode'], message['episodes']
        if mode not in ('original', 'official_only', 'strict_official') or type(count) is not int or count not in (1, 50):
            raise ValueError('Invalid paired experiment mode/count')
        if message.get('overlap_validation') and count != 1:
            raise ValueError('Overlapping validation is limited to one episode')
        if count == 50:
            task_config = (bench/'task/RoboDojo/config/_task.yml').read_text()
            block = re.search(r'(?m)^  '+re.escape(task)+r':\n((?:    .*\n|\s*\n)*)', task_config)
            configured = re.search(r'(?m)^    eval_nums:\s*(\d+)\s*$', block[1]) if block else None
            if configured is None or int(configured[1]) != count:
                raise ValueError('Formal run requires an explicit native task eval_nums: 50; EVAL_NUM only caps the native default')
        interval = message.get('traceback_interval_s', 0)
        if type(interval) is not int or not 0 <= interval <= 3600 or (count == 50 and interval):
            raise ValueError('Traceback diagnostics are limited to one-episode controls')
        config_name = message.get('agent_config_name', 'agent.gpt62sol.json')
        if not isinstance(config_name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+\.json', config_name):
            raise ValueError('Invalid Agent configuration filename')
        values = dict(Q0_BENCH_ROOT=str(bench), Q0_CLUSTER_PREFIX=message['cluster_prefix'],
            Q0_TASK_NAME=task,
            Q0_AGENT_CONFIG=str(bench/'XPolicyLab/policy/Q0_Preview/config'/config_name),
            Q0_REMOTE_RUN_DIR=str(service), Q0_PREVIEW_TASK_INFORMATION_MODE=mode,
            ROBODOJO_RUN_ID='q0-'+name, EVAL_NUM=str(count),
            Q0_SIM_TRACEBACK_INTERVAL_S=str(interval),
            Q0_PREVIEW_CONTROLLER_PYTHONPATH=str(root/'code/overlay'))
        lines = ['#!/bin/bash -l', 'set -euo pipefail', 'umask 077',
                 'source '+shlex.quote(message['site_setup'])]
        lines += ['export '+key+'='+shlex.quote(value) for key,value in values.items()]
        lines += ['exec bash '+shlex.quote(str(bench/'XPolicyLab/policy/Q0_Preview/experiments/push_t/run_external_arm.sh'))]
        launcher = root/'code'/('launch-'+name+'.sh')
        script = '\n'.join(lines)+'\n'
        if launcher.exists():
            if launcher.read_text() != script:
                raise ValueError('Conflicting launcher under an existing run name')
        else:
            with launcher.open('x') as file:
                file.write(script)
        request = root/'requests'/(name+'.json')
        expected = dict(argv=['bash','-l',str(launcher)], cwd=str(root))
        if message.get('overlap_validation'):
            expected['validation_episodes'] = 1
        if request.exists():
            if json.loads(request.read_text()) != expected:
                raise ValueError('Conflicting request under an existing run name')
            return dict(submitted=name, service_root=str(service), already_submitted=True)
        temporary = request.with_suffix('.tmp')
        temporary.write_text(json.dumps(expected))
        temporary.replace(request)
        return dict(submitted=name, service_root=str(service))
    if action == 'status':
        path = root/'results'/(name+'.json')
        value = dict(worker_result=json.loads(path.read_text()) if path.exists() else None)
        result = bench/f'eval_result/RoboDojo/{task}/Q0_Preview/arx_x5/1_ckpt_name=api,action_type=joint'/('q0-'+name)/'_result.json'
        value['evaluator_result'] = json.loads(result.read_text()) if result.exists() else None
        log = root/'logs'/(name+'.log')
        value['log_tail'] = tail(log, 6000)
        server_log = service/'server.log'
        value['server_log_tail'] = tail(server_log, 4000)
        return value
    raise ValueError('Unknown action')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.root.resolve(), json.load(sys.stdin))))


if __name__ == '__main__':
    main()
