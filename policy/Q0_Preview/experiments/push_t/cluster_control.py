"""Trusted SSH deployment helper for this experiment's private Slurm queue."""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys

from freeze_layouts import layout_manifest


def run(root, message):
    action = message['action']
    bench = root/'code/benchmark'
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
    if action == 'stop':
        if service.is_dir():
            (service/'STOP').touch()
        return dict(stopped=name)
    if action == 'submit':
        mode, count = message['mode'], message['episodes']
        if mode not in ('original', 'official_only') or type(count) is not int or count not in (1, 50):
            raise ValueError('Invalid paired experiment mode/count')
        values = dict(Q0_BENCH_ROOT=str(bench), Q0_CLUSTER_PREFIX=message['cluster_prefix'],
            Q0_REMOTE_RUN_DIR=str(service), Q0_PREVIEW_TASK_INFORMATION_MODE=mode,
            ROBODOJO_RUN_ID='q0-'+name, EVAL_NUM=str(count),
            Q0_PREVIEW_CONTROLLER_PYTHONPATH=str(root/'code/overlay'))
        lines = ['#!/bin/bash -l', 'set -euo pipefail', 'umask 077',
                 'source '+shlex.quote(message['site_setup'])]
        lines += ['export '+key+'='+shlex.quote(value) for key,value in values.items()]
        lines += ['exec bash '+shlex.quote(str(bench/'XPolicyLab/policy/Q0_Preview/experiments/push_t/run_external_arm.sh'))]
        launcher = root/'code'/('launch-'+name+'.sh')
        with launcher.open('x') as file:
            file.write('\n'.join(lines)+'\n')
        request = root/'requests'/(name+'.json')
        if request.exists():
            raise ValueError('Request already exists; use a fresh run name')
        temporary = request.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(argv=['bash','-l',str(launcher)], cwd=str(root))))
        temporary.replace(request)
        return dict(submitted=name, service_root=str(service))
    if action == 'status':
        path = root/'results'/(name+'.json')
        value = dict(worker_result=json.loads(path.read_text()) if path.exists() else None)
        result = bench/'eval_result/RoboDojo/push_T/Q0_Preview/arx_x5/1_ckpt_name=api,action_type=joint'/('q0-'+name)/'_result.json'
        value['evaluator_result'] = json.loads(result.read_text()) if result.exists() else None
        log = root/'logs'/(name+'.log')
        value['log_tail'] = log.read_text(errors='replace')[-6000:] if log.exists() else ''
        return value
    raise ValueError('Unknown action')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.root.resolve(), json.load(sys.stdin))))


if __name__ == '__main__':
    main()
