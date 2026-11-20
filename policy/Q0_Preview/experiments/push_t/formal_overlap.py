"""Trusted, explicitly authorized official-only overlap in an existing allocation.

This helper is deployed outside the immutable simulator snapshot. It reserves
its queue claim before publishing the request, so the serial worker can never
also launch it. It owns only its own srun process group, never the allocation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


AUTHORIZATION = 'same_gpu_official_only_50'


def save(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def exact(path, value):
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)+'\n'
    if path.exists():
        if path.read_text() != content:
            raise ValueError('Conflicting overlap artifact: '+str(path))
    else:
        with path.open('x') as file:
            file.write(content)


def validate(message):
    if (message.get('authorization') != AUTHORIZATION
            or message.get('mode') != 'official_only' or message.get('episodes') != 50
            or message.get('gpu') != 0 or not message.get('user_request')):
        raise ValueError('Explicit same-GPU official-only 50-episode authorization required')
    for key in ('name', 'benchmark_name', 'co_run'):
        if not isinstance(message.get(key), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', message[key]):
            raise ValueError('Invalid overlap identity: '+key)
    if not isinstance(message.get('job_id'), str) or not message['job_id'].isdigit():
        raise ValueError('Invalid allocation identity')
    if message['name'] == message['co_run']:
        raise ValueError('Overlap must have its own service identity')


def gpu_headroom(output, expected_uuid, minimum_free_mib=11000):
    rows = [row.strip().split(',') for row in output.strip().splitlines() if row.strip()]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError('Expected exactly one allocation-visible GPU')
    index, uuid, used, total = [value.strip() for value in rows[0]]
    used, total = int(used), int(total)
    if index != '0' or uuid != expected_uuid or total-used < minimum_free_mib:
        raise ValueError('Same-GPU identity/headroom gate failed')
    return dict(index=0, uuid=uuid, used_mib=used, total_mib=total,
                free_mib=total-used, minimum_free_mib=minimum_free_mib)


def load_control(root, benchmark):
    folder = root/'code'/benchmark/'XPolicyLab/policy/Q0_Preview/experiments/push_t'
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('frozen_overlap_control', folder/'cluster_control.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reserve_identity(root, message, commit=True):
    """One arm at a time; a terminal direct predecessor may transfer its lock."""
    lock = root/'code/formal-overlap-official_only.json'
    identity = {key: message[key] for key in ('name', 'job_id', 'benchmark_name', 'co_run', 'authorization')}
    if not lock.exists():
        if commit:
            exact(lock, identity)
        return
    previous = json.loads(lock.read_text())
    if previous == identity:
        return
    source = message.get('resume_source_service')
    if (source != previous['name'] or any(previous[key] != identity[key]
            for key in ('job_id', 'benchmark_name', 'co_run', 'authorization'))):
        raise ValueError('Conflicting overlap identity without a direct resume chain')
    worker = json.loads((root/'results'/(source+'.json')).read_text())
    claim = json.loads((root/'claims'/(source+'.json')).read_text())
    certificate = json.loads((root/'code'/('resume-'+message['name']+'.json')).read_text())
    if ('returncode' not in worker or worker.get('job_id') != identity['job_id']
            or not claim.get('formal_overlap') or claim.get('job_id') != identity['job_id']
            or certificate != message.get('resume_record')
            or certificate.get('source_service') != source
            or certificate.get('target_service') != message['name']
            or certificate.get('benchmark_name') != message['benchmark_name']
            or certificate.get('interrupted_episode_counted') is not False
            or not 0 < certificate.get('completed', 0) < 50):
        raise ValueError('Missing terminal predecessor/native resume proof')
    # Preserve every lock transition for operator audit, then transfer it.
    if commit:
        exact(root/'code'/('formal-overlap-predecessor-'+message['name']+'.json'), previous)
        save(lock, identity)


def prepare(root, message):
    validate(message)
    root = Path(root).resolve()
    name, job_id = message['name'], message['job_id']
    manifest = root/'code'/('formal-overlap-'+name+'.json')
    # Idempotent management retries cannot start a second official-only arm.
    reserve_identity(root, message, commit=False)
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['request'] != message:
            raise ValueError('Conflicting overlap request')
        return previous
    if (root/'runs'/name).exists() or (root/'results'/(name+'.json')).exists():
        raise ValueError('Formal overlap requires a fresh service')
    co_claim = json.loads((root/'claims'/(message['co_run']+'.json')).read_text())
    if co_claim.get('job_id') != job_id or ((root/'results'/(message['co_run']+'.json')).exists()
            and not message.get('resume_source_service')):
        raise ValueError('The original arm must be active in the same allocation')
    state = subprocess.check_output(['squeue', '-h', '-j', job_id, '-o', '%T'], text=True, timeout=30).strip()
    if state != 'RUNNING':
        raise ValueError('The existing Slurm allocation is not running')
    output = subprocess.check_output(['srun', '--jobid='+job_id, '--overlap', '--nodes=1',
        '--ntasks=1', '--cpus-per-task=1', 'nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.total', '--format=csv,noheader,nounits'],
        text=True, timeout=60)
    hardware = gpu_headroom(output, message['gpu_uuid'])
    control = load_control(root, message['benchmark_name'])
    deployment = control.run(root, dict(action='deployment', benchmark_name=message['benchmark_name'],
                                      site_setup=message['site_setup']))
    if deployment != message['deployment']:
        raise ValueError('Immutable simulator snapshot differs from validated deployment')
    if control.run(root, dict(action='freeze', benchmark_name=message['benchmark_name'])) != message['layouts']:
        raise ValueError('Frozen layouts differ from original arm')
    # Transfer ownership only after all read-only scheduler/hardware/code gates.
    reserve_identity(root, message)
    record = dict(request=message, hardware=hardware, prepared_unix=time.time(),
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Reserve the claim FIRST. The serial allocation worker skips claimed files.
    claim = root/'claims'/(name+'.json')
    exact(claim, dict(job_id=job_id, formal_overlap=True, authorization=AUTHORIZATION,
                      name=name, prepared_unix=record['prepared_unix']))
    control.run(root, dict(action='submit', name=name, mode='official_only', episodes=50,
        benchmark_name=message['benchmark_name'], agent_config_name=message['agent_config_name'],
        cluster_prefix=message['cluster_prefix'], site_setup=message['site_setup']))
    save(manifest, record)
    return record


def start(root, message):
    root = Path(root).resolve()
    record = prepare(root, message)
    marker = root/'claims'/(message['name']+'.formal-launch.json')
    if marker.exists():
        return dict(record=record, launch=json.loads(marker.read_text()), already_launched=True)
    with marker.open('x') as file:
        launch = dict(job_id=message['job_id'], launched_unix=time.time())
        json.dump(launch, file)
    with (root/'logs'/(message['name']+'-scheduler.log')).open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
            '--root', str(root), '--execute', message['name'], '--job-id', message['job_id']],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch['scheduler_pid'] = process.pid
    save(marker, launch)
    return dict(record=record, launch=launch)


def execute(root, name, job_id):
    root = Path(root).resolve()
    proof = json.loads((root/'code'/('formal-overlap-'+name+'.json')).read_text())
    message = proof['request']
    validate(message)
    if message['name'] != name or message['job_id'] != job_id:
        raise ValueError('Formal overlap execution identity mismatch')
    request = json.loads((root/'requests'/(name+'.json')).read_text())
    launcher = str(root/'code'/('launch-'+name+'.sh'))
    if request != dict(argv=['bash', '-l', launcher], cwd=str(root)):
        raise ValueError('Formal overlap launcher/cwd changed')
    child = None

    def shutdown(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    started = time.time()
    with (root/'logs'/(name+'.log')).open('x') as log:
        child = subprocess.Popen(['srun', '--jobid='+job_id, '--overlap', '--nodes=1',
            '--ntasks=1', '--cpus-per-task=4', *request['argv']], cwd=root,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result = child.wait()
    save(root/'results'/(name+'.json'), dict(returncode=result, job_id=job_id,
        started_unix=started, finished_unix=time.time(), formal_overlap=True,
        authorization=AUTHORIZATION))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--execute')
    parser.add_argument('--job-id')
    args = parser.parse_args()
    if args.execute:
        execute(args.root, args.execute, args.job_id)
    else:
        print(json.dumps(start(args.root, json.load(sys.stdin))))
