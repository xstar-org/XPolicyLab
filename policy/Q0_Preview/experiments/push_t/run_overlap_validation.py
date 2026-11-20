"""Claim a one-episode private queue request in an existing Slurm allocation.

Used only after checking available GPU memory, so the API arm keeps running
while a separate CLI validation scene runs. Never releases the allocation.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from slurm_worker import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', args.name) or not args.job_id.isdigit():
        raise ValueError('Invalid validation request/job identity')
    root = args.root.resolve()
    request = json.loads((root/'requests'/(args.name+'.json')).read_text())
    if request.get('validation_episodes') != 1:
        raise ValueError('Only an explicit one-episode validation may overlap')
    cwd = Path(request['cwd']).resolve()
    if not cwd.is_relative_to(root) or not cwd.is_dir():
        raise ValueError('Validation must run inside the private experiment')
    with (root/'claims'/(args.name+'.json')).open('x') as file:
        json.dump(dict(job_id=args.job_id, started_unix=time.time(), overlap_validation=True), file)
    started = time.time()
    child = None

    def shutdown(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    with (root/'logs'/(args.name+'.log')).open('x') as log:
        child = subprocess.Popen(['srun', '--jobid='+args.job_id, '--overlap', '--nodes=1',
            '--ntasks=1', '--cpus-per-task=8', *request['argv']], cwd=cwd,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        code = child.wait()
    save(root/'results'/(args.name+'.json'), dict(returncode=code, job_id=args.job_id,
         started_unix=started, finished_unix=time.time(), overlap_validation=True))


if __name__ == '__main__':
    main()
