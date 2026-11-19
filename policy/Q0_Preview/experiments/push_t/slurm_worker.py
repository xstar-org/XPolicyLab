"""Private filesystem request queue inside one explicitly allocated Slurm job."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sim-python', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    job = root/'jobs'/os.environ['SLURM_JOB_ID']
    job.mkdir(parents=True, exist_ok=False)
    for name in ('requests', 'claims', 'results', 'logs'):
        (root/name).mkdir(exist_ok=True)
    child = None

    def shutdown(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    save(job/'allocation.json', dict(job_id=os.environ['SLURM_JOB_ID'],
         host=os.uname().nodename, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
         status='qualifying', started_unix=time.time()))
    probe = [args.sim_python, '-u', str(Path(__file__).with_name('sim_probe.py')),
             '--output', str(job/'qualification')]
    with (job/'qualification.log').open('w') as log:
        child = subprocess.Popen(probe, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        code = child.wait()
    if code:
        save(job/'FAILED.json', dict(stage='simulation_qualification', returncode=code))
        raise SystemExit(code)
    save(job/'READY.json', dict(job_id=os.environ['SLURM_JOB_ID'],
         status='ready', qualified_unix=time.time(), api_required=False))
    print(f'SIMULATION_READY job={job.name}', flush=True)
    while True:
        for request in sorted((root/'requests').glob('*.json')):
            claim = root/'claims'/request.name
            if claim.exists() or (root/'results'/request.name).exists():
                continue
            try:
                data = json.loads(request.read_text())
                argv, cwd = data['argv'], Path(data['cwd']).resolve()
                if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
                    raise ValueError('argv must be a nonempty list of strings')
                if not cwd.is_relative_to(root) or not cwd.is_dir():
                    raise ValueError('cwd must be inside this experiment')
                with claim.open('x') as file:
                    json.dump(dict(job_id=job.name, started_unix=time.time()), file)
                started = time.time()
                with (root/'logs'/f'{request.stem}.log').open('w') as log:
                    child = subprocess.Popen(argv, cwd=cwd, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    result = child.wait()
                save(root/'results'/request.name, dict(returncode=result, job_id=job.name,
                     started_unix=started, finished_unix=time.time()))
            except (ValueError, KeyError, OSError) as error:
                save(root/'results'/request.name, dict(error=str(error), job_id=job.name))
        time.sleep(2)


if __name__ == '__main__':
    main()
