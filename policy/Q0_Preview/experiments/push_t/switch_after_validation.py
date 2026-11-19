"""Keep API active until CLI validation passes, then switch at a completed episode.

Run on the Japan host. The trusted plan contains paths and argv, never secrets.
Any failed prerequisite leaves the API service alone. The Slurm job is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def model_source_hash(policy):
    files = {}
    for folder in ('harness', 'runtime', 'tools', 'assets'):
        for path in (policy/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files[str(path.relative_to(policy))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validation_root = Path(plan['validation_root'])
    state = args.plan.parent/'transition-state.json'
    ssh = ['ssh', '-F', plan['ssh_config'], '-o', 'BatchMode=yes', plan['ssh_host']]
    remote_policy = plan['remote_root']+'/code/'+plan['benchmark_name']+'/XPolicyLab/policy/Q0_Preview'

    def run_ssh(command, content):
        for attempt in range(4):
            try:
                result = subprocess.run(ssh+[command], input=content, text=True,
                                        capture_output=True, timeout=90)
            except subprocess.TimeoutExpired:
                if attempt == 3:
                    raise
            else:
                if result.returncode == 0:
                    return json.loads(result.stdout)
                if result.returncode != 255 or attempt == 3:
                    raise RuntimeError('Transition SSH failed: '+result.stderr[-2000:])
            time.sleep(5*(attempt+1))

    def control(message):
        command = shlex.join(['python3', remote_policy+'/experiments/push_t/cluster_control.py',
                              '--root', plan['remote_root']])
        return run_ssh(command, json.dumps(message))

    def api_status():
        details = dict(root=plan['remote_root'], name=plan['api_service'], run_id=plan['api_result_run_id'])
        program = '''import json
from pathlib import Path
d=json.loads(DATA); r=Path(d['root'])
p=r/'code/benchmark/eval_result/RoboDojo/push_T/Q0_Preview/arx_x5/1_ckpt_name=api,action_type=joint'/d['run_id']/'_result.json'
w=r/'results'/(d['name']+'.json')
a=r/'api-pilot-completed.json'
print(json.dumps(dict(evaluator_result=json.loads(p.read_text()) if p.exists() else None,worker_result=json.loads(w.read_text()) if w.exists() else None,service_stopped=(r/'runs'/d['name']/'STOP').exists(),boundary_archive=json.loads(a.read_text()) if a.exists() else None)))
'''.replace('DATA', repr(json.dumps(details)))
        return run_ssh('python3 -', program)

    def record(phase, **extra):
        value = dict(phase=phase, unix=time.time(), **extra)
        temporary = state.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(state)
        print(phase, flush=True)

    try:
        record('WAITING_FOR_CLI_VALIDATION')
        while not (validation_root/'VALIDATION_PASSED.json').exists():
            if (validation_root/'PIPELINE_ERROR.json').exists():
                raise RuntimeError('CLI validation failed; API service was not stopped')
            time.sleep(5)
        proof = json.loads((validation_root/'VALIDATION_PASSED.json').read_text())
        cfg_path = Path(plan['agent_config'])
        cfg = json.loads(cfg_path.read_text())
        if (cfg.get('provider') != 'codex_cli' or proof['transport']['provider'] != 'codex_cli'
                or proof['transport']['authentication'] != 'chatgpt_login'
                or proof['transport']['model'] != cfg['model']
                or proof['transport']['agent_config_sha256'] != hashlib.sha256(cfg_path.read_bytes()).hexdigest()
                or proof['remote_root'] != plan['remote_root']):
            raise ValueError('CLI validation does not match this transition')
        if model_source_hash(cfg_path.parents[1]) != plan['model_source_sha256']:
            raise ValueError('Local CLI model/tool source changed during validation; API was not stopped')
        deployment = control(dict(action='deployment', benchmark_name=plan['benchmark_name'], site_setup=plan['site_setup']))
        if deployment != proof['deployment'] or control(dict(action='freeze', benchmark_name=plan['benchmark_name'])) != proof['layouts']:
            raise ValueError('CLI deployment changed after validation; API service was not stopped')
        command = plan['formal_command']
        if ('--key-file' in command or '--preflight-only' in command or '--overlap-preflights' in command
                or '--validated-preflight' not in command):
            raise ValueError('Formal CLI command must reuse validation without an API key or another preflight')
        for flag, expected in [('--agent-config', str(cfg_path)), ('--validated-preflight', str(validation_root)),
                               ('--root', plan['formal_root']), ('--benchmark-name', plan['benchmark_name'])]:
            if flag not in command or command[command.index(flag)+1] != expected:
                raise ValueError('Formal CLI command differs from its validated configuration')
        if Path(plan['formal_root']).exists() or Path(plan['formal_log']).exists():
            raise ValueError('Formal CLI archive/log must be fresh before stopping the API service')
        status = api_status()
        completed = (status.get('evaluator_result') or {}).get('eval_time', 0)
        boundary = completed if status['service_stopped'] else completed+1
        if boundary < 1:
            raise ValueError('Cannot switch an API service without a completed episode boundary')
        if not status['service_stopped']:
            # Both launch marker and certificate are idempotent across lost SSH replies.
            details = dict(root=plan['remote_root'], service=plan['api_service'],
                policy=remote_policy, validation=proof, boundary=boundary,
                result_run_id=plan['api_result_run_id'])
            program = '''import json,subprocess,sys
from pathlib import Path
d=json.loads(DATA)
r=Path(d['root']); marker=r/'code'/('switch-'+d['service']+'.json')
if not marker.exists():
 with marker.open('x') as f: json.dump(d,f)
 proof=r/'code'/('validated-cli-'+d['service']+'.json');proof.write_text(json.dumps(d['validation']))
 result=r/'code/benchmark/eval_result/RoboDojo/push_T/Q0_Preview/arx_x5/1_ckpt_name=api,action_type=joint'/d['result_run_id']/'_result.json'
 with (r/'logs'/('switch-'+d['service']+'.log')).open('x') as log:
  subprocess.Popen([sys.executable,'-u',d['policy']+'/experiments/push_t/stop_at_boundary.py','--result',str(result),'--service',str(r/'runs'/d['service']),'--archive',str(r/'api-pilot-completed.json'),'--minimum-completed',str(d['boundary']),'--validated-preflight',str(proof)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
else:
 if json.loads(marker.read_text()) != d: raise RuntimeError('Conflicting transition marker')
print(json.dumps({'armed':True}))
'''.replace('DATA', repr(json.dumps(details)))
            run_ssh('python3 -', program)
            record('CLI_VALIDATED_WAITING_FOR_API_BOUNDARY', minimum_completed=boundary)
            while True:
                status = api_status()
                if status['service_stopped'] and status['worker_result'] is not None:
                    break
                time.sleep(5)
            archive = status.get('boundary_archive')
            if (not archive or archive.get('cli_validation') != proof
                    or archive.get('native_result', {}).get('eval_time', 0) < boundary):
                raise RuntimeError('API stopped without the requested completed boundary archive')
            (args.plan.parent/'api-completed-before-switch.json').write_text(json.dumps(archive, indent=2))
        else:
            (args.plan.parent/'api-completed-before-switch.json').write_text(json.dumps(status, indent=2))
        record('API_STOPPED_AFTER_CLI_VALIDATION', completed_boundary=boundary)
        record('RUNNING_CLI_PAIRED_EXPERIMENT')
        active = Path(plan['formal_root']).parent/'active-pipeline.json'
        active.write_text(json.dumps(dict(name=Path(plan['formal_root']).name,
            command=command, provider='codex_cli', validation_root=str(validation_root)), indent=2))
        with Path(plan['formal_log']).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        record('CLI_PAIRED_EXPERIMENT_COMPLETE')
    except BaseException as error:
        record('TRANSITION_ERROR', error=repr(error))
        raise


if __name__ == '__main__':
    main()
