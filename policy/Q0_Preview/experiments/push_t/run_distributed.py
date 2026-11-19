"""Durable gated sequence: API-free smoke, real-model preflight, then 50+50.

Run on the API host in tmux/systemd. All credentials remain on this host.
Any infrastructure failure stops the sequence; no automatic episode reruns.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from report_pair import outcomes, compare
from native_resume import completed_sessions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True, help='New private local experiment directory')
    p.add_argument('--remote-root', required=True, help='Existing private allocation queue root')
    p.add_argument('--ssh-config', required=True)
    p.add_argument('--ssh-host', required=True)
    p.add_argument('--site-setup', required=True, help='Cluster-side CUDA/compiler setup script')
    p.add_argument('--cluster-prefix', required=True)
    p.add_argument('--python-prefix', default=sys.prefix)
    p.add_argument('--agent-config', type=Path)
    p.add_argument('--benchmark-name', default='benchmark', help='Private code snapshot directory under remote-root/code')
    p.add_argument('--key-file', type=Path, help='Required only for API transports; never used by Codex CLI')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--overlap-preflights', action='store_true', help='One-episode validation alongside the active API arm')
    p.add_argument('--job-id', help='Existing allocation, required for overlapping validation')
    p.add_argument('--validated-preflight', type=Path, help='Reuse this completed validation with identical source/config hashes')
    p.add_argument('--resume-from', type=Path, help='Retain only native-confirmed completed episodes from this interrupted CLI run')
    args = p.parse_args()
    if args.overlap_preflights and (not args.preflight_only or not args.job_id or not args.job_id.isdigit()):
        p.error('Overlap requires --preflight-only and an existing numeric --job-id')
    if args.validated_preflight and args.preflight_only:
        p.error('Choose a new preflight or a validated preflight to reuse')
    if args.resume_from and (args.preflight_only or args.overlap_preflights or not args.validated_preflight):
        p.error('Resume requires a validated formal run, without overlapping/preflight stages')
    root = args.root.resolve()
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    policy = Path(__file__).resolve().parents[2]
    agent_config = (args.agent_config or policy/'config/agent.gpt62sol.json').resolve()
    cfg = json.loads(agent_config.read_text())
    provider = cfg['provider']
    if provider not in ('openai', 'anthropic', 'codex_cli'):
        raise ValueError('Unknown model transport')
    remote_policy = args.remote_root+'/code/'+args.benchmark_name+'/XPolicyLab/policy/Q0_Preview'
    ssh = ['ssh','-F',args.ssh_config,'-o','BatchMode=yes','-o','ControlMaster=no',
           '-o','ControlPath=none','-o','ControlPersist=no','-o','ConnectTimeout=15','-o','ServerAliveInterval=10',
           '-o','ServerAliveCountMax=3',args.ssh_host]
    secret = None
    if provider != 'codex_cli':
        if args.key_file is None:
            raise ValueError('API transport requires --key-file')
        secret = args.key_file.read_text().strip()
        if not secret or args.key_file.stat().st_mode & 0o077:
            raise ValueError('API key file must be nonempty and owner-only')
    elif args.key_file is not None:
        raise ValueError('Codex CLI must use ChatGPT login, not --key-file')
    transport = dict(provider=provider, model=cfg['model'],
        agent_config_sha256=hashlib.sha256(agent_config.read_bytes()).hexdigest(),
        authentication='chatgpt_login' if provider == 'codex_cli' else 'api_key')
    (root/'transport.json').write_text(json.dumps(transport, indent=2))
    source_root = args.resume_from.resolve() if args.resume_from else None
    if source_root and json.loads((source_root/'transport.json').read_text()) != transport:
        raise ValueError('Resume transport/model/configuration differs from the source run')
    lineage = dict(source_root=str(source_root) if source_root else None, stages={})
    (root/'transport-implementation.json').write_text(json.dumps(dict(
        bridge_sha256=hashlib.sha256((policy/'tools/remote_agent.py').read_bytes()).hexdigest(),
        recovery_preserves_model_inputs=True), indent=2))
    remote_env = os.environ.copy()
    remote_env.pop('Q0_PREVIEW_API_KEY', None)
    remote_env.pop('OPENAI_API_KEY', None)

    def remote(command, message):
        for attempt in range(4):
            try:
                result = subprocess.run(ssh+[command],input=json.dumps(message),text=True,
                    capture_output=True,timeout=90,env=remote_env)
            except subprocess.TimeoutExpired:
                if attempt == 3:
                    raise
            else:
                if result.returncode == 0:
                    return json.loads(result.stdout)
                if result.returncode != 255 or attempt == 3:
                    raise RuntimeError('Cluster control failed: '+result.stderr[-4000:])
            # Retry only SSH transport. Submission is idempotent for an identical
            # run name/config, and the worker still claims each request once.
            time.sleep(5*(attempt+1))

    def control(message):
        message = dict(message, benchmark_name=args.benchmark_name)
        command = shlex.join(['python3',remote_policy+'/experiments/push_t/cluster_control.py',
                              '--root',args.remote_root])
        return remote(command, message)

    bridge = None
    active = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        frozen = control(dict(action='freeze'))
        if source_root and json.loads((source_root/'layouts.json').read_text()) != frozen:
            raise ValueError('Resume frozen layouts differ from the source run')
        (root/'layouts.json').write_text(json.dumps(frozen, indent=2))
        deployment = control(dict(action='deployment', site_setup=args.site_setup))
        (root/'deployment.json').write_text(json.dumps(deployment, indent=2))
        validation = dict(transport=transport, layouts=frozen, deployment=deployment, remote_root=args.remote_root)
        stages = [('smoke','original',1,True),('model-preflight','original',1,False),
                  ('original','original',50,False),('official_only','official_only',50,False)]
        if args.validated_preflight:
            proof = json.loads((args.validated_preflight/'VALIDATION_PASSED.json').read_text())
            if any(proof.get(k) != value for k, value in validation.items()):
                raise ValueError('Validated preflight source, config or layouts differ from this deployment')
            (root/'validation-source.json').write_text(json.dumps(dict(path=str(args.validated_preflight.resolve()), proof=proof), indent=2))
            stages = stages[2:]
        if args.preflight_only:
            stages = stages[:2]
        for label, mode, count, smoke in stages:
            if control(dict(action='freeze')) != frozen:
                raise ValueError('Layouts changed between stages')
            name = root.name+'-'+label
            inherited = []
            if source_root:
                complete = source_root/(label+'-result.json')
                if complete.exists():
                    outcomes(complete)
                    inherited = completed_sessions(source_root, label, count)
                    (root/(label+'-result.json')).write_bytes(complete.read_bytes())
                    lineage['stages'][label] = dict(initial_completed=count, inherited_sessions=inherited)
                    (root/'resume-lineage.json').write_text(json.dumps(lineage, indent=2))
                    print(f'STAGE_INHERITED {label} episodes={count}', flush=True)
                    continue
                prior = source_root/(label+'-status.json')
                prior_result = (json.loads(prior.read_text()).get('evaluator_result') or {}) if prior.exists() else {}
                if prior_result.get('eval_time', 0):
                    program = Path(__file__).with_name('native_resume.py').read_text()
                    record = remote(shlex.join(['python3','-c',program]), dict(root=args.remote_root,
                        benchmark_name=args.benchmark_name, source_service=source_root.name+'-'+label,
                        target_service=name))
                    inherited = completed_sessions(source_root, label, record['completed'])
                    lineage['stages'][label] = dict(native=record, initial_completed=record['completed'],
                                                    inherited_sessions=inherited)
                    (root/'resume-lineage.json').write_text(json.dumps(lineage, indent=2))
            active = name
            agent_root = root/label
            env = os.environ.copy()
            env['Q0_PREVIEW_TASK_INFORMATION_MODE'] = mode
            if secret is None:
                for key in ('Q0_PREVIEW_API_KEY', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'ANTHROPIC_AUTH_TOKEN'):
                    env.pop(key, None)
            else:
                env[cfg.get('api_key_env', 'Q0_PREVIEW_API_KEY')] = secret
            cmd = [str(Path(args.python_prefix)/'bin/python'),'-u',str(policy/'tools/remote_agent.py'),
                '--ssh-config',args.ssh_config,'--ssh-host',args.ssh_host,
                '--remote-root',args.remote_root+'/runs/'+name,
                '--remote-code',remote_policy+'/tools/bridge_exchange.py','--root',str(agent_root),
                '--config',str(agent_config),'--python-prefix',args.python_prefix]
            if smoke:
                cmd += ['--smoke','--two-group']
            with (root/(label+'-bridge.log')).open('w') as log:
                bridge = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            control(dict(action='submit',name=name,mode=mode,episodes=count,
                         agent_config_name=agent_config.name,
                         overlap_validation=args.overlap_preflights,
                         cluster_prefix=args.cluster_prefix,site_setup=args.site_setup))
            if args.overlap_preflights:
                control(dict(action='launch_overlap', name=name, job_id=args.job_id))
            print(f'STAGE_STARTED {label} episodes={count}',flush=True)
            stage_started = time.monotonic()
            while True:
                if bridge.poll() not in (None,0):
                    raise RuntimeError(f'Bridge failed in {label}; see local archive')
                if (time.monotonic()-stage_started > 1200
                        and not list((agent_root/'sessions').glob('*/episode.json'))):
                    raise TimeoutError('Simulator produced no policy observation in 20 minutes')
                status = control(dict(action='status',name=name))
                (root/(label+'-status.json')).write_text(json.dumps(status,indent=2))
                if status['worker_result'] is not None:
                    if status['worker_result'].get('returncode') != 0:
                        raise RuntimeError(f'Cluster stage {label} failed; see status/logs')
                    break
                time.sleep(10)
            bridge.wait(timeout=90)
            if bridge.returncode:
                raise RuntimeError(f'Bridge failed while closing {label}')
            bridge = None
            result = status['evaluator_result']
            if not result or result.get('eval_time') != count:
                raise ValueError(f'{label}: expected {count} completed evaluator episodes')
            if control(dict(action='freeze')) != frozen:
                raise ValueError('Layouts changed during the stage')
            (root/(label+'-result.json')).write_text(json.dumps(result,indent=2))
            episodes = list((agent_root/'sessions').iterdir())
            if len(episodes)+len(inherited) != count:
                raise ValueError(f'{label}: episode count mismatch')
            for ep in episodes+[Path(ep) for ep in inherited]:
                for error in (ep/'errors').glob('*.json'):
                    if json.loads(error.read_text()).get('infrastructure'):
                        raise ValueError(f'{label}: Agent infrastructure failure')
            if smoke:
                for i, decision in [(1,'CONTINUE'),(2,'REPLAN')]:
                    event = json.loads((episodes[0]/f'smoke_checkpoint_{i}.json').read_text())
                    if event['verdict']['decision'] != decision:
                        raise ValueError('Both checkpoint branches must be validated')
            elif label == 'model-preflight':
                end = json.loads((episodes[0]/'END.json').read_text())
                if end.get('groups_completed',0) < 1 or not list((episodes[0]/'api').glob('*/attempt_*/response.json')):
                    raise ValueError('Model preflight did not complete a model-generated action group')
            print(f'STAGE_PASSED {label}',flush=True)
            active = None
            if label == 'model-preflight':
                if control(dict(action='deployment', site_setup=args.site_setup)) != deployment:
                    raise ValueError('Deployment source/configuration changed during validation')
                (root/'VALIDATION_PASSED.json').write_text(json.dumps(validation, indent=2))
        if args.preflight_only:
            print('PREFLIGHT_VALIDATION_COMPLETE', flush=True)
            return
        report = compare(outcomes(root/'original-result.json'),outcomes(root/'official_only-result.json'))
        (root/'report.json').write_text(json.dumps(report,indent=2))
        print('PAIRED_EXPERIMENT_COMPLETE '+json.dumps(report),flush=True)
    except BaseException as error:
        (root/'PIPELINE_ERROR.json').write_text(json.dumps(dict(error=repr(error),stage=active,unix=time.time()),indent=2))
        raise
    finally:
        if active:
            try:
                control(dict(action='stop',name=active))
            except Exception:
                pass
        if bridge is not None and bridge.poll() is None:
            os.killpg(bridge.pid, signal.SIGTERM)
            try:
                bridge.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(bridge.pid,signal.SIGKILL)
                bridge.wait()


if __name__ == '__main__':
    main()
