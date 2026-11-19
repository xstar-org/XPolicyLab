"""Durable gated sequence: API-free smoke, real-model preflight, then 50+50.

Run on the API host in tmux/systemd. All credentials remain on this host.
Any infrastructure failure stops the sequence; no automatic episode reruns.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from report_pair import outcomes, compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True, help='New private local experiment directory')
    p.add_argument('--remote-root', required=True, help='Existing private allocation queue root')
    p.add_argument('--ssh-config', required=True)
    p.add_argument('--ssh-host', required=True)
    p.add_argument('--site-setup', required=True, help='Cluster-side CUDA/compiler setup script')
    p.add_argument('--cluster-prefix', required=True)
    p.add_argument('--python-prefix', default=sys.prefix)
    p.add_argument('--key-file', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    policy = Path(__file__).resolve().parents[2]
    remote_policy = args.remote_root+'/code/benchmark/XPolicyLab/policy/Q0_Preview'
    ssh = ['ssh','-F',args.ssh_config,'-o','BatchMode=yes',args.ssh_host]
    secret = args.key_file.read_text().strip()
    if not secret or args.key_file.stat().st_mode & 0o077:
        raise ValueError('API key file must be nonempty and owner-only')
    remote_env = os.environ.copy()
    remote_env.pop('Q0_PREVIEW_API_KEY', None)
    remote_env.pop('OPENAI_API_KEY', None)

    def control(message):
        command = shlex.join(['python3',remote_policy+'/experiments/push_t/cluster_control.py',
                              '--root',args.remote_root])
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

    bridge = None
    active = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        frozen = control(dict(action='freeze'))
        (root/'layouts.json').write_text(json.dumps(frozen, indent=2))
        for label, mode, count, smoke in [('smoke','original',1,True),('model-preflight','original',1,False),
                                          ('original','original',50,False),('official_only','official_only',50,False)]:
            if control(dict(action='freeze')) != frozen:
                raise ValueError('Layouts changed between stages')
            name = root.name+'-'+label
            active = name
            agent_root = root/label
            env = os.environ.copy()
            env.update(Q0_PREVIEW_API_KEY=secret, Q0_PREVIEW_TASK_INFORMATION_MODE=mode)
            cmd = [str(Path(args.python_prefix)/'bin/python'),'-u',str(policy/'tools/remote_agent.py'),
                '--ssh-config',args.ssh_config,'--ssh-host',args.ssh_host,
                '--remote-root',args.remote_root+'/runs/'+name,
                '--remote-code',remote_policy+'/tools/bridge_exchange.py','--root',str(agent_root),
                '--config',str(policy/'config/agent.gpt62sol.json'),'--python-prefix',args.python_prefix]
            if smoke:
                cmd += ['--smoke','--two-group']
            with (root/(label+'-bridge.log')).open('w') as log:
                bridge = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            control(dict(action='submit',name=name,mode=mode,episodes=count,
                         cluster_prefix=args.cluster_prefix,site_setup=args.site_setup))
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
            if len(episodes) != count:
                raise ValueError(f'{label}: episode count mismatch')
            for ep in episodes:
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
