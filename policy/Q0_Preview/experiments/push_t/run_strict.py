"""Fresh strict arm on a dedicated one-GPU queue, with a mandatory smoke gate."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from audit_strict import audit
from report_pair import outcomes
from strict_resume import audit_view, resume_evidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--ssh-config', required=True)
    p.add_argument('--ssh-host', default='q0-cluster')
    p.add_argument('--job-id', required=True)
    p.add_argument('--site-setup', required=True)
    p.add_argument('--cluster-prefix', required=True)
    p.add_argument('--python-prefix', required=True)
    p.add_argument('--expected-layouts', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--validated-smoke', type=Path)
    p.add_argument('--validated-inputs', type=Path)
    p.add_argument('--resume-from', type=Path, help='Terminal transport-interrupted strict run; preserve its audited native prefix')
    args = p.parse_args()
    if not args.smoke and not (args.validated_smoke and args.validated_inputs):
        p.error('Formal run requires --validated-smoke and --validated-inputs')
    if args.smoke and args.resume_from:
        p.error('Smoke runs cannot inherit formal results')
    policy = Path(__file__).resolve().parents[2]
    root = args.root.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    ssh = ['ssh', '-F', args.ssh_config, '-o', 'BatchMode=yes', '-o', 'ControlPath=none',
           '-o', 'ConnectTimeout=12', '-o', 'ServerAliveInterval=10',
           '-o', 'ServerAliveCountMax=3', args.ssh_host]
    remote_policy = args.remote_root+'/code/benchmark/XPolicyLab/policy/Q0_Preview'
    helper = remote_policy+'/experiments/push_t/cluster_control.py'

    def remote(argv, message=None):
        # Every RPC here is a read or an idempotent queue/STOP write. A management
        # timeout is not evidence that the live simulator stopped.
        attempt = 0
        while True:
            attempt += 1
            try:
                result = subprocess.run(ssh+[shlex.join(argv)], input=json.dumps(message) if message else None,
                                        text=True, capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                error = 'management SSH observation timeout'
            else:
                if result.returncode == 0:
                    return json.loads(result.stdout)
                if result.returncode != 255:
                    raise RuntimeError('Cluster RPC failed: '+result.stderr[-2000:])
                error = result.stderr[-1000:]
            (root/'management-retry.json').write_text(json.dumps(dict(attempt=attempt,
                unix=time.time(), error=error, action=(message or {}).get('action'))))
            print('MANAGEMENT_RETRY '+str(attempt)+' '+error, flush=True)
            time.sleep(min(15, attempt*2))

    def control(action, name=None, **kwargs):
        return remote(['python3', helper, '--root', args.remote_root],
                      dict(action=action, benchmark_name='benchmark', name=name or root.name, **kwargs))

    queue = remote(['python3', '-c',
        'import subprocess,json; print(json.dumps(subprocess.check_output('
        '["squeue","-h","-j",'+repr(args.job_id)+',"-o","%T"],text=True).strip()))'])
    if queue != 'RUNNING':
        raise ValueError('Requested allocation is not RUNNING: '+repr(queue))
    qualification = remote(['python3', '-c',
        'import json; print(json.dumps(json.load(open('+repr(args.remote_root+'/jobs/'+args.job_id+'/READY.json')+'))))'])
    if qualification.get('job_id') != args.job_id or qualification.get('status') != 'ready':
        raise ValueError('Single-GPU physics/RGB qualification missing')
    layouts = control('freeze')
    if layouts != json.loads(args.expected_layouts.read_text()):
        raise ValueError('Layouts differ from the previous 50 paired cases')
    deployment = control('deployment', site_setup=args.site_setup, include_overlay=True)
    dependencies = remote(['env', 'PYTHONPATH='+args.remote_root+'/code/overlay',
        args.cluster_prefix+'/bin/python', '-c',
        'import json,websockets; from websockets.asyncio.server import serve; '
        'print(json.dumps({"websockets":websockets.__version__}))'])
    (root/'controller-dependencies.json').write_text(json.dumps(dependencies, indent=2))
    local_assets = audit([])
    if not args.smoke:
        gate = json.loads((args.validated_smoke/'SMOKE_PASSED.json').read_text())
        if gate['deployment'] != deployment or gate['asset_sha256'] != local_assets['asset_sha256']:
            raise ValueError('Deployment or strict inputs changed since smoke validation')
        inputs = json.loads((args.validated_inputs/'INPUT_PREFLIGHT_PASSED.json').read_text())
        if not inputs.get('passed') or inputs.get('model_calls') != 0:
            raise ValueError('Missing actual sandbox/payload preflight')
        aliases = {'geometry_reference.md': 'assets/geometry/strict_geometry.md',
                   'geometry_interfaces.md': 'assets/geometry/geometry_interfaces.md',
                   'vlm_geom.py': 'harness/helpers/vlm_geom.py',
                   'observation_io.py': 'harness/helpers/observation_io.py'}
        if inputs['sandbox_assets_sha256'] != {name:local_assets['asset_sha256'][source]
                                              for name,source in aliases.items()}:
            raise ValueError('Sandbox assets changed since input preflight')
    for name, value in [('layouts', layouts), ('deployment', deployment), ('input-audit', local_assets)]:
        (root/(name+'.json')).write_text(json.dumps(value, indent=2))
    inherited = []
    if args.resume_from:
        source = args.resume_from.resolve()
        if (json.loads((source/'deployment.json').read_text()) != deployment
                or json.loads((source/'layouts.json').read_text()) != layouts
                or json.loads((source/'input-audit.json').read_text())['asset_sha256'] != local_assets['asset_sha256']):
            raise ValueError('Resume deployment, layouts or strict inputs changed')
        prior = control('status', name=source.name)
        (root/'resume-source-status.json').write_text(json.dumps(prior, indent=2))
        if prior.get('worker_result') is None or 'returncode' not in prior['worker_result']:
            raise ValueError('Source worker must be terminal before resuming')
        evidence = resume_evidence(source, prior['evaluator_result'])
        inherited = [Path(p) for p in evidence['inherited_sessions']]
        audit(inherited)
        request = dict(root=args.remote_root, benchmark_name='benchmark',
                       source_service=source.name, target_service=root.name)
        if evidence['exclude_native_tail']:
            request['exclude_infrastructure_tail'] = evidence['transport_interruption']
        record = remote(['python3', '-c', Path(__file__).with_name('native_resume.py').read_text()], request)
        if record['completed'] != evidence['completed']:
            raise ValueError('Native resumed prefix differs from closed archives')
        lineage = dict(stages={'strict_official':dict(native=record,
            initial_completed=evidence['completed'], **evidence)})
        (root/'resume-lineage.json').write_text(json.dumps(lineage, indent=2))
    env = os.environ.copy()
    env['Q0_PREVIEW_TASK_INFORMATION_MODE'] = 'strict_official'
    for key in ('OPENAI_API_KEY', 'Q0_PREVIEW_API_KEY', 'OPENAI_BASE_URL', 'ANTHROPIC_AUTH_TOKEN'):
        env.pop(key, None)
    command = [str(Path(args.python_prefix)/'bin/python'), '-u', str(policy/'tools/remote_agent.py'),
        '--ssh-config', args.ssh_config, '--ssh-host', args.ssh_host,
        '--remote-root', args.remote_root+'/runs/'+root.name, '--remote-code', remote_policy+'/tools/bridge_exchange.py',
        '--root', str(root/'strict_official'), '--python-prefix', args.python_prefix,
        '--config', str(policy/'config/agent.gpt62sol.strict.json')]
    if args.smoke:
        command += ['--smoke', '--two-group']
    bridge = None
    try:
        with (root/'bridge.log').open('x') as log:
            bridge = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root/'processes.json').write_text(json.dumps(dict(driver_pid=os.getpid(), bridge_pid=bridge.pid,
            job_id=args.job_id, service=root.name), indent=2))
        control('submit', mode='strict_official', episodes=1 if args.smoke else 50,
                site_setup=args.site_setup, cluster_prefix=args.cluster_prefix,
                agent_config_name='agent.gpt62sol.strict.json')
        began = time.monotonic()
        while True:
            if bridge.poll() not in (None, 0):
                raise RuntimeError('Strict agent bridge exited unexpectedly')
            status = control('status')
            (root/'status.json').write_text(json.dumps(status, indent=2))
            if status['worker_result'] is not None:
                if status['worker_result'].get('job_id') != args.job_id or status['worker_result'].get('returncode') != 0:
                    raise RuntimeError('Strict simulation worker failed or used another allocation')
                break
            if time.monotonic()-began > 1200 and not list((root/'strict_official/sessions').glob('*/episode.json')):
                raise TimeoutError('No first observation in 20 minutes')
            time.sleep(10)
        bridge.wait(timeout=90)
        if bridge.returncode:
            raise RuntimeError('Strict bridge failed while closing')
        sessions = inherited+sorted((root/'strict_official/sessions').glob('*'))
        checked = audit(sessions)
        (root/'final-input-audit.json').write_text(json.dumps(checked, indent=2))
        if args.smoke:
            if len(sessions) != 1 or len(list(sessions[0].glob('smoke_checkpoint_*.json'))) != 2:
                raise ValueError('Missing two-group continuation/replan smoke evidence')
            end = json.loads((sessions[0]/'END.json').read_text())
            if not end['final'] or end['groups_completed'] < 3 or end['reason'] != 'integration_smoke_complete':
                raise ValueError('Strict controller motion smoke did not close normally')
            (root/'SMOKE_PASSED.json').write_text(json.dumps(dict(deployment=deployment,
                asset_sha256=local_assets['asset_sha256'], audit=checked, end=end), indent=2))
        else:
            result = root/'strict-result.pending.json'
            result.write_text(json.dumps(status['evaluator_result'], indent=2))
            if len(outcomes(result)) != 50 or len(sessions) != 50 or not all(e['closed'] for e in checked['episodes']):
                raise ValueError('Exactly 50 closed native cases required')
            audit_view(root, sessions)
            result.replace(root/'strict-result.json')
        print('STRICT_SMOKE_COMPLETE' if args.smoke else 'STRICT_FIFTY_COMPLETE', flush=True)
    except BaseException as exc:
        (root/'ERROR.json').write_text(json.dumps(dict(error=repr(exc), unix=time.time()), indent=2))
        raise
    finally:
        if bridge is not None and bridge.poll() is None:
            control('stop')
            os.killpg(bridge.pid, signal.SIGTERM)
            bridge.wait(timeout=40)
        # The seven-day allocation remains available. Never scancel it here.


if __name__ == '__main__':
    main()
