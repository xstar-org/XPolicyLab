"""Run the explicitly requested official-only 50 cases alongside the original arm.

Only the trusted scheduler helper is deployed outside the frozen benchmark.
This driver owns its distinct bridge/service, never the original driver or job.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from formal_overlap import AUTHORIZATION
from native_resume import completed_sessions, infrastructure_tail_proof
from report_pair import outcomes


def resume_archive_proof(source, native_count):
    """Prove the closed prefix and exactly one unclosed infrastructure tail."""
    if type(native_count) is not int or not 0 < native_count <= 50:
        raise ValueError('Native resume requires a nonempty completed prefix')
    try:
        completed_sessions(source, 'official_only', native_count)
        valid = native_count
    except ValueError:
        valid = native_count-1
        completed_sessions(source, 'official_only', valid)
    if not 0 < valid < 50:
        raise ValueError('Resume prefix must be nonempty and incomplete')
    return valid, infrastructure_tail_proof(source, 'official_only', valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--validated-preflight', type=Path, required=True)
    parser.add_argument('--fast-validation', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--benchmark-name', required=True)
    parser.add_argument('--ssh-config', required=True)
    parser.add_argument('--ssh-host', required=True)
    parser.add_argument('--site-setup', required=True)
    parser.add_argument('--cluster-prefix', required=True)
    parser.add_argument('--python-prefix', required=True)
    parser.add_argument('--agent-config', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--user-request', required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--existing-worker-queue', action='store_true')
    args = parser.parse_args()
    if args.existing_worker_queue and not args.resume_from:
        parser.error('Existing-worker dispatch requires an audited resume source')
    root = args.root.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    policy = Path(__file__).resolve().parents[2]
    cfg = json.loads(args.agent_config.read_text())
    if (cfg.get('provider') != 'codex_cli' or cfg.get('model') != 'gpt-6.2-sol'
            or cfg.get('codex_service_tier') != 'fast'):
        raise ValueError('Parallel official-only arm requires subscription GPT-6.2 Sol Fast')
    # The existing closed Fast probe must include both ordinary text and a Q0
    # tool handoff, so tier selection is proven beyond a config-only assertion.
    receipts = [json.loads(path.read_text()) for path in args.fast_validation.glob('api/*/attempt_*/cli_receipt.json')]
    if (len(receipts) < 2 or any(r.get('model') != cfg['model'] or
            r.get('auth_method') != 'chatgpt' or r.get('requested_service_tier') != 'fast' or
            r.get('thread_service_tier') not in ('priority', 'fast') or not r.get('response_complete')
            for r in receipts) or {r.get('completion_source') for r in receipts} !=
            {'native_completed_turn', 'native_dynamic_tool_handoff'}):
        raise ValueError('Missing verified subscription Fast text/tool probe')
    original = args.original_root.resolve()
    preflight = json.loads((args.validated_preflight/'VALIDATION_PASSED.json').read_text())
    layouts = json.loads((original/'layouts.json').read_text())
    if layouts != preflight['layouts']:
        raise ValueError('Original/preflight layouts differ')
    original_cfg = json.loads((original/'original/agent_config.json').read_text())
    for key in ('provider', 'model', 'codex_service_tier', 'context_history', 'context_max_bytes', 'stream', 'sandbox'):
        if cfg.get(key) != original_cfg.get(key):
            raise ValueError('Parallel policy settings differ: '+key)
    remote_policy = args.remote_root+'/code/'+args.benchmark_name+'/XPolicyLab/policy/Q0_Preview'
    name = root.name+'-official_only'
    ssh = ['ssh', '-F', args.ssh_config, '-o', 'BatchMode=yes', '-o', 'ControlPath=none',
           '-o', 'ControlMaster=no', '-o', 'ControlPersist=no', '-o', 'ConnectTimeout=15',
           '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3', args.ssh_host]

    def remote(command, message):
        for attempt in range(4):
            try:
                result = subprocess.run(ssh+[command], input=json.dumps(message), text=True,
                    capture_output=True, timeout=90)
            except subprocess.TimeoutExpired:
                if attempt == 3:
                    raise
            else:
                if result.returncode == 0:
                    return json.loads(result.stdout)
                if result.returncode != 255 or attempt == 3:
                    raise RuntimeError('Parallel cluster control failed: '+result.stderr[-3000:])
            time.sleep(5*(attempt+1))

    def control(message):
        if args.existing_worker_queue and message.get('action') == 'status' and message.get('name') == name:
            return remote(shlex.join(['python3', remote_helper, '--root', args.remote_root]),
                          dict(authorization_message, operation='queue_status'))
        return remote(shlex.join(['python3', remote_policy+'/experiments/push_t/cluster_control.py',
                                 '--root', args.remote_root]), dict(message, benchmark_name=args.benchmark_name))

    if control(dict(action='freeze')) != layouts:
        raise ValueError('Remote frozen layouts changed')
    deployment = control(dict(action='deployment', site_setup=args.site_setup))
    if deployment != preflight['deployment']:
        raise ValueError('Immutable simulator deployment differs from validated snapshot')
    (root/'layouts.json').write_text(json.dumps(layouts, indent=2))
    (root/'deployment.json').write_text(json.dumps(deployment, indent=2))
    (root/'transport.json').write_text(json.dumps(dict(provider='codex_cli', model=cfg['model'],
        service_tier='fast', authentication='chatgpt_login',
        agent_config_sha256=hashlib.sha256(args.agent_config.read_bytes()).hexdigest()), indent=2))
    (root/'validation-source.json').write_text(json.dumps(dict(preflight=str(args.validated_preflight.resolve()),
        fast_probe=str(args.fast_validation.resolve()), original_root=str(original)), indent=2))
    message = dict(name=name, co_run=original.name+'-original', job_id=args.job_id,
        mode='official_only', episodes=50, authorization=AUTHORIZATION, gpu=0,
        gpu_uuid=args.gpu_uuid, user_request=args.user_request, benchmark_name=args.benchmark_name,
        site_setup=args.site_setup, cluster_prefix=args.cluster_prefix,
        agent_config_name=args.agent_config.name, deployment=deployment, layouts=layouts)
    if args.resume_from:
        source = args.resume_from.resolve()
        expected_cfg = dict(cfg, task_information_mode='official_only',
            python_prefix=str(Path(args.python_prefix).resolve()), base_url='',
            api_key_env=cfg.get('api_key_env', 'Q0_PREVIEW_API_KEY'))
        if (json.loads((source/'layouts.json').read_text()) != layouts
                or json.loads((source/'deployment.json').read_text()) != deployment
                or json.loads((source/'official_only/agent_config.json').read_text()) != expected_cfg):
            raise ValueError('Resume source layouts, deployment or policy config differs')
        prior = control(dict(action='status', name=source.name+'-official_only'))
        (root/'resume-source-status.json').write_text(json.dumps(prior, indent=2))
        if prior.get('worker_result') is None:
            raise ValueError('Interrupted predecessor must be terminal before resuming')
        count = (prior.get('evaluator_result') or {}).get('eval_time', 0)
        # Some native interruptions leave the tail uncounted; others write a
        # failure before closing. Only a proven unclosed infrastructure tail
        # may be removed, never a closed genuine failure.
        valid, tail = resume_archive_proof(source, count)
        (root/'interrupted-tail-proof.json').write_text(json.dumps(tail, indent=2))
        request = dict(root=args.remote_root, benchmark_name=args.benchmark_name,
            source_service=source.name+'-official_only', target_service=name)
        if count == valid+1:
            request['exclude_infrastructure_tail'] = tail
        program = Path(__file__).with_name('native_resume.py').read_text()
        record = remote(shlex.join(['python3', '-c', program]), request)
        if record['completed'] != valid:
            raise ValueError('Native resume prefix does not match closed archives')
        lineage = dict(stages=dict(official_only=dict(native=record, initial_completed=valid,
            inherited_sessions=completed_sessions(source, 'official_only', valid))))
        (root/'resume-lineage.json').write_text(json.dumps(lineage, indent=2))
        message.update(resume_source_service=request['source_service'], resume_record=record)
    if args.existing_worker_queue:
        message['existing_worker_queue'] = True
    authorization_message = message
    (root/'formal-authorization.json').write_text(json.dumps(message, indent=2, ensure_ascii=False))
    helper = Path(__file__).with_name('formal_overlap.py').read_bytes()
    remote_helper = args.remote_root+'/code/formal_overlap-'+hashlib.sha256(helper).hexdigest()[:16]+'.py'
    upload = "import base64,json,sys;from pathlib import Path;m=json.load(sys.stdin);p=Path(m['path']);b=base64.b64decode(m['source']);assert not p.exists() or p.read_bytes()==b;p.write_bytes(b);print(json.dumps({'sha256':__import__('hashlib').sha256(b).hexdigest()}))"
    remote(shlex.join(['python3', '-c', upload]), dict(path=remote_helper, source=base64.b64encode(helper).decode()))
    bridge = None
    stage_active = False

    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        env = os.environ.copy()
        env['Q0_PREVIEW_TASK_INFORMATION_MODE'] = 'official_only'
        for key in ('Q0_PREVIEW_API_KEY', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'ANTHROPIC_AUTH_TOKEN'):
            env.pop(key, None)
        cmd = [str(Path(args.python_prefix)/'bin/python'), '-u', str(policy/'tools/remote_agent.py'),
            '--ssh-config', args.ssh_config, '--ssh-host', args.ssh_host,
            '--remote-root', args.remote_root+'/runs/'+name,
            '--remote-code', remote_policy+'/tools/bridge_exchange.py', '--root', str(root/'official_only'),
            '--config', str(args.agent_config.resolve()), '--python-prefix', args.python_prefix]
        with (root/'official_only-bridge.log').open('x') as log:
            bridge = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root/'processes.json').write_text(json.dumps(dict(driver_pid=os.getpid(), bridge_pid=bridge.pid,
            service=name, job_id=args.job_id), indent=2))
        stage_active = True
        proof = remote(shlex.join(['python3', remote_helper, '--root', args.remote_root]), message)
        (root/'overlap-launch.json').write_text(json.dumps(proof, indent=2))
        stage_active = True
        event = 'OFFICIAL_ONLY_QUEUED' if args.existing_worker_queue else 'PARALLEL_OFFICIAL_ONLY_STARTED'
        print(event+' episodes=50 '+json.dumps(proof['launch']), flush=True)
        started = time.monotonic()
        claimed_at = None
        while True:
            if bridge.poll() not in (None, 0):
                raise RuntimeError('Parallel official-only bridge failed')
            status = control(dict(action='status', name=name))
            claim = status.get('allocation_claim')
            if claim:
                if claim.get('job_id') != args.job_id:
                    raise ValueError('Queued arm was claimed by a different allocation')
                if claimed_at is None:
                    claimed_at = time.monotonic()
            if args.existing_worker_queue:
                status['queue_wait_s'] = (claimed_at or time.monotonic())-started
                status['claimed_startup_elapsed_s'] = (time.monotonic()-claimed_at
                    if claimed_at is not None else None)
            waiting_since = claimed_at if args.existing_worker_queue else started
            if (waiting_since is not None and time.monotonic()-waiting_since > 1200
                    and not list((root/'official_only/sessions').glob('*/episode.json'))):
                raise TimeoutError('Claimed simulator produced no observation in 20 minutes')
            temporary = root/'official_only-status.tmp'
            temporary.write_text(json.dumps(status, indent=2))
            temporary.replace(root/'official_only-status.json')
            if status['worker_result'] is not None:
                if status['worker_result'].get('returncode') != 0:
                    raise RuntimeError('Parallel official-only simulator failed')
                break
            time.sleep(10)
        bridge.wait(timeout=90)
        if bridge.returncode:
            raise RuntimeError('Parallel official-only bridge failed while closing')
        result = status['evaluator_result']
        path = root/'official_only-result.pending.json'
        path.write_text(json.dumps(result, indent=2))
        outcomes(path)
        completed_sessions(root, 'official_only', 50)
        if control(dict(action='freeze')) != layouts:
            raise ValueError('Layouts changed during the parallel arm')
        path.replace(root/'official_only-result.json')
        stage_active = False
        print('PARALLEL_OFFICIAL_ONLY_COMPLETE', flush=True)
    except BaseException as error:
        (root/'PIPELINE_ERROR.json').write_text(json.dumps(dict(error=repr(error), unix=time.time()), indent=2))
        raise
    finally:
        if stage_active:
            try:
                control(dict(action='stop', name=name))
            except Exception:
                pass
        if bridge is not None and bridge.poll() is None:
            os.killpg(bridge.pid, signal.SIGTERM)
            try:
                bridge.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(bridge.pid, signal.SIGKILL)
                bridge.wait()


if __name__ == '__main__':
    main()
