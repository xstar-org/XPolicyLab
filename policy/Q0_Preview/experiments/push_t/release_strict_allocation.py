"""Release this experiment's GPU only after complete native results are saved.

Run on the trusted agent host, outside the policy sandbox. This is an explicit
post-run operation; it never changes policy inputs, outcomes or the simulator.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import shlex
import subprocess
import time


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('Missing or invalid completion evidence: '+str(path)) from error


def validate_completed_run(root, job_id, native_result, native_proof):
    root = Path(root)
    if not job_id.isdigit():
        raise ValueError('A single numeric Slurm allocation ID is required')
    result = read_json(root/'strict-result.json')
    rows = result.get('details', {})
    if (result.get('eval_time') != 50 or len(rows) != 50
            or set(rows) != {str(i) for i in range(50)}
            or any(type(r.get('layout_id')) is not int or r['layout_id'] != int(k)
                   or type(r.get('success')) is not bool for k, r in rows.items())):
        raise ValueError('Exactly 50 unique typed native outcomes are required')
    if not math.isclose(result['success_rate'], sum(r['success'] for r in rows.values())/50,
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError('Native aggregate and per-episode outcomes disagree')
    processes = read_json(root/'processes.json')
    status = read_json(root/'status.json')
    worker = status.get('worker_result') or {}
    if (processes.get('job_id') != job_id or worker.get('job_id') != job_id
            or worker.get('returncode') != 0 or status.get('evaluator_result') != result):
        raise ValueError('The same allocation must have finished simulation normally')
    audit = read_json(root/'final-input-audit.json')
    if (audit.get('episodes_checked') != 50 or audit.get('runtime_verified') is not True
            or audit.get('input_checks_passed') is not True
            or len(audit.get('episodes', [])) != 50
            or not all(r.get('closed') is True for r in audit['episodes'])):
        raise ValueError('All 50 runtime archives must be closed and input-audited')
    raw = Path(native_result).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    proof = read_json(native_proof)
    checks = proof.get('checks', {})
    if (json.loads(raw) != result or proof.get('native_sha256') != digest
            or proof.get('count') != 50 or proof.get('native_matches_driver') is not True
            or set(checks) != {'deployment', 'freeze'}
            or not all(v.get('matches_formal_launch') is True for v in checks.values())
            or checks['deployment'].get('value') != read_json(root/'deployment.json')
            or checks['freeze'].get('value') != read_json(root/'layouts.json')):
        raise ValueError('Saved cluster-native result and frozen-source proof are required')
    return dict(native_sha256=digest, episodes=50,
                successes=sum(r['success'] for r in rows.values()),
                simulation_worker_returncode=0, source_service=root.name)


def validate_job_identity(snapshot, job_id, remote_root):
    """Also run immediately before scancel on the cluster; mismatch fails closed."""
    if not snapshot['queue_present']:
        return
    fields = snapshot['job_fields']
    expected = remote_root.rstrip('/')
    tres = dict(item.split('=', 1) for item in fields.get('AllocTRES', '').split(',')
                if '=' in item)
    if (fields.get('JobId') != job_id or fields.get('JobName') != 'q0-strict-info-50'
            or fields.get('UserId') != snapshot['current_user_id']
            or fields.get('Partition') != 'gpu_4090' or fields.get('NumNodes') != '1'
            or tres.get('gres/gpu') != '1'
            or fields.get('Command') != expected+'/code/allocation.sbatch'
            or fields.get('StdOut') != expected+'/logs/allocation-'+job_id+'.log'):
        raise ValueError('Refusing to cancel an allocation with a different identity')


REMOTE_SNAPSHOT = '''
import json,os,pwd,subprocess,sys
m=json.load(sys.stdin); job=m['job_id']; username=pwd.getpwuid(os.getuid()).pw_name
q=subprocess.run(['squeue','--noheader','--user='+username,'--format=%i|%T|%j'],text=True,capture_output=True)
if q.returncode:raise RuntimeError('Scheduler absence cannot be verified: '+q.stderr)
lines=[line for line in q.stdout.splitlines() if line.split('|',1)[0].strip()==job]
j=subprocess.run(['scontrol','show','job',job,'-o'],text=True,capture_output=True)
fields=dict(part.split('=',1) for part in j.stdout.split() if '=' in part)
snapshot=dict(queue_present=bool(lines),queue_rows=lines,job_fields=fields,
              current_user_id=username+'('+str(os.getuid())+')',scheduler_query_returncode=q.returncode,
              job_query_returncode=j.returncode)
'''


def cluster_request(ssh_config, ssh_host, message):
    # Re-check identity remotely in the same process that issues scancel.
    program = REMOTE_SNAPSHOT+'\n'+inspect.getsource(validate_job_identity)+'''
if m['action']=='cancel':
 validate_job_identity(snapshot,job,m['remote_root'])
 if snapshot['queue_present']:
  p=subprocess.run(['scancel',job],text=True,capture_output=True)
  snapshot['scancel']=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
elif m['action']!='inspect':raise ValueError('Unknown scheduler operation')
print(json.dumps(snapshot))
'''
    command = ['ssh', '-F', str(ssh_config), '-o', 'BatchMode=yes', '-o', 'ControlPath=none',
               '-o', 'ConnectTimeout=12', ssh_host, shlex.join(['python3', '-c', program])]
    result = subprocess.run(command, input=json.dumps(message), text=True,
                            capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError('Scheduler operation failed: '+result.stderr[-1000:])
    return json.loads(result.stdout)


def save_proof(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def release(args, request=None):
    verified = validate_completed_run(args.root, args.job_id, args.native_result, args.native_proof)
    if not args.remote_root.startswith('/') or '..' in Path(args.remote_root).parts:
        raise ValueError('An absolute experiment root without parent traversal is required')
    request = request or (lambda m: cluster_request(args.ssh_config, args.ssh_host, m))
    payload = dict(job_id=args.job_id, remote_root=args.remote_root)
    before = request(dict(payload, action='inspect'))
    validate_job_identity(before, args.job_id, args.remote_root)
    proof = dict(job_id=args.job_id, released=False, verified_completed_run=verified,
                 requested_after_complete_fifty=True, before=before,
                 cancel_requested=False, scheduler_absence_checks=[])
    save_proof(args.output, proof)
    if before['queue_present']:
        cancelled = request(dict(payload, action='cancel'))
        proof['cancel_operation'] = cancelled
        proof['cancel_requested'] = 'scancel' in cancelled
        save_proof(args.output, proof)
    deadline = time.monotonic()+args.wait_seconds
    while True:
        after = request(dict(payload, action='inspect'))
        proof['scheduler_absence_checks'].append(dict(unix=time.time(), snapshot=after))
        if not after['queue_present']:
            proof.update(released=True, verified_unix=time.time(),
                         allocation_absent_from_scheduler=True)
            save_proof(args.output, proof)
            return proof
        save_proof(args.output, proof)
        if time.monotonic() >= deadline:
            raise RuntimeError('Allocation is still visible; cancellation is not proven complete')
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'native-result', 'native-proof', 'ssh-config', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['job-id', 'remote-root']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--ssh-host', default='q0-cluster')
    parser.add_argument('--wait-seconds', type=int, default=120)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 300:
        parser.error('--wait-seconds must be between 0 and 300')
    proof = release(args)
    print(json.dumps(dict(job_id=proof['job_id'], released=proof['released'],
                          native_sha256=proof['verified_completed_run']['native_sha256'])))


if __name__ == '__main__':
    main()
