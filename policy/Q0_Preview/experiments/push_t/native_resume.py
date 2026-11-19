"""Resume only a verified native result prefix; never turn an interruption into a failure.

This trusted helper can run over SSH using stdlib Python. No agent tool exposes it.
Source archives stay immutable; target services and local agent workspaces are fresh.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid


def atomic_exact(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Conflicting existing resume artifact: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)


def prepare(message):
    root = Path(message['root']).resolve()
    source, target = message['source_service'], message['target_service']
    benchmark = message['benchmark_name']
    source_benchmark = message.get('source_benchmark_name', benchmark)
    for name in (source, target, benchmark, source_benchmark):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', name):
            raise ValueError('Invalid resume service/snapshot identity')
    if source == target:
        raise ValueError('Resume requires a fresh service')
    certificate = root/'code'/('resume-'+target+'.json')
    if certificate.exists():
        record = json.loads(certificate.read_text())
        if record['source_service'] != source or record['benchmark_name'] != benchmark:
            raise ValueError('Conflicting resume certificate')
        if record.get('excluded_infrastructure_tail') != message.get('exclude_infrastructure_tail'):
            raise ValueError('Conflicting infrastructure exclusion proof')
        if record.get('source_benchmark_name', benchmark) != source_benchmark:
            raise ValueError('Conflicting source benchmark')
        if record.get('completed_prefix_proof') != message.get('completed_prefix_proof'):
            raise ValueError('Conflicting completed prefix proof')
        return record
    worker = root/'results'/(source+'.json')
    if not worker.exists() or 'returncode' not in json.loads(worker.read_text()):
        raise ValueError('Source worker must be terminal before native resume')
    if (root/'runs'/target).exists():
        raise ValueError('Target controller service already exists')
    relative = 'eval_result/RoboDojo/push_T/Q0_Preview/arx_x5/1_ckpt_name=api,action_type=joint'
    base = root/'code'/benchmark/relative
    source_base = root/'code'/source_benchmark/relative
    old_id, new_id = 'q0-'+source, 'q0-'+target
    source_result = source_base/old_id/'_result.json'
    source_checkpoint = source_base/('_resume_'+old_id+'.json')
    result_bytes = source_result.read_bytes()
    result = json.loads(result_bytes)
    checkpoint_bytes = source_checkpoint.read_bytes() if source_checkpoint.exists() else None
    checkpoint = json.loads(checkpoint_bytes) if checkpoint_bytes is not None else None
    count = result.get('eval_time')
    details = result.get('details', {})
    if (type(count) is not int or not 0 < count <= 50 or len(details) != count
            or (count == 50 and message.get('exclude_infrastructure_tail') is None)):
        raise ValueError('Resume requires a nonempty incomplete native result prefix')
    ids = [row.get('layout_id') for row in details.values()]
    if (any(type(i) is not int for i in ids) or set(ids) != set(range(count))
            or any(type(row.get('success')) is not bool for row in details.values())):
        raise ValueError('Native resume requires unique completed frozen layout IDs')
    prefix_proof = message.get('completed_prefix_proof')
    if prefix_proof is not None:
        # A normal native completion deletes its resume manifest. Extend only
        # the explicitly audited 25-case push_T default to the requested 50.
        if (checkpoint is not None or count != 25 or source_benchmark == benchmark
                or json.loads(worker.read_text()).get('returncode') != 0
                or message.get('exclude_infrastructure_tail') is not None
                or prefix_proof.get('completed') != count
                or not re.fullmatch(r'[0-9a-f]{64}', prefix_proof.get('closed_archives_sha256', ''))):
            raise ValueError('Invalid completed-prefix extension')
        old_config = (root/'code'/source_benchmark/'task/RoboDojo/config/_task.yml').read_bytes()
        new_config = (root/'code'/benchmark/'task/RoboDojo/config/_task.yml').read_bytes()
        pattern = rb'(\r?\n  push_T:\r?\n(?:(?!\r?\n  [^ ]).)*?\r?\n    eval_nums: )25(?=\s*\n)'
        expected, substitutions = re.subn(pattern, rb'\g<1>50', old_config, flags=re.S)
        if substitutions != 1 or new_config != expected:
            raise ValueError('Extension must change only push_T evaluation count from 25 to 50')
        log_bytes = (root/'logs'/(source+'.log')).read_bytes()
        if b'(eval completed)' not in log_bytes or b'[MAIN] external-agent eval finished' not in log_bytes:
            raise ValueError('Source did not finish native evaluation normally')
        successes = sum(row['success'] for row in details.values())
        if (any(row.get('score') != float(row['success']) for row in details.values())
                or result.get('success_rate') != successes/count
                or abs(result.get('score', -1)-100*successes/count) > 1e-9):
            raise ValueError('Completed prefix counters disagree')
        checkpoint = dict(run_id=old_id, task_name='push_T', policy_name='Q0_Preview',
            config_name='arx_x5', eval_seed=1, additional_info='ckpt_name=api,action_type=joint',
            details=details, completed_layout_ids=list(range(count)), abandoned_layout_ids=[],
            success_nums=successes, fail_nums=count-successes, total_score=successes,
            unstable_nums=0, restart_count=0)
    elif checkpoint is None:
        raise ValueError('Missing native checkpoint; completed extension requires explicit proof')
    if (checkpoint.get('run_id') != old_id or checkpoint.get('details') != details
            or checkpoint.get('abandoned_layout_ids') or checkpoint.get('completed_layout_ids') != list(range(count))
            or checkpoint.get('success_nums') != sum(row['success'] for row in details.values())
            or checkpoint.get('fail_nums') != sum(not row['success'] for row in details.values())):
        raise ValueError('Native checkpoint and official completed results disagree')
    for key, expected in [('task_name','push_T'),('policy_name','Q0_Preview'),('config_name','arx_x5'),
                          ('eval_seed',1),('additional_info','ckpt_name=api,action_type=joint')]:
        if checkpoint.get(key) != expected:
            raise ValueError('Native checkpoint configuration changed: '+key)
    exclusion = message.get('exclude_infrastructure_tail')
    if exclusion is not None:
        # The native evaluator can write a failure while the controller closes
        # after an adapter error. Only the last, proven infrastructure case may
        # be removed in a NEW result; preserve both source artifacts verbatim.
        if (not isinstance(exclusion, dict) or exclusion.get('layout_id') != count-1
                or count < 2 or exclusion.get('infrastructure') is not True
                or exclusion.get('archive_closed') is not False
                or not re.fullmatch(r'[0-9a-f]{64}', exclusion.get('error_sha256', ''))
                or not exclusion.get('error_path') or not exclusion.get('session_path')):
            raise ValueError('Invalid infrastructure tail exclusion proof')
        tail = next(row for row in details.values() if row['layout_id'] == count-1)
        if tail['success'] is not False:
            raise ValueError('Cannot exclude a native successful episode')
        details = {key: row for key, row in details.items() if row['layout_id'] < count-1}
        count -= 1
        ids = list(range(count))
        successes = sum(row['success'] for row in details.values())
        result.update(details=details, eval_time=count, success_rate=successes/count,
                      score=100*successes/count)
        checkpoint.update(details=details, completed_layout_ids=ids, success_nums=successes,
                          fail_nums=count-successes, total_score=sum(row['score'] for row in details.values()))
    checkpoint.update(run_id=new_id,
        save_dir=str((base/new_id).relative_to(root/'code'/benchmark)),
        restart_count=int(checkpoint.get('restart_count', 0))+1)
    record = dict(source_service=source, target_service=target, benchmark_name=benchmark,
        completed=count, completed_layout_ids=sorted(ids), source_result=str(source_result),
        source_result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        source_checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest() if checkpoint_bytes is not None else None,
        target_result=str(base/new_id/'_result.json'),
        interrupted_episode_counted=False)
    if exclusion is not None:
        record['excluded_infrastructure_tail'] = exclusion
    if source_benchmark != benchmark:
        record['source_benchmark_name'] = source_benchmark
    if prefix_proof is not None:
        record.update(completed_prefix_proof=prefix_proof,
            checkpoint_reconstructed_after_native_completion=True,
            source_completion_log_sha256=hashlib.sha256(log_bytes).hexdigest(),
            source_task_config_sha256=hashlib.sha256(old_config).hexdigest(),
            target_task_config_sha256=hashlib.sha256(new_config).hexdigest())
    target_bytes = json.dumps(result, indent=2).encode() if exclusion is not None else result_bytes
    atomic_exact(base/new_id/'_result.json', target_bytes)
    atomic_exact(base/('_resume_'+new_id+'.json'), json.dumps(checkpoint, indent=2).encode())
    atomic_exact(certificate, json.dumps(record, indent=2).encode())
    return record


def completed_prefix_proof(source_root, label, total):
    selected = completed_sessions(source_root, label, total)
    payload = [dict(path=path, end_sha256=hashlib.sha256((Path(path)/'END.json').read_bytes()).hexdigest())
               for path in selected]
    return dict(completed=total, closed_archives_sha256=hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest())


def infrastructure_tail_proof(source_root, label, completed):
    """Build trusted operator evidence; never available to the policy/model."""
    source_root = Path(source_root).resolve()
    completed_sessions(source_root, label, completed)
    lineage_file = source_root/'resume-lineage.json'
    lineage = json.loads(lineage_file.read_text()) if lineage_file.exists() else {}
    inherited = lineage.get('stages', {}).get(label, {}).get('inherited_sessions', [])
    local = sorted((source_root/label/'sessions').glob('*'),
        key=lambda ep: json.loads((ep/'episode.json').read_text())['wall_started_unix_s'])
    remaining = local[completed-len(inherited):]
    if len(remaining) != 1:
        raise ValueError('Exclusion requires exactly one interrupted tail session')
    episode = remaining[0]
    if (episode/'END.json').exists():
        raise ValueError('Cannot exclude a closed policy episode')
    meta = json.loads((episode/'episode.json').read_text())
    if meta.get('task') != 'push_T' or meta.get('task_information_mode') != label:
        raise ValueError('Infrastructure tail task/mode mismatch')
    errors = [p for p in (episode/'errors').glob('*.json')
              if json.loads(p.read_text()).get('infrastructure') is True]
    if not errors:
        raise ValueError('No infrastructure failure proof for tail episode')
    error = sorted(errors)[0]
    return dict(layout_id=completed, infrastructure=True, archive_closed=False,
        session_path=str(episode), error_path=str(error),
        error_sha256=hashlib.sha256(error.read_bytes()).hexdigest())


def completed_sessions(source_root, label, total):
    """Collect native-confirmed prefix episodes across any earlier resume segments."""
    source_root = Path(source_root).resolve()
    lineage_file = source_root/'resume-lineage.json'
    lineage = json.loads(lineage_file.read_text()) if lineage_file.exists() else {}
    inherited = [Path(path) for path in lineage.get('stages', {}).get(label, {}).get('inherited_sessions', [])]
    needed = total-len(inherited)
    if needed < 0:
        raise ValueError('Native completed count precedes the inherited prefix')
    local = sorted((source_root/label/'sessions').glob('*'),
        key=lambda ep: json.loads((ep/'episode.json').read_text())['wall_started_unix_s'])
    if len(local) < needed:
        raise ValueError('Missing source agent episode archives')
    selected = inherited+local[:needed]
    if len(set(selected)) != total:
        raise ValueError('Duplicate inherited episode archives')
    for ep in selected:
        meta = json.loads((ep/'episode.json').read_text())
        if meta.get('task') != 'push_T' or meta.get('task_information_mode') != label:
            raise ValueError('Inherited policy task/information mode differs from the resumed arm')
        if not (ep/'END.json').is_file():
            raise ValueError('Official completed episode lacks a closed policy archive')
        for error in (ep/'errors').glob('*.json'):
            if json.loads(error.read_text()).get('infrastructure'):
                raise ValueError('An infrastructure failure cannot be inherited as an evaluated episode')
    return [str(ep) for ep in selected]


if __name__ == '__main__':
    print(json.dumps(prepare(json.load(sys.stdin))))
