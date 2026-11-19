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
    for name in (source, target, benchmark):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', name):
            raise ValueError('Invalid resume service/snapshot identity')
    if source == target:
        raise ValueError('Resume requires a fresh service')
    certificate = root/'code'/('resume-'+target+'.json')
    if certificate.exists():
        record = json.loads(certificate.read_text())
        if record['source_service'] != source or record['benchmark_name'] != benchmark:
            raise ValueError('Conflicting resume certificate')
        return record
    worker = root/'results'/(source+'.json')
    if not worker.exists() or 'returncode' not in json.loads(worker.read_text()):
        raise ValueError('Source worker must be terminal before native resume')
    if (root/'runs'/target).exists():
        raise ValueError('Target controller service already exists')
    base = root/'code'/benchmark/'eval_result/RoboDojo/push_T/Q0_Preview/arx_x5/1_ckpt_name=api,action_type=joint'
    old_id, new_id = 'q0-'+source, 'q0-'+target
    source_result = base/old_id/'_result.json'
    source_checkpoint = base/('_resume_'+old_id+'.json')
    result_bytes, checkpoint_bytes = source_result.read_bytes(), source_checkpoint.read_bytes()
    result, checkpoint = json.loads(result_bytes), json.loads(checkpoint_bytes)
    count = result.get('eval_time')
    details = result.get('details', {})
    if type(count) is not int or not 0 < count < 50 or len(details) != count:
        raise ValueError('Resume requires a nonempty incomplete native result prefix')
    ids = [row.get('layout_id') for row in details.values()]
    if (any(type(i) is not int for i in ids) or set(ids) != set(range(count))
            or any(type(row.get('success')) is not bool for row in details.values())):
        raise ValueError('Native resume requires unique completed frozen layout IDs')
    if (checkpoint.get('run_id') != old_id or checkpoint.get('details') != details
            or checkpoint.get('abandoned_layout_ids') or checkpoint.get('completed_layout_ids') != list(range(count))
            or checkpoint.get('success_nums') != sum(row['success'] for row in details.values())
            or checkpoint.get('fail_nums') != sum(not row['success'] for row in details.values())):
        raise ValueError('Native checkpoint and official completed results disagree')
    for key, expected in [('task_name','push_T'),('policy_name','Q0_Preview'),('config_name','arx_x5'),
                          ('eval_seed',1),('additional_info','ckpt_name=api,action_type=joint')]:
        if checkpoint.get(key) != expected:
            raise ValueError('Native checkpoint configuration changed: '+key)
    checkpoint.update(run_id=new_id,
        save_dir=str((base/new_id).relative_to(root/'code'/benchmark)),
        restart_count=int(checkpoint.get('restart_count', 0))+1)
    record = dict(source_service=source, target_service=target, benchmark_name=benchmark,
        completed=count, completed_layout_ids=sorted(ids), source_result=str(source_result),
        source_result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        source_checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        target_result=str(base/new_id/'_result.json'),
        interrupted_episode_counted=False)
    atomic_exact(base/new_id/'_result.json', result_bytes)
    atomic_exact(base/('_resume_'+new_id+'.json'), json.dumps(checkpoint, indent=2).encode())
    atomic_exact(certificate, json.dumps(record, indent=2).encode())
    return record


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
