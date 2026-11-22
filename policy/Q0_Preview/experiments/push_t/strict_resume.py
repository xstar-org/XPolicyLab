"""Trusted evidence for a strict run interrupted by a terminal SSH bridge error.

No policy can read this helper or its certificates. Source archives are read-only.
An observation timeout alone never authorizes a restart or outcome exclusion.
"""
import hashlib
import json
import ast
import base64
import math
from pathlib import Path

from native_resume import completed_sessions


def resume_evidence(source, native):
    source = Path(source).resolve()
    label = 'strict_official'
    before = json.loads((source/'status.json').read_text())['evaluator_result']
    completed = before['eval_time']
    if type(completed) is not int or not 0 < completed < 50:
        raise ValueError('A nonempty incomplete native prefix is required')
    inherited = [Path(p) for p in completed_sessions(source, label, completed)]
    if not all(json.loads((p/'END.json').read_text()).get('final') is True for p in inherited):
        raise ValueError('Every inherited episode must have a final END')
    old = json.loads((source/'resume-lineage.json').read_text()) if (source/'resume-lineage.json').exists() else {}
    prior = old.get('stages', {}).get(label, {}).get('inherited_sessions', [])
    local = sorted((source/label/'sessions').iterdir(),
                   key=lambda p: json.loads((p/'episode.json').read_text())['wall_started_unix_s'])
    tail = local[completed-len(prior):]
    if len(tail) != 1 or (tail[0]/'END.json').exists():
        raise ValueError('Exactly one unclosed transport-interrupted tail is required')
    episode = tail[0]
    meta = json.loads((episode/'episode.json').read_text())
    error_path = source/label/'BRIDGE_ERROR.json'
    error = json.loads(error_path.read_text())
    driver_error = json.loads((source/'ERROR.json').read_text())
    state = json.loads((source/label/'TRANSPORT_STATE.json').read_text())
    if (error.get('error') != "TimeoutError('SSH transport recovery window exhausted')"
            or driver_error.get('error') != "RuntimeError('Strict agent bridge exited unexpectedly')"
            or state.get('event') != 'recovery_exhausted'
            or not meta['wall_started_unix_s'] < state['unix'] <= error['unix'] <= driver_error['unix']
            or meta.get('task') != 'push_T' or meta.get('task_information_mode') != label
            or not (episode/'STOP').exists()):
        raise ValueError('Missing terminal transport-failure evidence')
    expected_ids = set(range(completed))
    if (len(before['details']) != completed
            or {r['layout_id'] for r in before['details'].values()} != expected_ids
            or native.get('eval_time') not in (completed, completed+1)
            or len(native.get('details', {})) != native['eval_time']
            or {r['layout_id'] for r in native['details'].values()} != set(range(native['eval_time']))
            or {k:r for k,r in native['details'].items() if r['layout_id'] < completed} != before['details']):
        raise ValueError('Native completed outcomes changed across the interruption')
    exclude = native['eval_time'] == completed+1
    if exclude and next(r for r in native['details'].values() if r['layout_id'] == completed)['success'] is not False:
        raise ValueError('A native successful episode cannot be excluded')
    proof = dict(layout_id=completed, infrastructure=True, archive_closed=False,
                 session_path=str(episode), error_path=str(error_path),
                 error_sha256=hashlib.sha256(error_path.read_bytes()).hexdigest(),
                 transport_state_sha256=hashlib.sha256((source/label/'TRANSPORT_STATE.json').read_bytes()).hexdigest(),
                 driver_error_sha256=hashlib.sha256((source/'ERROR.json').read_bytes()).hexdigest())
    return dict(completed=completed, inherited_sessions=[str(p) for p in inherited],
                transport_interruption=proof, exclude_native_tail=exclude)


def audit_view(root, sessions):
    """Expose exactly the retained archives to the existing full-request auditor."""
    sessions = [Path(p).resolve() for p in sessions]
    if len(sessions) != 50 or len(set(sessions)) != 50 or len({p.name for p in sessions}) != 50:
        raise ValueError('Exactly 50 distinct retained episode archives are required')
    if not all(json.loads((p/'END.json').read_text()).get('final') is True for p in sessions):
        raise ValueError('An unfinished archive cannot enter the final audit view')
    view = Path(root)/'audit-sessions'
    view.mkdir(exist_ok=False)
    for episode in sessions:
        (view/episode.name).symlink_to(episode, target_is_directory=True)
    return view


def storage_resume_evidence(source, native, certificate_path):
    """Retain the native prefix after independently evidenced observation I/O failures.

    The operator certificate stays outside the model sandbox. Unlike a policy
    failure, these episodes never produced native outcomes; none may be removed.
    """
    source = Path(source).resolve()
    certificate_path = Path(certificate_path).resolve()
    certificate = json.loads(certificate_path.read_text())
    if certificate.get('source_service') != source.name:
        raise ValueError('Storage certificate belongs to another service')

    def verified_file(key):
        record = certificate[key]
        data = Path(record['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Storage evidence hash mismatch: '+key)
        return data

    before = json.loads(verified_file('native_result'))
    cached = json.loads((source/'status.json').read_text())['evaluator_result']
    completed = before['eval_time']
    if (type(completed) is not int or not 0 < completed < 50 or native != before or cached != before
            or len(before['details']) != completed
            or {r['layout_id'] for r in before['details'].values()} != set(range(completed))):
        raise ValueError('Storage recovery must preserve every native completed outcome')
    server = verified_file('server_log').decode(errors='replace')
    if not all(s in server for s in ('np.save', 'write_array', 'OSError: 921600 requested and 0 written')):
        raise ValueError('Missing observation-array write failure traceback')
    probe = json.loads(verified_file('node_write_probe'))
    node = json.loads(probe.get('stdout', '{}'))
    checks = node.get('checks', [])
    if probe.get('returncode') != 0 or not node.get('node') or len(checks) < 3 or not all(c.get('ok') is True for c in checks):
        raise ValueError('Actual compute-node array writes must pass before recovery')
    inherited = [Path(p) for p in completed_sessions(source, 'strict_official', completed)]
    if not all(json.loads((p/'END.json').read_text()).get('final') is True for p in inherited):
        raise ValueError('Inherited episodes need final END records')
    lineage = json.loads((source/'resume-lineage.json').read_text()) if (source/'resume-lineage.json').exists() else {}
    previous = lineage.get('stages', {}).get('strict_official', {}).get('inherited_sessions', [])
    local = sorted((source/'strict_official/sessions').iterdir(),
                   key=lambda p: json.loads((p/'episode.json').read_text())['wall_started_unix_s'])
    tails = local[completed-len(previous):]
    if not tails or [p.name for p in tails] != certificate.get('interrupted_sessions'):
        raise ValueError('Storage recovery must account for exactly the unrecorded tail archives')
    truncated = set()
    for row in certificate.get('truncated_npy_files', []):
        header = base64.b64decode(row['header_base64'], validate=True)
        if (header[:8] != b'\x93NUMPY\x01\x00' or row['size_bytes'] != len(header)
                or hashlib.sha256(header).hexdigest() != row['sha256']):
            raise ValueError('Invalid truncated array evidence')
        size = 10+int.from_bytes(header[8:10], 'little')
        info = ast.literal_eval(header[10:size].decode('latin1'))
        shape = info['shape']
        path = Path(row['path'])
        if (size != len(header) or info['descr'] != '|u1' or len(shape) != 3 or shape[-1] != 3
                or not all(type(d) is int and d > 0 for d in shape)
                or row['size_bytes'] >= size+math.prod(shape)
                or '..' in path.parts or path.is_absolute()
                or not str(path).startswith('runs/'+source.name+'/sessions/'+row['session']+'/observations/')
                or not str(path).endswith('.rgb.npy')):
            raise ValueError('Evidence is not a header-only RGB array in this source session')
        truncated.add(row['session'])
    for i, episode in enumerate(tails):
        meta = json.loads((episode/'episode.json').read_text())
        if meta.get('task') != 'push_T' or meta.get('task_information_mode') != 'strict_official' or not (episode/'STOP').exists():
            raise ValueError('Storage tail attribution or terminal stop is missing')
        if (episode/'END.json').exists():
            end = json.loads((episode/'END.json').read_text())
            if end.get('final') is not True or end.get('reason') != 'policy_error' or episode.name not in truncated:
                raise ValueError('A closed tail requires its own truncated-array evidence')
        elif i != len(tails)-1:
            raise ValueError('Only the last operator-stopped tail may remain unclosed')
    return dict(completed=completed, inherited_sessions=[str(p) for p in inherited],
                exclude_native_tail=False, storage_interruption=dict(
                    certificate_sha256=hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
                    interrupted_sessions=[p.name for p in tails], native_outcomes_removed=0,
                    node_write_checks=len(checks)))
