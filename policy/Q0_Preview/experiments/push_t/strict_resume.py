"""Trusted evidence for a strict run interrupted by a terminal SSH bridge error.

No policy can read this helper or its certificates. Source archives are read-only.
An observation timeout alone never authorizes a restart or outcome exclusion.
"""
import hashlib
import json
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
