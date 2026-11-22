"""Audit archived raw observations and deterministic model-readable strict assets.

Run before formal dispatch and again across every completed strict archive.
This validates input provenance; it cannot certify model pretraining contents.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

POLICY = Path(__file__).resolve().parents[2]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, POLICY/relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def audit(sessions):
    boundary = module('strict_audit_boundary', 'runtime/strict_information.py')
    tasks = module('strict_audit_tasks', 'runtime/task_context.py')
    allowed = ('assets/prompts/strict_prompt.md', 'assets/prompts/checkpoint_prompt.md',
               'assets/geometry/strict_geometry.md', 'assets/geometry/geometry_interfaces.md',
               'harness/helpers/vlm_geom.py', 'harness/helpers/observation_io.py')
    manifest = {name: hashlib.sha256((POLICY/name).read_bytes()).hexdigest() for name in allowed}
    text = '\n'.join((POLICY/name).read_text() for name in allowed)
    for prohibited in ('0.765', '7 degrees', '7 mm', 'Coin probe', 'Bread probe',
                       'geometry_evidence.md', '/work/task_reference.json'):
        if prohibited in text:
            raise ValueError('Restricted input material: '+prohibited)
    expected = tasks.task_context('push_T', mode='strict_official')
    count = 0
    episodes = []
    for session in sessions:
        ep = json.loads((session/'episode.json').read_text())
        if ep['task_information_mode'] != 'strict_official' or ep['task_context'] != expected:
            raise ValueError('Strict mode/context mismatch: '+str(session))
        for filename in ('task_reference.json', 'geometry_evidence.md'):
            path = session/'sandbox'/filename
            if path.exists() or path.is_symlink():
                raise ValueError('Restricted sandbox material: '+str(path))
        observations = sorted((session/'observations').glob('*/observation.json'))
        for path in observations:
            observation = json.loads(path.read_text())
            boundary.validate_published(observation)
            if observation['task_requirements'] != expected:
                raise ValueError('Changed task description: '+str(path))
        count += len(observations)
        episodes.append({'session': session.name, 'observations': len(observations),
                         'closed': (session/'END.json').exists()})
    return {'mode': 'strict_official', 'asset_sha256': manifest,
            'task_description_sha256': hashlib.sha256(expected.encode()).hexdigest(),
            'episodes_checked': len(episodes), 'observations_checked': count,
            'episodes': episodes, 'input_checks_passed': True,
            'runtime_verified': bool(episodes) and count > 0,
            'limitation': 'Input-boundary audit; no claim about pretrained knowledge or official certification.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sessions = sorted(args.sessions.glob('*/episode.json')) if args.sessions else []
    value = audit([path.parent for path in sessions])
    with args.output.open('x') as file:
        json.dump(value, file, indent=2)


if __name__ == '__main__':
    main()
