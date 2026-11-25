"""Construct real sandbox and first model payload, without making a model call."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import ModuleType

POLICY = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python-prefix', required=True)
    args = parser.parse_args()
    package = ModuleType('_q0_strict_preflight')
    package.__path__ = [str(POLICY)]
    sys.modules[package.__name__] = package
    runner = importlib.import_module(package.__name__+'.harness.runner')
    boundary = importlib.import_module(package.__name__+'.runtime.strict_information')
    root = args.output.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    shutil.copy2(args.session/'episode.json', root/'episode.json')
    source = args.session/'observations/obs_00000'
    shutil.copytree(source, root/'observations/obs_00000')
    config = json.loads((POLICY/'config/agent.gpt62sol.strict.json').read_text())
    config['python_prefix'] = args.python_prefix
    agent = runner.Agent(root, config)
    try:
        content = agent.prepare(json.loads((source/'request.json').read_text()))
        code = ('import json, os, observation_io as obs, vlm_geom as geom\n'
                'obs.configure("/work/observations/obs_00000/manifest.json")\n'
                'geom.configure("/work/observations/obs_00000/observation.json")\n'
                'd=obs.state()\n'
                'assert "home_check" not in d\n'
                'assert all("near_home" not in a for a in d["state"].values())\n'
                'assert not os.path.lexists("/work/task_reference.json")\n'
                'assert not os.path.lexists("/work/geometry_evidence.md")\n'
                'assert all(obs.image(c).shape == (480,640,3) for c in '
                '("cam_head","cam_left_wrist","cam_right_wrist"))\n'
                'print("STRICT_INPUT_SANDBOX_READY")')
        output = agent.sandbox.execute(code, root/'sandbox-proof', time.time()+180)
        if 'STRICT_INPUT_SANDBOX_READY' not in output:
            raise ValueError('Strict input sandbox preflight failed: '+output[-2000:])
        system, tools = agent._planning_prefix()
        payload = agent.api.payload(system, content, tools)
        (root/'candidate-model-request.json').write_text(json.dumps(payload, indent=2))
        raw = json.loads((root/'sandbox/observations/obs_00000/observation.json').read_text())
        boundary.validate_published(raw)
        assets = {name: hashlib.sha256((root/'sandbox'/name).read_bytes()).hexdigest()
                  for name in ('geometry_reference.md', 'geometry_interfaces.md', 'vlm_geom.py', 'observation_io.py')}
        proof = dict(passed=True, model_calls=0, raw_observation_checked=True,
            sandbox_assets_sha256=assets, system_sha256=hashlib.sha256(system.encode()).hexdigest(),
            tools=[tool['name'] for tool in tools], model=payload['model'],
            reasoning=payload.get('reasoning'), source_episode=args.session.name)
        (root/'INPUT_PREFLIGHT_PASSED.json').write_text(json.dumps(proof, indent=2))
        print(json.dumps(proof), flush=True)
    finally:
        agent.sandbox.close()
        agent.lock.close()


if __name__ == '__main__':
    main()
