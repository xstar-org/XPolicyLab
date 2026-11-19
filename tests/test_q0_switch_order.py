"""Failed CLI validation must never stop the still-running API evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


FILE = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t/switch_after_validation.py'
spec = importlib.util.spec_from_file_location('q0_switch', FILE)
switch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(switch)


class SwitchOrderTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        validation = root/'validation'
        validation.mkdir()
        cfg = root/'cli.json'
        cfg.write_text(json.dumps(dict(provider='codex_cli',model='gpt-6.2-sol')))
        plan = dict(validation_root=str(validation), ssh_config='/private/ssh',ssh_host='cluster',
                    remote_root='/private/experiment',benchmark_name='benchmark-cli',agent_config=str(cfg),
                    site_setup='/private/experiment/code/site.sh')
        path = root/'plan.json'
        path.write_text(json.dumps(plan))
        proof = dict(transport=dict(provider='codex_cli',model='gpt-6.2-sol',authentication='chatgpt_login',
            agent_config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest()),remote_root=plan['remote_root'],
            deployment=dict(sha256='frozen-source'),layouts=dict(layouts=[0]))
        return root,validation,path,proof

    def test_failed_preflight_leaves_api_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root,validation,path,proof = self.fixture(directory)
            (validation/'PIPELINE_ERROR.json').write_text('{"error":"CLI failed"}')
            with patch.object(sys,'argv',['switch','--plan',str(path)]), patch.object(switch.subprocess,'run') as remote:
                with self.assertRaises(RuntimeError):
                    switch.main()
                remote.assert_not_called()
            self.assertEqual(json.loads((root/'transition-state.json').read_text())['phase'],'TRANSITION_ERROR')

    def test_invalid_authentication_certificate_leaves_api_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root,validation,path,proof = self.fixture(directory)
            proof['transport']['authentication'] = 'api_key'
            (validation/'VALIDATION_PASSED.json').write_text(json.dumps(proof))
            with patch.object(sys,'argv',['switch','--plan',str(path)]), patch.object(switch.subprocess,'run') as remote:
                with self.assertRaises(ValueError):
                    switch.main()
                remote.assert_not_called()

    def test_changed_remote_source_cannot_arm_a_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root,validation,path,proof = self.fixture(directory)
            (validation/'VALIDATION_PASSED.json').write_text(json.dumps(proof))
            with patch.object(sys,'argv',['switch','--plan',str(path)]), patch.object(switch.subprocess,'run',
                    return_value=SimpleNamespace(returncode=0,stdout='{"sha256":"changed"}',stderr='')) as remote:
                with self.assertRaises(ValueError):
                    switch.main()
                self.assertEqual(remote.call_count,1)
                self.assertEqual(json.loads(remote.call_args.kwargs['input'])['action'],'deployment')
