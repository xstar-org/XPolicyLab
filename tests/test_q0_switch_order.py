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
        (root/'config').mkdir()
        cfg = root/'config/cli.json'
        cfg.write_text(json.dumps(dict(provider='codex_cli',model='gpt-6.2-sol')))
        plan = dict(validation_root=str(validation), ssh_config='/private/ssh',ssh_host='cluster',
                    remote_root='/private/experiment',benchmark_name='benchmark-cli',agent_config=str(cfg),
                    site_setup='/private/experiment/code/site.sh',
                    model_source_sha256=switch.model_source_hash(cfg.parents[1]))
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

    def test_successful_switch_orders_validation_boundary_and_cli_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root,validation,path,proof = self.fixture(directory)
            plan = json.loads(path.read_text())
            plan.update(api_service='api-recovery',api_result_run_id='q0-original',
                formal_root=str(root/'formal'),formal_log=str(root/'formal.log'))
            command = ['python','driver.py','--agent-config',plan['agent_config'],
                '--validated-preflight',str(validation),'--root',plan['formal_root'],
                '--benchmark-name',plan['benchmark_name']]
            plan['formal_command'] = command
            path.write_text(json.dumps(plan))
            (validation/'VALIDATION_PASSED.json').write_text(json.dumps(proof))
            events = []
            reads = 0

            def run(argv, **kwargs):
                nonlocal reads
                if argv[0] != 'ssh':
                    self.assertEqual(argv,command)
                    events.append('formal_cli')
                    return SimpleNamespace(returncode=0)
                content = kwargs['input']
                if content.startswith('{'):
                    action = json.loads(content)['action']
                    events.append(action)
                    value = proof['deployment'] if action == 'deployment' else proof['layouts']
                elif 'marker=' in content:
                    events.append('arm_boundary')
                    value = dict(armed=True)
                else:
                    events.append('api_status')
                    reads += 1
                    value = dict(evaluator_result=dict(eval_time=4 if reads == 1 else 5),
                        service_stopped=reads > 1,worker_result=None if reads == 1 else dict(returncode=1),
                        boundary_archive=dict(cli_validation=proof,native_result=dict(eval_time=5)))
                return SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='')

            with patch.object(sys,'argv',['switch','--plan',str(path)]), patch.object(switch.subprocess,'run',side_effect=run):
                switch.main()
            self.assertEqual(events,['deployment','freeze','api_status','arm_boundary','api_status','formal_cli'])
            self.assertEqual(json.loads((root/'transition-state.json').read_text())['phase'],'CLI_PAIRED_EXPERIMENT_COMPLETE')
