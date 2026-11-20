import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import unittest
import tempfile


SOURCE = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t/formal_overlap.py'
spec = importlib.util.spec_from_file_location('formal_overlap', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def request():
    return dict(authorization=module.AUTHORIZATION, mode='official_only', episodes=50,
        gpu=0, gpu_uuid='GPU-test', user_request='Run official-only 50 on the same GPU',
        name='parallel-official-cli-001-official_only', co_run='paired-codexcli-004-original',
        benchmark_name='benchmark-codexcli-50-20261119', job_id='1836298',
        site_setup='/site.sh', deployment={'sha': 'frozen'}, layouts={'count': 50},
        agent_config_name='agent.gpt62sol.codexcli.json', cluster_prefix='/sim')


class FormalOverlapTests(unittest.TestCase):
    def test_only_explicit_separate_official_50_is_authorized(self):
        for change in (dict(authorization='yes'), dict(mode='original'), dict(episodes=1),
                       dict(gpu=1), dict(user_request=''), dict(job_id='1;kill'),
                       dict(name='paired-codexcli-004-original')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.validate(dict(request(), **change))

    def test_gpu_identity_and_capacity(self):
        self.assertEqual(module.gpu_headroom('0, GPU-test, 9241, 23028', 'GPU-test')['free_mib'], 13787)
        for output in ('0, GPU-other, 9241, 23028', '0, GPU-test, 13000, 23028',
                       '0, GPU-test, 9241, 23028\n1, GPU-other, 0, 23028'):
            with self.assertRaises(ValueError):
                module.gpu_headroom(output, 'GPU-test')

    def test_claim_precedes_submission_and_retry_never_resubmits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            message = request()
            for folder in ('code', 'claims', 'runs', 'results'):
                (root/folder).mkdir()
            (root/'claims'/(message['co_run']+'.json')).write_text(json.dumps({'job_id': '1836298'}))
            submitted = []

            class Control:
                @staticmethod
                def run(root, payload):
                    if payload['action'] == 'deployment':
                        return message['deployment']
                    if payload['action'] == 'freeze':
                        return message['layouts']
                    assert (root/'claims'/(message['name']+'.json')).exists()
                    submitted.append(payload)

            with patch.object(module, 'load_control', return_value=Control), patch.object(
                    module.subprocess, 'check_output', side_effect=['RUNNING', '0, GPU-test, 9241, 23028']):
                proof = module.prepare(root, message)
                self.assertEqual(module.prepare(root, message), proof)
                self.assertEqual(len(submitted), 1)
                self.assertEqual(submitted[0]['mode'], 'official_only')
                self.assertEqual(submitted[0]['episodes'], 50)
                with self.assertRaisesRegex(ValueError, 'Conflicting'):
                    module.prepare(root, dict(message, name='another-official-only'))
