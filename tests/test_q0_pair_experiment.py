"""Regression tests for paired layout identity and evaluator-only reporting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freeze = load('freeze_layouts')
report = load('report_pair')
sys.path.insert(0, str(ROOT))
control = load('cluster_control')


class PairTests(unittest.TestCase):
    def test_cli_validation_has_separate_snapshot_and_single_episode_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'code').mkdir()
            (root/'requests').mkdir()
            request = dict(action='submit', name='cli-preflight', mode='original', episodes=1,
                cluster_prefix='/python', site_setup='/private/site.sh',
                benchmark_name='benchmark-cli', agent_config_name='agent.codexcli.json',
                overlap_validation=True)
            control.run(root, request)
            launcher = (root/'code/launch-cli-preflight.sh').read_text()
            self.assertIn(str(root/'code/benchmark-cli'), launcher)
            self.assertIn('agent.codexcli.json', launcher)
            queued = json.loads((root/'requests/cli-preflight.json').read_text())
            self.assertEqual(queued['validation_episodes'], 1)
            (root/'claims').mkdir()
            (root/'logs').mkdir()
            with patch.object(control.subprocess, 'Popen', return_value=SimpleNamespace(pid=123)) as launch:
                control.run(root, dict(action='launch_overlap', name='cli-preflight', job_id='1234'))
                again = control.run(root, dict(action='launch_overlap', name='cli-preflight', job_id='1234'))
                self.assertTrue(again['already_launched'])
                self.assertEqual(launch.call_count, 1)
                with self.assertRaises(ValueError):
                    control.run(root, dict(action='launch_overlap', name='cli-preflight', job_id='5678'))
            with self.assertRaises(ValueError):
                control.run(root, request | {'name':'not-validation', 'episodes':50})
            self.assertFalse((root/'requests/not-validation.json').exists())
            with self.assertRaises(ValueError):
                control.run(root, request | {'name':'bad-snapshot','benchmark_name':'../outside'})

    def test_validation_source_hash_ignores_results_but_detects_code_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bench = root/'code/benchmark-cli'
            bench.mkdir(parents=True)
            (bench/'eval_result').mkdir()
            (bench/'model.py').write_text('version = 1')
            site = root/'code/site.sh'
            site.write_text('load compiler')
            request = dict(action='deployment', benchmark_name='benchmark-cli', site_setup=str(site))
            before = control.run(root, request)
            (bench/'eval_result/result.json').write_text('{"eval_time":1}')
            self.assertEqual(control.run(root, request), before)
            (bench/'model.py').write_text('version = 2')
            self.assertNotEqual(control.run(root, request), before)

    def test_lost_submission_ack_is_safe_to_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'code').mkdir()
            (root/'requests').mkdir()
            request = dict(action='submit', name='paired-test-original', mode='original', episodes=50,
                           cluster_prefix='/python', site_setup='/private/site.sh')
            control.run(root, request)
            path = root/'requests/paired-test-original.json'
            before = path.stat().st_mtime_ns
            again = control.run(root, request)
            self.assertTrue(again['already_submitted'])
            self.assertEqual(path.stat().st_mtime_ns, before)
            with self.assertRaises(ValueError):
                control.run(root, request | {'mode':'official_only'})

    def test_layouts_use_numeric_order_and_detect_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in reversed(range(50)):
                (root/f'push_T_{i}.json').write_text(str(i))
            (root/'push_T_random_0.json').write_text('{}')
            before = freeze.layout_manifest(root)
            self.assertEqual(before['layouts'][10]['filename'], 'push_T_10.json')
            (root/'push_T_10.json').write_text('changed')
            self.assertNotEqual(before, freeze.layout_manifest(root))
            (root/'push_T_49.json').unlink()
            with self.assertRaises(ValueError):
                freeze.layout_manifest(root)

    def test_extra_layouts_match_native_first_fifty_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(55):
                (root/f'push_T_{i}.json').write_text(str(i))
            value = freeze.layout_manifest(root)
            self.assertEqual(value['available_layout_count'], 55)
            self.assertEqual(len(value['layouts']), 50)
            self.assertEqual(value['layouts'][-1]['filename'], 'push_T_49.json')

    def test_incomplete_or_ambiguous_evaluator_results_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            data = dict(eval_time=50, details={str(i): dict(layout_id=i, success=i < 30)
                                              for i in range(50)})
            path.write_text(json.dumps(data))
            self.assertEqual(sum(report.outcomes(path).values()), 30)
            data['details']['49']['layout_id'] = 0
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                report.outcomes(path)
            data['eval_time'] = 49
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                report.outcomes(path)

    def test_pairing_uses_layout_ids_and_exact_test(self):
        a = {i: i < 40 for i in range(50)}
        b = {i: i < 30 for i in reversed(range(50))}
        result = report.compare(a, b)
        self.assertEqual(result['original']['successes'], 40)
        self.assertEqual(result['official_only']['successes'], 30)
        self.assertEqual(result['paired']['original_only'], 10)
        self.assertEqual(result['paired']['both_fail'], 10)
        self.assertAlmostEqual(result['paired']['mcnemar_exact_two_sided_p'], 2/1024)
        self.assertAlmostEqual(result['difference_original_minus_official'], .2)
        self.assertEqual(report.compare(a, a)['paired']['mcnemar_exact_two_sided_p'], 1)


if __name__ == '__main__':
    unittest.main()
