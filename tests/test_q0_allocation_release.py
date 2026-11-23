"""Allocation cancellation requires a complete run and the exact owned job."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t/release_strict_allocation.py'
spec = importlib.util.spec_from_file_location('q0_release_allocation', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AllocationReleaseTests(unittest.TestCase):
    def fixture(self, root):
        def save(name, value):
            (root/name).write_text(json.dumps(value))
        rows = {str(i):dict(layout_id=i, success=i != 7, score=float(i != 7)) for i in range(50)}
        native = dict(eval_time=50, success_rate=49/50, score=98., details=rows)
        save('strict-result.json', native)
        save('native.json', native)
        save('processes.json', dict(job_id='123'))
        save('status.json', dict(evaluator_result=native, worker_result=dict(job_id='123', returncode=0)))
        deployment = dict(sha256='frozen-source', file_count=380)
        layouts = dict(layout_ids=list(range(50)))
        save('deployment.json', deployment)
        save('layouts.json', layouts)
        save('final-input-audit.json', dict(episodes_checked=50, runtime_verified=True,
             input_checks_passed=True, episodes=[dict(session=str(i), closed=True) for i in range(50)]))
        save('native-proof.json', dict(native_sha256=hashlib.sha256((root/'native.json').read_bytes()).hexdigest(),
             count=50, native_matches_driver=True,
             checks={name:dict(matches_formal_launch=True, value=value)
                     for name, value in [('deployment', deployment), ('freeze', layouts)]}))
        return SimpleNamespace(root=root, job_id='123', native_result=root/'native.json',
             native_proof=root/'native-proof.json', remote_root='/experiment',
             output=root/'release.json', wait_seconds=0)

    def snapshot(self, present=True):
        return dict(queue_present=present, current_user_id='worker(1234)', job_fields={
            'JobId':'123', 'JobName':'q0-strict-info-50', 'UserId':'worker(1234)',
            'Partition':'gpu_4090', 'NumNodes':'1', 'AllocTRES':'cpu=8,gres/gpu=1',
            'Command':'/experiment/code/allocation.sbatch',
            'StdOut':'/experiment/logs/allocation-123.log'})

    def test_partial_or_invalid_native_results_never_contact_cluster(self):
        for mutation in ['partial', 'duplicate', 'nonboolean', 'aggregate']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                args = self.fixture(Path(tmp))
                data = module.read_json(args.root/'strict-result.json')
                if mutation == 'partial':data['eval_time'] = 49
                if mutation == 'duplicate':data['details']['49']['layout_id'] = 48
                if mutation == 'nonboolean':data['details']['49']['success'] = 1
                if mutation == 'aggregate':data['success_rate'] = 1.
                (args.root/'strict-result.json').write_text(json.dumps(data))
                calls = []
                with self.assertRaises(ValueError):module.release(args, calls.append)
                self.assertEqual(calls, [])
                self.assertFalse(args.output.exists())

    def test_wrong_job_unfinished_worker_and_unsaved_source_never_contact_cluster(self):
        for mutation in ['job', 'worker', 'digest', 'deployment', 'open_archive']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                args = self.fixture(Path(tmp))
                if mutation in ('job', 'worker'):
                    data = module.read_json(args.root/'status.json')
                    data['worker_result'].update(dict(job_id='999') if mutation == 'job' else dict(returncode=1))
                    (args.root/'status.json').write_text(json.dumps(data))
                elif mutation == 'open_archive':
                    data = module.read_json(args.root/'final-input-audit.json')
                    data['episodes'][-1]['closed'] = False
                    (args.root/'final-input-audit.json').write_text(json.dumps(data))
                else:
                    data = module.read_json(args.native_proof)
                    if mutation == 'digest':data['native_sha256'] = 'unverified'
                    else:data['checks']['deployment']['matches_formal_launch'] = False
                    args.native_proof.write_text(json.dumps(data))
                calls = []
                with self.assertRaises(ValueError):module.release(args, calls.append)
                self.assertEqual(calls, [])

    def test_another_or_shared_allocation_is_never_cancelled(self):
        for key, value in [('JobId','999'), ('JobName','another-task'), ('UserId','other(999)'),
                           ('Command','/another/code/allocation.sbatch'),
                           ('StdOut','/another/allocation-123.log'), ('AllocTRES','gres/gpu=4')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                args = self.fixture(Path(tmp))
                snapshot = self.snapshot()
                snapshot['job_fields'][key] = value
                calls = []
                def request(message):
                    calls.append(message['action'])
                    return snapshot
                with self.assertRaises(ValueError):module.release(args, request)
                self.assertEqual(calls, ['inspect'])

    def test_complete_run_with_genuine_failure_releases_without_changing_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            before = args.native_result.read_bytes()
            actions = []
            def request(message):
                actions.append(message['action'])
                if len(actions) == 3:return self.snapshot(False)
                result = self.snapshot()
                if message['action'] == 'cancel':result['scancel'] = dict(returncode=0)
                return result
            proof = module.release(args, request)
            self.assertEqual(actions, ['inspect', 'cancel', 'inspect'])
            self.assertTrue(proof['released'])
            self.assertTrue(module.read_json(args.output)['allocation_absent_from_scheduler'])
            self.assertEqual(args.native_result.read_bytes(), before)
            self.assertFalse(module.read_json(args.native_result)['details']['7']['success'])

    def test_cancel_request_alone_is_not_release_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            def request(message):
                result = self.snapshot()
                if message['action'] == 'cancel':result['scancel'] = dict(returncode=0)
                return result
            with self.assertRaises(RuntimeError):module.release(args, request)
            proof = module.read_json(args.output)
            self.assertTrue(proof['cancel_requested'])
            self.assertFalse(proof['released'])

    def test_already_absent_job_does_not_issue_scancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            actions = []
            def request(message):
                actions.append(message['action'])
                return self.snapshot(False)
            self.assertTrue(module.release(args, request)['released'])
            self.assertEqual(actions, ['inspect', 'inspect'])

    def test_transport_observation_failure_never_claims_resources_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            actions = []
            def request(message):
                actions.append(message['action'])
                if len(actions) == 3:raise TimeoutError('Scheduler observation expired')
                result = self.snapshot()
                if message['action'] == 'cancel':result['scancel'] = dict(returncode=0)
                return result
            with self.assertRaises(TimeoutError):module.release(args, request)
            self.assertFalse(module.read_json(args.output)['released'])


if __name__ == '__main__':
    unittest.main()
