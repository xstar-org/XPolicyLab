import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import unittest
import tempfile
import sys


SOURCE = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t/formal_overlap.py'
spec = importlib.util.spec_from_file_location('formal_overlap', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.insert(0, str(SOURCE.parent))
from run_parallel_official import resume_archive_proof


def request():
    return dict(authorization=module.AUTHORIZATION, mode='official_only', episodes=50,
        gpu=0, gpu_uuid='GPU-test', user_request='Run official-only 50 on the same GPU',
        name='parallel-official-cli-001-official_only', co_run='paired-codexcli-004-original',
        benchmark_name='benchmark-codexcli-50-20261119', job_id='1836298',
        site_setup='/site.sh', deployment={'sha': 'frozen'}, layouts={'count': 50},
        agent_config_name='agent.gpt62sol.codexcli.json', cluster_prefix='/sim')


class FormalOverlapTests(unittest.TestCase):
    def test_authorized_set_accepts_a_different_dispatch_gpu_from_probe(self):
        first = 'GPU-00000000-0000-0000-0000-000000000001'
        second = 'GPU-00000000-0000-0000-0000-000000000002'
        outside = 'GPU-00000000-0000-0000-0000-000000000003'
        message = dict(request(),cross_allocation_authorization=module.CROSS_ALLOCATION_AUTHORIZATION,
            resume_source_service='old-official_only',authorized_gpu_uuids=[first,second],
            gpu_uuid=first,step_gpu_index=None)
        module.validate(message)
        self.assertEqual(module.gpu_step_args(message),['--gpus-per-task=1','--gpu-bind=single:1'])
        probe = module.gpu_headroom('0, '+first+', 0, 24000', [first,second])
        with patch.dict(module.os.environ,dict(SLURM_JOB_ID='1836298',CUDA_VISIBLE_DEVICES='0',SLURM_STEP_GPUS='6')):
            with patch.object(module.subprocess,'check_output',return_value='0, '+second+', 0, 24000'):
                actual = module.verify_worker('1836298',first,True,[first,second])
            self.assertNotEqual(probe['uuid'],actual['hardware']['uuid'])
            self.assertEqual(actual['hardware']['uuid'],second)
            self.assertEqual(actual['slurm_step_gpus'],'6')
            with patch.object(module.subprocess,'check_output',return_value='0, '+outside+', 0, 24000'):
                with self.assertRaises(ValueError):module.verify_worker('1836298',first,True,[first,second])
            with patch.object(module.subprocess,'check_output',return_value='0, '+first+', 0, 24000\n1, '+second+', 0, 24000'):
                with self.assertRaises(ValueError):module.verify_worker('1836298',first,True,[first,second])

    def test_cross_allocation_transfer_requires_cancelled_explicit_user_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('code', 'claims', 'results'):(root/folder).mkdir()
            old = request()
            module.reserve_identity(root, old)
            (root/'results'/(old['name']+'.json')).write_text(json.dumps(dict(job_id=old['job_id'],returncode=1)))
            (root/'claims'/(old['name']+'.json')).write_text(json.dumps(dict(job_id=old['job_id'],formal_overlap=True)))
            new = dict(old,name='replacement-official_only',job_id='99',
                resume_source_service=old['name'],step_gpu_index=2,
                cross_allocation_authorization=module.CROSS_ALLOCATION_AUTHORIZATION)
            pause = dict(user_requested_pause=True,source_job_id=old['job_id'],
                source_allocation_state='CANCELLED',completed_before_pause=13,source_service=old['name'])
            record = dict(source_service=old['name'],target_service=new['name'],
                benchmark_name=old['benchmark_name'],interrupted_episode_counted=False,completed=13)
            certificate = root/'code'/('resume-'+new['name']+'.json')
            certificate.write_text(json.dumps(record));new['resume_record']=record
            with self.assertRaisesRegex(ValueError,'released user-pause'):
                module.reserve_identity(root,new)
            record['operator_pause']=pause;certificate.write_text(json.dumps(record))
            module.validate(new)
            self.assertEqual(module.gpu_step_args(new),['--gpus-per-task=1','--gpu-bind=map_gpu:2'])
            module.reserve_identity(root,new)
            self.assertEqual(json.loads((root/'code/formal-overlap-official_only.json').read_text())['job_id'],'99')
            with self.assertRaises(ValueError):module.validate(dict(new,step_gpu_index=None))

    def test_queued_resume_is_guarded_before_worker_can_claim_and_needs_no_slurm_rpc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('code', 'claims', 'results', 'requests', 'jobs/1836298'):
                (root/folder).mkdir(parents=True)
            old = request()
            module.reserve_identity(root, old)
            new = dict(old, name='parallel-official-cli-003-official_only',
                resume_source_service=old['name'], existing_worker_queue=True)
            record = dict(source_service=old['name'], target_service=new['name'],
                benchmark_name=old['benchmark_name'], interrupted_episode_counted=False, completed=13)
            new['resume_record'] = record
            (root/'code'/('resume-'+new['name']+'.json')).write_text(json.dumps(record))
            (root/'results'/(old['name']+'.json')).write_text(json.dumps(dict(job_id=old['job_id'], returncode=1)))
            (root/'claims'/(old['name']+'.json')).write_text(json.dumps(dict(job_id=old['job_id'], formal_overlap=True)))
            (root/'claims'/(old['co_run']+'.json')).write_text(json.dumps(dict(job_id=old['job_id'])))
            (root/'jobs/1836298/allocation.json').write_text(json.dumps(dict(job_id=old['job_id'], cuda_visible_devices='0')))
            (root/'jobs/1836298/READY.json').write_text(json.dumps(dict(job_id=old['job_id'], status='ready')))
            submissions = []

            class Control:
                @staticmethod
                def run(root, payload):
                    if payload['action'] == 'deployment':return new['deployment']
                    if payload['action'] == 'freeze':return new['layouts']
                    claim = json.loads((root/'claims'/(new['name']+'.json')).read_text())
                    assert claim['queue_hold'] is True
                    submissions.append(payload)
                    (root/'code'/('launch-'+new['name']+'.sh')).write_text('#!/bin/bash\nexec bash frozen.sh\n')
                    (root/'requests'/(new['name']+'.json')).write_text('{}')

            with patch.object(module, 'load_control', return_value=Control), patch.object(
                    module.subprocess, 'check_output', side_effect=AssertionError('no Slurm RPC')), patch.object(
                    module.subprocess, 'Popen', side_effect=AssertionError('no new scheduler')):
                proof = module.start(root, new)
                self.assertTrue(proof['launch']['existing_worker_queue'])
                self.assertFalse((root/'claims'/(new['name']+'.json')).exists())
                script = (root/'code'/('launch-'+new['name']+'.sh')).read_text()
                self.assertLess(script.index('--verify-worker-job 1836298'), script.index('exec bash frozen.sh'))
                module.start(root, new)
                self.assertEqual(len(submissions), 1)

    def test_worker_dispatch_rejects_another_allocation_before_gpu_or_simulator(self):
        with patch.dict(module.os.environ, dict(SLURM_JOB_ID='another', CUDA_VISIBLE_DEVICES='0')):
            with patch.object(module.subprocess, 'check_output', side_effect=AssertionError('must not run')):
                with self.assertRaisesRegex(ValueError, 'different allocation'):
                    module.verify_worker('1836298', 'GPU-test')

    def test_counted_and_uncounted_infrastructure_tail_keep_the_same_closed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(4):
                episode = root/'official_only/sessions'/str(index)
                episode.mkdir(parents=True)
                (episode/'episode.json').write_text(json.dumps(dict(task='push_T',
                    task_information_mode='official_only', wall_started_unix_s=index)))
                if index < 3:
                    (episode/'END.json').write_text('{}')
                else:
                    (episode/'errors').mkdir()
                    (episode/'errors/checkpoint.json').write_text('{"infrastructure":true}')
            for count in (3, 4):
                valid, tail = resume_archive_proof(root, count)
                self.assertEqual(valid, 3)
                self.assertEqual(tail['layout_id'], 3)
                self.assertTrue(tail['infrastructure'])
            with self.assertRaises(ValueError):
                resume_archive_proof(root, 2)
            (root/'official_only/sessions/3/END.json').write_text('{}')
            with self.assertRaises(ValueError):
                resume_archive_proof(root, 4)

    def test_lock_transfer_requires_terminal_direct_predecessor_and_native_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('code', 'claims', 'results'):
                (root/folder).mkdir()
            old = request()
            module.reserve_identity(root, old)
            new = dict(old, name='parallel-official-cli-002-official_only',
                       resume_source_service=old['name'])
            record = dict(source_service=old['name'], target_service=new['name'],
                benchmark_name=old['benchmark_name'], interrupted_episode_counted=False, completed=13)
            new['resume_record'] = record
            worker = root/'results'/(old['name']+'.json')
            claim = root/'claims'/(old['name']+'.json')
            certificate = root/'code'/('resume-'+new['name']+'.json')
            worker.write_text(json.dumps(dict(job_id=old['job_id'])))
            claim.write_text(json.dumps(dict(job_id=old['job_id'], formal_overlap=True)))
            certificate.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'terminal predecessor'):
                module.reserve_identity(root, new)
            worker.write_text(json.dumps(dict(job_id=old['job_id'], returncode=1)))
            with self.assertRaisesRegex(ValueError, 'terminal predecessor'):
                module.reserve_identity(root, dict(new, resume_record=dict(record, completed=12)))
            (root/'claims'/(old['co_run']+'.json')).write_text(json.dumps(dict(job_id=old['job_id'])))
            with patch.object(module.subprocess, 'check_output', side_effect=RuntimeError('controller unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'controller unavailable'):
                    module.prepare(root, new)
            self.assertEqual(json.loads((root/'code/formal-overlap-official_only.json').read_text())['name'], old['name'])
            module.reserve_identity(root, new)
            module.reserve_identity(root, new)
            self.assertEqual(json.loads((root/'code/formal-overlap-official_only.json').read_text())['name'], new['name'])
            self.assertEqual(json.loads((root/'code'/('formal-overlap-predecessor-'+new['name']+'.json')).read_text())['name'], old['name'])
            with self.assertRaisesRegex(ValueError, 'direct resume chain'):
                module.reserve_identity(root, dict(new, name='parallel-official-cli-003-official_only'))

    def test_only_explicit_separate_official_50_is_authorized(self):
        for change in (dict(authorization='yes'), dict(mode='original'), dict(episodes=1),
                       dict(gpu=1), dict(user_request=''), dict(job_id='1;kill'),
                       dict(name='paired-codexcli-004-original')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.validate(dict(request(), **change))

    def test_gpu_identity_and_capacity(self):
        self.assertEqual(module.gpu_headroom('0, GPU-test, 9241, 23028', 'GPU-test')['free_mib'], 13787)
        self.assertEqual(module.gpu_headroom('2, GPU-test, 9241, 23028', 'GPU-test',
            require_index_zero=False)['logical_cuda_index'], 0)
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
