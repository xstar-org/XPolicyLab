"""An explicit pre-stop certificate excludes only the administrative tail."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t/native_resume.py'
spec = importlib.util.spec_from_file_location('pause_resume', FILE)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class OperatorPauseTests(unittest.TestCase):
    def fixture(self, directory, counted=False, success=False):
        root = Path(directory)
        (root/'results').mkdir()
        source, target = 'old-official_only', 'new-official_only'
        (root/'results'/(source+'.json')).write_text('{"returncode":1,"job_id":"42"}')
        base = root/'code/benchmark/eval_result/RoboDojo/push_T/Q0_Preview/arx_x5/1_ckpt_name=api,action_type=joint'
        (base/('q0-'+source)).mkdir(parents=True)
        prefix = {str(i):dict(layout_id=i,success=i!=1,score=float(i!=1)) for i in range(3)}
        details = copy.deepcopy(prefix)
        if counted:details['3'] = dict(layout_id=3,success=success,score=float(success))
        total, successes = len(details), sum(r['success'] for r in details.values())
        result = base/('q0-'+source)/'_result.json'
        result.write_text(json.dumps(dict(eval_time=total,success_rate=successes/total,
            score=100*successes/total,details=details)))
        checkpoint = base/('_resume_q0-'+source+'.json')
        checkpoint.write_text(json.dumps(dict(run_id='q0-'+source,task_name='push_T',
            policy_name='Q0_Preview',config_name='arx_x5',eval_seed=1,
            additional_info='ckpt_name=api,action_type=joint',completed_layout_ids=list(range(total)),
            abandoned_layout_ids=[],success_nums=successes,fail_nums=total-successes,
            total_score=successes,details=details)))
        proof = dict(user_requested_pause=True,source_service=source,source_job_id='42',
            source_allocation_state='CANCELLED by user',archive_closed_before_stop=False,
            completed_before_pause=3,layout_id=3,details_before_pause=prefix,pause_unix=20,
            session_path='/operator/archive/tail',pause_certificate_sha256='a'*64,
            released_allocation_sha256='b'*64,closed_archives_sha256='c'*64)
        message = dict(root=str(root),benchmark_name='benchmark',source_service=source,
                       target_service=target,operator_pause=proof)
        return root, result, checkpoint, message

    def test_uncounted_stop_preserves_native_bytes_and_genuine_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root, result, checkpoint, message = self.fixture(directory)
            before = result.read_bytes(), checkpoint.read_bytes()
            record = resume.prepare(message)
            self.assertEqual(record['completed'], 3)
            self.assertFalse(record['excluded_operator_pause_tail'])
            self.assertEqual(Path(record['target_result']).read_bytes(), before[0])
            self.assertEqual((result.read_bytes(),checkpoint.read_bytes()), before)
            self.assertFalse(json.loads(Path(record['target_result']).read_text())['details']['1']['success'])
            self.assertEqual(resume.prepare(message), record)

    def test_only_counted_failed_administrative_tail_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, result, checkpoint, message = self.fixture(directory, counted=True)
            before = result.read_bytes(), checkpoint.read_bytes()
            record = resume.prepare(message)
            final = json.loads(Path(record['target_result']).read_text())
            self.assertEqual(final['eval_time'], 3)
            self.assertFalse(final['details']['1']['success'])
            self.assertTrue(record['excluded_operator_pause_tail'])
            self.assertEqual((result.read_bytes(),checkpoint.read_bytes()), before)

    def test_successful_tail_or_changed_completed_prefix_cannot_be_excluded(self):
        for successful_tail in (False, True):
            with self.subTest(successful_tail=successful_tail), tempfile.TemporaryDirectory() as directory:
                root, result, checkpoint, message = self.fixture(directory, counted=True, success=successful_tail)
                if not successful_tail:
                    message['operator_pause']['details_before_pause']['1']['success'] = True
                with self.assertRaises(ValueError):resume.prepare(message)

    def test_local_pause_proof_rejects_an_episode_closed_before_user_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'old'
            prefix = {}
            for index in range(4):
                episode = root/'official_only/sessions'/str(index)
                episode.mkdir(parents=True)
                meta = dict(task='push_T',task_information_mode='official_only',wall_started_unix_s=index)
                (episode/'episode.json').write_text(json.dumps(meta))
                if index < 3:
                    (episode/'END.json').write_text('{}')
                    prefix[str(index)] = dict(layout_id=index,success=index!=1,score=float(index!=1))
            tail = root/'official_only/sessions/3'
            pause = dict(user_requested_pause=True,official_root=str(root),job_id='42',
                active_session=str(tail),active_session_end_before_stop=False,unix=20,
                active_episode=json.loads((tail/'episode.json').read_text()),valid_official_prefix=3,
                official_status_before_stop=dict(evaluator_result=dict(eval_time=3,details=prefix)))
            released = dict(status='paused',user_requested=True,job_id='42',slurm_state='CANCELLED',
                squeue_empty=True,official_only=dict(valid_completed=3))
            certificate, release = root/'pause.json',root/'released.json'
            certificate.write_text(json.dumps(pause));release.write_text(json.dumps(released))
            proof = resume.operator_pause_proof(root,'official_only',certificate,release)
            self.assertEqual(proof['completed_before_pause'], 3)
            (tail/'END.json').write_text('{"wall_seconds":30}')
            resume.operator_pause_proof(root,'official_only',certificate,release)
            (tail/'END.json').write_text('{"wall_seconds":5}')
            with self.assertRaisesRegex(ValueError, 'closed before'):
                resume.operator_pause_proof(root,'official_only',certificate,release)
