"""Native recovery preserves outcomes and excludes the interrupted policy archive."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


FILE = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t/native_resume.py'
spec = importlib.util.spec_from_file_location('q0_native_resume', FILE)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class NativeResumeTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root/'results').mkdir()
        (root/'results/old-original.json').write_text('{"returncode":1}')
        base = root/'code/benchmark-cli/eval_result/RoboDojo/push_T/Q0_Preview/arx_x5/1_ckpt_name=api,action_type=joint'
        (base/'q0-old-original').mkdir(parents=True)
        details = {str(i):dict(layout_id=i, success=i>0, score=float(i>0)) for i in range(3)}
        result = dict(eval_time=3, success_rate=2/3,score=200/3, details=details)
        source = base/'q0-old-original/_result.json'
        source.write_text(json.dumps(result,indent=2))
        checkpoint = base/'_resume_q0-old-original.json'
        checkpoint.write_text(json.dumps(dict(run_id='q0-old-original',save_dir='old',
            task_name='push_T',policy_name='Q0_Preview',config_name='arx_x5',eval_seed=1,
            additional_info='ckpt_name=api,action_type=joint',success_nums=2,fail_nums=1,
            total_score=2,completed_layout_ids=[0,1,2],abandoned_layout_ids=[],details=details,restart_count=0)))
        request = dict(root=str(root),benchmark_name='benchmark-cli',source_service='old-original',target_service='new-original')
        return root,base,source,checkpoint,request

    def test_resume_preserves_official_outcomes_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root,base,source,checkpoint,request = self.fixture(directory)
            before = source.read_bytes(),checkpoint.read_bytes()
            record = resume.prepare(request)
            self.assertEqual(record['completed'],3)
            self.assertFalse(record['interrupted_episode_counted'])
            target = base/'q0-new-original/_result.json'
            self.assertEqual(target.read_bytes(),before[0])
            state = json.loads((base/'_resume_q0-new-original.json').read_text())
            self.assertEqual(state['run_id'],'q0-new-original')
            self.assertEqual(state['completed_layout_ids'],[0,1,2])
            self.assertEqual(state['restart_count'],1)
            self.assertEqual((source.read_bytes(),checkpoint.read_bytes()),before)
            stamp = target.stat().st_mtime_ns
            self.assertEqual(resume.prepare(request),record)
            self.assertEqual(target.stat().st_mtime_ns,stamp)

    def test_live_worker_and_mismatched_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root,base,source,checkpoint,request = self.fixture(directory)
            worker = root/'results/old-original.json'
            worker.unlink()
            with self.assertRaises(ValueError):resume.prepare(request)
            worker.write_text('{"returncode":1}')
            state = json.loads(checkpoint.read_text());state['success_nums']=3
            checkpoint.write_text(json.dumps(state))
            with self.assertRaises(ValueError):resume.prepare(request)
            self.assertFalse((base/'q0-new-original/_result.json').exists())

    def test_recursive_policy_prefix_excludes_interrupted_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root/'first'
            for i in range(4):
                ep = first/'original/sessions'/str(i);ep.mkdir(parents=True)
                (ep/'episode.json').write_text(json.dumps(dict(wall_started_unix_s=i,task='push_T',task_information_mode='original')))
                (ep/'END.json').write_text('{}')
            bad = first/'original/sessions/3/errors';bad.mkdir()
            (bad/'infra.json').write_text('{"infrastructure":true}')
            inherited = resume.completed_sessions(first,'original',3)
            self.assertEqual(len(inherited),3)
            second = root/'second';second.mkdir()
            (second/'resume-lineage.json').write_text(json.dumps(dict(stages=dict(original=dict(inherited_sessions=inherited)))))
            for i in range(3):
                ep = second/'original/sessions'/str(i);ep.mkdir(parents=True)
                (ep/'episode.json').write_text(json.dumps(dict(wall_started_unix_s=10+i,task='push_T',task_information_mode='original')))
                (ep/'END.json').write_text('{}')
            all_done = resume.completed_sessions(second,'original',5)
            self.assertEqual(all_done[:3],inherited)
            self.assertEqual(len(all_done),5)
            with self.assertRaises(ValueError):resume.completed_sessions(first,'original',4)
