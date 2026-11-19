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

    def completed_fixture(self, directory):
        root,base,source,checkpoint,request = self.fixture(directory)
        details = {str(i):dict(layout_id=i,success=i>0,score=float(i>0)) for i in range(25)}
        source.write_text(json.dumps(dict(eval_time=25,success_rate=24/25,score=96.0,details=details)))
        checkpoint.unlink()
        (root/'results/old-original.json').write_text('{"returncode":0}')
        (root/'logs').mkdir()
        (root/'logs/old-original.log').write_text('[main] removed resume manifest (eval completed)\n[MAIN] external-agent eval finished\n')
        old = root/'code/benchmark-cli/task/RoboDojo/config/_task.yml'
        new = root/'code/benchmark-50/task/RoboDojo/config/_task.yml'
        old.parent.mkdir(parents=True);new.parent.mkdir(parents=True)
        old.write_text('tasks:\n  push_T:\n    data_source: "teleop"\n    eval_nums: 25\n\n  other:\n    eval_nums: 25\n')
        new.write_text(old.read_text().replace('eval_nums: 25','eval_nums: 50',1))
        request.update(benchmark_name='benchmark-50',source_benchmark_name='benchmark-cli',
            completed_prefix_proof=dict(completed=25,closed_archives_sha256='a'*64))
        return root,base,source,checkpoint,request,new

    def test_completed_default_prefix_extension_keeps_all_native_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root,base,source,checkpoint,request,new=self.completed_fixture(directory)
            before=source.read_bytes()
            record=resume.prepare(request)
            self.assertEqual(record['completed'],25)
            self.assertTrue(record['checkpoint_reconstructed_after_native_completion'])
            self.assertIsNone(record['source_checkpoint_sha256'])
            self.assertEqual(Path(record['target_result']).read_bytes(),before)
            self.assertEqual(source.read_bytes(),before)
            self.assertFalse(checkpoint.exists())
            state=json.loads((Path(record['target_result']).parent.parent/'_resume_q0-new-original.json').read_text())
            self.assertEqual(state['completed_layout_ids'],list(range(25)))
            self.assertEqual((state['success_nums'],state['fail_nums']),(24,1))
            self.assertEqual(resume.prepare(request),record)

    def test_completed_extension_rejects_changed_rules_and_missing_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root,base,source,checkpoint,request,new=self.completed_fixture(directory)
            content=new.read_text();new.write_text(content+'# changed rules\n')
            with self.assertRaises(ValueError):resume.prepare(request)
            new.write_text(content)
            proof=request.pop('completed_prefix_proof')
            with self.assertRaises(ValueError):resume.prepare(request)
            request['completed_prefix_proof']=proof
            (root/'results/old-original.json').write_text('{"returncode":1}')
            with self.assertRaises(ValueError):resume.prepare(request)

    def test_completed_extension_preserves_crlf_task_config_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root,base,source,checkpoint,request,new=self.completed_fixture(directory)
            old=root/'code/benchmark-cli/task/RoboDojo/config/_task.yml'
            for path in (old,new):path.write_bytes(path.read_bytes().replace(b'\n',b'\r\n'))
            record=resume.prepare(request)
            self.assertEqual(record['completed'],25)

    def test_only_proven_failed_tail_is_excluded_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root,base,source,checkpoint,request = self.fixture(directory)
            result=json.loads(source.read_text());result['details']['2'].update(success=False,score=0.0)
            result.update(success_rate=1/3,score=100/3);source.write_text(json.dumps(result))
            state=json.loads(checkpoint.read_text());state.update(details=result['details'],success_nums=1,fail_nums=2,total_score=1)
            checkpoint.write_text(json.dumps(state));before=source.read_bytes(),checkpoint.read_bytes()
            proof=dict(layout_id=2,infrastructure=True,archive_closed=False,error_sha256='a'*64,
                       error_path='/private/episode/errors/obs.json',session_path='/private/episode')
            request['exclude_infrastructure_tail']=proof
            record=resume.prepare(request)
            self.assertEqual(record['completed'],2)
            self.assertEqual(record['excluded_infrastructure_tail'],proof)
            target=json.loads((base/'q0-new-original/_result.json').read_text())
            self.assertEqual(target['details'],{k:v for k,v in result['details'].items() if k!='2'})
            self.assertEqual(target['success_rate'],.5)
            self.assertFalse(target['details']['0']['success'])
            state=json.loads((base/'_resume_q0-new-original.json').read_text())
            self.assertEqual((state['success_nums'],state['fail_nums'],state['total_score']),(1,1,1))
            self.assertEqual((source.read_bytes(),checkpoint.read_bytes()),before)
            self.assertEqual(resume.prepare(request),record)
            request['exclude_infrastructure_tail']={**proof,'layout_id':1}
            with self.assertRaises(ValueError):resume.prepare(request)

    def test_successful_tail_cannot_be_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root,base,source,checkpoint,request = self.fixture(directory)
            request['exclude_infrastructure_tail']=dict(layout_id=2,infrastructure=True,
                archive_closed=False,error_sha256='a'*64,error_path='/error',session_path='/episode')
            with self.assertRaises(ValueError):resume.prepare(request)

    def test_tail_proof_requires_unclosed_single_infrastructure_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for i in range(3):
                ep=root/'original/sessions'/str(i);ep.mkdir(parents=True)
                (ep/'episode.json').write_text(json.dumps(dict(wall_started_unix_s=i,task='push_T',task_information_mode='original')))
                if i<2:(ep/'END.json').write_text('{}')
            ep=root/'original/sessions/2';(ep/'errors').mkdir()
            with self.assertRaises(ValueError):resume.infrastructure_tail_proof(root,'original',2)
            (ep/'errors/error.json').write_text('{"infrastructure":true}')
            proof=resume.infrastructure_tail_proof(root,'original',2)
            self.assertEqual(proof['layout_id'],2)
            self.assertEqual(len(proof['error_sha256']),64)
            (ep/'END.json').write_text('{}')
            with self.assertRaises(ValueError):resume.infrastructure_tail_proof(root,'original',2)

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
