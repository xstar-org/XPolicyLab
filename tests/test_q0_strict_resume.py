"""Transport recovery must preserve every completed outcome and its provenance."""
import importlib.util
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('q0_strict_resume', SCRIPTS/'strict_resume.py')
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class StrictResumeTests(unittest.TestCase):
    def fixture(self, root):
        label = root/'strict_official'
        for i in range(3):
            ep = label/'sessions'/str(i)
            ep.mkdir(parents=True)
            (ep/'episode.json').write_text(json.dumps(dict(task='push_T',
                task_information_mode='strict_official', wall_started_unix_s=i)))
            if i < 2:
                (ep/'END.json').write_text('{"final":true}')
        (label/'sessions/2/STOP').touch()
        (label/'BRIDGE_ERROR.json').write_text(json.dumps(dict(unix=4,
            error="TimeoutError('SSH transport recovery window exhausted')")))
        (root/'ERROR.json').write_text(json.dumps(dict(unix=5,
            error="RuntimeError('Strict agent bridge exited unexpectedly')")))
        (label/'TRANSPORT_STATE.json').write_text('{"event":"recovery_exhausted","unix":3}')
        rows = {'0':dict(layout_id=0, success=False), '1':dict(layout_id=1, success=True)}
        native = dict(eval_time=2, details=rows)
        (root/'status.json').write_text(json.dumps(dict(evaluator_result=native)))
        return native

    def test_transport_tail_can_be_retried_while_completed_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = self.fixture(root)
            before = {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            proof = resume.resume_evidence(root, native)
            self.assertEqual(proof['completed'], 2)
            self.assertFalse(proof['exclude_native_tail'])
            self.assertEqual([Path(p).name for p in proof['inherited_sessions']], ['0', '1'])
            native = dict(eval_time=3, details={**native['details'], '2':dict(layout_id=2, success=False)})
            self.assertTrue(resume.resume_evidence(root, native)['exclude_native_tail'])
            self.assertEqual(before, {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            native['details']['2']['success'] = True
            with self.assertRaises(ValueError):
                resume.resume_evidence(root, native)

    def test_timeout_alone_closed_tail_and_changed_prefix_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = self.fixture(root)
            state = root/'strict_official/TRANSPORT_STATE.json'
            original = state.read_bytes()
            state.write_text('{"event":"reconnect_attempt","unix":3}')
            with self.assertRaises(ValueError):resume.resume_evidence(root, native)
            state.write_bytes(original)
            tail = root/'strict_official/sessions/2/END.json'
            tail.write_text('{"final":true}')
            with self.assertRaises(ValueError):resume.resume_evidence(root, native)
            tail.unlink()
            native['details']['0']['success'] = True
            with self.assertRaises(ValueError):resume.resume_evidence(root, native)

    def test_only_fifty_distinct_final_archives_enter_audit_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = []
            for i in range(50):
                ep = root/'archives'/str(i)
                ep.mkdir(parents=True)
                (ep/'END.json').write_text('{"final":true}')
                episodes.append(ep)
            with self.assertRaises(ValueError):resume.audit_view(root, episodes[:-1])
            with self.assertRaises(ValueError):resume.audit_view(root, episodes[:-1]+[episodes[0]])
            (episodes[-1]/'END.json').write_text('{"final":false}')
            with self.assertRaises(ValueError):resume.audit_view(root, episodes)
            (episodes[-1]/'END.json').write_text('{"final":true}')
            view = resume.audit_view(root, episodes)
            self.assertEqual({p.resolve() for p in view.iterdir()}, set(episodes))

    def storage_fixture(self, root):
        native = self.fixture(root)
        (root/'strict_official/sessions/2/END.json').write_text('{"final":true,"reason":"policy_error"}')
        files = {}
        for key, value in [('native_result', json.dumps(native)),
                           ('server_log', 'np.save write_array OSError: 921600 requested and 0 written'),
                           ('node_write_probe', json.dumps(dict(returncode=0, stdout=json.dumps(
                               dict(node='test-compute', checks=[dict(ok=True)]*3)))) )]:
            path = root/(key+'.evidence')
            path.write_text(value)
            files[key] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        meta = b"{'descr': '|u1', 'fortran_order': False, 'shape': (480, 640, 3), }\n"
        header = b'\x93NUMPY\x01\x00'+len(meta).to_bytes(2, 'little')+meta
        cert = dict(source_service=root.name, interrupted_sessions=['2'], **files,
                    truncated_npy_files=[dict(session='2', size_bytes=len(header),
                        header_base64=base64.b64encode(header).decode(), sha256=hashlib.sha256(header).hexdigest(),
                        path='runs/'+root.name+'/sessions/2/observations/obs_1/cam_head.rgb.npy')])
        path = root/'storage-certificate.json'
        path.write_text(json.dumps(cert))
        return native, cert, path

    def test_storage_resume_keeps_native_failure_and_never_removes_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native, cert, path = self.storage_fixture(root)
            before = {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = resume.storage_resume_evidence(root, native, path)
            self.assertEqual(result['completed'], 2)
            self.assertEqual(result['storage_interruption']['native_outcomes_removed'], 0)
            self.assertEqual([Path(p).name for p in result['inherited_sessions']], ['0', '1'])
            self.assertEqual(before, {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            changed = json.loads(json.dumps(native));changed['details']['0']['success'] = True
            with self.assertRaises(ValueError):resume.storage_resume_evidence(root, changed, path)

    def test_storage_resume_requires_partial_file_for_each_closed_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native, cert, path = self.storage_fixture(root)
            records = cert['truncated_npy_files']
            cert['truncated_npy_files'] = []
            path.write_text(json.dumps(cert))
            with self.assertRaises(ValueError):resume.storage_resume_evidence(root, native, path)
            cert['truncated_npy_files'] = records
            path.write_text(json.dumps(cert))
            (root/'strict_official/sessions/2/END.json').write_text('{"final":true,"reason":"client_closed"}')
            with self.assertRaises(ValueError):resume.storage_resume_evidence(root, native, path)

    def test_storage_resume_rejects_changed_evidence_or_failing_node_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native, cert, path = self.storage_fixture(root)
            log = Path(cert['server_log']['path']);log.write_text('changed')
            with self.assertRaises(ValueError):resume.storage_resume_evidence(root, native, path)
            log.write_text('np.save write_array OSError: 921600 requested and 0 written')
            probe = Path(cert['node_write_probe']['path'])
            probe.write_text(json.dumps(dict(returncode=0, stdout=json.dumps(dict(node='test-compute', checks=[dict(ok=False)]*3)))))
            cert['node_write_probe']['sha256'] = hashlib.sha256(probe.read_bytes()).hexdigest()
            path.write_text(json.dumps(cert))
            with self.assertRaises(ValueError):resume.storage_resume_evidence(root, native, path)
