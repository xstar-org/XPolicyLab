"""Reject foreign/reordered model context and provisional END records."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t'
sys.path.insert(0, str(SCRIPTS))
audit = importlib.import_module('audit_strict')
requests = importlib.import_module('audit_strict_requests')


class StrictRequestAuditTests(unittest.TestCase):
    def test_unreviewed_cli_developer_instructions_require_review(self):
        def wrapper(text):
            return [{'type':'message', 'role':'developer',
                     'content':[{'type':'input_text', 'text':text}]}]
        self.assertEqual(requests.verify_recorded_wrappers([], 'audited prompt')['requested_instructions'], 0)
        self.assertEqual(requests.verify_recorded_wrappers(wrapper('audited prompt'),
                         'audited prompt')['requested_instructions'], 1)
        with self.assertRaises(ValueError):
            requests.verify_recorded_wrappers(wrapper('extra task hint'), 'audited prompt')
        with patch.dict(requests.REVIEWED_GENERIC_CLI_WRAPPERS,
                        {requests.hashlib.sha256(b'generic permissions').hexdigest():'reviewed test fixture'}):
            self.assertEqual(requests.verify_recorded_wrappers(wrapper('generic permissions'),
                             'audited prompt')['reviewed_generic_blocks'], 1)

    def test_changed_actual_sandbox_material_requires_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'sandbox').mkdir()
            for name, source in requests.SANDBOX_ASSETS.items():
                shutil.copy2(audit.POLICY/source, root/'sandbox'/name)
            self.assertEqual(len(requests.verify_sandbox_assets(root)), 4)
            (root/'sandbox/geometry_reference.md').write_text('Extra task hints')
            with self.assertRaises(ValueError):
                requests.verify_sandbox_assets(root)

    def test_subsequence_preserves_provenance_with_duplicate_messages(self):
        a, b, c = ({'text':x} for x in ('a', 'b', 'c'))
        ids = {requests.digest(a):[0, 2], requests.digest(b):[1], requests.digest(c):[3]}
        requests.match_journal([a, b, a, c], ids)
        requests.match_journal([a, c], ids)
        with self.assertRaises(ValueError):
            requests.match_journal([c, a], ids)
        with self.assertRaises(ValueError):
            requests.match_journal([{'text':'foreign episode memory'}], ids)

    def test_provisional_end_does_not_close_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = audit.module('test_strict_audit_task', 'runtime/task_context.py')
            (root/'episode.json').write_text(json.dumps({'task_information_mode':'strict_official',
                'task_context':task.task_context('push_T', mode='strict_official')}))
            (root/'END.json').write_text('{"final": false}')
            self.assertFalse(audit.audit([root])['episodes'][0]['closed'])
            (root/'END.json').write_text('{"final": true}')
            self.assertTrue(audit.audit([root])['episodes'][0]['closed'])
            self.assertFalse(audit.audit([root])['runtime_verified'])


if __name__ == '__main__':
    unittest.main()
