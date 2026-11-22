"""Reject foreign/reordered model context and provisional END records."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t'
sys.path.insert(0, str(SCRIPTS))
audit = importlib.import_module('audit_strict')
requests = importlib.import_module('audit_strict_requests')


class StrictRequestAuditTests(unittest.TestCase):
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
