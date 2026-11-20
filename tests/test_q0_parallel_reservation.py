"""Delegation blocks duplicate submission before queueing, without stopping work."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/experiments/push_t'
sys.path.insert(0, str(ROOT))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reservation = load('reserve_parallel_arm')
control = load('cluster_control')


class ParallelReservationTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root/'code').mkdir()
        (root/'requests').mkdir()
        config = root/'code/benchmark/task/RoboDojo/config/_task.yml'
        config.parent.mkdir(parents=True)
        config.write_text('tasks:\n  push_T:\n    eval_nums: 50\n')
        message = dict(root=str(root), serial_service='serial-official_only',
            parallel_service='parallel-official_only', operator_authorized_parallel=True)
        return root, message

    def test_delegation_is_idempotent_and_blocks_before_queue_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, message = self.fixture(directory)
            original = root/'runs/serial-original'
            original.mkdir(parents=True)
            (original/'LIVE').write_text('keep running')
            result = reservation.reserve(message)
            self.assertEqual(reservation.reserve(message), result)
            with self.assertRaisesRegex(ValueError, 'Conflicting launcher'):
                control.run(root, dict(action='submit', name='serial-official_only',
                    mode='official_only', episodes=50, cluster_prefix='/python', site_setup='/site.sh'))
            self.assertFalse((root/'requests/serial-official_only.json').exists())
            self.assertFalse((original/'STOP').exists())
            self.assertEqual((original/'LIVE').read_text(), 'keep running')

    def test_requires_authorization_and_rejects_existing_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root, message = self.fixture(directory)
            with self.assertRaises(ValueError):
                reservation.reserve(message | dict(operator_authorized_parallel=False))
            (root/'requests/serial-official_only.json').write_text('{}')
            with self.assertRaises(ValueError):
                reservation.reserve(message)
            self.assertFalse((root/'code/launch-serial-official_only.sh').exists())


if __name__ == '__main__':
    unittest.main()
