"""The strict arm must not recover original task hints via text, files or fallback."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

POLICY = Path(__file__).resolve().parents[1]/'policy/Q0_Preview'
spec = importlib.util.spec_from_file_location('q0_task_context', POLICY/'runtime/task_context.py')
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)


class TaskInformationTests(unittest.TestCase):
    def test_original_keeps_restricted_thresholds(self):
        text = tasks.task_context('push_T')
        self.assertIn('7 mm', text)
        self.assertIn('7 degrees', text)
        self.assertIn('1 cm', text)

    def test_official_is_exact_description_and_public_scoring(self):
        card = json.loads((POLICY/'assets/tasks/official_task_descriptions.json').read_text())['cards']['push_T']
        text = tasks.task_context('push_T', mode='official_only')
        self.assertEqual(text, card['description'] + '\n\nPublic scoring:\n' + '\n'.join(card['scoring']))
        for secret in ('7 mm', '7 degrees', '1 cm', 'Optional considerations', 'Evaluation notes'):
            self.assertNotIn(secret, text)

    def test_official_does_not_open_original_assets(self):
        original = Path.read_text
        def guarded(path, *args, **kwargs):
            if path.name in ('task_cards.json', 'task_reference.json'):
                raise AssertionError('Original task asset read in strict mode')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guarded):
            self.assertIn('Push', tasks.task_context('push_T', mode='official_only'))
            self.assertEqual(tasks.task_step_limit('push_T', mode='official_only'), 600)

    def test_unknown_task_and_mode_fail_closed(self):
        with self.assertRaises(ValueError):
            tasks.task_context('make_kong', mode='official_only')
        with self.assertRaises(ValueError):
            tasks.task_context('push_T', mode='typo')
        with self.assertRaises(ValueError):
            tasks.task_reference('push_T', mode='official_only')

    def test_reference_exposure_and_stale_workspace_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            tasks.stage_task_reference(work, 'push_T', 'official_only')
            self.assertFalse((work/'task_reference.json').exists())
            tasks.stage_task_reference(work, 'push_T', 'original')
            self.assertIn('7 mm', (work/'task_reference.json').read_text())
            with self.assertRaises(RuntimeError):
                tasks.stage_task_reference(work, 'push_T', 'official_only')

    def test_stale_reference_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work/'task_reference.json').symlink_to(work/'missing')
            with self.assertRaises(RuntimeError):
                tasks.stage_task_reference(work, 'push_T', 'official_only')

    def test_original_prompt_unchanged_strict_prompt_removes_reference(self):
        text = (POLICY/'assets/prompts/prompt.md').read_text()
        self.assertEqual(tasks.planning_prompt(text), text)
        strict = tasks.planning_prompt(text, 'official_only')
        self.assertNotIn('/work/task_reference.json', strict)
        self.assertNotIn('Required strategy', strict)
        self.assertNotIn('Evaluation notes', strict)
        self.assertEqual(strict[strict.index('## Gripper pose reference points'):],
                         text[text.index('## Gripper pose reference points'):])


if __name__ == '__main__':
    unittest.main()
