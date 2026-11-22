"""Regression checks for paths missed by the original text-only ablation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

POLICY = Path(__file__).resolve().parents[1]/'policy/Q0_Preview'


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, POLICY/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boundary = load('strict_boundary', 'runtime/strict_information.py')
tasks = load('strict_tasks', 'runtime/task_context.py')


class StrictInformationTests(unittest.TestCase):
    def test_removes_raw_not_only_display_fields(self):
        raw = {'state': {'left': {'near_home': True, 'home_position_error_m': .1,
                                'home_orientation_error_deg': 1, 'home_grasp_point': [1, 2, 3]}},
               'home_check': {'position_per_axis_m': .15, 'orientation_deg': 20}}
        result = boundary.strict_observation(raw)
        self.assertEqual(result, {'state': {'left': {'home_grasp_point': [1, 2, 3]}}})
        self.assertIn('home_check', raw)

    def test_new_reward_or_object_state_fields_fail_closed(self):
        for key in ('score', 'reward', 'object_pose', 'future_field'):
            with self.assertRaises(ValueError):
                boundary.strict_observation({'state': {}, key: 0})
            with self.assertRaises(ValueError):
                boundary.strict_observation({'state': {'left': {key: 0}}})

    def test_feedback_and_raw_tool_observation_are_checked(self):
        for key in boundary.FORBIDDEN_KEYS:
            with self.assertRaises(ValueError):
                boundary.validate_published({'last_execution': {'nested': [{key: 0}]}})
        boundary.validate_published({'last_execution': {'arrival': {'position_error_mm': 0}}})

    def test_only_description_no_scoring_appendix(self):
        card = json.loads((POLICY/'assets/tasks/official_task_descriptions.json').read_text())['cards']['push_T']
        self.assertEqual(tasks.task_context('push_T', mode='strict_official'), card['description'])

    def test_no_original_asset_reads_or_fallback(self):
        original = Path.read_text
        def guarded(path, *args, **kwargs):
            if path.name in ('task_reference.json', 'task_cards.json'):
                self.fail('Restricted asset read')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guarded):
            tasks.task_context('push_T', mode='strict_official')
            with self.assertRaises(ValueError):
                tasks.task_context('unknown_task', mode='strict_official')
            with self.assertRaises(ValueError):
                tasks.task_reference('push_T', mode='strict_official')

    def test_no_historical_prompt_or_stale_reference(self):
        result = tasks.planning_prompt('UNTRUSTED OLD TASK STRATEGY', 'strict_official')
        self.assertEqual(result, (POLICY/'assets/prompts/strict_prompt.md').read_text())
        for text in ('0.765', '7 mm', '7 degrees', 'Coin probe', 'Bread probe', 'geometry_evidence.md'):
            self.assertNotIn(text, result + (POLICY/'assets/geometry/strict_geometry.md').read_text())
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            tasks.stage_task_reference(work, 'push_T', 'strict_official')
            (work/'task_reference.json').symlink_to(work/'missing')
            with self.assertRaises(RuntimeError):
                tasks.stage_task_reference(work, 'push_T', 'strict_official')


if __name__ == '__main__':
    unittest.main()
