"""Simulator-free checks of active environment and dense-history ordering."""
import importlib.util
from pathlib import Path
import unittest
import os
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('vpp2_deploy', Path(__file__).resolve().parents[1] / 'policy/VPP2/deploy.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class Env:
    def __init__(self):
        self.steps = {2: 0, 7: 0}
        self.limits = {2: 1, 7: 5}
        self.actions = []

    def get_running_env_idx_list(self):
        return [i for i in self.steps if self.steps[i] < self.limits[i]]

    def is_episode_end(self):
        return not self.get_running_env_idx_list()

    def get_obs_batch(self, ids):
        return [dict(env_idx=i, step=self.steps[i]) for i in ids]

    def take_action_batch(self, actions, ids):
        for action, i in zip(actions, ids):
            assert action == i
            assert i in self.get_running_env_idx_list()
            self.actions.append((i, self.steps[i]))
            self.steps[i] += 1


class Client:
    def __init__(self):
        self.observations = []
        self.batches = []

    def call(self, func_name, obs=None):
        if func_name == 'update_obs_batch':
            self.observations.extend((o['env_idx'], o['step']) for o in obs)
        if func_name == 'get_action_batch':
            self.batches.append(obs['env_idx_list'][:])
            return [[i] * 3 for i in obs['env_idx_list']]


class DeployTests(unittest.TestCase):
    def test_early_end_and_tail_keep_exact_dense_history(self):
        for overlap in ("0", "1"):
            with self.subTest(overlap=overlap), patch.dict(
                os.environ, {"VPP2_OBSERVATION_UPLOAD_OVERLAP": overlap}
            ):
                env, client = Env(), Client()
                deploy.eval_one_episode_batch(TASK_ENV=env, model_client=client)
                self.assertEqual(client.batches, [[2, 7], [7]])
                self.assertEqual([s for i, s in client.observations if i == 7], [0, 1, 2, 3, 4])
                self.assertEqual([s for i, s in client.observations if i == 2], [0])
                self.assertEqual(env.steps, {2: 1, 7: 5})

    def test_invalid_chunk_rejected_before_actions(self):
        env = Env()
        class BadClient(Client):
            def call(self, func_name, obs=None):
                if func_name == 'get_action_batch':
                    return [[2], [7, 7]]
                return super().call(func_name, obs)
        with self.assertRaises(ValueError):
            deploy.eval_one_episode_batch(env, BadClient())
        self.assertEqual(env.actions, [])


if __name__ == '__main__':
    unittest.main()
