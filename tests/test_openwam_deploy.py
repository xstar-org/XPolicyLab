"""Exercise observation boundaries, chunk sizes and per-env early completion."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

from client_server.ws.model_client import WsModelClient

policy = Path(__file__).resolve().parents[1] / 'policy/OpenWAM'
package_spec = importlib.util.spec_from_file_location('test_openwam_policy', policy / '__init__.py')
package = importlib.util.module_from_spec(package_spec)
sys.modules[package_spec.name] = package
package_spec.loader.exec_module(package)
spec = importlib.util.spec_from_file_location('test_openwam_policy.deploy', policy / 'deploy.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class Env:
    def __init__(self, ends):
        self.counts = dict.fromkeys(ends, 0)
        self.ends = ends
        self.reads = []

    def get_running_env_idx_list(self):
        return [e for e in self.ends if self.counts[e] < self.ends[e]]

    def is_episode_end(self):
        return not self.get_running_env_idx_list()

    def get_obs_batch(self, ids):
        self.reads.append([(e, self.counts[e]) for e in ids])
        return [
            dict(env_idx=e, step=self.counts[e], vision={'head': {'color': (e, self.counts[e])}})
            for e in ids
        ]

    def get_obs(self):
        return self.get_obs_batch([0])[0]

    def take_action_batch(self, actions, ids):
        for action, e in zip(actions, ids):
            assert action == (e, self.counts[e])
            assert self.counts[e] < self.ends[e]
            self.counts[e] += 1

    def take_action(self, action):
        self.take_action_batch([action], [0])


class Client:
    def __init__(self, size):
        self.size = size
        self.latest = {}

    def call(self, func_name, obs=None):
        if func_name == 'reset':
            self.latest = {}
        elif func_name.startswith('update_obs'):
            self.latest = {o['env_idx']: o['step'] for o in (obs if isinstance(obs, list) else [obs])}
        else:
            ids = obs if func_name == 'get_action_batch' else [0]
            chunks = [[(e, self.latest[e] + j) for j in range(self.size)] for e in ids]
            return chunks if func_name == 'get_action_batch' else chunks[0]


class CadenceTests(unittest.TestCase):
    def test_single_cadence_and_termination(self):
        for size in (1, 4, 7, 32):
            with self.subTest(size=size):
                env = Env({0: 11})
                deploy.eval_one_episode(env, Client(size))
                self.assertEqual(env.counts, {0: 11})
                self.assertEqual(env.reads, [[(0, i)] for i in range(0, 11, size)])

    def test_batch_retirement_keeps_actions_and_observations_aligned(self):
        env = Env({0: 2, 4: 9})
        deploy.eval_one_episode_batch(env, Client(4))
        self.assertEqual(env.counts, {0: 2, 4: 9})
        self.assertEqual(env.reads, [[(0, 0), (4, 0)], [(4, 4)], [(4, 8)]])

    def test_empty_chunks_fail_without_spinning(self):
        for loop in (deploy.eval_one_episode, deploy.eval_one_episode_batch):
            with self.assertRaises(ValueError):
                loop(Env({0: 3}), Client(0))

    def test_batch_history_uses_exact_action_indices_after_one_env_retires(self):
        class HistoryProtocol:
            async def connect(self, **kwargs):
                pass

            async def close(self):
                pass

            async def reset(self, **kwargs):
                return SimpleNamespace(payload={'result': {'openwam_fixed_history': {'stride': 25, 'clip_frames': 9}}})

            async def call(self, func_name, obs, **kwargs):
                assert func_name == 'infer_fixed_history_batch'
                current = [row for row in obs if row.get('history_current')]
                for row in current:
                    env, step = row['env_idx'], row['frame_idx']
                    expected = {0, *(max(0, step - 25 * offset) for offset in range(8))}
                    local = [item for item in obs if item['env_idx'] == env]
                    assert {item['frame_idx'] for item in local} == expected
                    assert all(item['vision']['head']['color'] == (env, item['frame_idx']) for item in local)
                chunks = [[(row['env_idx'], row['frame_idx'] + j) for j in range(48)] for row in current]
                return SimpleNamespace(payload={'result': chunks})

        for loop, ends in (
            (deploy.eval_one_episode, {0: 307}),
            (deploy.eval_one_episode_batch, {8: 77, 3: 307}),
        ):
            with self.subTest(entry_point=loop.__name__):
                env = Env(ends)
                with WsModelClient(url='ws://unused', evaluation_id='eval', trial_id='trial', client=HistoryProtocol()) as client:
                    loop(env, client)
                self.assertEqual(env.counts, env.ends)
                self.assertIn([(e, 23) for e in ends], env.reads)
                last_env = next(reversed(ends))
                self.assertIn([(last_env, 288)], env.reads)
                self.assertNotIn([(last_env, 24)], env.reads)


if __name__ == '__main__':
    unittest.main()
