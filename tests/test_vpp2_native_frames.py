"""Test actual client routing, early termination, default dense capture, and guards."""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from XPolicyLab.policy.VPP2 import deploy
from XPolicyLab.policy.VPP2.native_frames import NATIVE_STEP_FIELD, needs_frame


class Env:
    physx_monitor_enabled = True
    def __init__(self):
        self.take_action_cnt = [0, 0]
        self.captured = []
        self.guarded = []
        self.actions = []
    def get_running_env_idx_list(self):
        return [e for e, limit in enumerate([53, 97]) if self.take_action_cnt[e] < limit]
    def is_episode_end(self): return not self.get_running_env_idx_list()
    def get_obs_batch(self, ids):
        self.captured.extend((e,self.take_action_cnt[e]) for e in ids)
        return [{'env_idx':e} for e in ids]
    def take_action_batch(self, actions, ids):
        for e,a in zip(ids,actions):
            assert a == e
            self.actions.append((e,self.take_action_cnt[e]))
            self.take_action_cnt[e] += 1
    def _check_physx_broken_envs(self): pass
    def _check_endpose_finite(self, ids):
        self.guarded.extend((e,self.take_action_cnt[e]) for e in ids)


class Client:
    def __init__(self): self.uploads=[]; self.calls=[]
    def call(self, func_name, obs=None):
        self.calls.append(func_name)
        if func_name=='update_obs_batch': self.uploads.extend(obs)
        if func_name=='get_action_batch': return [[e]*24 for e in obs['env_idx_list']]


def test_native_clock_early_end_reset_and_finite_guards():
    env,client=Env(),Client()
    with patch.dict('os.environ',VPP2_NATIVE_FRAME_SAMPLING='1'):
        deploy.eval_one_episode_batch(env,client)
    assert env.take_action_cnt==[53,97] and len(env.actions)==150
    assert [(o['env_idx'],o[NATIVE_STEP_FIELD]) for o in client.uploads]==env.captured
    assert all(needs_frame(t) for _,t in env.captured)
    assert all(not needs_frame(t) for _,t in env.guarded)
    assert len(env.captured)<60 and env.guarded
    assert len({o[deploy._CLIENT_ID_FIELD] for o in client.uploads})==1
    assert client.calls[:2]==['reset','reset_client']


@pytest.mark.parametrize('enabled', [None, '0'])
def test_default_and_disabled_keep_dense_native_frames(enabled):
    env,client=Env(),Client()
    settings = {} if enabled is None else {'VPP2_NATIVE_FRAME_SAMPLING': enabled}
    with patch.dict('os.environ', settings, clear=True):
        deploy.eval_one_episode_batch(env,client)
    assert all(NATIVE_STEP_FIELD not in obs for obs in client.uploads)
    assert len(env.captured)==len(client.uploads)==150
    assert [t for e,t in env.captured if e==1]==list(range(97))


@pytest.mark.parametrize('chunk_size', [0, 23, 25])
def test_native_replan_length_rejected_before_actions(chunk_size):
    class WrongLength(Client):
        def call(self, func_name, obs=None):
            if func_name == 'get_action_batch':
                return [[e] * chunk_size for e in obs['env_idx_list']]
            return super().call(func_name, obs)
    env = Env()
    with patch.dict('os.environ', VPP2_NATIVE_FRAME_SAMPLING='1'):
        with pytest.raises(ValueError):
            deploy.eval_one_episode_batch(env, WrongLength())
    assert not env.actions


def test_native_wrong_replan_boundary_rejected_before_observation():
    env = Env()
    env.take_action_cnt = [1, 1]
    with patch.dict('os.environ', VPP2_NATIVE_FRAME_SAMPLING='1'):
        with pytest.raises(ValueError, match='24-action boundary'):
            deploy.eval_one_episode_batch(env, Client())
    assert not env.actions and not env.captured


def test_native_observation_identity_mismatch_rejected_before_actions():
    class Reordered(Env):
        def get_obs_batch(self, ids):
            return super().get_obs_batch(ids)[::-1]
    env = Reordered()
    with patch.dict('os.environ', VPP2_NATIVE_FRAME_SAMPLING='1'):
        with pytest.raises(ValueError, match='identities'):
            deploy.eval_one_episode_batch(env, Client())
    assert not env.actions
