import ast
from pathlib import Path
from typing import Any
from collections import deque
from types import SimpleNamespace
import unittest
import numpy as np

p = Path(__file__).resolve().parents[1] / "policy/VPP2/model.py"
tree = ast.parse(p.read_text())
keep = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
cls = next(n for n in keep if isinstance(n, ast.ClassDef))
cls.bases = []
init = next(
    n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
)
for i, n in enumerate(init.body):
    if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "checkpoint_path" for t in n.targets
    ):
        init.body = init.body[:i]
        break
def _pack_robot_state(obs, _action_type, dims):
    state = obs["state"]
    return np.concatenate([
        state["left_ee_pose"], state["left_ee_joint_state"],
        state["right_ee_pose"], state["right_ee_joint_state"],
    ])


def _unpack_robot_state(actions, _action_type, dims):
    return [
        {
            "left_ee_pose": action[:7],
            "left_ee_joint_state": action[7:8],
            "right_ee_pose": action[8:15],
            "right_ee_joint_state": action[15:16],
        }
        for action in actions
    ]


ns = dict(
    np=np,
    Any=Any,
    Path=Path,
    POLICY_DIR=Path("policy/VPP2"),
    resolve_checkpoint_root=lambda *_args, **_kwargs: Path("checkpoints/joint2b_s100000"),
    get_robot_action_dim_info=lambda _env: {"arm_dim": [6, 6], "ee_dim": [1, 1]},
    pack_robot_state=_pack_robot_state,
    unpack_robot_state=_unpack_robot_state,
)
exec(compile(ast.Module(body=keep, type_ignores=[]), str(p), "exec"), ns)


class Backend:
    history_condition_frames = 8

    def __init__(self):
        self._history_by_env = {}
        self._history_step_by_env = {}
        self.calls = []
        self.resets = []

    def observe(self, o):
        i = o["env_idx"]
        h = self._history_by_env.setdefault(i, deque())
        step = self._history_step_by_env.get(i, -1) + 1
        h.append((step, None))
        self._history_step_by_env[i] = step

    def _infer_action_chunks(self, observations, instructions):
        self.calls.append(([o["env_idx"] for o in observations], instructions))
        return [np.full((32, 16), o["env_idx"], dtype=np.float32) for o in observations]

    def reset_env(self, i):
        self.resets.append(i)
        self._history_by_env.pop(i, None)
        self._history_step_by_env.pop(i, None)


class ClientIsolation(unittest.TestCase):
    def test_interleaved_batch_clients_and_reset(self):
        m = ns["Model"](dict(action_type="ee", env_cfg_type="arx_x5"))
        m.model = Backend()

        def update(client, step):
            obs = []
            for i in range(2):
                obs.append(
                    dict(
                        _vpp2_client_id=client,
                        env_idx=i,
                        _vpp2_native_step=step,
                        description=client,
                        vision={
                            n: dict(color=np.zeros((2, 2, 3), dtype=np.uint8))
                            for n in ("cam_head", "cam_left_wrist", "cam_right_wrist")
                        },
                        state={
                            k: np.zeros(7 if "pose" in k else 1)
                            for k in (
                                "left_ee_pose",
                                "right_ee_pose",
                                "left_ee_joint_state",
                                "right_ee_joint_state",
                            )
                        },
                    )
                )
            m.update_obs_batch(obs)

        update("A", 0)
        a = m.get_action_batch(dict(_vpp2_client_id="A", env_idx_list=[0, 1]))
        aids = m.model.calls[-1][0]
        update("B", 0)
        b = m.get_action_batch(dict(_vpp2_client_id="B", env_idx_list=[0, 1]))
        bids = m.model.calls[-1][0]
        self.assertTrue(set(aids).isdisjoint(bids))
        self.assertEqual(len(a[0]), 24)
        self.assertEqual(len(b[0]), 24)
        m.reset_client(dict(_vpp2_client_id="A"))
        self.assertEqual(m.model.resets, aids)
        self.assertTrue(all(i in m.model._history_by_env for i in bids))
        m.reset()
        self.assertTrue(all(i in m.model._history_by_env for i in bids))
        update("B", 24)
        m.get_action_batch(dict(_vpp2_client_id="B", env_idx_list=[1, 0]))
        self.assertEqual(m.model.calls[-1][0], bids[::-1])
        self.assertEqual(m.history_guard_mismatches, 0)


if __name__ == "__main__":
    unittest.main()
