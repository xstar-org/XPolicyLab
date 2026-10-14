import importlib.util
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("observation_pipeline", Path(__file__).parents[1] / "observation_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class Client:
    def __init__(self):
        self.events = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def call(self, func_name=None, obs=None, **kwargs):
        if func_name == "update_obs_batch":
            self.entered.set()
            self.release.wait(5)
        self.events.append((func_name, obs))
        return "response"


class PipelineTests(unittest.TestCase):
    def test_owned_frames_fifo_and_rpc_barriers(self):
        client = Client()
        original = client.call
        client.release.clear()
        def run(env, c):
            obs = [{"pixels": bytearray(b"RGB"), "state": [1]}]
            c.call(func_name="update_obs_batch", obs=obs)
            self.assertTrue(client.entered.wait(2))
            obs[0]["pixels"][0] = 0
            obs[0]["state"][0] = 9
            c.call(func_name="update_obs_batch", obs=[{"state": [2]}])
            client.release.set()
            self.assertEqual(c.call(func_name="get_action_batch", obs={}), "response")
            c.call(func_name="reset_client", obs={})
            return 7
        self.assertEqual(pipeline.run_with_observation_pipeline(None, client, run), 7)
        self.assertEqual([x[0] for x in client.events], ["update_obs_batch", "update_obs_batch", "get_action_batch", "reset_client"])
        self.assertEqual(client.events[0][1][0], {"pixels": bytearray(b"RGB"), "state": [1]})
        self.assertEqual(client.call, original)

    def test_exit_flushes_even_when_rollout_raises(self):
        client = Client()
        original = client.call
        def run(env, c):
            c.call(func_name="update_obs_batch", obs=[1])
            raise ValueError("simulation failure")
        with self.assertRaisesRegex(ValueError, "simulation failure"):
            pipeline.run_with_observation_pipeline(None, client, run)
        self.assertEqual(client.events, [("update_obs_batch", [1])])
        self.assertEqual(client.call, original)

    def test_upload_failure_prevents_inference_and_restores_client(self):
        class Failing(Client):
            def call(self, func_name=None, obs=None, **kwargs):
                if func_name == "update_obs_batch":
                    raise RuntimeError("upload failure")
                return super().call(func_name, obs, **kwargs)
        client = Failing()
        original = client.call
        def run(env, c):
            c.call(func_name="update_obs_batch", obs=[1])
            c.call(func_name="get_action_batch")
        with self.assertRaisesRegex(RuntimeError, "upload failure"):
            pipeline.run_with_observation_pipeline(None, client, run)
        self.assertFalse(client.events)
        self.assertEqual(client.call, original)

    def test_feature_is_opt_in(self):
        client = Client()
        original = client.call
        @pipeline.overlap_observation_uploads
        def run(env, c):
            return c.call == original
        with patch.dict(os.environ, {"VPP2_OBSERVATION_UPLOAD_OVERLAP": "0"}):
            self.assertTrue(run(None, client))
        with patch.dict(os.environ, {"VPP2_OBSERVATION_UPLOAD_OVERLAP": "1"}):
            self.assertFalse(run(None, client))
        self.assertEqual(client.call, original)


if __name__ == "__main__":
    unittest.main()
