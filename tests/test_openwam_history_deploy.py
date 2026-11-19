def test_robodojo_observes_during_chunks_and_removes_finished_slots():
    from policy.OpenWAMHistory.deploy import eval_one_episode_batch

    class Env:
        step = 0

        def get_running_env_idx_list(self):
            return [0, 1] if self.step < 4 else [1]

        def get_obs_batch(self, slots):
            return [{"env_idx": i} for i in slots]

        def take_action_batch(self, actions, slots):
            assert slots == self.get_running_env_idx_list()
            assert actions == [0] * len(slots)
            self.step += 1

        def is_episode_end(self):
            return self.step == 15

    traces = []

    class Client:
        def call(self, func_name, obs=None):
            if func_name == "get_action_batch":
                return [[0] * 7 for _ in obs]
            if func_name == "update_obs_batch":
                traces.append((obs[0]["frame_id"], [o["env_idx"] for o in obs]))

    eval_one_episode_batch(Env(), Client())
    assert traces == [(0, [0, 1]), (5, [1]), (7, [1]), (10, [1]), (14, [1])]
