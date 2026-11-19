"""XPolicyLab adapter for the history ablation, using its native EE conversion."""

import numpy as np
from XPolicyLab.policy.OpenWAM.model import Model as OpenWAMAdapter


class Model(OpenWAMAdapter):
    def reset(self):
        super().reset()
        self._banks = {}
        self._conditions_by_env = {}

    def update_obs_batch(self, obs_list):
        from openwam.history import ObservationHistory

        for obs in obs_list:
            slot = int(obs["env_idx"])
            conditions = self._conditions(self._encode_obs(obs))
            if hasattr(self._engine.architecture, "history_context"):
                if slot not in self._banks:
                    self._banks[slot] = ObservationHistory(self._engine.cfg.dataloader.history_mode)
                conditions["history_snapshot"] = self._banks[slot].observe(
                    conditions["first_frame_image"][0], int(obs["frame_id"])
                )
            self._conditions_by_env[slot] = conditions

    def get_action_batch(self, env_idx_list):
        chunks = []
        for slot in env_idx_list:
            result = self._engine.generate(self._conditions_by_env[int(slot)])
            actions = np.array(result["actions"][: self.replan_steps], copy=True)
            dims = getattr(self._engine.architecture, "binary_command_dims", ())
            if dims:
                actions[..., dims] = np.where(actions[..., dims] > 0.5, 1.0, -1.0)
            chunks.append(self._eef20_chunk_to_native(actions))
        return chunks
