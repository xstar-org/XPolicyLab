"""Cache RGB observations for OpenWAM rollouts."""


class OpenWAMRolloutClient:
    """Adapt standard rollout calls when the checkpoint advertises fixed history."""

    def __init__(self, model_client):
        self._client = model_client
        self._obs_cache = None

    def call(self, func_name, obs=None):
        if func_name == "reset":
            result = self._client.call(func_name=func_name)
            spec = result.get("openwam_fixed_history") if isinstance(result, dict) else None
            self._obs_cache = ObsCache(**spec) if spec is not None else None
            self.step = 0
            self.observations = {}
            return result

        if self._obs_cache is not None:
            if func_name in {"update_obs", "update_obs_batch"}:
                return self._update_obs_batch(obs)
            if func_name == "get_action":
                chunks = self._get_actions()
                return chunks[0] if len(chunks) else chunks
            if func_name == "get_action_batch":
                return self._get_actions(obs)

        return self._client.call(func_name=func_name, obs=obs)

    def _update_obs_batch(self, observations):
        if isinstance(observations, dict):
            observations = [observations]
        self.observations = {}
        for index, obs in enumerate(observations):
            env_idx = int(obs.get("env_idx", index))
            self.observations[env_idx] = {**obs, "env_idx": env_idx}

    def after_action(self, read_observations, *, chunk_size):
        if self._obs_cache is None:
            return
        self.step += 1
        if self._obs_cache.capture_between_replans(self.step, chunk_size):
            observations = read_observations()
            if isinstance(observations, dict):
                observations = [observations]
            self._obs_cache.record(self.step, observations)

    def _get_actions(self, env_ids=None):
        observations = self.observations.values() if env_ids is None else (
            self.observations[int(env_idx)] for env_idx in env_ids
        )
        payload = self._obs_cache.request(self.step, list(observations))
        return self._client.call(func_name="infer_batch_from_cached_obs", obs=payload)


class ObsCache:
    """Store per-environment RGB observations at the steps needed for replanning."""

    def __init__(self, stride, clip_frames):
        if stride <= 0 or clip_frames < 2:
            raise ValueError("Fixed history requires a positive stride and an episode anchor")
        self.stride = stride
        self.offsets = range(clip_frames - 2, -1, -1)
        self.frames = {}

    def capture_between_replans(self, step, chunk_size):
        if step % chunk_size == 0:
            return False
        return any((step + offset * self.stride) % chunk_size == 0 for offset in self.offsets)

    def record(self, step, observations):
        for obs in observations:
            cameras = {
                name: {"color": camera["color"]} for name, camera in obs["vision"].items()
                if "color" in camera
            }
            self.frames[(int(obs["env_idx"]), step)] = cameras

    def request(self, step, observations):
        self.record(step, observations)
        ordered_ids = [0, *(max(0, step - offset * self.stride) for offset in self.offsets)]
        past_ids = dict.fromkeys(ordered_ids)
        del past_ids[step]
        # The wire decoder accepts a flat list of observations. Each image is
        # sent once even when the clip repeats the first frame near the reset.
        payload = []
        for obs in observations:
            env_idx = int(obs["env_idx"])
            for frame in past_ids:
                payload.append({
                    "env_idx": env_idx, "frame_idx": frame, "vision": self.frames[(env_idx, frame)],
                })
            current = {key: obs[key] for key in ("state", "instruction", "task_instruction") if key in obs}
            payload.append({
                **current, "env_idx": env_idx, "frame_idx": step,
                "vision": self.frames[(env_idx, step)], "cur_obs": True,
                "history_frame_ids": ordered_ids,
            })
        active = {int(obs["env_idx"]) for obs in observations}
        oldest = step - max(self.offsets) * self.stride
        self.frames = {
            (env_idx, frame): vision for (env_idx, frame), vision in self.frames.items()
            if env_idx in active and (frame == 0 or frame >= oldest)
        }
        return payload
