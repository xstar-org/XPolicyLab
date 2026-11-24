"""Collect the RGB observations used by OpenWAM's fixed-history checkpoint."""

class FixedHistoryClient:
    """Collect local control-step observations before sending each inference request."""

    def __init__(self, spec):
        self.history = FixedHistory(**spec)
        self.step = 0
        self.observations = {}

    def update_obs_batch(self, observations):
        if isinstance(observations, dict):
            observations = [observations]
        self.observations = {}
        for index, obs in enumerate(observations):
            env_idx = int(obs.get("env_idx", index))
            self.observations[env_idx] = {**obs, "env_idx": env_idx}

    def after_action(self, read_observations):
        self.step += 1
        if self.history.capture_between_replans(self.step):
            observations = read_observations()
            if isinstance(observations, dict):
                observations = [observations]
            self.history.record(self.step, observations)

    def get_actions(self, call_model, env_ids=None):
        observations = self.observations.values() if env_ids is None else (
            self.observations[int(env_idx)] for env_idx in env_ids
        )
        payload = self.history.request(self.step, list(observations))
        actions = call_model("infer_fixed_history_batch", payload)
        if len(actions) and len(actions[0]):
            self.history.set_horizon(len(actions[0]))
        return actions


class FixedHistory:
    def __init__(self, stride, clip_frames):
        if stride <= 0 or clip_frames < 2:
            raise ValueError("Fixed history requires a positive stride and an episode anchor")
        self.stride = stride
        self.offsets = range(clip_frames - 2, -1, -1)
        self.horizon = None
        self.frames = {}

    def set_horizon(self, horizon):
        if horizon <= 0 or (self.horizon is not None and self.horizon != horizon):
            raise ValueError("Fixed-history rollout requires a constant positive action chunk length")
        self.horizon = horizon

    def capture_between_replans(self, step):
        if self.horizon is None:
            raise RuntimeError("The action chunk length is unknown")
        if step % self.horizon == 0:
            return False
        return any((step + offset * self.stride) % self.horizon == 0 for offset in self.offsets)

    def record(self, step, observations):
        for obs in observations:
            cameras = {
                name: {"color": camera["color"]} for name, camera in obs["vision"].items()
                if "color" in camera
            }
            self.frames[(int(obs["env_idx"]), step)] = cameras

    def request(self, step, observations):
        self.record(step, observations)
        wanted = {0, *(max(0, step - offset * self.stride) for offset in self.offsets)}
        # The wire decoder accepts a flat list of observations. Each image is
        # sent once even when the clip repeats the first frame near the reset.
        payload = []
        for obs in observations:
            env_idx = int(obs["env_idx"])
            for frame in sorted(wanted - {step}):
                payload.append({
                    "env_idx": env_idx, "frame_idx": frame, "vision": self.frames[(env_idx, frame)],
                })
            current = {key: obs[key] for key in ("state", "instruction", "task_instruction") if key in obs}
            payload.append({
                **current, "env_idx": env_idx, "frame_idx": step,
                "vision": self.frames[(env_idx, step)], "history_current": True,
            })
        active = {int(obs["env_idx"]) for obs in observations}
        oldest = step - max(self.offsets) * self.stride
        self.frames = {
            (env_idx, frame): vision for (env_idx, frame), vision in self.frames.items()
            if env_idx in active and (frame == 0 or frame >= oldest)
        }
        return payload
