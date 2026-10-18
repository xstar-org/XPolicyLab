"""Opt-in capture of genuine frames needed by VPP2's frozen history schedule."""
import os

NATIVE_STEP_FIELD = "_vpp2_history_native_step"
PROTOCOL_VERSION = 1


def native_step(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Native step must be a nonnegative integer")
    return value


def needs_frame(step):
    step = native_step(step)
    return any((step + offset * 25) % 24 == 0 for offset in range(8))


def sampling_enabled():
    return os.environ.get("VPP2_NATIVE_FRAME_SAMPLING", "0").lower() in {"1", "true", "yes"}


def prepare_native_frames(env, client, request_actions, client_context):
    """Bind native-frame capture/guards once; the episode loop remains shared."""
    version = client.call(func_name="native_frame_protocol", obs=client_context)
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise ValueError("Server does not implement VPP2 native-frame protocol v1")

    def capture(env_ids):
        selected = [e for e in env_ids if needs_frame(int(env.take_action_cnt[e]))]
        omitted = [e for e in env_ids if e not in selected]
        if omitted and getattr(env, "physx_monitor_enabled", False):
            env._check_physx_broken_envs()
            env._check_endpose_finite(omitted)
        if not selected:
            return []
        observations = env.get_obs_batch(selected)
        if [int(o["env_idx"]) for o in observations] != selected:
            raise ValueError("Observation identities do not match the requested environments")
        return [dict(obs, **{NATIVE_STEP_FIELD: int(env.take_action_cnt[int(obs["env_idx"])])})
                for obs in observations]

    def request_native_actions(env_ids):
        if any(int(env.take_action_cnt[e]) % 24 for e in env_ids):
            raise ValueError("Native-frame inference must occur at a 24-action boundary")
        actions = request_actions(env_ids)
        if len(actions[0]) != 24:
            raise ValueError("Native-frame v1 requires exactly 24 executed actions per replan")
        return actions

    return capture, request_native_actions
