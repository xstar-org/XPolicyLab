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
    enabled = os.environ.get("VPP2_NATIVE_FRAME_SAMPLING", "0").lower() in {"1", "true", "yes"}
    video = os.environ.get("ROBODOJO_SAVE_EVAL_VIDEOS", "true").lower() not in {"0", "false", "no"}
    return enabled and not video


def evaluate_native_frames(env, client, client_id_field, client_id, tag_observation):
    version = client.call(func_name="native_frame_protocol", obs={client_id_field: client_id})
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise ValueError("Server does not implement VPP2 native-frame protocol v1")
    client.call(func_name="reset")
    client.call(func_name="reset_client", obs={client_id_field: client_id})

    def upload(ids):
        selected = [e for e in ids if needs_frame(int(env.take_action_cnt[e]))]
        omitted = [e for e in ids if e not in selected]
        # Skipping a camera read must not suppress the original safety checks.
        if omitted and getattr(env, "physx_monitor_enabled", False):
            env._check_physx_broken_envs()
            env._check_endpose_finite(omitted)
        if selected:
            observations = env.get_obs_batch(selected)
            if [int(o["env_idx"]) for o in observations] != selected:
                raise ValueError("Observation identities do not match the requested environments")
            tagged = []
            for obs in observations:
                obs = dict(obs)
                obs[NATIVE_STEP_FIELD] = int(env.take_action_cnt[int(obs["env_idx"])])
                tagged.append(tag_observation(obs))
            client.call(func_name="update_obs_batch", obs=tagged)

    while not env.is_episode_end():
        ids = env.get_running_env_idx_list()
        if not ids:
            break
        if any(int(env.take_action_cnt[e]) % 24 for e in ids):
            raise ValueError("Native-frame inference must occur at a 24-action boundary")
        upload(ids)
        actions = client.call(func_name="get_action_batch", obs={
            "env_idx_list": ids, client_id_field: client_id,
        })
        if len(actions) != len(ids) or any(len(chunk) != 24 for chunk in actions):
            raise ValueError("Native-frame v1 requires exactly 24 executed actions per replan")
        for tick in range(24):
            env.take_action_batch([chunk[tick] for chunk in actions], ids)
            if env.is_episode_end() or tick == 23:
                break
            running = set(env.get_running_env_idx_list())
            keep = [i for i, e in enumerate(ids) if e in running]
            ids = [ids[i] for i in keep]
            actions = [actions[i] for i in keep]
            if not ids:
                break
            upload(ids)
