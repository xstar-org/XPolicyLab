import os
import socket
import uuid

from XPolicyLab.policy.VPP2.native_frames import (
    NATIVE_STEP_FIELD, PROTOCOL_VERSION, needs_frame, sampling_enabled,
)

# VPP2 keeps per-episode observation history on the policy server.  Every
# executed action records an observation in dense mode; native-frame mode
# records only the required genuine frames. Each call carries this process's
# client id so one server can serve several
# simulator clients without mixing their histories.  A single-client official
# evaluation behaves exactly as without the id.
_CLIENT_ID_FIELD = "_vpp2_client_id"
_CLIENT_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _with_client_id(obs):
    tagged = dict(obs)
    tagged[_CLIENT_ID_FIELD] = _CLIENT_ID
    return tagged


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    model_client.call(func_name="reset_client", obs={_CLIENT_ID_FIELD: _CLIENT_ID})

    while not TASK_ENV.is_episode_end():  # Check whether the episode ends
        obs = TASK_ENV.get_obs()  # Get Observation
        model_client.call(func_name="update_obs", obs=_with_client_id(obs))

        actions = model_client.call(func_name="get_action", obs={_CLIENT_ID_FIELD: _CLIENT_ID})
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=_with_client_id(obs))


def eval_one_episode_batch(TASK_ENV, model_client):
    """Execute independent environments; dense is the default."""
    sample_native = sampling_enabled()
    if sample_native:
        version = model_client.call(func_name="native_frame_protocol", obs={_CLIENT_ID_FIELD: _CLIENT_ID})
        if type(version) is not int or version != PROTOCOL_VERSION:
            raise ValueError("Server does not implement VPP2 native-frame protocol v1")

    def upload_observations(env_ids):
        selected = env_ids
        if sample_native:
            selected = [e for e in env_ids if needs_frame(int(TASK_ENV.take_action_cnt[e]))]
            omitted = [e for e in env_ids if e not in selected]
            # Skipping camera reads must retain the original safety checks.
            if omitted and getattr(TASK_ENV, "physx_monitor_enabled", False):
                TASK_ENV._check_physx_broken_envs()
                TASK_ENV._check_endpose_finite(omitted)
        if not selected:
            return
        observations = TASK_ENV.get_obs_batch(selected)
        if sample_native and [int(o["env_idx"]) for o in observations] != selected:
            raise ValueError("Observation identities do not match the requested environments")
        tagged = [_with_client_id(obs) for obs in observations]
        if sample_native:
            for obs in tagged:
                obs[NATIVE_STEP_FIELD] = int(TASK_ENV.take_action_cnt[int(obs["env_idx"])])
        model_client.call(func_name="update_obs_batch", obs=tagged)

    model_client.call(func_name="reset")
    model_client.call(func_name="reset_client", obs={_CLIENT_ID_FIELD: _CLIENT_ID})
    while not TASK_ENV.is_episode_end():
        env_ids = TASK_ENV.get_running_env_idx_list()
        if not env_ids:
            break
        if sample_native and any(int(TASK_ENV.take_action_cnt[e]) % 24 for e in env_ids):
            raise ValueError("Native-frame inference must occur at a 24-action boundary")
        upload_observations(env_ids)
        actions = model_client.call(func_name="get_action_batch", obs={
            "env_idx_list": env_ids, _CLIENT_ID_FIELD: _CLIENT_ID,
        })
        if len(actions) != len(env_ids) or any(not len(chunk) for chunk in actions):
            raise ValueError("Policy must return a nonempty action chunk for each active environment")
        lengths = {len(chunk) for chunk in actions}
        if len(lengths) != 1:
            raise ValueError("Policy action chunks must have equal length")
        chunk_size = len(actions[0])
        if sample_native and chunk_size != 24:
            raise ValueError("Native-frame v1 requires exactly 24 executed actions per replan")
        for step in range(chunk_size):
            TASK_ENV.take_action_batch([chunk[step] for chunk in actions], env_ids)
            if TASK_ENV.is_episode_end() or step + 1 == chunk_size:
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            keep = [i for i, env_id in enumerate(env_ids) if env_id in running]
            actions = [actions[i] for i in keep]
            env_ids = [env_ids[i] for i in keep]
            if not env_ids:
                break
            upload_observations(env_ids)
