import os
import socket
import uuid

from XPolicyLab.policy.VPP2.native_frames import (
    collect_native_frames, sampling_enabled,
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
    """Share one observation/action loop; dense capture is the default."""
    sample_native = sampling_enabled()
    model_client.call(func_name="reset")
    model_client.call(func_name="reset_client", obs={_CLIENT_ID_FIELD: _CLIENT_ID})
    while not TASK_ENV.is_episode_end():
        env_ids = TASK_ENV.get_running_env_idx_list()
        if not env_ids:
            break
        actions = None
        step = 0
        while True:
            if sample_native:
                observations = collect_native_frames(TASK_ENV, env_ids, replan=actions is None)
            else:
                observations = TASK_ENV.get_obs_batch(env_ids)
            if observations:
                model_client.call(func_name="update_obs_batch", obs=[
                    _with_client_id(obs) for obs in observations
                ])

            if actions is None:
                actions = model_client.call(func_name="get_action_batch", obs={
                    "env_idx_list": env_ids, _CLIENT_ID_FIELD: _CLIENT_ID,
                })
                if len(actions) != len(env_ids) or any(not len(chunk) for chunk in actions):
                    raise ValueError("Policy must return a nonempty action chunk for each active environment")
                if len({len(chunk) for chunk in actions}) != 1:
                    raise ValueError("Policy action chunks must have equal length")
                if sample_native and len(actions[0]) != 24:
                    raise ValueError("Native-frame v1 requires exactly 24 executed actions per replan")

            TASK_ENV.take_action_batch([chunk[step] for chunk in actions], env_ids)
            if TASK_ENV.is_episode_end() or step + 1 == len(actions[0]):
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            keep = [i for i, env_id in enumerate(env_ids) if env_id in running]
            actions = [actions[i] for i in keep]
            env_ids = [env_ids[i] for i in keep]
            if not env_ids:
                break
            step += 1
