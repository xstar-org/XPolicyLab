"""OpenWAM synchronous RoboDojo rollout."""

import os

from .fixed_history import FixedHistory


def _uses_fixed_history(env):
    cfg = getattr(env, "deploy_cfg", {})
    value = os.environ.get("OPENWAM_FIXED_HISTORY", cfg.get("fixed_history", False))
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def eval_one_episode(TASK_ENV, model_client):
    if _uses_fixed_history(TASK_ENV):
        return eval_one_episode_batch(TASK_ENV, model_client)
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        if not len(actions):
            raise ValueError("OpenWAM returned an empty action chunk")
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    history = None
    if _uses_fixed_history(TASK_ENV):
        history = FixedHistory(**model_client.call(func_name="get_fixed_history_spec"))
    else:
        model_client.call(func_name="reset")
    step = 0
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        observations = TASK_ENV.get_obs_batch(env_idx_list)
        if history is None:
            model_client.call(func_name="update_obs_batch", obs=observations)
            actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        else:
            actions = model_client.call(func_name="infer_fixed_history_batch", obs=history.request(step, observations))
        if len(actions) != len(env_idx_list) or not actions or not len(actions[0]):
            raise ValueError("OpenWAM returned an empty or misaligned action batch")
        chunk_size = len(actions[0])
        if any(len(chunk) != chunk_size for chunk in actions):
            raise ValueError("OpenWAM returned unequal action chunk lengths")
        if history is not None:
            history.set_horizon(chunk_size)
        for action_idx in range(chunk_size):
            current_action_list = [chunk[action_idx] for chunk in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            step += 1
            if TASK_ENV.is_episode_end():
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[i] for i in active]
            env_idx_list = [env_idx_list[i] for i in active]
            if history is not None and history.capture_between_replans(step):
                history.record(step, TASK_ENV.get_obs_batch(env_idx_list))
