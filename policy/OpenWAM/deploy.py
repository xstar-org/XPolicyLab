"""OpenWAM sync inference consumes only the observation at each replan."""


def eval_one_episode(TASK_ENV, model_client):
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
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        if len(actions) != len(env_idx_list) or not actions or not len(actions[0]):
            raise ValueError("OpenWAM returned an empty or misaligned action batch")
        chunk_size = len(actions[0])
        if any(len(chunk) != chunk_size for chunk in actions):
            raise ValueError("OpenWAM returned unequal action chunk lengths")
        for action_idx in range(chunk_size):
            current_action_list = [chunk[action_idx] for chunk in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            if TASK_ENV.is_episode_end():
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[i] for i in active]
            env_idx_list = [env_idx_list[i] for i in active]
