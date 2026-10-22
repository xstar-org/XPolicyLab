import os


def _chunk_boundary_obs_only() -> bool:
    """OpenWAM replans on full chunks: only the chunk-boundary observation is
    consumed (update_obs_batch overwrites the previous one and get_action_batch
    reads the latest frame). With OPENWAM_CHUNK_BOUNDARY_OBS=1 the per-step
    update_obs_batch calls inside a chunk are skipped; the loop head keeps
    capturing one observation per chunk. Video streams then record one frame
    per chunk instead of one per action step."""
    return os.environ.get("OPENWAM_CHUNK_BOUNDARY_OBS", "0") == "1"


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():  # Check whether the episode ends
        obs = TASK_ENV.get_obs()  # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation

        actions = model_client.call(func_name="get_action")  # Get Action chunk
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if action_idx != len(actions) - 1:
                obs = TASK_ENV.get_obs()  # Get Observation
                model_client.call(func_name="update_obs", obs=obs)


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():  # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)  # Get Batch Observation

        model_client.call(func_name="update_obs_batch", obs=obs_list)  # Update Observation
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)  # Get Action chunk

        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]

            TASK_ENV.take_action_batch(current_action_list, env_idx_list)

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]

            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            if _chunk_boundary_obs_only():
                # Keep the official per-step video cadence without uploading
                # intermediate observations the policy never reads.
                stream = getattr(TASK_ENV, "stream_video_batch", None)
                if stream is not None:
                    stream(env_idx_list)
                continue
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
