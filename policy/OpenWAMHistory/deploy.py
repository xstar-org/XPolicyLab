"""Collect observed 5 Hz candidates even while an action chunk is executing."""


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    step = 0

    def observe(slots):
        observations = TASK_ENV.get_obs_batch(slots)
        for obs in observations:
            obs["frame_id"] = step
        model_client.call(func_name="update_obs_batch", obs=observations)

    while not TASK_ENV.is_episode_end():
        slots = list(TASK_ENV.get_running_env_idx_list())
        observe(slots)
        chunks = model_client.call(func_name="get_action_batch", obs=slots)
        if len(chunks) != len(slots) or not chunks or not chunks[0]:
            raise ValueError("RoboDojo returned empty or misaligned action chunks")
        horizon = len(chunks[0])
        if any(len(chunk) != horizon for chunk in chunks):
            raise ValueError("Mismatched action horizons")
        for offset in range(horizon):
            TASK_ENV.take_action_batch([chunk[offset] for chunk in chunks], slots)
            step += 1
            if TASK_ENV.is_episode_end():
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [(slot, chunk) for slot, chunk in zip(slots, chunks) if slot in running]
            slots = [slot for slot, _ in active]
            chunks = [chunk for _, chunk in active]
            if step % 5 == 0 and offset + 1 < horizon:
                observe(slots)
