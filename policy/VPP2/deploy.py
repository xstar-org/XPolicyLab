import os
import socket
import time
import uuid

from XPolicyLab.policy.VPP2.observation_pipeline import overlap_observation_uploads

_CLIENT_ID_FIELD = "_vpp2_client_id"
_NATIVE_STEP_FIELD = "_vpp2_native_step"
_CLIENT_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _with_client_id(obs, native_step=None):
    tagged = dict(obs)
    tagged[_CLIENT_ID_FIELD] = _CLIENT_ID
    if native_step is not None:
        tagged[_NATIVE_STEP_FIELD] = int(native_step)
    return tagged


def _history_frame_needed(native_step, replan_steps, history_stride, history_frames):
    """Return whether this exact native step is selected at a future replan.

    VPP2 replans every ``replan_steps`` and asks for the current frame plus
    ``history_frames - 1`` frames spaced by ``history_stride``.  Saving the
    union of those exact indices preserves the model condition while avoiding
    camera/RPC work for frames which can never be selected.
    """
    native_step = int(native_step)
    if native_step <= 0:
        return True
    max_offset = (int(history_frames) - 1) * int(history_stride)
    first_boundary = ((native_step + replan_steps - 1) // replan_steps) * replan_steps
    wanted_offsets = set(range(0, max_offset + 1, int(history_stride)))
    return any(
        boundary - native_step in wanted_offsets
        for boundary in range(first_boundary, native_step + max_offset + 1, replan_steps)
    )


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    model_client.call(func_name="reset_client", obs={_CLIENT_ID_FIELD: _CLIENT_ID})

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=_with_client_id(obs))
        actions = model_client.call(func_name="get_action", obs={_CLIENT_ID_FIELD: _CLIENT_ID})
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if action_idx != len(actions) - 1:
                obs = TASK_ENV.get_obs()
                model_client.call(func_name="update_obs", obs=_with_client_id(obs))


@overlap_observation_uploads
def eval_one_episode_batch(TASK_ENV, model_client):
    profile = {"obs": 0.0, "update": 0.0, "policy": 0.0, "step": 0.0}
    calls = {"obs": 0, "update": 0, "policy": 0, "step": 0}
    sparse = _truthy(os.environ.get("VPP2_EXACT_HISTORY_CAPTURE", "0"))
    history_stride = int(os.environ.get("VPP2_HISTORY_STRIDE", "25"))
    history_frames = int(os.environ.get("VPP2_HISTORY_FRAMES", "8"))
    native_step = 0
    started = time.perf_counter()
    model_client.call(func_name="reset")
    model_client.call(func_name="reset_client", obs={_CLIENT_ID_FIELD: _CLIENT_ID})

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        if not env_idx_list:
            break
        t0 = time.perf_counter()
        obs_list = [
            _with_client_id(obs, native_step if sparse else None)
            for obs in TASK_ENV.get_obs_batch(env_idx_list)
        ]
        profile["obs"] += time.perf_counter() - t0
        calls["obs"] += 1
        t0 = time.perf_counter()
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        profile["update"] += time.perf_counter() - t0
        calls["update"] += 1
        t0 = time.perf_counter()
        actions = model_client.call(
            func_name="get_action_batch",
            obs={"env_idx_list": env_idx_list, _CLIENT_ID_FIELD: _CLIENT_ID},
        )
        profile["policy"] += time.perf_counter() - t0
        calls["policy"] += 1
        chunk_size = min(len(env_actions) for env_actions in actions)

        for action_idx in range(chunk_size):
            t0 = time.perf_counter()
            TASK_ENV.take_action_batch(
                [env_actions[action_idx] for env_actions in actions], env_idx_list
            )
            profile["step"] += time.perf_counter() - t0
            calls["step"] += 1
            native_step += 1
            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[i] for i in active]
            env_idx_list = [env_idx_list[i] for i in active]
            if not env_idx_list:
                break
            if sparse and not _history_frame_needed(
                native_step, chunk_size, history_stride, history_frames
            ):
                continue
            t0 = time.perf_counter()
            obs_list = [
                _with_client_id(obs, native_step if sparse else None)
                for obs in TASK_ENV.get_obs_batch(env_idx_list)
            ]
            profile["obs"] += time.perf_counter() - t0
            calls["obs"] += 1
            t0 = time.perf_counter()
            model_client.call(func_name="update_obs_batch", obs=obs_list)
            profile["update"] += time.perf_counter() - t0
            calls["update"] += 1

    total = time.perf_counter() - started
    print(
        "[VPP2 batch profile] "
        f"mode={'exact-history' if sparse else 'dense'} total={total:.3f}s "
        f"native_steps={native_step} obs={profile['obs']:.3f}s/{calls['obs']} "
        f"update_rpc={profile['update']:.3f}s/{calls['update']} "
        f"policy_rpc={profile['policy']:.3f}s/{calls['policy']} "
        f"step_render={profile['step']:.3f}s/{calls['step']}",
        flush=True,
    )
