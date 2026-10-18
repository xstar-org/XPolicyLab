"""Opt-in capture of genuine frames needed by VPP2's frozen history schedule."""
import os

NATIVE_STEP_FIELD = "_vpp2_history_native_step"


def native_step(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Native step must be a nonnegative integer")
    return value


def needs_frame(step):
    step = native_step(step)
    return any((step + offset * 25) % 24 == 0 for offset in range(8))


def sampling_enabled():
    return os.environ.get("VPP2_NATIVE_FRAME_SAMPLING", "0").lower() in {"1", "true", "yes"}


def collect_native_frames(env, env_ids, *, replan):
    """Capture required genuine frames and retain checks for skipped reads."""
    if replan and any(int(env.take_action_cnt[e]) % 24 for e in env_ids):
        raise ValueError("Native-frame inference must occur at a 24-action boundary")
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
