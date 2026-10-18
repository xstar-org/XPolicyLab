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
