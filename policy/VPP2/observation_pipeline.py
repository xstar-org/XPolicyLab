"""Ordered observation uploads overlapping the current simulation action chunk."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import wraps
import os


def run_with_observation_pipeline(env, client, run):
    original = client.call
    pending = deque()
    # Only this worker may call the synchronous client until a flush completes.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='vpp2-obs-upload') as worker:
        def flush():
            while pending:
                pending.popleft().result()

        def call(func_name=None, obs=None, **kwargs):
            if func_name == 'update_obs_batch':
                # Camera views and state buffers may be reused by the next step.
                owned = deepcopy(obs)
                if len(pending) >= 2:
                    pending.popleft().result()
                pending.append(worker.submit(original, func_name=func_name, obs=owned, **kwargs))
                return None
            # No action, reset, or custom RPC may overtake an observation.
            flush()
            return original(func_name=func_name, obs=obs, **kwargs)

        client.call = call
        try:
            return run(TASK_ENV, model_client)
        finally:
            try:
                flush()
            finally:
                client.call = original


def overlap_observation_uploads(run):
    """Opt in for batch evaluation; synchronous mode remains the default."""
    @wraps(run)
    def wrapped(TASK_ENV, model_client):
        enabled = os.environ.get("VPP2_OBSERVATION_UPLOAD_OVERLAP", "0").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return run(TASK_ENV, model_client)
        return run_with_observation_pipeline(TASK_ENV, model_client, run)
    return wrapped
