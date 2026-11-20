"""Trusted deployment helper: archive complete native results, then stop a service.

Never exposed to the evaluated agent. Does not cancel the Slurm allocation.
"""
import argparse
import json
from pathlib import Path
import time


def completed_result(path, minimum):
    try:
        result = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    count = result.get('eval_time', 0)
    details = result.get('details', {})
    if type(count) is not int or count < minimum or len(details) != count:
        return None
    if any(type(item.get('success')) is not bool or type(item.get('layout_id')) is not int
           for item in details.values()):
        raise ValueError('Native result has ambiguous episode identities/outcomes')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--service', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--minimum-completed', type=int, required=True)
    parser.add_argument('--validated-preflight', type=Path, required=True,
                        help='CLI validation certificate; API must keep running until this exists')
    parser.add_argument('--timeout', type=float, default=10800)
    args = parser.parse_args()
    if args.minimum_completed < 1 or args.timeout <= 0:
        parser.error('Positive episode boundary and timeout required')
    proof = json.loads(args.validated_preflight.read_text())
    if (proof.get('transport', {}).get('provider') != 'codex_cli'
            or proof.get('transport', {}).get('authentication') != 'chatgpt_login'
            or not proof.get('deployment', {}).get('sha256') or not proof.get('layouts')):
        raise ValueError('A completed Codex CLI preflight is required before stopping API evaluation')
    until = time.monotonic()+args.timeout
    while time.monotonic() < until:
        result = completed_result(args.result, args.minimum_completed)
        if result is not None:
            # Commit the untouched evaluator snapshot before stopping the service.
            record = dict(reason='switch_to_codex_cli_at_episode_boundary', unix=time.time(),
                          provider='openai_api', native_result=result,
                          cli_validation=proof,
                          source=str(args.result), excluded_from_cli_comparison=True)
            with args.archive.open('x') as file:
                json.dump(record, file, indent=2)
            (args.service/'STOP').touch()
            print('STOPPED_AFTER_COMPLETE_EPISODES '+str(result['eval_time']), flush=True)
            return
        if (args.service/'STOP').exists():
            raise RuntimeError('Service stopped before the requested episode boundary')
        time.sleep(.5)
    raise TimeoutError('No completed episode boundary before timeout')


if __name__ == '__main__':
    main()
