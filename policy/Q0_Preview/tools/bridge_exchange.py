"""Controller-side stdio protocol, reachable only through an authenticated SSH session.

Exposes published observations and accepts attributed replies. No API credentials,
task source, evaluator scores, arbitrary paths, or simulator internals cross here.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import uuid

PROTOCOL = 1
IDENTITY = ('run_id', 'request_id', 'observation_id')
EPISODE = re.compile(r'[a-zA-Z0-9_-]{1,128}')
OBSERVATION = re.compile(r'obs_\d{5,}')


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def episode_path(root, name):
    if not isinstance(name, str) or not EPISODE.fullmatch(name):
        raise ValueError('Invalid episode name')
    path = root/'sessions'/name
    if path.is_symlink() or not path.resolve().is_relative_to((root/'sessions').resolve()):
        raise ValueError('Episode outside sessions')
    return path


def snapshot(ep, req):
    if not OBSERVATION.fullmatch(req['observation_id']):
        raise ValueError('Invalid observation name')
    folder = ep/'observations'/req['observation_id']
    manifest = read(folder/'manifest.json')
    if any(manifest.get(k) != req[k] for k in (*IDENTITY, 'task', 'kind')):
        raise ValueError('Manifest attribution mismatch')
    allowed = {'observation.json'} | {camera+suffix for camera in
               ('cam_head', 'cam_left_wrist', 'cam_right_wrist') for suffix in ('.jpg', '.rgb.npy')}
    if {item['name'] for item in manifest['files'].values()} != allowed:
        raise ValueError('Unexpected observation files')
    result = {}
    for name in sorted(allowed | {'manifest.json', 'request.json'}):
        path = folder/name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Observation outside published folder')
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if len(content) > 32*1024**2:
            raise ValueError('Oversized observation file')
        expected = next((item['sha256'] for item in manifest['files'].values() if item['name'] == name), digest)
        if digest != expected:
            raise ValueError('Observation hash mismatch')
        result[name] = dict(sha256=digest, data=base64.b64encode(content).decode('ascii'))
    return result


def accept_reply(root, item, now):
    ep = episode_path(root, item['episode'])
    reply = item['reply']
    req = read(ep/'READY.json', {})
    if (not req or any(reply.get(k) != req.get(k) for k in IDENTITY)
            or now >= req['deadline'] or (ep/'END.json').exists() or (ep/'STOP').exists()):
        return dict(episode=ep.name, accepted=False, reason='stale_or_ended')
    if not OBSERVATION.fullmatch(req['observation_id']):
        raise ValueError('Invalid observation name')
    path = ep/'observations'/req['observation_id']/'reply.json'
    previous = read(path)
    if previous is not None and previous != reply:
        raise ValueError('Refusing conflicting reply')
    if previous is None:
        atomic(path, reply)
    return dict(episode=ep.name, request_id=req['request_id'], accepted=True)


def exchange(root, message):
    if message.get('protocol') != PROTOCOL:
        raise ValueError('Unsupported bridge protocol')
    now = time.time()
    if abs(now-message['client_unix']) > 2:
        raise ValueError('Host clocks differ by more than two seconds')
    output = dict(protocol=PROTOCOL, server_unix=now, episodes={}, replies=[])
    if not root.exists():
        output['waiting_for_service'] = True
        return output
    for item in message.get('replies', []):
        output['replies'].append(accept_reply(root, item, now))
    for name in message.get('stop_episodes', []):
        ep = episode_path(root, name)
        if (ep/'episode.json').exists():
            (ep/'STOP').touch()
    atomic(root/'EXTERNAL_AGENT.json', dict(last_exchange_unix=now, protocol=PROTOCOL))
    known = message.get('known', {})
    for ep in sorted((root/'sessions').glob('*')):
        if not ep.is_dir() or not (ep/'episode.json').exists():
            continue
        episode_path(root, ep.name)
        meta = read(ep/'episode.json')
        if meta['run_id'] != ep.name:
            raise ValueError('Episode identity mismatch')
        req = read(ep/'READY.json')
        status = dict(episode=meta, ready=req, end=read(ep/'END.json'), stop=(ep/'STOP').exists())
        if (req and not status['end'] and not status['stop']
                and known.get(ep.name) != req['request_id']):
            status['observation'] = snapshot(ep, req)
        output['episodes'][ep.name] = status
    output['service_stopped'] = (root/'STOP').exists()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps(dict(protocol=PROTOCOL, hello=True, server_unix=time.time())), flush=True)
    for line in sys.stdin:
        response = exchange(root, json.loads(line))
        print(json.dumps(response, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
