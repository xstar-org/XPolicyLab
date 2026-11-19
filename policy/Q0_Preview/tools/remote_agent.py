"""API-host supervisor: pull published observations over SSH, run local Agents.

SSH connects outward to the cluster login node; the controller allocation uses
the same private shared filesystem. No port forwarding or model endpoint relay.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import time
import zlib

from bridge_exchange import PROTOCOL, IDENTITY, OBSERVATION, atomic, read
from runtime import POLICY, config, create_run_directory
from processes import stop


def receive(rpc, timeout=90):
    until = time.monotonic()+timeout
    chunks, size = [], 0
    while time.monotonic() < until:
        if not select.select([rpc.stdout], [], [], max(0, until-time.monotonic()))[0]:
            break
        chunk = os.read(rpc.stdout.fileno(), 1024*1024)
        if not chunk:
            raise RuntimeError('SSH bridge disconnected; see transport.stderr')
        chunks.append(chunk)
        size += len(chunk)
        if size > 128*1024**2:
            raise ValueError('SSH response exceeds 128 MiB')
        if b'\n' in chunk:
            return json.loads(b''.join(chunks))
    raise TimeoutError('SSH exchange stalled before a complete response')


def import_observation(ep, req, files):
    if not OBSERVATION.fullmatch(req['observation_id']):
        raise ValueError('Invalid observation name')
    folder = ep/'observations'/req['observation_id']
    folder.mkdir(parents=True, exist_ok=True)
    for name, item in files.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid observation filename')
        content = base64.b64decode(item['data'], validate=True)
        if item.get('encoding') == 'zlib':
            expected = item.get('raw_bytes')
            if type(expected) is not int or not 0 <= expected <= 32*1024**2:
                raise ValueError('Invalid decompressed observation size')
            try:
                decoder = zlib.decompressobj()
                content = decoder.decompress(content, expected+1)
            except zlib.error as error:
                raise ValueError('Invalid compressed observation') from error
            if len(content) != expected or not decoder.eof or decoder.unused_data:
                raise ValueError('Compressed observation size mismatch')
        elif item.get('encoding') not in (None, 'base64'):
            raise ValueError('Unknown observation encoding')
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('Transferred observation hash mismatch')
        path = folder/name
        if path.exists() and path.read_bytes() != content:
            raise ValueError('Refusing to replace an immutable observation')
        path.write_bytes(content)
    manifest = read(folder/'manifest.json')
    if any(manifest.get(k) != req[k] for k in (*IDENTITY, 'task', 'kind')):
        raise ValueError('Transferred observation attribution mismatch')
    for item in manifest['files'].values():
        name = item['name']
        if Path(name).name != name or hashlib.sha256((folder/name).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Transferred manifest integrity mismatch')
    # READY is always committed last; Agents can never read a partial observation.
    atomic(ep/'READY.json', req)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssh-config', type=Path, required=True, help='Private SSH config with host/key; never API keys')
    parser.add_argument('--ssh-host', required=True)
    parser.add_argument('--remote-root', required=True, help='Controller service run root, not allocation root')
    parser.add_argument('--remote-code', required=True, help='Controller-side tools/bridge_exchange.py')
    parser.add_argument('--root', required=True, help='New local Agent archive root')
    parser.add_argument('--config', required=True)
    parser.add_argument('--python-prefix', default=sys.prefix)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--two-group', action='store_true')
    args = parser.parse_args()
    cfg = config(args.python_prefix, args.config)
    if not args.smoke and not os.environ.get(cfg['api_key_env']):
        parser.error('Missing API key in local Agent environment')
    root = create_run_directory(args.root)
    (root/'sessions').mkdir()
    atomic(root/'agent_config.json', cfg)
    remote_command = shlex.join(['python3', '-u', args.remote_code, '--root', args.remote_root])
    ssh_command = ['ssh', '-F', str(args.ssh_config), '-o', 'BatchMode=yes',
                   '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                   args.ssh_host, remote_command]
    ssh_env = os.environ.copy()
    ssh_env.pop(cfg['api_key_env'], None)
    children, known, sent = {}, {}, set()
    rpc = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (root/'transport.stderr').open('w') as log:
        connecting_at = time.time()
        rpc = subprocess.Popen(ssh_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log, env=ssh_env, start_new_session=True)
        try:
            hello = receive(rpc)
            if (not hello.get('hello') or hello.get('protocol') != PROTOCOL
                    or not connecting_at-2 <= hello['server_unix'] <= time.time()+2):
                raise ValueError('SSH handshake protocol/clock mismatch')
            atomic(root/'BRIDGE_STARTED.json', dict(remote_root=args.remote_root, protocol=PROTOCOL))
            while not (root/'STOP').exists():
                replies = []
                for name, req_id in known.items():
                    ep = root/'sessions'/name
                    req = read(ep/'READY.json', {})
                    reply = read(ep/'observations'/req.get('observation_id', '_')/'reply.json')
                    if reply and req_id not in sent:
                        if any(reply.get(k) != req.get(k) for k in IDENTITY):
                            raise ValueError('Local reply attribution mismatch')
                        replies.append(dict(episode=name, reply=reply))
                message = dict(protocol=PROTOCOL, client_unix=time.time(), known=known, replies=replies)
                rpc.stdin.write((json.dumps(message)+'\n').encode())
                rpc.stdin.flush()
                result = receive(rpc)
                if (result.get('protocol') != PROTOCOL
                        or not message['client_unix']-2 <= result['server_unix'] <= time.time()+2):
                    raise ValueError('Bridge protocol/clock/latency mismatch')
                for verdict in result['replies']:
                    if verdict.get('accepted'):
                        sent.add(verdict['request_id'])
                        with (root/'replies.jsonl').open('a') as file:
                            file.write(json.dumps(verdict)+'\n')
                for name, status in result['episodes'].items():
                    if not name or Path(name).name != name or name in ('.', '..'):
                        raise ValueError('Invalid episode name')
                    ep = root/'sessions'/name
                    ep.mkdir(exist_ok=True)
                    meta = status['episode']
                    if meta.get('task_information_mode', 'original') != cfg['task_information_mode']:
                        raise ValueError('Controller/Agent information modes disagree')
                    if meta['run_id'] != name:
                        raise ValueError('Controller/Agent episode identity mismatch')
                    atomic(ep/'episode.json', meta)
                    if status['end'] is not None:
                        atomic(ep/'END.json', status['end'])
                    if status['stop']:
                        (ep/'STOP').touch()
                    if 'observation' in status:
                        import_observation(ep, status['ready'], status['observation'])
                        known[name] = status['ready']['request_id']
                    if name not in children and not status['end'] and not status['stop']:
                        cmd = [str(Path(cfg['python_prefix'])/'bin/python'), '-u', str(POLICY/'tools/agent_entry.py'),
                               'smoke_agent' if args.smoke else 'runner', '--run-dir', str(ep)]
                        if not args.smoke:
                            cmd += ['--config', str(root/'agent_config.json')]
                            if cfg['provider'] == 'openai':
                                cmd += ['--enable-openai']
                        elif args.two_group:
                            cmd += ['--two-group']
                        with (ep/'agent.log').open('w') as agent_log:
                            children[name] = subprocess.Popen(cmd, stdout=agent_log,
                                stderr=subprocess.STDOUT, start_new_session=True)
                        print(f'AGENT_STARTED {name}', flush=True)
                    if name in children and children[name].poll() is not None and not status['end'] and not status['stop']:
                        raise RuntimeError(f'Agent exited before episode end: {name}; see agent.log')
                    for error_path in (ep/'errors').glob('*.json'):
                        if read(error_path, {}).get('infrastructure'):
                            raise RuntimeError(f'Agent infrastructure failure: {name}; see {error_path.name}')
                if result.get('service_stopped'):
                    break
                atomic(root/'BRIDGE_HEALTH.json', dict(last_exchange_unix=time.time(),
                     episodes=len(children), accepted_replies=len(sent)))
                time.sleep(.5)
        except BaseException as error:
            atomic(root/'BRIDGE_ERROR.json', dict(error=repr(error), unix=time.time()))
            raise
        finally:
            if rpc.poll() is None:
                try:
                    rpc.stdin.write((json.dumps(dict(protocol=PROTOCOL, client_unix=time.time(),
                        known=known, stop_episodes=list(children)))+'\n').encode())
                    rpc.stdin.flush()
                except (OSError, BrokenPipeError):
                    pass
            for name, child in children.items():
                (root/'sessions'/name/'STOP').touch()
                stop(child)
            stop(rpc)


if __name__ == '__main__':
    main()
