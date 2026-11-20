"""API-host supervisor: pull published observations over SSH, run local Agents.

SSH connects outward to the cluster login node; the controller allocation uses
the same private shared filesystem. No port forwarding or model endpoint relay.
"""
import argparse
import base64
import hashlib
import io
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


class SSHDisconnected(ConnectionError):
    """An SSH byte stream ended, rather than a protocol validation failure."""


class SSHTransportTimeout(TimeoutError):
    """A bounded pipe read/write expired, rather than local archive I/O."""


def receive(rpc, timeout=90):
    until = time.monotonic()+timeout
    chunks, size = [], 0
    while time.monotonic() < until:
        try:
            ready = select.select([rpc.stdout], [], [], max(0, until-time.monotonic()))[0]
            chunk = os.read(rpc.stdout.fileno(), 1024*1024) if ready else None
        except OSError as error:
            raise SSHDisconnected('SSH output pipe failed') from error
        if not ready:
            break
        if not chunk:
            raise SSHDisconnected('SSH bridge disconnected; see transport.stderr')
        chunks.append(chunk)
        size += len(chunk)
        if size > 128*1024**2:
            raise ValueError('SSH response exceeds 128 MiB')
        if b'\n' in chunk:
            return json.loads(b''.join(chunks))
    raise SSHTransportTimeout('SSH exchange stalled before a complete response')


def bridge_ssh_command(ssh_config, ssh_host, remote_command):
    # Agent observations use their own TCP stream. Management/log RPCs may use
    # a multiplexed connection without blocking a checkpoint image behind them.
    return ['ssh', '-F', str(ssh_config), '-o', 'BatchMode=yes',
            '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ControlPersist=no',
            '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
            ssh_host, remote_command]


def send_line(rpc, message, timeout=10):
    """Bound pipe writes as well as reads; never buffer an unbounded send."""
    content = (json.dumps(message)+'\n').encode()
    fd = rpc.stdin.fileno()
    until = time.monotonic()+timeout
    offset = 0
    os.set_blocking(fd, False)
    try:
        while offset < len(content):
            remaining = until-time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise SSHTransportTimeout('SSH exchange stalled while sending a request')
            try:
                written = os.write(fd, content[offset:])
            except BlockingIOError:
                continue
            if written == 0:
                raise SSHDisconnected('SSH input pipe closed while sending a request')
            offset += written
    except (SSHDisconnected, SSHTransportTimeout):
        raise
    except OSError as error:
        raise SSHDisconnected('SSH input pipe failed') from error
    finally:
        os.set_blocking(fd, True)


class SSHBridgeTransport:
    """Recover the transport only, retaining the caller's live Agent state.

    If an acknowledgement is lost, the identical attributed reply may be sent
    again. The controller's accept_reply is idempotent and rejects stale IDs.
    Validation errors deliberately bypass this recovery loop.
    """
    def __init__(self, command, env, log, root, remote_root, recovery_window_s=60):
        self.command, self.env, self.log = command, env, log
        self.root, self.remote_root = Path(root), remote_root
        self.recovery_window_s = recovery_window_s
        self.rpc = None
        self.connections = self.reconnects = 0

    def record(self, event, **fields):
        value = dict(event=event, unix=time.time(), connections=self.connections,
                     successful_reconnects=self.reconnects, **fields)
        with (self.root/'transport-reconnects.jsonl').open('a') as file:
            file.write(json.dumps(value)+'\n')
        atomic(self.root/'TRANSPORT_STATE.json', value)

    def close(self):
        if self.rpc is not None:
            stop(self.rpc, grace_seconds=2)
            self.rpc.stdin.close()
            self.rpc.stdout.close()
            self.rpc = None

    def connect(self, budget):
        self.connections += 1
        started = time.time()
        self.rpc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, env=self.env, start_new_session=True)
        hello = receive(self.rpc, timeout=min(20, budget))
        if (not hello.get('hello') or hello.get('protocol') != PROTOCOL
                or not started-2 <= hello['server_unix'] <= time.time()+2):
            raise ValueError('SSH handshake protocol/clock mismatch')
        initial = not (self.root/'BRIDGE_STARTED.json').exists()
        if initial:
            atomic(self.root/'BRIDGE_STARTED.json', dict(remote_root=self.remote_root, protocol=PROTOCOL))
        else:
            self.reconnects += 1
        self.record('connected' if initial else 'connection_restored')

    def exchange(self, message):
        recovery_until = None
        failure = None
        while True:
            budget = None if recovery_until is None else recovery_until-time.monotonic()
            if budget is not None and budget <= 0:
                self.record('recovery_exhausted', error=repr(failure))
                raise TimeoutError('SSH transport recovery window exhausted') from failure
            try:
                if self.rpc is None:
                    self.connect(20 if budget is None else budget)
                budget = None if recovery_until is None else recovery_until-time.monotonic()
                if budget is not None and budget <= 0:
                    raise SSHTransportTimeout('SSH recovery window elapsed during connection')
                attempt = dict(message, client_unix=time.time())
                send_line(self.rpc, attempt, timeout=10 if budget is None else min(10,budget))
                # A reconnect does not grant a fresh unbounded read budget.
                budget = None if recovery_until is None else recovery_until-time.monotonic()
                if budget is not None and budget <= 0:
                    raise SSHTransportTimeout('SSH recovery window elapsed during connection')
                result = receive(self.rpc, timeout=90 if budget is None else min(90,budget))
            except (SSHDisconnected, SSHTransportTimeout) as error:
                failure = error
                if recovery_until is None:
                    recovery_until = time.monotonic()+self.recovery_window_s
                self.record('transport_lost', error=repr(error))
                self.close()
                remaining = recovery_until-time.monotonic()
                if remaining > 0:
                    self.record('reconnect_attempt', remaining_seconds=remaining)
                    time.sleep(min(1,remaining))
                continue
            if (result.get('protocol') != PROTOCOL
                    or not attempt['client_unix']-2 <= result['server_unix'] <= time.time()+2):
                raise ValueError('Bridge protocol/clock/latency mismatch')
            if recovery_until is not None:
                self.record('exchange_recovered')
            return result

    def send_stop(self, known, episodes):
        if self.rpc is not None and self.rpc.poll() is None:
            try:
                send_line(self.rpc, dict(protocol=PROTOCOL, client_unix=time.time(),
                    known=known, stop_episodes=episodes), timeout=2)
            except (OSError, TimeoutError):
                pass


def restore_rgb_npy(jpeg):
    # Use the benchmark's marker-aware decoder, exactly as Episode.publish does.
    shared = Path(os.environ.get('Q0_PREVIEW_XPL_ROOT', str(POLICY.parents[2])))
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))
    import cv2
    cv2.setNumThreads(1)
    import numpy as np
    from XPolicyLab.utils.process_data import decode_image_bit
    buffer = io.BytesIO()
    np.save(buffer, decode_image_bit(jpeg), allow_pickle=False)
    return buffer.getvalue()


def import_observation(ep, req, files):
    if not OBSERVATION.fullmatch(req['observation_id']):
        raise ValueError('Invalid observation name')
    folder = ep/'observations'/req['observation_id']
    folder.mkdir(parents=True, exist_ok=True)
    # JPEG sources must be verified and written before deriving RGB arrays.
    for name, item in sorted(files.items(), key=lambda pair: pair[1].get('encoding') == 'jpeg_rgb_npy'):
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid observation filename')
        if item.get('encoding') == 'jpeg_rgb_npy':
            if not name.endswith('.rgb.npy') or item.get('source') != name[:-8]+'.jpg':
                raise ValueError('Invalid RGB array source')
            content = restore_rgb_npy((folder/item['source']).read_bytes())
            if len(content) != item.get('raw_bytes') or len(content) > 32*1024**2:
                raise ValueError('Reconstructed RGB array size mismatch')
        else:
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
        elif item.get('encoding') not in (None, 'base64', 'jpeg_rgb_npy'):
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
    if not args.smoke and cfg['provider'] != 'codex_cli' and not os.environ.get(cfg['api_key_env']):
        parser.error('Missing API key in local Agent environment')
    root = create_run_directory(args.root)
    (root/'sessions').mkdir()
    atomic(root/'agent_config.json', cfg)
    remote_command = shlex.join(['python3', '-u', args.remote_code, '--root', args.remote_root])
    ssh_command = bridge_ssh_command(args.ssh_config, args.ssh_host, remote_command)
    ssh_env = os.environ.copy()
    ssh_env.pop(cfg['api_key_env'], None)
    children, known, sent = {}, {}, set()

    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (root/'transport.stderr').open('w') as log:
        transport = SSHBridgeTransport(ssh_command, ssh_env, log, root, args.remote_root)
        try:
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
                message = dict(protocol=PROTOCOL, client_unix=time.time(), known=known, replies=replies,
                               compact_images=True)
                result = transport.exchange(message)
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
            transport.send_stop(known, list(children))
            for name, child in children.items():
                (root/'sessions'/name/'STOP').touch()
                stop(child)
            transport.close()


if __name__ == '__main__':
    main()
