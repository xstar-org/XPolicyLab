"""Cross-host transport preserves publication order, bytes, identity and deadlines."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/tools'
sys.path.insert(0, str(TOOLS))
import bridge_exchange as bridge
from remote_agent import import_observation, receive, SSHBridgeTransport, bridge_ssh_command


class BridgeTests(unittest.TestCase):
    def test_observation_transport_never_uses_the_management_master(self):
        command = bridge_ssh_command('/private/config', 'cluster', 'python bridge.py')
        for option in ('ControlMaster=no','ControlPath=none','ControlPersist=no',
                       'ConnectTimeout=15','ServerAliveInterval=10','ServerAliveCountMax=3'):
            self.assertIn(option, command)

    def test_lost_reply_ack_reconnects_without_duplicate_publication_or_caller_state_change(self):
        script = '''import json,os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import bridge_exchange as b
root=Path(sys.argv[2]);marker=root/'ack-dropped.json'
print(json.dumps(dict(hello=True,protocol=b.PROTOCOL,server_unix=time.time())),flush=True)
for line in sys.stdin:
 message=json.loads(line);answer=b.exchange(root,message)
 if not marker.exists():
  path=root/'sessions/episode_1/observations/obs_00000/reply.json'
  marker.write_text(json.dumps(dict(reply_mtime_ns=path.stat().st_mtime_ns)))
  os._exit(0)
 print(json.dumps(answer),flush=True)
'''
        reply = {k:self.req[k] for k in bridge.IDENTITY} | {'payload':{'groups':[]}}
        known = {'episode_1':'request_1'}
        sent = set()
        message = self.message(replies=[dict(episode='episode_1',reply=reply)])
        message['known'] = known
        before = json.dumps(message,sort_keys=True)
        local = Path(self.temp.name)/'local-transport'
        local.mkdir()
        with (local/'transport.stderr').open('w') as log:
            transport = SSHBridgeTransport([sys.executable,'-u','-c',script,str(TOOLS),str(self.root)],
                {},log,local,str(self.root),recovery_window_s=5)
            try:
                result = transport.exchange(message)
                self.assertTrue(result['replies'][0]['accepted'])
                self.assertEqual(transport.reconnects,1)
            finally:
                transport.close()
        self.assertEqual(json.dumps(message,sort_keys=True),before)
        self.assertEqual(known, {'episode_1':'request_1'})
        self.assertEqual(sent,set())
        original = json.loads((self.root/'ack-dropped.json').read_text())['reply_mtime_ns']
        self.assertEqual((self.folder/'reply.json').stat().st_mtime_ns,original)
        self.assertEqual(json.loads((self.folder/'reply.json').read_text()),reply)
        events = [json.loads(line)['event'] for line in (local/'transport-reconnects.jsonl').read_text().splitlines()]
        self.assertIn('transport_lost',events)
        self.assertIn('connection_restored',events)
        self.assertEqual(events[-1],'exchange_recovered')

    def test_protocol_json_and_clock_failures_never_reconnect(self):
        script = '''import json,sys,time
mode=sys.argv[1]
print(json.dumps(dict(hello=True,protocol=1,server_unix=time.time())),flush=True)
sys.stdin.readline()
if mode=='json': print('invalid-json',flush=True)
else: print(json.dumps(dict(protocol=2 if mode=='protocol' else 1,server_unix=0 if mode=='clock' else time.time())),flush=True)
'''
        for mode in ('protocol','json','clock'):
            with self.subTest(mode=mode):
                local = Path(self.temp.name)/('invalid-'+mode)
                local.mkdir()
                with (local/'transport.stderr').open('w') as log:
                    transport = SSHBridgeTransport([sys.executable,'-u','-c',script,mode],{},log,local,'unused')
                    try:
                        with self.assertRaises(ValueError):
                            transport.exchange(self.message())
                        self.assertEqual(transport.connections,1)
                    finally:
                        transport.close()
                events = [json.loads(line)['event'] for line in (local/'transport-reconnects.jsonl').read_text().splitlines()]
                self.assertNotIn('reconnect_attempt',events)

    def test_repeated_transport_failure_has_a_bounded_recovery_window(self):
        script = '''import json,sys,time
print(json.dumps(dict(hello=True,protocol=1,server_unix=time.time())),flush=True)
sys.stdin.readline()
'''
        local = Path(self.temp.name)/'exhausted'
        local.mkdir()
        began = time.monotonic()
        with (local/'transport.stderr').open('w') as log:
            transport = SSHBridgeTransport([sys.executable,'-u','-c',script],{},log,local,'unused',recovery_window_s=.05)
            try:
                with self.assertRaisesRegex(TimeoutError,'recovery window exhausted'):
                    transport.exchange(self.message())
            finally:
                transport.close()
        self.assertLess(time.monotonic()-began,2)
        self.assertEqual(json.loads((local/'TRANSPORT_STATE.json').read_text())['event'],'recovery_exhausted')

    def test_local_archive_or_spawn_errors_are_not_transport_retries(self):
        local = Path(self.temp.name)/'local-io-failure'
        local.mkdir()
        with (local/'transport.stderr').open('w') as log:
            transport = SSHBridgeTransport(['unused'],{},log,local,'unused')
            with patch.object(transport,'connect',side_effect=PermissionError('local archive denied')) as connect:
                with self.assertRaises(PermissionError):
                    transport.exchange(self.message())
                connect.assert_called_once()
        self.assertFalse((local/'transport-reconnects.jsonl').exists())

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('cv2'), 'image dependencies required')
    def test_compact_rgb_arrays_restore_exact_published_bytes(self):
        import numpy as np
        from remote_agent import restore_rgb_npy
        shared = Path(self.temp.name)/'shared'
        shared.mkdir()
        (shared/'XPolicyLab').symlink_to(TOOLS.parents[2], target_is_directory=True)
        with patch.dict('os.environ', {'Q0_PREVIEW_XPL_ROOT':str(shared)}):
            import sys
            sys.path.insert(0, str(shared))
            from XPolicyLab.utils.process_data import encode_image_bit
            for camera in ('cam_head','cam_left_wrist','cam_right_wrist'):
                jpeg = encode_image_bit(np.arange(64*48*3, dtype=np.uint8).reshape(48,64,3), quality=95)
                (self.folder/(camera+'.jpg')).write_bytes(jpeg)
                (self.folder/(camera+'.rgb.npy')).write_bytes(restore_rgb_npy(jpeg))
            manifest = json.loads((self.folder/'manifest.json').read_text())
            for item in manifest['files'].values():
                item['sha256'] = hashlib.sha256((self.folder/item['name']).read_bytes()).hexdigest()
            bridge.atomic(self.folder/'manifest.json', manifest)
            snapshot = bridge.snapshot(self.ep, self.req, compact_images=True)
            self.assertNotIn('data', snapshot['cam_head.rgb.npy'])
            local = Path(self.temp.name)/'compact'
            import_observation(local, self.req, snapshot)
            for name in snapshot:
                self.assertEqual((local/'observations/obs_00000'/name).read_bytes(), (self.folder/name).read_bytes())
            snapshot['cam_head.rgb.npy']['sha256'] = 'invalid'
            fresh = Path(self.temp.name)/'wrong-decoder'
            with self.assertRaises(ValueError):
                import_observation(fresh, self.req, snapshot)
            self.assertFalse((fresh/'READY.json').exists())

    def test_partial_response_cannot_block_past_timeout(self):
        child = subprocess.Popen([sys.executable,'-c',
            'import sys,time; sys.stdout.write("{\\\"partial\\\":"); sys.stdout.flush(); time.sleep(10)'],
            stdout=subprocess.PIPE)
        try:
            with self.assertRaises(TimeoutError):
                receive(child, timeout=.1)
        finally:
            child.kill()
            child.wait()
            child.stdout.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'remote'
        self.ep = self.root/'sessions'/'episode_1'
        self.folder = self.ep/'observations'/'obs_00000'
        self.folder.mkdir(parents=True)
        self.req = dict(run_id='episode_1', request_id='request_1', observation_id='obs_00000',
                        task='push_T', kind='plan', deadline=time.time()+100)
        bridge.atomic(self.ep/'episode.json', dict(run_id='episode_1', task='push_T', task_information_mode='official_only'))
        bridge.atomic(self.ep/'READY.json', self.req)
        bridge.atomic(self.folder/'request.json', self.req)
        files = {}
        for name in ['observation.json']+[c+s for c in ('cam_head','cam_left_wrist','cam_right_wrist') for s in ('.jpg','.rgb.npy')]:
            content = b'{}' if name.endswith('.json') else b'exact binary bytes\x00\xff'
            (self.folder/name).write_bytes(content)
            files[name] = dict(name=name, sha256=hashlib.sha256(content).hexdigest())
        bridge.atomic(self.folder/'manifest.json', {k:self.req[k] for k in (*bridge.IDENTITY,'task','kind')} | {'files':files})

    def message(self, **extra):
        return dict(protocol=bridge.PROTOCOL, client_unix=time.time(), known={}, **extra)

    def test_exact_bytes_and_only_publish_ready_after_verification(self):
        status = bridge.exchange(self.root, self.message())['episodes']['episode_1']
        local = Path(self.temp.name)/'local'
        import_observation(local, self.req, status['observation'])
        self.assertEqual(json.loads((local/'READY.json').read_text()), self.req)
        for name in status['observation']:
            self.assertEqual((local/'observations'/'obs_00000'/name).read_bytes(), (self.folder/name).read_bytes())
        snapshot = status['observation']
        snapshot['cam_head.jpg']['data'] = base64.b64encode(b'corrupted').decode()
        fresh = Path(self.temp.name)/'corrupt'
        with self.assertRaises(ValueError):
            import_observation(fresh, self.req, snapshot)
        self.assertFalse((fresh/'READY.json').exists())

    def test_unchanged_requests_do_not_retransfer_images(self):
        msg = self.message()
        msg['known'] = {'episode_1':'request_1'}
        status = bridge.exchange(self.root, msg)['episodes']['episode_1']
        self.assertNotIn('observation', status)

    def test_stdio_handshake_precedes_first_exchange(self):
        result = subprocess.run([sys.executable, str(TOOLS/'bridge_exchange.py'),
            '--root',str(self.root)],input=json.dumps(self.message())+'\n',
            text=True,capture_output=True,check=True)
        hello, response = map(json.loads, result.stdout.splitlines())
        self.assertTrue(hello['hello'])
        self.assertEqual(hello['protocol'], bridge.PROTOCOL)
        self.assertIn('observation', response['episodes']['episode_1'])
        self.assertEqual(result.stderr, '')

    def test_valid_reply_is_idempotent_stale_and_ended_rejected(self):
        reply = {k:self.req[k] for k in bridge.IDENTITY} | {'payload':{'groups':[]}}
        item = dict(episode='episode_1', reply=reply)
        self.assertTrue(bridge.accept_reply(self.root, item, time.time())['accepted'])
        self.assertTrue(bridge.accept_reply(self.root, item, time.time())['accepted'])
        item['reply'] = reply | {'payload':{'different':True}}
        with self.assertRaises(ValueError):
            bridge.accept_reply(self.root, item, time.time())
        item['reply'] = reply | {'request_id':'stale'}
        self.assertFalse(bridge.accept_reply(self.root, item, time.time())['accepted'])
        item['reply'] = reply
        self.assertFalse(bridge.accept_reply(self.root, item, self.req['deadline']+1)['accepted'])
        bridge.atomic(self.ep/'END.json', {'reason':'finished'})
        self.assertFalse(bridge.accept_reply(self.root, item, time.time())['accepted'])

    def test_clocks_paths_and_private_files_rejected(self):
        with self.assertRaises(ValueError):
            bridge.exchange(self.root, self.message() | {'client_unix':0})
        # A delayed packet must not be confused with two unsynchronized clocks.
        result = bridge.exchange(self.root, self.message() | {'client_unix':time.time()-12})
        self.assertIn('episode_1', result['episodes'])
        with self.assertRaises(ValueError):
            bridge.episode_path(self.root, '../outside')
        manifest = json.loads((self.folder/'manifest.json').read_text())
        manifest['files']['secret'] = dict(name='task_reference.json', sha256='x')
        bridge.atomic(self.folder/'manifest.json', manifest)
        with self.assertRaises(ValueError):
            bridge.snapshot(self.ep, self.req)


if __name__ == '__main__':
    unittest.main()
