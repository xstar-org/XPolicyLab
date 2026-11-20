"""Subscription CLI isolation, history transport, privacy and failure attribution."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


POLICY = Path(__file__).resolve().parents[1]/'policy/Q0_Preview'
PACKAGE = '_q0_cli_tests'
package = ModuleType(PACKAGE)
package.__path__ = [str(POLICY)]
sys.modules[PACKAGE] = package
cli = importlib.import_module(PACKAGE+'.harness.codex_cli')
api_module = importlib.import_module(PACKAGE+'.harness.api')
archive = importlib.import_module(PACKAGE+'.harness.archive')
runner = importlib.import_module(PACKAGE+'.harness.runner')


class CodexCLITests(unittest.TestCase):
    def call_with_native_events(self, root, native_events, service_tier=None, returned_tier='priority'):
        auth = root/'login'
        auth.mkdir()
        (auth/'auth.json').write_text('{}')
        (auth/'auth.json').chmod(0o600)
        expected = {'submit_trajectory'}
        catalog = {'method':'rawResponseItem/completed', 'params':{'item':{
            'type':'additional_tools', 'tools':[{'type':'namespace','name':'functions',
            'tools':[dict(name=name) for name in sorted(expected | {'request_user_input'})]}]}}}
        responses = {
            'initialize':{}, 'account/read':dict(account=dict(type='chatgpt')),
            'account/rateLimits/read':{},
            'thread/start':dict(model='gpt-6.2-sol',thread=dict(id='native-test-thread')),
            'thread/inject_items':{}, 'turn/start':{}}
        if service_tier == 'fast':
            responses['thread/start']['serviceTier'] = returned_tier
        client = SimpleNamespace(events=[catalog],call=Mock(side_effect=lambda method, params: responses[method]),
            send=Mock(), receive=Mock(side_effect=native_events), close=Mock())
        api = api_module.API(root,dict(provider='codex_cli',model='gpt-6.2-sol',codex_auth_home=str(auth)))
        if service_tier is not None:
            api.config['codex_service_tier'] = service_tier
        tool = dict(name='submit_trajectory',description='Q0 trajectory',schema=dict(type='object'))
        payload = api.payload('Original Q0 system instruction',
            [dict(role='user',content=[dict(type='input_text',text='Original observation')])],[tool])
        req = dict(kind='plan',observation_id='obs_00000',deadline=time.time()+60)
        with patch.object(cli.shutil,'which',return_value='/codex'), \
                patch.object(cli,'restricted_catalog',return_value={}), \
                patch.object(cli.subprocess,'check_output',return_value='codex-cli test'), \
                patch.object(cli,'StdioClient',return_value=client):
            return api, api.call(payload,req), client

    def test_fast_requests_priority_without_changing_model_or_reasoning(self):
        message=dict(type='message',role='assistant',status='completed',
            content=[dict(type='output_text',text='Fast probe')])
        events=[dict(method='rawResponseItem/completed',params=dict(item=message)),
                dict(method='turn/completed',params=dict(turn=dict(status='completed',error=None)))]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            api,response,client=self.call_with_native_events(root,events,service_tier='fast')
            thread=next(c.args[1] for c in client.call.call_args_list if c.args[0]=='thread/start')
            self.assertEqual(thread['serviceTier'],'priority')
            self.assertEqual(thread['model'],'gpt-6.2-sol')
            receipt=json.loads(next((root/'api').glob('*/attempt_*/cli_receipt.json')).read_text())
            self.assertEqual(receipt['requested_service_tier'],'fast')
            self.assertEqual(receipt['thread_service_tier'],'priority')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(api_module.APIError):
                self.call_with_native_events(Path(directory),events,service_tier='fast',returned_tier=None)

    def test_native_text_only_completion_returns_to_existing_no_tool_harness(self):
        reasoning = dict(type='reasoning',summary=[],encrypted_content='opaque-original')
        message = dict(type='message',role='assistant',status='completed',
            content=[dict(type='output_text',text='The block remains slightly offset.')])
        events = [dict(method='rawResponseItem/completed',params=dict(item=item))
                  for item in (reasoning,message)]
        events.append(dict(method='turn/completed',params=dict(turn=dict(status='completed',error=None))))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api, response, client = self.call_with_native_events(root,events)
            self.assertEqual(response['output'],[reasoning,message])
            self.assertEqual(api.calls(response),[])
            history = []
            api.append_response(history,response)
            self.assertEqual(history,[reasoning,message])
            folder = root/'api/00001_plan_obs_00000/attempt_00'
            receipt = json.loads((folder/'cli_receipt.json').read_text())
            manifest = json.loads((folder/'archive_manifest.json').read_text())
            self.assertTrue(receipt['native_model_response_complete'])
            self.assertEqual(manifest['response_status'],'native_completed_turn')
            self.assertTrue(manifest['native_model_response_complete'])
            self.assertEqual(manifest['completion_source'],'native_completed_turn')
            client.close.assert_called_once()

    def test_failed_native_turn_and_missing_tool_handoff_still_fail_closed(self):
        failures = [
            [dict(method='turn/completed',params=dict(turn=dict(status='failed',error=dict(message='capacity'))))],
            [dict(method='rawResponseItem/completed',params=dict(item=dict(type='function_call',
                name='submit_trajectory',call_id='call-missing',arguments='{}'))),
             dict(method='turn/completed',params=dict(turn=dict(status='completed',error=None)))]
        ]
        for events in failures:
            with tempfile.TemporaryDirectory() as directory, self.subTest(events=events):
                with self.assertRaises(api_module.APIError) as caught:
                    self.call_with_native_events(Path(directory),events)
                self.assertEqual(caught.exception.reason,'codex_cli_transport_error')
                self.assertTrue('did not complete' in str(caught.exception) or
                                'without delivering' in str(caught.exception))

    def test_api_credentials_and_task_identity_cannot_reach_cli(self):
        initial = dict(OPENAI_API_KEY='do-not-inherit', OPENAI_BASE_URL='https://unused',
                       Q0_PREVIEW_API_KEY='also-private', CODEX_API_KEY='private',
                       CODEX_TOOL_CALL_ID='parent-identity', PATH='/usr/bin')
        with patch.dict(os.environ, initial, clear=True):
            env = cli.cli_environment(Path('/private/cli-home'))
            self.assertEqual(dict(os.environ), initial)
        for key in initial:
            if key != 'PATH':
                self.assertNotIn(key, env)
        self.assertEqual(env['CODEX_HOME'], '/private/cli-home')

    def test_response_history_preserves_reasoning_calls_images_and_roles(self):
        history = [dict(role='user',content=[dict(type='input_text',text='original')]),
                   dict(type='reasoning',encrypted_content='opaque-signed-original',summary=[]),
                   dict(type='function_call',name='run_python',call_id='call-1',arguments='{"code":"print(1)"}'),
                   dict(type='function_call_output',call_id='call-1',output='1'),
                   dict(role='user',content=[dict(type='input_image',image_url='data:image/png;base64,exact'),
                                            dict(type='input_text',text='next')])]
        before = copy.deepcopy(history)
        injected, turn = cli.last_user_input(history)
        self.assertEqual(history, before)
        self.assertEqual(injected[0], before[0] | {'type':'message'})
        self.assertEqual(injected[1:], before[1:-1])
        self.assertEqual(turn[0], dict(type='image',url='data:image/png;base64,exact'))
        with self.assertRaises(ValueError):
            cli.last_user_input(history[:-1])

    def test_actual_catalog_rejects_extra_execution_or_mcp_tools(self):
        good = dict(tools=[dict(type='namespace',name='functions',tools=[
            dict(name='run_python'),dict(name='request_user_input')])])
        self.assertEqual(cli.audit_tool_catalog(good, {'run_python'})['blocked_native_tools'], ['request_user_input'])
        extra = copy.deepcopy(good)
        extra['tools'][0]['tools'].append(dict(name='exec_command'))
        with self.assertRaises(ValueError):
            cli.audit_tool_catalog(extra, {'run_python'})
        extra['tools'][0]['name'] = 'private_mcp'
        with self.assertRaises(ValueError):
            cli.audit_tool_catalog(extra, {'run_python'})
        with self.assertRaises(ValueError):
            cli.audit_tool_catalog(good, {'submit_trajectory'})

    def test_account_rpc_results_and_notifications_are_not_archived(self):
        script = '''import sys,json
for line in sys.stdin:
 x=json.loads(line)
 print(json.dumps({'method':'account/updated','params':{'email':'private@example.invalid'}}),flush=True)
 print(json.dumps({'id':x['id'],'result':{'account':{'type':'chatgpt','email':'private@example.invalid','tokens':'private-sentinel'}}}),flush=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = cli.StdioClient([sys.executable,'-u','-c',script],os.environ.copy(),root,root,lambda:None)
            try:
                self.assertEqual(client.call('account/read', {})['account']['type'], 'chatgpt')
            finally:
                client.close()
            self.assertEqual((root/'cli-events.jsonl').read_text(), '')

    def test_cli_transport_errors_are_api_infrastructure_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            api = api_module.API(Path(directory),dict(provider='codex_cli',model='gpt-6.2-sol'))
            with patch.object(cli,'call_codex',side_effect=RuntimeError('catalog rejected')):
                with self.assertRaises(api_module.APIError) as caught:
                    api.call({}, {})
            self.assertEqual(caught.exception.reason, 'codex_cli_transport_error')

    def test_native_handoff_archive_does_not_claim_http_or_native_turn_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'cli_receipt.json').write_text(json.dumps(dict(response_complete=True,auth_method='chatgpt')))
            (root/'response.json').write_text('{"status":"completed"}')
            archive.finalize_attempt(root)
            result = json.loads((root/'archive_manifest.json').read_text())
            self.assertEqual(result['response_status'], 'native_tool_handoff')
            self.assertTrue(result['complete_json_response'])
            self.assertFalse(result['native_model_response_complete'])
            self.assertIn('response.json',result['files'])

    def test_cli_checkpoint_infrastructure_failure_cannot_be_silently_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            for provider in ('codex_cli','openai'):
                agent = runner.Agent.__new__(runner.Agent)
                agent.root = Path(directory)/provider
                agent.prepare = lambda req: []
                agent.history = SimpleNamespace(round_id=0)
                agent.api = SimpleNamespace(provider=provider,payload=Mock(return_value={}),
                    call=Mock(side_effect=api_module.APIError('codex_cli_transport_error','login failed')))
                result = agent._review(dict(observation_id='obs_00001',deadline=time.time()+60,request_id='r'))
                self.assertEqual(result['decision'],'REPLAN')
                error = json.loads((agent.root/'errors/obs_00001.json').read_text())
                self.assertEqual(error.get('infrastructure',False), provider == 'codex_cli')
