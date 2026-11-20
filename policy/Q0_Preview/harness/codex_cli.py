"""Subscription-authenticated Codex CLI transport for the existing Q0 tools.

Each call starts a private stdio app-server, registers the supplied dynamic
tools, injects the original Responses items, and returns at the first native
tool handoff. Only the outer Q0 harness executes tools, through its existing
offline sandbox and robot controller. No model prose is interpreted as a tool.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import tempfile
import time

from ..runtime.io_utils import atomic


DISABLED_FEATURES = ('shell_tool', 'unified_exec', 'view_image', 'apps', 'plugins',
                     'image_generation', 'multi_agent', 'multi_agent_v2',
                     'skill_search', 'code_mode', 'code_mode_host', 'browser_use',
                     'computer_use', 'goals')
BUILTIN_NONEXECUTING_TOOLS = {'request_user_input'}


def toml_literal(value):
    if isinstance(value, dict):
        return '{'+','.join(json.dumps(k)+'='+toml_literal(v) for k,v in value.items())+'}'
    if isinstance(value, list):
        return '['+','.join(toml_literal(v) for v in value)+']'
    return json.dumps(value)


def cli_environment(home):
    env = os.environ.copy()
    for key in list(env):
        if (key.startswith(('OPENAI_', 'Q0_PREVIEW_API_', 'CODEX_API_'))
                or key in ('CODEX_SESSION_ID', 'CODEX_THREAD_ID', 'CODEX_TOOL_CALL_ID',
                           'CODEX_TASK_WORKSPACE_VERIFYING_IDENTITY')):
            env.pop(key, None)
    env.update(CODEX_HOME=str(home), CODEX_SQLITE_HOME=str(home/'sqlite'),
               PYTHONIOENCODING='utf-8')
    return env


def last_user_input(messages):
    """Preserve prior role/tool/reasoning items; convert only the final user input."""
    if not messages or messages[-1].get('role') != 'user':
        raise ValueError('Codex CLI requires the final Responses item to be a user message')
    content = messages[-1]['content']
    if isinstance(content, str):
        content = [dict(type='input_text', text=content)]
    result = []
    for block in content:
        if block.get('type') in ('input_text', 'text'):
            result.append(dict(type='text', text=block['text']))
        elif block.get('type') == 'input_image':
            if not isinstance(block.get('image_url'), str) or not block['image_url'].startswith('data:image/'):
                raise ValueError('Codex CLI observation images must be inline data URLs')
            item = dict(type='image', url=block['image_url'])
            if block.get('detail'):
                item['detail'] = block['detail']
            result.append(item)
        else:
            raise ValueError('Unsupported final user content for Codex CLI')
    if not result:
        raise ValueError('Codex CLI requires nonempty final user content')
    history = copy.deepcopy(messages[:-1])
    # Responses accepts the short {role, content} message form. The app-server's
    # typed ResponseItem parser requires its equivalent explicit discriminator.
    # Tool IDs, image bytes and encrypted reasoning remain untouched.
    for item in history:
        if 'type' not in item and 'role' in item and 'content' in item:
            item['type'] = 'message'
    return history, result


def dynamic_tools(tools):
    names = [tool['name'] for tool in tools]
    if not names or len(names) != len(set(names)) or any(name in BUILTIN_NONEXECUTING_TOOLS for name in names):
        raise ValueError('Invalid Codex CLI dynamic tool set')
    return [dict(type='function', name=tool['name'], description=tool['description'],
                 inputSchema=copy.deepcopy(tool['parameters'])) for tool in tools]


def audit_tool_catalog(item, expected):
    """Raw input events expose the actual catalog, including native tools."""
    found = set()
    for tool in item.get('tools', []):
        if tool.get('type') == 'namespace':
            if tool.get('name') != 'functions':
                raise ValueError('Unexpected native tool namespace: '+str(tool.get('name')))
            found.update(child['name'] for child in tool.get('tools', []))
        elif tool.get('type') == 'function':
            found.add(tool['name'])
        else:
            raise ValueError('Unexpected native tool kind: '+str(tool.get('type')))
    if found-expected-BUILTIN_NONEXECUTING_TOOLS or not expected.issubset(found):
        raise ValueError('Codex CLI tool catalog differs from the supplied Q0 tools')
    return dict(registered_tools=sorted(found), executable_q0_tools=sorted(expected),
                blocked_native_tools=sorted(found & BUILTIN_NONEXECUTING_TOOLS))


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class StdioClient:
    def __init__(self, command, env, cwd, folder, alive):
        self.folder, self.alive = folder, alive
        self.pending, self.serial, self.events = b'', 0, []
        self.private_ids = set()
        self.stderr = (folder/'cli-stderr.log').open('wb')
        self.wire = (folder/'cli-events.jsonl').open('w')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, cwd=cwd, env=env, start_new_session=True, bufsize=0)

    def send(self, value):
        self.process.stdin.write((json.dumps(value)+'\n').encode())
        self.process.stdin.flush()

    def receive(self):
        while b'\n' not in self.pending:
            self.alive()
            if not select.select([self.process.stdout], [], [], .2)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError('Codex app-server exited before its tool handoff')
                continue
            chunk = os.read(self.process.stdout.fileno(), 1024*1024)
            if not chunk:
                raise RuntimeError('Codex app-server closed its output')
            self.pending += chunk
            if len(self.pending) > 128*1024**2:
                raise ValueError('Codex event exceeds 128 MiB')
        line, self.pending = self.pending.split(b'\n', 1)
        value = json.loads(line)
        # Account records can contain email/identifiers. They are not model input
        # and must not enter the experiment archive.
        if (value.get('method') not in ('account/updated', 'account/rateLimits/updated')
                and value.get('id') not in self.private_ids):
            self.wire.write(json.dumps(value)+'\n')
            self.wire.flush()
        return value

    def call(self, method, params):
        self.serial += 1
        if method.startswith('account/'):
            self.private_ids.add(self.serial)
        self.send(dict(id=self.serial, method=method, params=params))
        while True:
            item = self.receive()
            if item.get('id') == self.serial and 'method' not in item:
                if 'error' in item:
                    raise RuntimeError('Codex '+method+' failed: '+str(item['error']))
                return item['result']
            if 'id' in item and 'method' in item:
                raise RuntimeError('Unexpected Codex server request before tool handoff')
            self.events.append(item)

    def close(self):
        stop(self.process)
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()
        self.wire.close()


def restricted_catalog(executable, env, home, target, model, timeout):
    source = subprocess.run([executable, 'debug', 'models'], capture_output=True,
        text=True, env=env, cwd=home, timeout=timeout)
    if source.returncode:
        raise RuntimeError('Codex model catalog unavailable; no API fallback is permitted')
    models = json.loads(source.stdout)['models']
    found = next((copy.deepcopy(m) for m in models if m.get('slug') == model), None)
    if found is None:
        raise ValueError('Requested model is absent from the authenticated Codex model catalog')
    original = copy.deepcopy(found)
    # These fields select tool rendering, not a different model or checkpoint.
    found.update(apply_patch_tool_type=None, experimental_supported_tools=[],
                 tool_mode='direct', supports_search_tool=False, multi_agent_version=None)
    atomic(target, dict(models=[found]))
    return dict(slug=model, supported_reasoning_levels=original.get('supported_reasoning_levels'),
                restricted_fields={k:found[k] for k in ('apply_patch_tool_type',
                    'experimental_supported_tools', 'tool_mode', 'supports_search_tool', 'multi_agent_version')})


def call_codex(api, payload, req):
    """Return an OpenAI-shaped compatibility envelope from a native CLI handoff."""
    service_tier = api.config.get('codex_service_tier', 'default')
    if service_tier not in ('default', 'fast'):
        raise ValueError('Codex service tier must be default or fast')
    api.index += 1
    folder = api.root/'api'/f'{api.index:05d}_{req["kind"]}_{req["observation_id"]}'/'attempt_00'
    folder.mkdir(parents=True, exist_ok=False)
    atomic(folder/'request.json', payload)
    executable = shutil.which(api.config.get('codex_executable', 'codex'))
    if not executable:
        raise RuntimeError('Codex CLI executable is unavailable')
    source_home = Path(api.config.get('codex_auth_home') or os.environ.get('CODEX_HOME') or Path.home()/'.codex').resolve()
    auth = source_home/'auth.json'
    if not auth.is_file() or auth.stat().st_mode & 0o077:
        raise ValueError('Codex requires an existing owner-only ChatGPT login')
    history, turn_input = last_user_input(payload['input'])
    tools = dynamic_tools(payload['tools'])
    names = {tool['name'] for tool in tools}
    deadline = req['deadline']-1
    api.check_alive(deadline)
    receipt = dict(provider='codex_cli', model=api.model, auth_method='chatgpt',
        transport='private_stdio_app_server', response_complete=False,
        native_model_response_complete=False, api_fallback=False,
        history_transport='thread/inject_items; original Responses items',
        completion_source='native_dynamic_tool_handoff', started_unix=time.time())
    receipt['requested_service_tier'] = service_tier
    atomic(folder/'cli_receipt.json', receipt)
    client = temporary = None
    try:
        # Only a filesystem reference is made to the existing native CLI login.
        # No credential contents are read, copied, mounted into Python, or archived.
        temporary = tempfile.TemporaryDirectory(prefix='q0-codex-', dir=api.root)
        home = Path(temporary.name)
        home.chmod(0o700)
        (home/'auth.json').symlink_to(auth)
        cwd = home/'workspace'
        cwd.mkdir()
        env = cli_environment(home)
        base = dict(forced_login_method='chatgpt', model_provider='openai',
                    project_doc_max_bytes=0, web_search='disabled',
                    features={name:False for name in DISABLED_FEATURES})
        # Refresh/auth operations remain native CLI behavior; no API key is
        # accepted. Authentication and user configuration have separate homes.
        catalog = folder/'cli-model-catalog.json'
        receipt['model_catalog'] = restricted_catalog(executable, env, home, catalog,
            api.model, min(60, max(1, deadline-time.time())))
        base['model_catalog_json'] = str(catalog)
        if service_tier == 'fast':
            base['service_tier'] = 'fast'
        command = [executable, 'app-server', '--stdio']
        for key, value in base.items():
            command += ['-c', key+'='+toml_literal(value)]
        receipt['cli_version'] = subprocess.check_output([executable,'--version'],
            env=env, text=True, timeout=10).strip()
        client = StdioClient(command, env, cwd, folder, lambda: api.check_alive(deadline))
        client.call('initialize', dict(clientInfo=dict(name='q0_policy',version='1'),
            capabilities=dict(experimentalApi=True)))
        client.send(dict(method='initialized'))
        account = client.call('account/read', dict(refreshToken=False)).get('account') or {}
        if account.get('type') != 'chatgpt':
            raise ValueError('Codex CLI must authenticate using ChatGPT, never an API key')
        # Only subscription budget percentages/times are retained, not account
        # email, IDs or authentication responses.
        limits = client.call('account/rateLimits/read', {})
        receipt['rate_limits'] = {
            key:{field:window.get(field) for field in ('usedPercent','windowDurationMins','resetsAt')}
            for key in ('primary','secondary')
            if isinstance(window := (limits.get('rateLimits') or {}).get(key), dict)}
        skills = [dict(path=str(path),enabled=False) for path in (home/'skills').rglob('SKILL.md')]
        params = dict(model=api.model, modelProvider='openai', cwd=str(cwd),
            sandbox='read-only', approvalPolicy='never', ephemeral=True,
            experimentalRawEvents=True, allowProviderModelFallback=False,
            baseInstructions=payload['instructions'], developerInstructions='',
            dynamicTools=tools, environments=[], config=dict(skills=dict(config=skills)))
        if service_tier == 'fast':
            params['serviceTier'] = 'priority'
        started = client.call('thread/start', params)
        if started.get('model') != api.model:
            raise ValueError('Codex substituted a different model')
        receipt['thread_service_tier'] = started.get('serviceTier')
        if service_tier == 'fast' and started.get('serviceTier') not in ('fast', 'priority'):
            raise ValueError('Codex did not acknowledge the requested Fast service tier')
        thread = started['thread']['id']
        receipt['thread_id'] = thread
        if history:
            client.call('thread/inject_items', dict(threadId=thread,items=history))
        client.call('turn/start', dict(threadId=thread,input=turn_input,
            effort=payload['reasoning']['effort']))
        # Notifications generated during history injection are archival input,
        # never assistant output. Only the current turn's subsequent handoff
        # is allowed to produce an executable function call.
        output, raw_calls, catalog_audit, wrappers = [], {}, None, []
        for event in client.events:
            item = event.get('params', {}).get('item', {})
            if item.get('type') == 'additional_tools':
                catalog_audit = audit_tool_catalog(item, names)
        while True:
            event = client.receive()
            method, params = event.get('method'), event.get('params', {})
            if method == 'rawResponseItem/completed':
                item = params['item']
                if item.get('type') == 'additional_tools':
                    catalog_audit = audit_tool_catalog(item, names)
                elif item.get('type') == 'reasoning':
                    output.append(item)
                elif item.get('type') == 'function_call':
                    raw_calls[item['call_id']] = item
                elif item.get('type') == 'message':
                    if item.get('role') == 'assistant':
                        output.append(item)
                    elif item.get('role') in ('developer', 'system'):
                        wrappers.append(item)
            if 'id' in event and method:
                if method != 'item/tool/call' or params.get('tool') not in names:
                    raise ValueError('Non-Q0 Codex server request rejected: '+str(method))
                if params.get('namespace') not in (None, 'functions'):
                    raise ValueError('Unexpected Codex dynamic tool namespace')
                raw = raw_calls.get(params['callId'])
                if raw is None or raw.get('name') != params['tool']:
                    raise ValueError('Missing native completed function-call artifact')
                if json.loads(raw['arguments']) != params['arguments']:
                    raise ValueError('CLI raw/function-request arguments disagree')
                if catalog_audit is None:
                    raise ValueError('CLI did not publish its model-visible tool catalog')
                output.append(raw)
                result = dict(status='completed', model=api.model, output=output,
                    source='codex_cli_dynamic_tool_handoff', usage=None)
                receipt.update(response_complete=True, completed_unix=time.time(),
                    tool_name=params['tool'], tool_call_id=params['callId'],
                    tool_catalog=catalog_audit,
                    differences=['Codex read-only permission/date wrappers',
                                 'native request_user_input is advertised but rejected',
                                 'CLI is interrupted at handoff; no native tool executes'])
                atomic(folder/'cli-context-wrappers.json', wrappers)
                atomic(folder/'response.json', result)
                atomic(folder/'cli_receipt.json', receipt)
                api.validate_envelope(result)
                return result
            if method == 'turn/completed':
                turn = params.get('turn', {})
                if turn.get('status') != 'completed' or turn.get('error') is not None:
                    raise RuntimeError('Codex turn did not complete: '+str(turn))
                if raw_calls:
                    raise RuntimeError('Codex completed without delivering its Q0 tool handoff')
                if catalog_audit is None:
                    raise ValueError('CLI did not publish its model-visible tool catalog')
                # A normal text-only generation is valid Responses output. The
                # unchanged Q0 harness records "no_tool", adds its existing tool
                # reminder, and continues under the same planning deadline.
                result = dict(status='completed', model=api.model, output=output,
                    source='codex_cli_completed_turn', usage=None)
                receipt.update(response_complete=True, native_model_response_complete=True,
                    completion_source='native_completed_turn', completed_unix=time.time(),
                    tool_catalog=catalog_audit,
                    differences=['Codex read-only permission/date wrappers',
                                 'native request_user_input is advertised but rejected'])
                atomic(folder/'cli-context-wrappers.json', wrappers)
                atomic(folder/'response.json', result)
                atomic(folder/'cli_receipt.json', receipt)
                api.validate_envelope(result)
                return result
            if method == 'error':
                raise RuntimeError('Codex turn error: '+str(params.get('error')))
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error), completed_unix=time.time())
        atomic(folder/'cli_receipt.json', receipt)
        raise
    finally:
        if client is not None:
            client.close()
        if temporary is not None:
            temporary.cleanup()
