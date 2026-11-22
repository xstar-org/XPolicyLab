"""Check complete strict request archives, images, history and native CLI receipts.

This complements raw-observation/source review; it does not infer that missing
keywords prove semantic compliance. Model-generated content remains attributable
to its episode journal rather than being classified as external instructions.
"""
import argparse
import base64
from bisect import bisect_right
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

from audit_strict import audit, POLICY

SANDBOX_ASSETS = {'geometry_reference.md':'assets/geometry/strict_geometry.md',
                  'geometry_interfaces.md':'assets/geometry/geometry_interfaces.md',
                  'vlm_geom.py':'harness/helpers/vlm_geom.py',
                  'observation_io.py':'harness/helpers/observation_io.py'}


def verify_sandbox_assets(session):
    hashes = {}
    for name, source in SANDBOX_ASSETS.items():
        actual = (session/'sandbox'/name).read_bytes()
        if actual != (POLICY/source).read_bytes():
            raise ValueError('Strict sandbox asset changed; requires provenance review: '+name)
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return hashes


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def match_journal(items, journal_ids):
    previous = -1
    for item in items:
        ids = journal_ids.get(digest(item), [])
        position = bisect_right(ids, previous)
        if position == len(ids):
            raise ValueError('Input item has no ordered episode-journal provenance')
        previous = ids[position]


def audit_requests(session, expected_system):
    assets = verify_sandbox_assets(session)
    package_name = '_q0_strict_request_audit'
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(POLICY)]
        sys.modules[package_name] = package
    archive = importlib.import_module(package_name+'.harness.archive')
    history = importlib.import_module(package_name+'.harness.history')
    runner = importlib.import_module(package_name+'.harness.runner')
    messages, _ = archive.read_history_journal(session/'history_journal')
    journal_ids = {}
    for index, message in enumerate(messages):
        journal_ids.setdefault(digest(message), []).append(index)
    prefix = json.loads((session/'planning_prefix.json').read_text())
    if prefix['system'] != expected_system or prefix['tools'] != runner.PLAN_TOOLS:
        raise ValueError('Actual planning prefix differs from audited inputs: '+session.name)
    if prefix['sha256'] != history.fingerprint({k:prefix[k] for k in ('system', 'tools')}):
        raise ValueError('Invalid planning prefix digest')
    images = set()
    for parent in ('observations', 'tools'):
        for path in (session/parent).rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                images.add(hashlib.sha256(path.read_bytes()).hexdigest())
    counts = {'calls': 0, 'planning_calls': 0, 'checkpoint_calls': 0, 'images_verified': 0,
              'planning_contexts_reconstructed': 0, 'checkpoint_contexts_reconstructed': 0,
              'planning_contexts_matched_to_episode_journal': 0}
    attempts = []
    checkpoint_prompt = (POLICY/'assets/prompts/checkpoint_prompt.md').read_text()
    for folder in sorted((session/'api').glob('*/attempt_*')):
        manifest = archive.verify_attempt(folder)
        if manifest['complete_json_response'] is not True:
            raise ValueError('Incomplete model attempt requires separate infrastructure review: '+str(folder))
        request = json.loads((folder/'request.json').read_text())
        receipt = json.loads((folder/'cli_receipt.json').read_text())
        selection_path = folder/'context_selection.json'
        selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
        checkpoint = folder.parent.name.split('_', 2)[1] == 'checkpoint'
        kind = 'checkpoint' if checkpoint else 'planning'
        if (request['model'] != 'gpt-6.2-sol' or receipt['model'] != 'gpt-6.2-sol'
                or receipt.get('auth_method') != 'chatgpt'
                or receipt.get('requested_service_tier') != 'fast'
                or receipt.get('thread_service_tier') not in ('fast', 'priority')
                or not receipt.get('response_complete') or receipt.get('api_fallback') is not False
                or receipt.get('error')):
            raise ValueError('Model/transport/response invariant failed: '+str(folder))
        expected_tools = runner.CHECK_TOOLS if checkpoint else runner.PLAN_TOOLS
        expected_payload = runner.API(session, {'provider':'codex_cli', 'model':'gpt-6.2-sol'}).payload(
            checkpoint_prompt if checkpoint else expected_system, [], expected_tools, checkpoint=checkpoint)
        for key in ('instructions', 'tools', 'reasoning', 'store', 'stream', 'parallel_tool_calls'):
            if request[key] != expected_payload[key]:
                raise ValueError('Unexpected model request '+key+': '+str(folder))
        tools = {tool['name'] for tool in expected_tools}
        catalog = receipt['tool_catalog']
        if (set(catalog['executable_q0_tools']) != tools
                or set(catalog['registered_tools']) - tools - {'request_user_input'}):
            raise ValueError('Unexpected native tool exposure: '+str(folder))
        if checkpoint:
            observation_id = folder.parent.name.split('_', 2)[2]
            review = json.loads((session/'checkpoint_reviews'/(observation_id+'.json')).read_text())
            raw = json.loads((session/'observations'/observation_id/'observation.json').read_text())
            visible = runner.model_observation(raw, observation_id, checking=True)
            content = [runner.text_content(runner.model_json(visible), 'codex_cli')]
            for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                content += [runner.text_content('CURRENT '+observation_id+' '+camera, 'codex_cli'),
                            runner.image_content(session/'observations'/observation_id/(camera+'.jpg'), 'codex_cli')]
            context = review['request']
            content.append(runner.text_content(runner.model_json({k:context[k] for k in
                ('expectation', 'completed_group', 'remaining_groups')}), 'codex_cli'))
            if request['input'] != [{'role':'user', 'content':content}]:
                raise ValueError('Checkpoint contains unaccounted input: '+str(folder))
            counts['checkpoint_contexts_reconstructed'] += 1
        else:
            if selection is not None:
                if history.reconstruct_active(messages, selection) != request['input']:
                    raise ValueError('Planning input differs from episode-local journal: '+str(folder))
                counts['planning_contexts_reconstructed'] += 1
            else:
                # CLI 0.166 transport retains the exact request and journal but
                # not the per-call rolling-window selection metadata. Establish
                # every item's episode-local provenance and order; do not claim
                # to have reconstructed the unrecorded window selection.
                match_journal(request['input'], journal_ids)
            counts['planning_contexts_matched_to_episode_journal'] += 1

        def verify_images(value):
            if isinstance(value, dict):
                if value.get('type') == 'input_image':
                    header, data = value['image_url'].split(',', 1)
                    if not header.startswith('data:image/') or ';base64' not in header:
                        raise ValueError('External image URL in strict request')
                    if hashlib.sha256(base64.b64decode(data, validate=True)).hexdigest() not in images:
                        raise ValueError('Image without canonical/tool-output provenance')
                    counts['images_verified'] += 1
                for child in value.values():
                    verify_images(child)
            elif isinstance(value, list):
                for child in value:
                    verify_images(child)
        verify_images(request['input'])
        counts['calls'] += 1
        counts[kind+'_calls'] += 1
        attempts.append({'attempt':str(folder.relative_to(session)),
                         'request_sha256':archive.file_record(folder/'request.json')['sha256'],
                         'receipt_sha256':archive.file_record(folder/'cli_receipt.json')['sha256']})
    return {'session':session.name, **counts, 'attempt_manifest_sha256':digest(attempts),
            'sandbox_assets_sha256':assets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=Path, required=True)
    parser.add_argument('--input-preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-fifty', action='store_true')
    args = parser.parse_args()
    sessions = [p.parent for p in sorted(args.sessions.glob('*/episode.json'))]
    raw = audit(sessions)
    if args.require_fifty and (len(sessions) != 50 or not all(e['closed'] for e in raw['episodes'])):
        raise ValueError('Exactly 50 final closed strict archives required')
    payload = json.loads((args.input_preflight/'candidate-model-request.json').read_text())
    reports = [audit_requests(session, payload['instructions']) for session in sessions]
    with args.output.open('x') as file:
        json.dump({'raw_inputs':raw, 'request_audits':reports,
                   'calls_checked':sum(r['calls'] for r in reports)}, file, indent=2)


if __name__ == '__main__':
    main()
