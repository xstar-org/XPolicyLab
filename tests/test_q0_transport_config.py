"""CLI deployment must not require or inherit the API endpoint configuration."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


FILE = Path(__file__).resolve().parents[1]/'policy/Q0_Preview/tools/runtime.py'
spec = importlib.util.spec_from_file_location('q0_launch_runtime', FILE)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class TransportConfigTests(unittest.TestCase):
    def test_cli_uses_login_and_ignores_api_endpoint_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(dict(provider='codex_cli', model='gpt-6.2-sol', sandbox='bwrap')))
            with patch.dict('os.environ', {'Q0_PREVIEW_API_BASE_URL':'https://unused.example/v1'}, clear=True):
                cfg = runtime.config(sys.prefix, path)
            self.assertEqual(cfg['base_url'], '')
            self.assertEqual(cfg['provider'], 'codex_cli')
            path.write_text(json.dumps(dict(provider='codex_cli', base_url='https://api.example/v1')))
            with self.assertRaises(ValueError):
                runtime.config(sys.prefix, path)

    def test_api_endpoint_and_transport_typos_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            for provider, url in [('openai','http://plain.example'),('codex_typo','https://valid.example')]:
                path.write_text(json.dumps(dict(provider=provider, base_url=url)))
                with patch.dict('os.environ', {}, clear=True), self.assertRaises(ValueError):
                    runtime.config(sys.prefix, path)
