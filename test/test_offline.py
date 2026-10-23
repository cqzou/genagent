"""Offline tests: no API keys or network needed."""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_import_without_openai_key():
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY")}
    # Stub out load_dotenv so a local .env can't put the key back.
    code = "import dotenv; dotenv.load_dotenv = lambda *a, **k: False; import genagent"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr[-400:]


def test_openai_client_can_be_replaced():
    from genagent.llm_utils import OpenAIProvider
    provider = OpenAIProvider()
    fake = object()
    provider.client = fake
    assert provider.client is fake


def test_parse_json_keeps_escaped_quotes():
    from genagent import parse_json
    raw = '{"summary": "She said \\"hello\\" to me."}'
    assert parse_json(raw, ["summary"]) == {"summary": 'She said "hello" to me.'}


def test_parse_json_still_handles_double_escaped():
    from genagent import parse_json
    assert parse_json('Sure: {\\"a\\": \\"x\\"}', ["a"]) == {"a": "x"}
    assert parse_json('Sure! {"a": "plain"}', ["a"]) == {"a": "plain"}
