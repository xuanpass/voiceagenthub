import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import server.backends.openai_compat as oaic_mod
from server.backends.openai_compat import OpenAICompatBackend


# --- httpx streaming mock ------------------------------------------------

def _sse_lines(*chunks):
    out = []
    for c in chunks:
        # json.dumps (not repr) so the embedded string is valid JSON
        out.append('data: {"choices":[{"delta":{"content":%s}}]}' % json.dumps(c))
    out.append("data: [DONE]")
    return out


class _FakeStream:
    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aclose(self):
        pass


class _FakeStreamCM:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return _FakeStream(self._lines)

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    def __init__(self, lines, **kw):
        self._lines = lines
        self.sent = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, method, url, headers=None, json=None, **kw):
        self.sent = (method, url, json)
        return _FakeStreamCM(self._lines)

    async def aclose(self):
        pass


@pytest.fixture
def fake_client(monkeypatch):
    fc = _FakeClient(_sse_lines("你好", "世界"))
    monkeypatch.setattr(oaic_mod.httpx, "AsyncClient", lambda **kw: fc)
    return fc


async def _collect(gen):
    return "".join([p async for p in gen])


def test_openai_compat_keeps_history(monkeypatch, fake_client):
    b = OpenAICompatBackend(name="hermes", endpoint="http://x", session_prefix="voice-hermes")
    # turn 1
    r1 = asyncio.run(_collect(b.send("第一句", "voice-hermesCONV1")))
    assert r1 == "你好世界"
    # turn 2 should include turn 1 in the messages sent to the model
    asyncio.run(_collect(b.send("第二句", "voice-hermesCONV1")))
    sent_msgs = fake_client.sent[2]["messages"]
    roles = [m["role"] for m in sent_msgs]
    # user, assistant, user  (history of turn1 + current user)
    assert roles.count("user") == 2
    assert roles.count("assistant") == 1
    assert sent_msgs[0]["content"] == "第一句"
    assert sent_msgs[1]["content"] == "你好世界"
    assert sent_msgs[2]["content"] == "第二句"


def test_openai_compat_isolates_sessions(monkeypatch, fake_client):
    b = OpenAICompatBackend(name="hermes", endpoint="http://x", session_prefix="voice-hermes")
    asyncio.run(_collect(b.send("会话A", "voice-hermesAAA")))
    asyncio.run(_collect(b.send("会话B", "voice-hermesBBB")))
    sent_msgs = fake_client.sent[2]["messages"]
    # session B must NOT carry session A's history
    assert len(sent_msgs) == 1
    assert sent_msgs[0]["content"] == "会话B"


def test_openai_compat_caps_history(monkeypatch, fake_client):
    b = OpenAICompatBackend(name="h", endpoint="http://x", max_history=4)
    for i in range(6):
        asyncio.run(_collect(b.send(f"t{i}", "sess")))
    assert len(b._histories["sess"]) == 4  # capped


def test_openai_compat_context_passthrough(monkeypatch, fake_client):
    b = OpenAICompatBackend(name="h", endpoint="http://x")
    from server.backends.base import Message

    ctx = [Message("user", "A答")]
    asyncio.run(_collect(b.send("评一下", "sessX", context=ctx)))
    sent_msgs = fake_client.sent[2]["messages"]
    # when context is supplied, history is NOT merged and NOT stored
    assert len(sent_msgs) == 2  # ctx + current user
    assert "sessX" not in b._histories


# --- RouterProcessor._session --------------------------------------------

def test_routerprocessor_session_format():
    from server.main import RouterProcessor

    class FakeB:
        def __init__(self, p):
            self.session_prefix = p

    rp = RouterProcessor.__new__(RouterProcessor)
    rp.backends = {"hermes": FakeB("voice-hermes"), "openclaw": FakeB("voice-openclaw")}
    rp.conv_id = "CONV9"
    assert rp._session("hermes") == "voice-hermesCONV9"
    assert rp._session("openclaw") == "voice-openclawCONV9"
