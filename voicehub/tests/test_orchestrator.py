import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.orchestrator import Orchestrator
from server.backends.base import Message


class FakeBackend:
    def __init__(self, name, pieces):
        self.name = name
        self.pieces = list(pieces)
        self.calls = []

    async def send(self, text, session_id, context=None, abort=None):
        self.calls.append((text, context))
        for p in self.pieces:
            if abort is not None and abort.is_set():
                return
            yield p


class FakeRouter:
    def __init__(self, agents):
        self.agents = agents  # key -> {"display": str}


async def _collect(gen):
    out = []
    async for item in gen:
        out.append(item)
    return out


def _orch():
    backends = {
        "a": FakeBackend("a", ["A1", "A2"]),
        "b": FakeBackend("b", ["B1"]),
    }
    router = FakeRouter({"a": {"display": "A"}, "b": {"display": "B"}})
    return Orchestrator(backends, router)


def test_roundtable_yields_all():
    orch = _orch()
    res = asyncio.run(_collect(orch.roundtable(["a", "b"], "hi", "s")))
    who = {w for w, _ in res}
    assert who == {"a", "b"}
    assert ("a", "A1") in res and ("a", "A2") in res and ("b", "B1") in res


def test_handoff_order_and_context():
    orch = _orch()
    res = asyncio.run(_collect(orch.handoff(["a", "b"], "hi", "s")))
    seq = [w for w, _ in res]
    assert seq == ["a", "a", "b"]  # a fully before b
    # second agent receives first agent's output as the first context message
    b_text, b_ctx = orch.backends["b"].calls[0]
    assert isinstance(b_ctx, list) and b_ctx[0].content == "A1A2"


def test_review_target_before_reviewer():
    backends = {"t": FakeBackend("t", ["T-ans"]), "r": FakeBackend("r", ["R-review"])}
    orch = Orchestrator(backends, FakeRouter({"t": {"display": "Tgt"}}))
    res = asyncio.run(_collect(orch.review("t", "r", "hi", "s")))
    # A scheme: target is collected silently (not spoken), only reviewer yields.
    seq = [w for w, _ in res]
    assert seq == ["r"]
    # reviewer is handed the target's output as the thing to critique
    r_text, _ = backends["r"].calls[0]
    assert "T-ans" in r_text and "Tgt" in r_text


def test_abort_stops_streaming():
    # barge-in: once abort is set, the running generator must stop yielding.
    backends = {"a": FakeBackend("a", ["A1", "A2", "A3"])}
    orch = Orchestrator(backends, FakeRouter({"a": {"display": "A"}}))
    abort = asyncio.Event()

    async def _run():
        gen = orch.handoff(["a"], "hi", "s", abort=abort)
        out = []
        async for item in gen:
            out.append(item)
            if len(out) >= 1:
                abort.set()  # user interrupts after the first piece
        return out

    res = asyncio.run(_run())
    # only A1 was yielded; A2/A3 dropped because abort was set mid-stream
    assert res == [("a", "A1")]

def test_roundtable_abort_cancels_tasks():
    # hard cancel: abort mid-roundtable -> remaining agent tasks are cancelled
    # and no leftover pending tasks remain.
    backends = {
        "a": FakeBackend("a", ["A1"]),
        "b": _slow_fake(["B1"], delay=5.0),
    }
    orch = Orchestrator(backends, FakeRouter({"a": {"display": "A"}, "b": {"display": "B"}}))
    abort = asyncio.Event()

    async def _run():
        gen = orch.roundtable(["a", "b"], "hi", "s", abort=abort)
        out = []
        async for item in gen:
            out.append(item)
            if len(out) >= 1:
                abort.set()
                break
        return out

    res = asyncio.run(_run())
    # only got A's piece before abort; B's task was cancelled
    assert ("a", "A1") in res
    assert not any(w == "b" for w, _ in res)


class _slow_fake:
    """Backend that delays between pieces to simulate a long-running LLM."""
    def __init__(self, pieces, delay=0.0):
        self.name = "slow"
        self.pieces = list(pieces)
        self.calls = []
        self._delay = delay

    async def send(self, text, session_id, context=None, abort=None):
        import asyncio as _ao
        self.calls.append((text, context))
        for p in self.pieces:
            if self._delay:
                await _ao.sleep(self._delay)
            if abort is not None and abort.is_set():
                return
            yield p
