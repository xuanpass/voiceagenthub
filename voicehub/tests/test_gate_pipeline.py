"""Pipeline integration test for GateProcessor (Pipecat 1.12 compatible).

Tests the actual decision flow:
- partial frames: push_frame called (forwarded for live captions)
- final block: push_frame NOT called (dropped)
- final pass: push_frame called (forwarded)
- no match: reviewer consulted; push_frame called iff reviewer says respond

Uses a GateProcessor subclass that records push_frame invocations.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GATE_LOG_DIR", "/tmp/voicehub-gate-test-logs")

from pipecat.frames.frames import TranscriptionFrame, InterimTranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from server.gate.rules import RuleEngine
from server.gate.logger import GateLogger, GateRecord
from server.gate.reviewer import BaseReviewer
from server.gate.processor import GateProcessor


class MockReviewer(BaseReviewer):
    def __init__(self, verdict="respond"):
        self._verdict = verdict
        self.calls = []

    async def review(self, text):
        self.calls.append(text)
        return self._verdict


class CaptureGate(GateProcessor):
    """Subclass that captures push_frame invocations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)


async def main():
    results = {}

    engine = RuleEngine()

    # Test 1: partial TranscriptionFrame (is_partial=True) -> forwarded
    logger = GateLogger(buffer_size=50, flush_interval=300)
    gate = CaptureGate(engine, logger, MockReviewer(), "test-conv-1")
    partial_tf = TranscriptionFrame(text="今天天气", user_id="u", timestamp="t1")
    setattr(partial_tf, "is_partial", True)
    await gate.process_frame(partial_tf, FrameDirection.DOWNSTREAM)
    results["partial_forwarded"] = len(gate.pushed) >= 1

    # Test 2: InterimTranscriptionFrame -> forwarded
    gate.pushed.clear()
    interim_tf = InterimTranscriptionFrame(text="今天天气怎么样", user_id="u", timestamp="t2")
    await gate.process_frame(interim_tf, FrameDirection.DOWNSTREAM)
    results["interim_forwarded"] = len(gate.pushed) >= 1

    # Test 3: final "嗯" -> block -> dropped
    gate.pushed.clear()
    final_block = TranscriptionFrame(text="嗯", user_id="u", timestamp="t3")
    setattr(final_block, "is_partial", False)
    await gate.process_frame(final_block, FrameDirection.DOWNSTREAM)
    results["block_dropped"] = len(gate.pushed) == 0

    # Test 4: final "小克在吗" -> pass (agent alias) -> forwarded
    gate.pushed.clear()
    final_pass = TranscriptionFrame(text="小克在吗", user_id="u", timestamp="t4")
    setattr(final_pass, "is_partial", False)
    await gate.process_frame(final_pass, FrameDirection.DOWNSTREAM)
    results["pass_forwarded"] = len(gate.pushed) >= 1

    # Test 5: final "这个呢" -> no rule -> LLM reviewer called -> fail-open
    reviewer5 = MockReviewer(verdict="respond")
    logger5 = GateLogger(buffer_size=50, flush_interval=300)
    gate5 = CaptureGate(engine, logger5, reviewer5, "test-conv-5")
    final_review = TranscriptionFrame(text="这个呢", user_id="u", timestamp="t5")
    setattr(final_review, "is_partial", False)
    await gate5.process_frame(final_review, FrameDirection.DOWNSTREAM)
    results["reviewer_called"] = "这个呢" in reviewer5.calls
    results["reviewer_fail_open"] = len(gate5.pushed) >= 1

    # Test 6: LLM reviewer says ignore -> dropped
    reviewer6 = MockReviewer(verdict="ignore")
    logger6 = GateLogger(buffer_size=50, flush_interval=300)
    gate6 = CaptureGate(engine, logger6, reviewer6, "test-conv-6")
    final_ig = TranscriptionFrame(text="那个东西啊", user_id="u", timestamp="t6")
    setattr(final_ig, "is_partial", False)
    await gate6.process_frame(final_ig, FrameDirection.DOWNSTREAM)
    results["llm_ignore_dropped"] = len(gate6.pushed) == 0

    # Test 7: reviewer=None (rules-only), no match -> fail-open
    logger7 = GateLogger(buffer_size=50, flush_interval=300)
    gate7 = CaptureGate(engine, logger7, None, "test-conv-7")
    final_ambiguous = TranscriptionFrame(text="怎么办", user_id="u", timestamp="t7")
    setattr(final_ambiguous, "is_partial", False)
    await gate7.process_frame(final_ambiguous, FrameDirection.DOWNSTREAM)
    results["no_reviewer_fail_open"] = len(gate7.pushed) >= 1

    # Test 8: logger recorded decisions correctly
    await logger.flush()
    records = logger.read_records(days=1)
    results["logged_block"] = any(r.text == "嗯" and r.final_verdict == "ignore" for r in records)
    results["logged_pass"]   = any(r.text == "小克在吗" and r.final_verdict == "respond" for r in records)

    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    print("\n=== Pipeline Integration Test Results ===")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")
    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
