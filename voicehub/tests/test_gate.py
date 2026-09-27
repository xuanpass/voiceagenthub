"""
Unit + Integration tests for VoiceHub Gate module.

Coverage:
- Rule engine: all 8 rules (block/pass) + boundary cases + priority ordering
- Pipeline: partial pass-through, block drop, pass forward, LLM review, fail-open
- Optimizer: suggestion flow, apply add/remove, version bump
- Logger: JSONL write, read_records, get_stats, flush
Run on server: /home/wangxuan/voicehub/.venv/bin/python tests/test_gate.py
"""
import sys
import os
import json
import tempfile
import asyncio

# Ensure server venv path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point log dir to a temp location so we don't pollute production logs
_TEST_LOG_DIR = tempfile.mkdtemp(prefix="gate-test-")
os.environ["GATE_LOG_DIR"] = _TEST_LOG_DIR

from pipecat.frames.frames import TranscriptionFrame, InterimTranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from server.gate.rules import RuleEngine, Rule, INITIAL_RULES
from server.gate.logger import GateLogger, GateRecord
from server.gate.reviewer import BaseReviewer, LLMReviewer
from server.gate.processor import GateProcessor
from server.gate.optimizer import RuleOptimizer, RuleSuggestion


# ─────────────────────────────────────────────────
#  Mock helpers
# ─────────────────────────────────────────────────
class MockReviewer(BaseReviewer):
    def __init__(self, verdict="respond"):
        self._verdict = verdict
        self.calls = []

    async def review(self, text):
        self.calls.append(text)
        return self._verdict


class CaptureGate(GateProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)


# ─────────────────────────────────────────────────
#  Rule engine tests
# ─────────────────────────────────────────────────
class TestRuleEngine:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_eq(self, actual, expected, msg):
        if actual == expected:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"  FAIL: {msg} — expected {expected!r}, got {actual!r}")

    def assert_true(self, cond, msg):
        self.assert_eq(cond, True, msg)

    def run_all(self):
        engine = RuleEngine()

        # --- 4 BLOCK rules ---
        block_cases = [
            ("self_talk_filler", "嗯"),
            ("self_talk_filler", "嗯嗯"),
            ("self_talk_filler", "嗯..."),
            ("self_talk_filler", "哦"),
            ("self_talk_filler", "哦哦"),
            ("self_talk_filler", "噢"),
            ("self_talk_filler", "啊"),
            ("self_talk_filler", "好吧"),
            ("self_talk_filler", "好吧。"),
            ("self_talk_filler", "好吧好吧"),
            ("self_talk_filler", "算了"),
            ("self_talk_filler", "随便"),
            ("self_talk_filler", "无所谓"),
            ("self_talk_filler", "哎呀"),
            ("self_talk_filler", "我的天"),
            ("self_talk_filler", "好家伙"),
            ("self_talk_filler", "绝了"),
            ("self_talk_filler", "无语"),
            ("self_talk_filler", "服了"),
            ("self_talk_filler", "真行"),
            ("self_talk_filler", "可以啊"),
            ("self_talk_filler", "好了好了"),
            ("self_talk_filler", "行了行了"),
            ("short_ack", "对"),
            ("short_ack", "对对"),
            ("short_ack", "对啊"),
            ("short_ack", "是"),
            ("short_ack", "是的"),
            ("short_ack", "行"),
            ("short_ack", "好"),
            ("short_ack", "好的"),
            ("short_ack", "好滴"),
            ("short_ack", "收到"),
            ("short_ack", "明白"),
            ("short_ack", "了解"),
            ("short_ack", "知道"),
            ("short_ack", "谢谢"),
            ("short_ack", "感谢"),
            ("short_ack", "嗯呢"),
            ("short_ack", "嗯呐"),
            ("short_ack", "嗯嗯嗯"),
            ("short_ack", "哈哈"),
            ("short_ack", "呵呵"),
            ("short_ack", "嘿嘿"),
            ("short_ack", "牛"),
            ("short_ack", "强"),
            ("short_ack", "厉害"),
            ("short_ack", "哇塞"),
            ("short_ack", "可不是"),
            ("short_ack", "那倒是"),
            ("short_ack", "确实"),
            ("short_ack", "当然"),
            ("third_person_address", "你说"),
            ("third_person_address", "你觉得怎么样"),
            ("third_person_address", "你干嘛"),
            ("third_person_address", "你怎么回事"),
            ("third_person_address", "你吃饱了吗"),
            ("third_person_address", "妈，吃饭了"),
            ("third_person_address", "爸，我回来了"),
            ("third_person_address", "老婆，你看这个"),
            ("third_person_address", "哥们，帮个忙"),
            ("tv_broadcast", "观众朋友们大家好"),
            ("tv_broadcast", "欢迎收看本期节目"),
            ("tv_broadcast", "广告之后马上回来"),
            ("tv_broadcast", "不要走开"),
            ("tv_broadcast", "点赞关注一键三连"),
            ("tv_broadcast", "下期再见"),
            ("tv_broadcast", "感谢观看"),
        ]
        for rule_name, text in block_cases:
            result = engine.evaluate(text)
            verdict = result[0] if result else None
            self.assert_eq(verdict, "block", f"block[{rule_name}]: '{text}'")

        # --- 4 PASS rules ---
        pass_cases = [
            ("agent_alias", "小克"),
            ("agent_alias", "小克在吗"),
            ("agent_alias", "小克帮我查天气"),
            ("agent_alias", "openclaw"),
            ("agent_alias", "克劳德"),
            ("agent_alias", "赫尔墨斯"),
            ("agent_alias", "hermes"),
            ("agent_alias", "爱马仕"),
            ("agent_alias", "赫敏"),
            ("agent_alias", "小助"),
            ("agent_alias", "workbuddy"),
            ("agent_alias", "小樱"),
            ("agent_alias", "cherry"),
            ("agent_alias", "樱桃"),
            ("question_mark", "今天天气怎么样？"),
            ("question_mark", "你在吗？"),
            ("question_mark", "这是什么？"),
            ("question_mark", "这多少钱?"),
            ("question_words", "什么是量子计算"),
            ("question_words", "怎么弄"),
            ("question_words", "多少钱"),
            ("question_words", "为什么这样"),
            ("question_words", "哪个比较好"),
            ("question_words", "哪里可以买到"),
            ("question_words", "什么时候发货"),
            ("question_words", "谁来了"),
            ("question_words", "多少度"),
            ("imperative_request", "帮我查一下股价"),
            ("imperative_request", "给我播放音乐"),
            ("imperative_request", "打开客厅的灯"),
            ("imperative_request", "关闭空调"),
            ("imperative_request", "查一下天气"),
            ("imperative_request", "搜索附近餐厅"),
            ("imperative_request", "提醒我下午开会"),
            ("imperative_request", "记一下这个号码"),
            ("imperative_request", "设置明天闹钟"),
            ("imperative_request", "切换成英文"),
        ]
        for rule_name, text in pass_cases:
            result = engine.evaluate(text)
            verdict = result[0] if result else None
            rname = result[1] if result else None
            self.assert_eq(verdict, "pass", f"pass[{rule_name}]: '{text}'")
            self.assert_eq(rname, rule_name, f"rule_name match: '{text}'")

        # --- no_match (should go to LLM) ---
        # Note: "怎么办啊" contains "怎么" which matches question_words -> pass (not no_match)
        # "然后呢" is in self_talk_filler -> block
        no_match_cases = [
            "这个呢",
            "接着呢",
            "那个东西",
        ]
        for text in no_match_cases:
            result = engine.evaluate(text)
            # These may or may not match rules — the key test is that the ones listed above don't match block rules
            # and don't match agent_alias / imperative — so they go through or get pass
            self.assert_true(result is None or result[0] in ("pass", None), f"no_match or pass: '{text}' -> {result}")

        # --- priority: agent_alias (p105) overrides block rules ---
        # "嗯，小克帮我查一下" -> contains agent_alias -> should pass
        result = engine.evaluate("嗯，小克帮我查一下")
        self.assert_eq(result[0], "pass", "agent alias overrides self-talk (compound)")
        self.assert_eq(result[1], "agent_alias", "agent_alias rule name (compound)")

        # "好吧小克" -> agent alias -> pass
        result = engine.evaluate("好吧小克")
        self.assert_eq(result[0], "pass", "agent alias overrides short_ack")

        # --- imperative overrides self-talk ---
        result = engine.evaluate("帮我")
        self.assert_eq(result[0], "pass", "imperative '帮我' passes")
        result = engine.evaluate("给我")
        self.assert_eq(result[0], "pass", "imperative '给我' passes")

        # --- rule stats ---
        stats = engine.get_stats()
        self.assert_eq(stats["total_rules"], 8, "total_rules == 8")
        self.assert_eq(stats["block_rules"], 4, "block_rules == 4")
        self.assert_eq(stats["pass_rules"], 4, "pass_rules == 4")
        self.assert_eq(stats["version"], 0, "initial version == 0")

        # --- hot update: add rule ---
        new_rule = Rule(name="test_evolved", pattern="^测试$", verdict="pass", priority=30)
        engine.add_rule(new_rule)
        self.assert_eq(engine.get_stats()["total_rules"], 9, "rules count after add")
        self.assert_eq(engine.get_stats()["version"], 1, "version after add")
        result = engine.evaluate("测试")
        self.assert_eq(result[0], "pass", "newly added rule matches")

        # --- hot update: remove rule ---
        ok = engine.remove_rule("test_evolved")
        self.assert_true(ok, "remove_rule returns True")
        self.assert_eq(engine.get_stats()["total_rules"], 8, "rules count after remove")
        self.assert_eq(engine.get_stats()["version"], 2, "version after remove")
        result = engine.evaluate("测试")
        self.assert_eq(result, None, "removed rule no longer matches")

        # --- hot update: modify rule ---
        engine.update_rule("question_mark", pattern=r"^[?？]$")
        self.assert_eq(engine.get_stats()["version"], 3, "version after modify")


# ─────────────────────────────────────────────────
#  Pipeline decision flow tests
# ─────────────────────────────────────────────────
class TestPipelineFlow:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_eq(self, actual, expected, msg):
        if actual == expected:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"  FAIL: {msg} — expected {expected!r}, got {actual!r}")

    def assert_true(self, cond, msg):
        self.assert_eq(cond, True, msg)

    async def run_all(self):
        engine = RuleEngine()

        # 1. partial TranscriptionFrame -> forwarded
        logger = GateLogger(buffer_size=50, flush_interval=300)
        gate = CaptureGate(engine, logger, MockReviewer(), "p1")
        tf = TranscriptionFrame(text="今天天气", user_id="u", timestamp="t1")
        setattr(tf, "is_partial", True)
        await gate.process_frame(tf, FrameDirection.DOWNSTREAM)
        self.assert_eq(len(gate.pushed), 1, "partial forwarded (1 frame)")

        # 2. InterimTranscriptionFrame -> forwarded
        gate.pushed.clear()
        itf = InterimTranscriptionFrame(text="今天天气怎么样", user_id="u", timestamp="t2")
        await gate.process_frame(itf, FrameDirection.DOWNSTREAM)
        self.assert_eq(len(gate.pushed), 1, "interim forwarded (1 frame)")

        # 3. final block -> dropped
        gate.pushed.clear()
        tf = TranscriptionFrame(text="嗯", user_id="u", timestamp="t3")
        setattr(tf, "is_partial", False)
        await gate.process_frame(tf, FrameDirection.DOWNSTREAM)
        self.assert_eq(len(gate.pushed), 0, "block dropped (0 frames)")

        # 4. final pass -> forwarded
        gate.pushed.clear()
        tf = TranscriptionFrame(text="小克在吗", user_id="u", timestamp="t4")
        setattr(tf, "is_partial", False)
        await gate.process_frame(tf, FrameDirection.DOWNSTREAM)
        self.assert_eq(len(gate.pushed), 1, "pass forwarded (1 frame)")

        # 5. no match -> reviewer called -> fail-open forward
        reviewer5 = MockReviewer(verdict="respond")
        logger5 = GateLogger(buffer_size=50, flush_interval=300)
        gate5 = CaptureGate(engine, logger5, reviewer5, "p5")
        tf = TranscriptionFrame(text="这个呢", user_id="u", timestamp="t5")
        setattr(tf, "is_partial", False)
        await gate5.process_frame(tf, FrameDirection.DOWNSTREAM)
        self.assert_eq("这个呢" in reviewer5.calls, True, "reviewer was called")
        self.assert_eq(len(gate5.pushed), 1, "reviewer=respond -> fail-open forwarded")

        # 6. reviewer=ignore -> dropped
        reviewer6 = MockReviewer(verdict="ignore")
        logger6 = GateLogger(buffer_size=50, flush_interval=300)
        gate6 = CaptureGate(engine, logger6, reviewer6, "p6")
        tf = TranscriptionFrame(text="那个东西啊", user_id="u", timestamp="t6")
        setattr(tf, "is_partial", False)
        await gate6.process_frame(tf, FrameDirection.DOWNSTREAM)
        self.assert_eq(len(gate6.pushed), 0, "reviewer=ignore -> dropped")

        # 7. reviewer=None (rules-only) -> fail-open
        logger7 = GateLogger(buffer_size=50, flush_interval=300)
        gate7 = CaptureGate(engine, logger7, None, "p7")
        tf = TranscriptionFrame(text="怎么办", user_id="u", timestamp="t7")
        setattr(tf, "is_partial", False)
        await gate7.process_frame(tf, FrameDirection.DOWNSTREAM)
        self.assert_eq(len(gate7.pushed), 1, "no reviewer -> fail-open forwarded")

        # 8. empty text -> silently ignored
        gate.pushed.clear()
        tf = TranscriptionFrame(text="", user_id="u", timestamp="t8")
        setattr(tf, "is_partial", False)
        await gate.process_frame(tf, FrameDirection.DOWNSTREAM)
        self.assert_eq(len(gate.pushed), 0, "empty text -> no forward")

        # 9. whitespace text -> silently ignored
        gate.pushed.clear()
        tf = TranscriptionFrame(text="   ", user_id="u", timestamp="t9")
        setattr(tf, "is_partial", False)
        await gate.process_frame(tf, FrameDirection.DOWNSTREAM)
        self.assert_eq(len(gate.pushed), 0, "whitespace text -> no forward")


# ─────────────────────────────────────────────────
#  Logger tests
# ─────────────────────────────────────────────────
class TestLogger:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        # Use isolated temp dir per test run to avoid data pollution
        self._log_dir = tempfile.mkdtemp(prefix="gate-log-test-")
        os.environ["GATE_LOG_DIR"] = self._log_dir

    def assert_eq(self, actual, expected, msg):
        if actual == expected:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"  FAIL: {msg} — expected {expected!r}, got {actual!r}")

    def assert_true(self, cond, msg):
        self.assert_eq(cond, True, msg)

    async def run_all(self):
        logger = GateLogger(buffer_size=3, flush_interval=300)

        # log 2 records (below buffer_size)
        await logger.log(GateRecord(timestamp=1.0, text="嗯", rule_verdict="block", rule_name="r1", final_verdict="ignore"))
        await logger.log(GateRecord(timestamp=2.0, text="小克", rule_verdict="pass", rule_name="r2", final_verdict="respond"))

        # flush explicitly
        await logger.flush()

        # read records
        records = logger.read_records(days=1)
        self.assert_eq(len(records), 2, "read_records returns 2")
        self.assert_eq(records[0].text, "嗯", "first record text")
        self.assert_eq(records[1].text, "小克", "second record text")

        # get_stats
        stats = await logger.get_stats()
        self.assert_eq(stats["recent_total"], 2, "stats recent_total == 2")
        self.assert_eq(stats["ignored"], 1, "stats ignored == 1")

        # auto-flush on buffer_size — use isolated dir to avoid reading logger1's data
        log_dir2 = tempfile.mkdtemp(prefix="gate-log-test-2-")
        os.environ["GATE_LOG_DIR"] = log_dir2
        logger2 = GateLogger(buffer_size=2, flush_interval=300)
        await logger2.log(GateRecord(timestamp=1.0, text="a", rule_verdict="block", rule_name="r1", final_verdict="ignore"))
        # not yet flushed
        recs_before = logger2.read_records(days=1)
        self.assert_eq(len(recs_before), 0, "not flushed before buffer_size")
        # push second — triggers auto-flush
        await logger2.log(GateRecord(timestamp=2.0, text="b", rule_verdict="pass", rule_name="r2", final_verdict="respond"))
        await asyncio.sleep(0.1)
        recs_after = logger2.read_records(days=1)
        self.assert_eq(len(recs_after), 2, "auto-flush after buffer_size")


# ─────────────────────────────────────────────────
#  Optimizer tests
# ─────────────────────────────────────────────────
class TestOptimizer:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self._log_dir = tempfile.mkdtemp(prefix="gate-opt-test-")
        os.environ["GATE_LOG_DIR"] = self._log_dir

    def assert_eq(self, actual, expected, msg):
        if actual == expected:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"  FAIL: {msg} — expected {expected!r}, got {actual!r}")

    def assert_true(self, cond, msg):
        self.assert_eq(cond, True, msg)

    async def run_all(self):
        engine = RuleEngine()
        logger = GateLogger(buffer_size=50, flush_interval=300)
        opt = RuleOptimizer(engine, logger, llm=None)

        # Empty logs -> empty report
        report = await opt.analyze_and_propose(days=1)
        self.assert_eq(report.analyzed_records, 0, "no records analyzed")
        self.assert_eq(len(report.suggestions), 0, "no suggestions with empty logs")

        # Apply: add rule
        new_rule = Rule(name="test_rule", pattern="^hello$", verdict="pass", priority=20)
        sug = RuleSuggestion(action="add", rule=new_rule, reason="test", confidence=0.5)
        ok = await opt.apply_suggestion(sug)
        self.assert_true(ok, "apply add returns True")
        self.assert_eq(engine.get_stats()["total_rules"], 9, "engine has 9 rules after add")

        # Apply: remove rule
        sug2 = RuleSuggestion(action="remove", target_rule_name="test_rule")
        ok = await opt.apply_suggestion(sug2)
        self.assert_true(ok, "apply remove returns True")
        self.assert_eq(engine.get_stats()["total_rules"], 8, "back to 8 after remove")

        # Apply: modify rule
        mod_rule = Rule(name="question_mark", pattern=r"^[?？]$", verdict="pass", priority=45)
        sug3 = RuleSuggestion(action="modify", rule=mod_rule)
        ok = await opt.apply_suggestion(sug3)
        self.assert_true(ok, "apply modify returns True")

        # Apply: invalid action -> False
        sug4 = RuleSuggestion(action="invalid")
        ok = await opt.apply_suggestion(sug4)
        self.assert_eq(ok, False, "invalid action returns False")


# ─────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  VoiceHub Gate Module — Full Test Suite")
    print("=" * 60)

    total_pass = 0
    total_fail = 0

    # Rule engine
    print("\n▶ Rule Engine Tests")
    t1 = TestRuleEngine()
    t1.run_all()
    total_pass += t1.passed
    total_fail += t1.failed
    for e in t1.errors:
        print(e)
    print(f"  Passed: {t1.passed} | Failed: {t1.failed}")

    # Pipeline flow
    print("\n▶ Pipeline Flow Tests")
    t2 = TestPipelineFlow()
    await t2.run_all()
    total_pass += t2.passed
    total_fail += t2.failed
    for e in t2.errors:
        print(e)
    print(f"  Passed: {t2.passed} | Failed: {t2.failed}")

    # Logger
    print("\n▶ Logger Tests")
    t3 = TestLogger()
    await t3.run_all()
    total_pass += t3.passed
    total_fail += t3.failed
    for e in t3.errors:
        print(e)
    print(f"  Passed: {t3.passed} | Failed: {t3.failed}")

    # Optimizer
    print("\n▶ Optimizer Tests")
    t4 = TestOptimizer()
    await t4.run_all()
    total_pass += t4.passed
    total_fail += t4.failed
    for e in t4.errors:
        print(e)
    print(f"  Passed: {t4.passed} | Failed: {t4.failed}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  Total: {total_pass + total_fail} | Passed: {total_pass} | Failed: {total_fail}")
    if total_fail == 0:
        print("  ALL PASSED ✓")
    else:
        print(f"  {total_fail} FAILED ✗")
    print("=" * 60)

    return total_fail


if __name__ == "__main__":
    fails = asyncio.run(main())
    sys.exit(1 if fails > 0 else 0)
