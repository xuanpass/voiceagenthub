"""Rule self-evolution: analyze logged decisions -> propose rule changes.

Closed loop:
  1. Read GateRecords (rule verdict vs LLM verdict vs user_followup)
  2. Detect error patterns (wrong ignores, wrong passes)
  3. Ask LLM to propose regex rules from the error clusters
  4. Backtest proposals against historical records
  5. Human reviews -> apply_suggestion() hot-updates the engine

Runs offline (admin-triggered or cron); never mutates rules at runtime
without explicit apply.
"""
from __future__ import annotations
import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .rules import Rule, RuleEngine
from .logger import GateLogger, GateRecord

logger = logging.getLogger(__name__)


@dataclass
class RuleSuggestion:
    """A rule change proposal."""
    action: str  # "add" | "remove" | "modify"
    rule: Optional[Rule] = None
    target_rule_name: Optional[str] = None
    reason: str = ""
    confidence: float = 0.0
    backtest: dict = field(default_factory=dict)  # precision/recall on history


@dataclass
class OptimizationReport:
    analyzed_records: int = 0
    wrong_ignores: int = 0   # we ignored, user re-engaged (worst error)
    wrong_passes: int = 0    # we responded, was background noise
    suggestions: list[RuleSuggestion] = field(default_factory=list)


class RuleOptimizer:
    """Analyzes gate logs and proposes rule updates."""

    # Signals that a recent "ignore" was wrong: user insisted right after.
    _FOLLOWUP_PATTERNS = re.compile(
        r"(我在跟你说话|不是跟你们|跟你|听见了?吗|回答我|在吗|你说话呀|怎么不说话|喂)"
    )

    def __init__(self, rule_engine: RuleEngine, gate_logger: GateLogger, llm=None):
        self._engine = rule_engine
        self._logger = gate_logger
        self._llm = llm  # optional LLMBackend for rule mining

    async def analyze_and_propose(self, days: int = 1) -> OptimizationReport:
        report = OptimizationReport()
        loop = asyncio.get_event_loop()
        records = await loop.run_in_executor(None, self._logger.read_records, days)
        report.analyzed_records = len(records)
        if not records:
            return report

        # --- Step 1: find errors ---
        wrong_ignores: list[GateRecord] = []
        wrong_passes: list[GateRecord] = []
        for r in records:
            if r.user_followup and r.final_verdict == "ignore":
                wrong_ignores.append(r)
            # LLM said ignore but rules passed it through (noise leaked to agents)
            elif r.llm_verdict == "ignore" and r.final_verdict == "respond":
                wrong_passes.append(r)
        report.wrong_ignores = len(wrong_ignores)
        report.wrong_passes = len(wrong_passes)

        # --- Step 2: propose rules from error clusters ---
        if wrong_passes:
            sug = await self._mine_block_rule(wrong_passes)
            if sug:
                report.suggestions.append(sug)
        if wrong_ignores:
            sug = await self._mine_pass_rule(wrong_ignores)
            if sug:
                report.suggestions.append(sug)
        return report

    async def _mine_block_rule(self, records: list[GateRecord]) -> Optional[RuleSuggestion]:
        """LLM mines a block-rule from texts that leaked through (noise)."""
        texts = [r.text for r in records[:50]]
        if self._llm is None:
            return self._heuristic_block_rule(texts)
        prompt = (
            "以下是语音助手中被误应答的背景语音（自言自语/旁人对话/电视声）。\n"
            "请归纳出一个能匹配这些文本的中文正则表达式（Python re 语法，尽量精确避免误伤）。\n"
            "只输出正则本身，不要解释。\n\n"
            + "\n".join(texts)
        )
        try:
            reply = ""
            async for chunk in self._llm.send(text=prompt, session_id="gate-opt", abort=None):
                reply += chunk
            pattern = reply.strip().strip("`\"' \n")
            self._validate_regex(pattern)  # raises on bad pattern
        except Exception as e:
            logger.warning("[Gate] LLM rule mining failed: %s", e)
            return self._heuristic_block_rule(texts)
        rule = Rule(name=f"evolved_block_{len(texts)}", pattern=pattern, verdict="block",
                    priority=75, description="Evolved: block background speech")
        sug = RuleSuggestion(action="add", rule=rule,
                             reason=f"LLM-mined from {len(texts)} leaked noise samples",
                             confidence=0.7)
        sug.backtest = self._backtest(rule, records, expect="block")
        return sug

    async def _mine_pass_rule(self, records: list[GateRecord]) -> Optional[RuleSuggestion]:
        """LLM mines a pass-rule from texts wrongly blocked (user re-engaged)."""
        texts = [r.text for r in records[:50]]
        if self._llm is None:
            return None  # heuristic for pass-rules is risky; skip without LLM
        prompt = (
            "以下是语音助手中被误忽略的真实用户话语（用户随后表示不满）。\n"
            "请归纳出一个能匹配这类话语的中文正则表达式（Python re 语法，尽量精确）。\n"
            "只输出正则本身，不要解释。\n\n"
            + "\n".join(texts)
        )
        try:
            reply = ""
            async for chunk in self._llm.send(text=prompt, session_id="gate-opt", abort=None):
                reply += chunk
            pattern = reply.strip().strip("`\"' \n")
            self._validate_regex(pattern)
        except Exception as e:
            logger.warning("[Gate] LLM pass-rule mining failed: %s", e)
            return None
        rule = Rule(name=f"evolved_pass_{len(texts)}", pattern=pattern, verdict="pass",
                    priority=60, description="Evolved: rescue real user speech")
        sug = RuleSuggestion(action="add", rule=rule,
                             reason=f"LLM-mined from {len(texts)} wrongly-ignored samples",
                             confidence=0.8)
        sug.backtest = self._backtest(rule, records, expect="pass")
        return sug

    @staticmethod
    def _heuristic_block_rule(texts: list[str]) -> Optional[RuleSuggestion]:
        """Fallback without LLM: exact-match alternation of frequent noise."""
        from collections import Counter
        common = [t for t, c in Counter(texts).items() if c >= 2]
        if not common:
            return None
        escaped = [re.escape(t) for t in common]
        pattern = r"^(?:" + "|".join(escaped) + r")$"
        rule = Rule(name="evolved_block_exact", pattern=pattern, verdict="block",
                    priority=75, description="Evolved: repeated noise phrases")
        sug = RuleSuggestion(action="add", rule=rule,
                             reason=f"{len(common)} repeated noise phrases", confidence=0.5)
        return sug

    @staticmethod
    def _validate_regex(pattern: str) -> None:
        re.compile(pattern)  # raises re.error on invalid pattern

    @staticmethod
    def _backtest(rule: Rule, records: list[GateRecord], expect: str) -> dict:
        """Check the proposed rule against the error records it should fix."""
        hit = sum(1 for r in records if rule.matches(r.text))
        return {
            "targets": len(records),
            "would_fix": hit,
            "fix_rate": hit / max(len(records), 1),
        }

    async def apply_suggestion(self, suggestion: RuleSuggestion) -> bool:
        """Apply after human review. Hot-updates the engine (no restart)."""
        if suggestion.action == "add" and suggestion.rule:
            self._engine.add_rule(suggestion.rule)
            logger.info("[Gate] rule added: %s", suggestion.rule.name)
            return True
        elif suggestion.action == "remove" and suggestion.target_rule_name:
            ok = self._engine.remove_rule(suggestion.target_rule_name)
            if ok:
                logger.info("[Gate] rule removed: %s", suggestion.target_rule_name)
            return ok
        elif suggestion.action == "modify" and suggestion.rule:
            ok = self._engine.update_rule(
                suggestion.rule.name,
                pattern=suggestion.rule.pattern,
                verdict=suggestion.rule.verdict,
                priority=suggestion.rule.priority,
            )
            if ok:
                logger.info("[Gate] rule modified: %s", suggestion.rule.name)
            return ok
        return False
