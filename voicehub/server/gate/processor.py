"""GateProcessor: Pipecat frame processor for response gating.

Pipeline position: STT -> GateProcessor -> Router

Decision flow per final transcription:
  1. Rule pre-filter (fast regex, ~70% resolved)
     - block -> drop frame, log
     - pass  -> forward immediately (no LLM cost)
  2. No rule matched -> LLM review (ambiguous ~30%)
     - respond -> forward
     - ignore  -> drop, log
     - None (reviewer down) -> forward (fail-open)

Only final transcriptions are gated; partials pass through untouched so the
client keeps live caption updates.
"""
from __future__ import annotations
import time
import logging

from pipecat.frames.frames import TranscriptionFrame, InterimTranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .rules import RuleEngine
from .logger import GateLogger, GateRecord
from .reviewer import BaseReviewer
from .optimizer import RuleOptimizer

logger = logging.getLogger(__name__)


class GateProcessor(FrameProcessor):
    """Gate: rule pre-filter + LLM review + decision logging."""

    def __init__(
        self,
        rule_engine: RuleEngine,
        gate_logger: GateLogger,
        reviewer: BaseReviewer | None = None,
        conv_id: str = "",
    ):
        super().__init__()
        self.rule_engine = rule_engine
        self.gate_logger = gate_logger
        self.reviewer = reviewer
        self.conv_id = conv_id
        # rule engine + logger are shared server-wide; optimizer is per-gate
        self.optimizer = RuleOptimizer(rule_engine, gate_logger)

    async def process_frame(self, frame, direction: FrameDirection):
        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            text = frame.text.strip()
            if not text:
                return
            # Partial frames: forward untouched for live captions.
            # In pipecat 1.12, InterimTranscriptionFrame is partial by nature;
            # TranscriptionFrame may also carry is_partial=True from some STT services.
            if isinstance(frame, InterimTranscriptionFrame) or (hasattr(frame, "is_partial") and frame.is_partial):
                await self.push_frame(frame, direction)
                return
            result = self.rule_engine.evaluate(text)
            record = GateRecord(
                timestamp=time.time(),
                text=text,
                rule_verdict=result[0] if result else "no_match",
                rule_name=result[1] if result else "no_match",
                conv_id=self.conv_id,
            )

            if result and result[0] == "block":
                record.final_verdict = "ignore"
                await self.gate_logger.log(record)
                logger.debug("[Gate] BLOCK(%s): %s", record.rule_name, text[:40])
                return  # drop

            if result and result[0] == "pass":
                record.final_verdict = "respond"
                await self.gate_logger.log(record)
                await self.push_frame(frame, direction)
                return

            # --- no rule matched: LLM review ---
            if self.reviewer is not None:
                verdict = await self.reviewer.review(text)
                record.llm_verdict = verdict
                if verdict == "ignore":
                    record.final_verdict = "ignore"
                    await self.gate_logger.log(record)
                    logger.debug("[Gate] LLM-IGNORE: %s", text[:40])
                    return  # drop
                # respond or None (reviewer down) -> fail-open
                record.final_verdict = "respond"

            await self.gate_logger.log(record)
            await self.push_frame(frame, direction)
            return

        await super().process_frame(frame, direction)
