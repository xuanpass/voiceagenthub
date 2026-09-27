"""Gate reviewer: LLM review for rule-unmatched utterances.

Rules block ~70% obvious non-addressed speech; this reviews the ambiguous rest.
Verdict: respond | ignore. On failure/timeout, defaults to pass (never eat a
real user request just because the reviewer is down).
"""
from __future__ import annotations
import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseReviewer(ABC):
    @abstractmethod
    async def review(self, text: str) -> Optional[str]:
        """Return "respond" | "ignore" | None (None = unavailable, default pass)."""
        ...

    async def close(self) -> None:
        pass


class LLMReviewer(BaseReviewer):
    """One-shot LLM review on a dedicated session (no history pollution)."""

    SYSTEM_PROMPT = (
        "你是语音助手的应答门控。常开麦克风会拾取各种声音，判断这句话是否是对助手说的、值得回应。\n"
        "只输出一个词：respond 或 ignore。\n"
        "respond：对助手提问、下指令、点名求助（提到小克/小助/小樱等名字）\n"
        "ignore：自言自语、感叹、短应答（嗯/哦/好吧）、跟旁人聊天、电视/广播/电话声\n"
        "拿不准时输出 respond（宁可多答，不可漏答）\n"
    )

    def __init__(self, backend, timeout: float = 5.0):
        self._backend = backend  # any LLMBackend adapter with async send()
        self._timeout = timeout

    async def review(self, text: str) -> Optional[str]:
        prompt = f"{self.SYSTEM_PROMPT}\n用户话语：{text}\n只输出 respond 或 ignore："
        try:
            reply = ""
            async for chunk in self._backend.send(
                text=prompt, session_id="gate-review", abort=None
            ):
                reply += chunk
                if len(reply) > 40:  # verdict is one word; stop early
                    break
            m = re.search(r"respond|ignore", reply.strip().lower())
            if m:
                return m.group(0)
            logger.warning("[Gate] unparsable reviewer reply: %r", reply[:60])
            return None
        except asyncio.TimeoutError:
            logger.warning("[Gate] reviewer timeout, defaulting to pass")
            return None
        except Exception as e:
            logger.warning("[Gate] reviewer failed (default pass): %s", e)
            return None
