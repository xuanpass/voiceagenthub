"""Collaboration orchestrator: handoff / roundtable / review over agent backends.

星型拓扑: VoiceHub 作为中枢串联各 agent, agent 之间不直接互调。
- handoff: 串行, A 的输出喂给 B, 仅播报最终 agent 结果
- roundtable: 并行 fan-out, 各 agent 用自己音色逐个播报
- review: 把上轮对话注入目标 agent, 要批判性意见
"""
from __future__ import annotations
import asyncio
from typing import AsyncIterator, Optional
from .backends.base import AgentBackend, Message
from .router import Router


class Orchestrator:
    def __init__(self, backends: dict[str, AgentBackend], router: Router):
        self.backends = backends
        self.router = router

    async def handoff(
        self,
        steps: list[str],
        user_text: str,
        session: str,
        abort: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[tuple[str, str]]:
        ctx: list[Message] = []
        acc = user_text
        for key in steps:
            if abort is not None and abort.is_set():
                return
            parts: list[str] = []
            async for piece in self.backends[key].send(acc, f"{session}:{key}", ctx, abort):
                parts.append(piece)
                yield (key, piece)
                if abort is not None and abort.is_set():
                    return
            acc = "".join(parts)
            ctx.append(Message("user", acc or user_text))

    async def roundtable(
        self,
        agents: list[str],
        user_text: str,
        session: str,
        abort: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[tuple[str, str]]:
        # fan-out: every named agent answers in its own voice; stream each
        # agent's sentences as soon as that agent finishes (concurrent, not blocked).
        async def _collect(key: str) -> list[tuple[str, str]]:
            out: list[tuple[str, str]] = []
            async for piece in self.backends[key].send(user_text, f"{session}:{key}", abort=abort):
                out.append((key, piece))
            return out

        tasks = [asyncio.create_task(_collect(key)) for key in agents]
        for fut in asyncio.as_completed(tasks):
            items = await fut
            if abort is not None and abort.is_set():
                # hard cancel: cancel in-flight agent tasks AND await them so
                # the coroutines are properly unwound (not just marked cancel).
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                return
            for item in items:
                yield item

    async def review(
        self,
        target: str,
        reviewer: str,
        user_text: str,
        session: str,
        abort: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[tuple[str, str]]:
        # target answers (collected silently, NOT spoken); reviewer critiques.
        # Rationale: with verbose models the target's full reply + the critique
        # would make TTS audio very long; the user asked for a *review*, so we
        # only broadcast the reviewer's verdict, not the target's raw answer.
        disp = (self.router.agents.get(target) or {}).get("display", target)
        target_parts: list[str] = []
        async for piece in self.backends[target].send(user_text, f"{session}:{target}", abort=abort):
            target_parts.append(piece)  # collect but do NOT yield -> silent
            if abort is not None and abort.is_set():
                return
        prior = "".join(target_parts)
        prompt = (
            f"请批判性评审下面这段由 {disp} 给出的内容, 指出问题/风险/改进点:\n\n{prior}"
        )
        async for piece in self.backends[reviewer].send(prompt, f"{session}:review", abort=abort):
            yield (reviewer, piece)
            if abort is not None and abort.is_set():
                return
