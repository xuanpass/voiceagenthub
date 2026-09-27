"""WorkBuddy backend: shells out to the `codebuddy` CLI on the same host.
"""
from __future__ import annotations
import asyncio
import shutil
from typing import AsyncIterator, Optional
from .base import LLMBackend, Message


class WorkBuddyBackend(LLMBackend):
    def __init__(
        self,
        name: str,
        bin_path: str | None = None,
        model: str | None = None,
        timeout: int = 600,
        session_prefix: str = "",
    ):
        self.name = name
        self.bin = bin_path or "codebuddy"
        self.model = model
        self.timeout = timeout
        self.session_prefix = session_prefix

    async def warmup(self) -> None:
        if shutil.which(self.bin) is None and not self._looks_like_path(self.bin):
            raise FileNotFoundError(f"codebuddy CLI not found: {self.bin}")

    async def abort(self) -> None:
        pass

    async def send(
        self, text: str, session_id: str, context: Optional[list[Message]] = None
    ) -> AsyncIterator[str]:
        if shutil.which(self.bin) is None and not self._looks_like_path(self.bin):
            yield f"[VoiceHub] codebuddy CLI 未找到: {self.bin}"
            return
        cmd = [self.bin, "-p", text, "-y", "--output-format", "text",
               "--session-id", session_id]
        if self.model:
            cmd += ["--model", self.model]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await asyncio.wait_for(proc.communicate(), self.timeout)
        except asyncio.TimeoutError:
            yield f"[WorkBuddy 超时] >{self.timeout}s"
            return
        except FileNotFoundError:
            yield f"[VoiceHub] codebuddy CLI 未找到: {self.bin}"
            return

        if proc.returncode != 0:
            yield f"[WorkBuddy 错误] {err.decode('utf-8', 'ignore')[:300]}"
            return
        full = out.decode("utf-8", "ignore").strip()
        if full:
            yield full
        else:
            yield f"[WorkBuddy 空响应] {err.decode('utf-8', 'ignore')[:200]}"

    @staticmethod
    def _looks_like_path(s: str) -> bool:
        return "/" in s or "\\" in s or s.endswith(".exe")
