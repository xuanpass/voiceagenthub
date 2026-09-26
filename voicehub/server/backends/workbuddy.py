"""WorkBuddy backend: shells out to the `codebuddy` CLI on the same host.

WorkBuddy desktop does NOT expose a clean, token-free local chat API:
  - The internal "CodeBuddy Code API" daemon (:24878, ACP-style /api/v1/runs)
    requires an auth token that is not obtainable outside the desktop app.
  - The supported headless interface is the `codebuddy` CLI `-p/--print` mode.
    This adapter calls it the same way the OpenClaw adapter shells out to the
    openclaw CLI. Verified invocation:

        codebuddy -p "<text>" -y --output-format text --session-id <id>

NOTE: the standalone CLI needs the desktop app's authenticated session context.
If it hangs when invoked from VoiceHub, enable WorkBuddy's local "Code API"
(in app settings) and authenticate the CLI first. Once authenticated, this
adapter works unchanged.
"""
from __future__ import annotations
import asyncio
import shutil
from typing import AsyncIterator, Optional
from .base import Message


class WorkBuddyBackend:
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

    async def send(
        self, text: str, session_id: str, context: Optional[list[Message]] = None
    ) -> AsyncIterator[str]:
        if shutil.which(self.bin) is None and not _looks_like_path(self.bin):
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


def _looks_like_path(s: str) -> bool:
    return "/" in s or "\\" in s or s.endswith(".exe")
