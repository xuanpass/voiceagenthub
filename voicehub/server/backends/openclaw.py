from __future__ import annotations
import asyncio
import json
import os
import shutil
from typing import AsyncIterator, Optional
from .base import LLMBackend, Message

_LOCAL_BIN = os.path.expanduser("~/.local/bin")


class OpenClawBackend(LLMBackend):
    def __init__(self, name: str, bin_path: str | None = None, timeout: int = 600, session_prefix: str = ""):
        self.name = name
        self.timeout = timeout
        self.session_prefix = session_prefix
        if bin_path and shutil.which(bin_path):
            self.bin = bin_path
        elif shutil.which("openclaw"):
            self.bin = "openclaw"
        elif os.path.exists(os.path.join(_LOCAL_BIN, "openclaw")):
            self.bin = os.path.join(_LOCAL_BIN, "openclaw")
        else:
            self.bin = bin_path or "openclaw"

    async def warmup(self) -> None:
        if not (shutil.which(self.bin) or os.path.exists(self.bin)):
            raise FileNotFoundError(f"openclaw CLI not found: {self.bin}")

    async def abort(self) -> None:
        if hasattr(self, "_current_proc") and self._current_proc.returncode is None:
            try:
                self._current_proc.kill()
            except ProcessLookupError:
                pass

    async def send(self, text: str, session_id: str, context: Optional[list[Message]] = None, abort: Optional[asyncio.Event] = None) -> AsyncIterator[str]:
        # NOTE: `context` is intentionally NOT forwarded to the CLI. OpenClaw
        # maintains its own per-session history keyed by --session-id, so
        # replaying `context` here would duplicate the conversation it already
        # has. Signature stays aligned with LLMBackend.send() for LSP/dispatch.
        if not (shutil.which(self.bin) or os.path.exists(self.bin)):
            yield "[VoiceHub] openclaw CLI not found: " + self.bin
            return
        cmd = [self.bin, "agent", "-m", text, "--json", "--session-id", session_id]
        env = dict(os.environ)
        if _LOCAL_BIN not in env.get("PATH", ""):
            env["PATH"] = _LOCAL_BIN + ":" + env.get("PATH", "")
        try:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
            self._current_proc = proc
        except FileNotFoundError:
            yield f"[VoiceHub] openclaw CLI not found: {self.bin}"
            return

        async def _watch() -> None:
            if abort is None:
                return
            try:
                await abort.wait()
            except asyncio.CancelledError:
                return
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

        watch_task = asyncio.create_task(_watch()) if abort is not None else None
        try:
            out, err = await asyncio.wait_for(proc.communicate(), self.timeout)
        except asyncio.TimeoutError:
            if watch_task is not None:
                watch_task.cancel()
            yield f"[OpenClaw timeout] >{self.timeout}s"
            return
        finally:
            if watch_task is not None:
                watch_task.cancel()

        if proc.returncode != 0:
            if abort is not None and abort.is_set():
                return
            yield "[OpenClaw error] " + err.decode("utf-8", "ignore")[:300]
            return
        if abort is not None and abort.is_set():
            return
        try:
            data = json.loads(out.decode("utf-8", "ignore"))
            payloads = data.get("result", {}).get("payloads", [])
            full = "".join(p.get("text", "") for p in payloads if isinstance(p, dict))
            if full:
                yield full
            else:
                yield "[OpenClaw empty] " + str(data)[:200]
        except Exception as e:
            yield f"[OpenClaw parse error] {e}"
        finally:
            if hasattr(self, "_current_proc"):
                delattr(self, "_current_proc")
