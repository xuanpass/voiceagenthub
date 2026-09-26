"""OpenClaw backend: shells out to the `openclaw` CLI on the same host.
The CLI is already authenticated to the running Gateway, so no token handling here.
Verified: `openclaw agent -m "<text>" --json --session-id <id>` returns
{"result":{"payloads":[{"text": "..."}]}}.
Non-streaming by nature (CLI returns final JSON); TTS still plays sentence by sentence.

The openclaw binary lives at ~/.local/bin/openclaw (a pnpm shim that resolves its
own node from ~/.local/bin/node). The VoiceHub server is launched from a
non-interactive shell that does not source .bashrc, so ~/.local/bin is missing
from PATH. We therefore resolve the binary explicitly and inject ~/.local/bin
into the child's PATH so the shim can locate node.
"""
from __future__ import annotations
import asyncio
import json
import os
import shutil
from typing import AsyncIterator, Optional
from .base import LLMBackend, Message

_LOCAL_BIN = os.path.expanduser("~/.local/bin")


class OpenClawBackend(LLMBackend):
    def __init__(
        self,
        name: str,
        bin_path: str | None = None,
        timeout: int = 600,
        session_prefix: str = "",
    ):
        self.name = name
        self.timeout = timeout
        self.session_prefix = session_prefix
        # Resolve binary: explicit path > PATH lookup > ~/.local/bin fallback.
        if bin_path and shutil.which(bin_path):
            self.bin = bin_path
        elif shutil.which("openclaw"):
            self.bin = "openclaw"
        elif os.path.exists(os.path.join(_LOCAL_BIN, "openclaw")):
            self.bin = os.path.join(_LOCAL_BIN, "openclaw")
        else:
            self.bin = bin_path or "openclaw"  # send() reports the missing error

    async def send(
        self,
        text: str,
        session_id: str,
        context: Optional[list[Message]] = None,
        abort: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        if not (shutil.which(self.bin) or os.path.exists(self.bin)):
            yield (
                f"[VoiceHub] openclaw CLI 未找到: {self.bin} "
                "(请确认 ~/.local/bin 含 openclaw 与 node 软链)"
            )
            return
        cmd = [self.bin, "agent", "-m", text, "--json", "--session-id", session_id]
        env = dict(os.environ)
        if _LOCAL_BIN not in env.get("PATH", ""):
            env["PATH"] = f"{_LOCAL_BIN}:{env.get('PATH', '')}"
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self._current_proc = proc
        except FileNotFoundError:
            yield f"[VoiceHub] openclaw CLI 未找到: {self.bin}"
            return

        # hard-cancel watcher: if abort is set while the CLI is still running,
        # kill the child immediately instead of waiting for it to finish.
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
            yield f"[OpenClaw 超时] >{self.timeout}s"
            return
        finally:
            if watch_task is not None:
                watch_task.cancel()

        if proc.returncode != 0:
            # non-zero may mean we killed it on abort -> treat as cancelled, not error
            if abort is not None and abort.is_set():
                return
            yield f"[OpenClaw 错误] {err.decode('utf-8', 'ignore')[:300]}"
            return
        if abort is not None and abort.is_set():
            return  # aborted mid-run -> drop the stale CLI result
        try:
            data = json.loads(out.decode("utf-8", "ignore"))
            payloads = data.get("result", {}).get("payloads", [])
            full = "".join(p.get("text", "") for p in payloads if isinstance(p, dict))
            if full:
                yield full
            else:
                yield f"[OpenClaw 空响应] {str(data)[:200]}"
        except Exception as e:  # noqa: BLE001
            yield f"[OpenClaw 解析错误] {e}"
        finally:
            # Clear current proc reference
            if hasattr(self, '_current_proc'):
                delattr(self, '_current_proc')
