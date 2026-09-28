"""OpenAI-compatible HTTP backend (streaming SSE).
Used for Hermes (:8642), CherryStudio and WorkBuddy (via bridge)."""
from __future__ import annotations
import asyncio
import json
import httpx
from typing import AsyncIterator, Optional
from .base import LLMBackend, Message


class OpenAICompatBackend(LLMBackend):
    def __init__(
        self,
        name: str,
        endpoint: str,
        api_key: str | None = None,
        model: str = "auto",
        timeout: int = 120,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        session_prefix: str = "",
        max_history: int = 24,
    ):
        self.name = name
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.auth_header = auth_header
        self.auth_scheme = auth_scheme
        self.session_prefix = session_prefix
        self.max_history = max_history
        self._histories: dict[str, list[dict]] = {}
        # Shared client with connection pool: reuses TCP connections across
        # turns. Hard-cancel closes only the current response's connection
        # (via async-with __aexit__), leaving the pool intact for next turn.
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    def _history(self, session_id: str) -> list[dict]:
        return self._histories.setdefault(session_id, [])

    async def warmup(self) -> None:
        """Send a tiny request to warm up the model (load weights, establish
        TCP connection). Best-effort: failures are silently ignored."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            val = self.api_key if not self.auth_scheme else f"{self.auth_scheme} {self.api_key}"
            headers[self.auth_header] = val
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        try:
            async with self._client.stream("POST", self.endpoint, headers=headers, json=payload) as r:
                async for _ in r.aiter_lines():
                    pass
        except Exception:
            pass  # warmup is best-effort

    async def send(
        self,
        text: str,
        session_id: str,
        context: Optional[list[Message]] = None,
        abort: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        if context:
            messages = [{"role": m.role, "content": m.content} for m in context]
        else:
            messages = list(self._history(session_id))
        messages.append({"role": "user", "content": text})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            val = self.api_key if not self.auth_scheme else f"{self.auth_scheme} {self.api_key}"
            headers[self.auth_header] = val
        payload = {"model": self.model, "messages": messages, "stream": True}
        reply_parts: list[str] = []

        # httpx.stream() returns an async context manager (NOT awaitable).
        # Enter it to obtain the response, iterate SSE lines, and let __aexit__
        # return the connection to the pool (or close it on hard-cancel).
        try:
            async with self._client.stream(
                "POST", self.endpoint, headers=headers, json=payload
            ) as response:
                self._current_response = response
                async for line in response.aiter_lines():
                    if abort is not None and abort.is_set():
                        break  # hard-cancel: drop stream, __aexit__ closes conn
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        reply_parts.append(delta)
                        yield delta
        finally:
            # Clean up current response reference
            if hasattr(self, '_current_response'):
                delattr(self, '_current_response')
        # No finally: aclose() — client is shared. Normal completion returns
        # connection to pool; hard cancel (break) closes it via __aexit__.
        reply = "".join(reply_parts)
        aborted = abort is not None and abort.is_set()
        if reply and not context and not aborted:
            hist = self._history(session_id)
            hist.append({"role": "user", "content": text})
            hist.append({"role": "assistant", "content": reply})
            if len(hist) > self.max_history:
                del hist[: len(hist) - self.max_history]

    async def abort(self) -> None:
        """Abort the current streaming request."""
        if hasattr(self, '_current_response'):
            try:
                await self._current_response.aclose()
            except Exception:
                pass

    async def close(self) -> None:
        """Close the shared HTTP client (server shutdown)."""
        await self._client.aclose()
