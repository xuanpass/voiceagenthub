"""Gate decision logger: records every gate decision for offline optimization.

Buffered async writes to JSONL, one file per day.
user_followup is patched later by the processor when the user re-engages
(e.g. says "我在跟你说话" after a wrong ignore) - that's the training signal.
"""
from __future__ import annotations
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

def _get_log_dir():
    return Path(os.environ.get("GATE_LOG_DIR", "/home/wangxuan/voicehub/logs/gate"))


@dataclass
class GateRecord:
    """Single gate decision record."""
    timestamp: float
    text: str
    rule_verdict: str  # "pass" | "block" | "no_match"
    rule_name: str  # which rule triggered, or "no_match"
    llm_verdict: Optional[str] = None  # "respond" | "ignore" | None (not reviewed)
    final_verdict: str = ""  # "respond" | "ignore"
    response_text: Optional[str] = None  # what the agent actually replied
    user_followup: Optional[bool] = None  # user re-engaged after our decision?
    conv_id: str = ""  # which conversation


class GateLogger:
    """Async-buffered gate decision logger."""

    def __init__(self, buffer_size: int = 100, flush_interval: int = 60):
        self._buffer: list[GateRecord] = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._recent: list[GateRecord] = []  # last N records for stats/followup
        _get_log_dir().mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        if self._flush_task:
            self._flush_task.cancel()
            self._flush_task = None
        async with self._lock:
            await self._do_flush()

    async def log(self, record: GateRecord) -> None:
        async with self._lock:
            self._buffer.append(record)
            self._recent.append(record)
            if len(self._recent) > 200:
                del self._recent[: len(self._recent) - 200]
            if len(self._buffer) >= self._buffer_size:
                await self._do_flush()

    async def patch_followup(self, conv_id: str, text_hint: str) -> bool:
        """Mark the most recent ignore-decision in this conversation as wrong
        (user re-engaged right after we ignored them)."""
        async with self._lock:
            for r in reversed(self._recent):
                if r.conv_id == conv_id and r.final_verdict == "ignore" and r.user_followup is None:
                    if text_hint in r.text or r.text in text_hint:
                        r.user_followup = True
                        return True
            return False

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            async with self._lock:
                await self._do_flush()

    async def flush(self) -> None:
        """Flush the buffer (public for testing)."""
        async with self._lock:
            await self._do_flush()

    async def _do_flush(self) -> None:
        if not self._buffer:
            return
        records = self._buffer[:]
        self._buffer.clear()
        date_str = time.strftime("%Y-%m-%d")
        log_file = _get_log_dir() / f"gate_{date_str}.jsonl"
        lines = "\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in records)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._append_to_file, log_file, lines)

    @staticmethod
    def _append_to_file(path: Path, content: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(content + "\n")

    async def get_stats(self) -> dict:
        async with self._lock:
            total = len(self._recent)
            blocked = sum(1 for r in self._recent if r.final_verdict == "ignore")
            reviewed = sum(1 for r in self._recent if r.llm_verdict is not None)
            followup = sum(1 for r in self._recent if r.user_followup)
        return {
            "recent_total": total,
            "ignored": blocked,
            "llm_reviewed": reviewed,
            "wrong_ignores": followup,
            "block_rate": blocked / max(total, 1),
        }

    def read_records(self, days: int = 1) -> list[GateRecord]:
        """Read recent log files (for optimizer). Blocking I/O - call from executor."""
        records = []
        now = time.time()
        for d in range(days):
            day = time.localtime(now - d * 86400)
            f = _get_log_dir() / f"gate_{time.strftime('%Y-%m-%d', day)}.jsonl"
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(GateRecord(**obj))
                except (json.JSONDecodeError, TypeError):
                    continue
        return records
