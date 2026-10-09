"""声纹旁路处理器：VAD 切段 → 异步算声纹 → 数据通道推送身份事件。

设计要点（为什么是旁路）：
- 关键路径是 transport.input → vad → stt → router → transport.output，
  延迟敏感。本处理器只监听/缓冲，从不拦截或延迟任何帧——所有帧
  无条件 push_frame 转发；重计算（sherpa ~380ms/2s）走 asyncio.create_task
  后台任务，不占事件循环。
- 切段用 VAD 起止帧（说话开始清缓冲，说话结束封段），而非按固定窗口
  切——段即一次完整发言，语义完整、时长可控。
- 回声防护：缓冲窗口内出现过 TTSAudioRawFrame（机器人在播报）则丢弃
  本段不算声纹——麦克风采进来的扬声器声音会把机器人音色聚成假簇，
  污染"谁在说话"的判断。
- 事件去抖：同一说话人不每次发言都推事件（数据通道刷屏），仅标签
  变化或距上次 >5s 才推；攒段达标的 speaker_found 单独推一次。

事件协议（JSON，数据通道）：
  {"type":"speaker", "speaker": "<名字|vcxxx>", "confidence": 0.93,
   "enrolled": true/false, "cluster_id": "vcxxx|null", "connection_id": "..."}
  {"type":"speaker_found", "cluster_id": "vcxxx", "segments": 3,
   "connection_id": "..."}   # 新声音攒够段，前端弹命名条
"""
from __future__ import annotations

import asyncio
import logging
import time

from pipecat.frames.frames import (
    InputAudioRawFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger("voicehub")

SAMPLE_RATE = 16000
DEFAULT_MIN_S = 1.0
SPEAK_EVENT_COOLDOWN_S = 5.0


class SpeakerIdProcessor(FrameProcessor):
    """VAD 之后的声纹旁路：缓冲发言 → 后台提特征 → 推送身份事件。"""

    def __init__(
        self,
        embedder,          # SpeakerEmbedder（duck-typed，便于 fake 测试）
        registry,          # VoiceprintRegistry
        connection_id: str,
        send_msg,          # async (dict) -> None；main.py 注入 _send_client_msg 部分绑定
        min_s: float = DEFAULT_MIN_S,
    ):
        super().__init__()
        self._embedder = embedder
        self._registry = registry
        self._connection_id = connection_id
        self._send_msg = send_msg
        self._min_s = min_s

        self._speaking = False          # VAD 起止帧之间才收缓冲
        self._buf = bytearray()
        self._bot_bytes = 0             # 缓冲窗口内的机器人播报字节数（回声门）

        self._last_label: str | None = None
        self._last_sent = 0.0
        self._announced_ready: set[str] = set()  # 已推过 speaker_found 的簇
        self._tasks: set[asyncio.Task] = set()   # 持引用防 GC

    # ---------- 帧处理 ----------

    async def process_frame(self, frame, direction):
        if isinstance(frame, VADUserStartedSpeakingFrame):
            # 新发言开始：丢弃陈旧半截缓冲，回声计数归零
            self._speaking = True
            self._buf.clear()
            self._bot_bytes = 0
        elif isinstance(frame, InputAudioRawFrame):
            if self._speaking:
                audio = getattr(frame, "audio", None)
                if audio:
                    self._buf.extend(audio)
        elif isinstance(frame, TTSAudioRawFrame):
            # 机器人播报经过本节点（router 推下游，旁路也能看见）→ 记回声嫌疑
            audio = getattr(frame, "audio", None) or b""
            self._bot_bytes += len(audio)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._speaking:
                self._speaking = False
                pcm = bytes(self._buf)
                self._buf.clear()
                dur = len(pcm) / 2 / SAMPLE_RATE
                echo = self._bot_bytes > 0
                self._bot_bytes = 0
                if echo:
                    logger.debug("[PROBE-VOICEPRINT] skip echo segment dur=%.1fs", dur)
                elif dur >= self._min_s:
                    task = asyncio.create_task(self._process(pcm))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
        # 无条件转发：旁路绝不改变管线时序
        await self.push_frame(frame, direction)

    # ---------- 后台：提特征 + 归类 + 推事件 ----------

    async def _process(self, pcm: bytes) -> None:
        try:
            emb = await self._embedder.embed(pcm)
            if not emb:
                return
            res = self._registry.classify(emb)
            now = time.time()
            if res["label"] != self._last_label or (now - self._last_sent) >= SPEAK_EVENT_COOLDOWN_S:
                self._last_label = res["label"]
                self._last_sent = now
                await self._send_msg({
                    "type": "speaker",
                    "speaker": res["label"],
                    "confidence": res["confidence"],
                    "enrolled": res["enrolled"],
                    "cluster_id": res["cluster_id"],
                    "connection_id": self._connection_id,
                })
            cid = res.get("cluster_id")
            if (
                res.get("ready_now")
                and cid
                and cid not in self._announced_ready
                and self._registry.cluster_ready(cid)
            ):
                self._announced_ready.add(cid)
                await self._send_msg({
                    "type": "speaker_found",
                    "cluster_id": cid,
                    "segments": res["segments"],
                    "connection_id": self._connection_id,
                })
        except Exception as e:
            logger.warning("[PROBE-VOICEPRINT] process err %r", e)
