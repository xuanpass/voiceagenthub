"""Streaming STT using faster-whisper, emits partial/final TranscriptionFrame for Pipecat 1.11"""
from __future__ import annotations
import asyncio
import time
import numpy as np
import logging
from faster_whisper import WhisperModel
from pipecat.frames.frames import (
    TranscriptionFrame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from .base import STTBackend

_logger = logging.getLogger("voicehub.stt_probe")


class StreamingWhisperSTTService(STTBackend, FrameProcessor):
    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        super().__init__()
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._audio_buffer = b""
        self._last_partial_text = ""
        self._last_partial_time: float = 0.0
        self._idle_flush_scheduled: bool = False
        self._idle_flush_task: asyncio.Task | None = None
        self._silence_threshold = 0.01

    async def process_frame(self, frame, direction: FrameDirection):
        # 1. 处理音频帧，累积缓冲并实时推理
        if isinstance(frame, InputAudioRawFrame):
            # VoiceHub 打点：记录 STT 实际收到的音频帧累计时长（首 10 帧必打，避免跨连接取模漏记）
            _n = globals().get('_SWN', 0) + 1; globals()['_SWN'] = _n
            if _n <= 10:
                _logger.warning(f'[PROBE-STT-audio] conn-frame#={_n} bytes={len(frame.audio)} '
                               f'sr={frame.sample_rate} ch={frame.num_channels} '
                               f'buf_dur_ms={len(self._audio_buffer + frame.audio) / (2 * frame.sample_rate) * 1000:.0f}')
            self._audio_buffer += frame.audio
            # 每500ms处理一次音频（faster-whisper 对 150ms 短片段常判空，提到 0.5s 提高识别率）
            buffer_duration = len(self._audio_buffer) / (2 * frame.sample_rate)
            if buffer_duration >= 0.5:
                # ---- 防幻觉第一道闸：能量门控 ----
                # 实测用户说话平均 RMS 仅 2~6%、静音段可低到 <1%；faster-whisper 在
                # 近静音缓冲上会编造“字幕by索兰娅/点赞订阅”等水印短语。缓冲 RMS 低于
                # 阈值直接跳过 Whisper，省 CPU 也避免幻觉污染 Gate。_silence_threshold
                # 已在 __init__ 定义（默认 0.01 = 满量程 1%）。
                audio_int16 = np.frombuffer(self._audio_buffer, dtype=np.int16)
                rms = float(np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))) / 32768.0
                if rms >= self._silence_threshold:
                    audio = audio_int16.astype(np.float32) / 32768.0
                    # 锁定中文，避免短音频/噪声下 whisper auto-detect 把中文误判成泰文/英文
                    # 关键：faster-whisper 的 transcribe 是同步 CPU 密集调用，若直接在 async
                    # 协程里跑会阻塞整个事件循环，导致同进程内其它通话/ pipeline 启动被饿死
                    # （表现为 StartFrame 透传超时、AI 永不响应）。用 to_thread 把它挪到线程池，
                    # 不让推理卡住 asyncio。
                    # 防幻觉第二、三道闸：no_speech_threshold 整段无语音概率高则判空；
                    # logprob_threshold 模型不自信(低对数概率)判空；compression_ratio_threshold
                    # 异常重复(幻觉)判空。逐段再按 seg.no_speech_prob 过滤。
                    segments, info = await asyncio.to_thread(
                        self._model.transcribe, audio,
                        condition_on_previous_text=False, language="zh",
                        no_speech_threshold=0.6,
                        log_prob_threshold=-0.8,
                        compression_ratio_threshold=2.4,
                    )
                    info_ns = float(getattr(info, "no_speech_prob", 0.0)) if info else 0.0
                    kept = []
                    if info_ns < 0.7:
                        for seg in segments:
                            ns = float(getattr(seg, "no_speech_prob", 0.0))
                            if ns >= 0.5:
                                continue
                            t = (seg.text or "").strip()
                            if t:
                                kept.append(t)
                    current_partial = "".join(kept)
                    if _n <= 20:
                        _logger.warning(f'[PROBE-STT-transcribe] frame#={_n} rms={rms:.4f} info_ns={info_ns:.2f} out_text="{current_partial}"')
                    if current_partial and current_partial != self._last_partial_text:
                        partial_frame = TranscriptionFrame(current_partial, "", str(frame.pts or 0))
                        setattr(partial_frame, "is_partial", True)
                        _logger.warning(f'[PROBE-STT-push] before-push partial text="{current_partial[:40]}"')
                        await self.push_frame(partial_frame)
                        _logger.warning(f'[PROBE-STT-push] after-push partial text="{current_partial[:40]}"')
                        self._last_partial_text = current_partial
                        self._last_partial_time = time.monotonic()
                        self._idle_flush_scheduled = False
                        if self._idle_flush_task and not self._idle_flush_task.done():
                            self._idle_flush_task.cancel()
                            self._idle_flush_task = None
                        self._idle_flush_task = asyncio.create_task(self._idle_flush())
                else:
                    if _n <= 30:
                        _logger.warning(f'[PROBE-STT-skip] silent buffer rms={rms:.4f} < {self._silence_threshold} -> skip Whisper')
                self._audio_buffer = b""
            return
        # 2. 处理VAD结束帧：把累积的 partial 转成 final 并发出（仍向下游透传）
        #    仅在已累积 >=0.3s 音频时才 finalize —— 短噪声片段(<0.3s)触发的 VAD 停止
        #    不再误 finalize，让 idle-final / 下一轮 500ms 转写兜底，避免把环境噪声当输入。
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            _nv = globals().get('_SWV', 0) + 1; globals()['_SWV'] = _nv
            # STT 输入固定 16k/16-bit mono，用常量算缓冲时长（VAD 帧无 sample_rate 属性）
            buf_dur = len(self._audio_buffer) / (2 * 16000)
            _logger.warning(f'[PROBE-STT-vad] VADUserStoppedSpeakingFrame #{_nv} buf_dur_ms={buf_dur*1000:.0f} last_partial="{self._last_partial_text}"')
            if buf_dur >= 0.3 or self._last_partial_text:
                await self._finalize_partial(frame, "vad-final")
        # 3. 连接结束/通话挂断：把残留 partial 强制转成 final，避免“永远不发 final”
        elif isinstance(frame, EndFrame):
            await self._finalize_partial(frame, "end-final")
        # 4. 兜底：所有未被消费的帧（尤其 StartFrame / CancelFrame / InterruptionFrame
        #    等系统帧）必须向下游透传——否则 Gate/Router 的 process 任务永远不被
        #    启动，STT 推出的 TranscriptionFrame 会卡在 Gate 的入队队列里，AI 永不响应。
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def _finalize_partial(self, frame, tag: str) -> None:
        if self._last_partial_text:
            final_text = self._last_partial_text
            final_frame = TranscriptionFrame(final_text, "", str(getattr(frame, "pts", 0) or 0))
            setattr(final_frame, "is_partial", False)
            _logger.warning(f'[PROBE-STT-final] {tag} text="{final_text[:60]}"')
            await self.push_frame(final_frame)
            self._last_partial_text = ""
        if self._idle_flush_task and not self._idle_flush_task.done():
            self._idle_flush_task.cancel()
            self._idle_flush_task = None

    async def _idle_flush(self) -> None:
        try:
            await asyncio.sleep(1.5)
            if self._last_partial_text:
                await self._finalize_partial(None, "idle-final")
        except asyncio.CancelledError:
            pass

    async def warmup(self) -> None:
        """预加载模型资源"""
        # Model is already loaded in __init__, this is a no-op for backward compatibility
        pass

    async def transcribe(self, audio_chunk: bytes, final: bool = False) -> AsyncIterator[tuple[str, bool]]:
        """流式转录音频块，实现STTBackend接口"""
        self._audio_buffer += audio_chunk
        
        # 每100ms处理一次音频（平衡延迟和准确率）
        buffer_duration = len(self._audio_buffer) / (2 * 16000)  # 16kHz 16-bit mono
        
        if buffer_duration >= 0.15 or final:
            audio = np.frombuffer(self._audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self._model.transcribe_stream(audio, stream=True)
            current_partial = "".join([seg.text for seg in segments])
            
            if current_partial and current_partial != self._last_partial_text:
                self._last_partial_text = current_partial
                yield current_partial, False
            
            self._audio_buffer = b""
            
            if final and self._last_partial_text:
                yield self._last_partial_text, True
                self._last_partial_text = ""

    async def abort(self) -> None:
        """终止当前转录任务，清空缓冲"""
        self._audio_buffer = b""
        self._last_partial_text = ""

    async def close(self):
        """释放模型资源（仅服务关闭时由 BackendManager.close_all 调用一次）"""
        if hasattr(self, "_model"):
            del self._model