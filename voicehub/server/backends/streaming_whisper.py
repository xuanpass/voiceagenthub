"""Streaming STT using faster-whisper, emits partial/final TranscriptionFrame for Pipecat 1.11"""
from __future__ import annotations
import asyncio
import numpy as np
from faster_whisper import WhisperModel
from pipecat.frames.frames import TranscriptionFrame, InputAudioRawFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor
from .base import STTBackend


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
        # Silero VAD already outputs 16kHz mono audio, no resample needed
        # 直接使用Pipecat传入的16kHz单声道音频，无需额外处理

    async def process_frame(self, frame, direction: FrameDirection):
        # 1. 处理音频帧，累积缓冲并实时推理
        if isinstance(frame, RawAudioFrame):
            self._audio_buffer += frame.audio
            # 每100ms处理一次音频（平衡延迟和准确率）
            buffer_duration = len(self._audio_buffer) / (2 * self._sample_rate)
            # 调整缓冲时长为150ms，平衡延迟和识别准确率
            if buffer_duration >= 0.15:
                audio = np.frombuffer(self._audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                segments, _ = self._model.transcribe_stream(audio, stream=True)
                current_partial = "".join([seg.text for seg in segments])
                # 发出partial转录帧
                if current_partial and current_partial != self._last_partial_text:
                    partial_frame = TranscriptionFrame(current_partial, "", frame.timestamp)
                    setattr(partial_frame, "is_partial", True)
                    await self.push_frame(partial_frame)
                    self._last_partial_text = current_partial
                self._audio_buffer = b""
        # 2. 处理VAD结束帧，发出final转录
        elif isinstance(frame, VoiceActivityFrame) and not frame.is_speech:
            if self._last_partial_text:
                final_frame = TranscriptionFrame(self._last_partial_text, "", frame.timestamp)
                setattr(final_frame, "is_partial", False)
                await self.push_frame(final_frame)
                self._last_partial_text = ""
        # 3. 转发所有其他帧
        await super().process_frame(frame, direction)

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
        """释放模型资源"""
        del self._model