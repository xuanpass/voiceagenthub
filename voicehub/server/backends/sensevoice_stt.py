"""SenseVoiceSmall (INT8 ONNX) STT via sherpa-onnx —— Whisper 的中文替代方案。

为什么用它替换 faster-whisper small/int8：
1. **结构上不会幻觉**：SenseVoice 是非自回归(CTC)模型，输出由声学帧直接映射，
   不会像 Whisper(自回归解码器)那样在静音/噪声段凭语言模型先验编造
   “字幕by索兰娅 / 点赞订阅”这类字幕组水印文本。这是 Whisper 的已知缺陷，
   且 Whisper 自身的 no_speech 判定对这些幻觉很“自信”(no_speech_prob < 0.5)，
   用阈值过滤挡不住（实测已验证）。
2. **中文更强**：SenseVoice 针对中/粤/英/日/韩训练，中文准确率显著高于 whisper small。
3. **CPU 更快**：INT8 ONNX + 非自回归，实时率远高于 whisper-small int8。

注意：SenseVoice 输出带语言/情感/事件标签，形如
`<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好` —— 必须剥离这些 `<|...|>` 标签。
"""
from __future__ import annotations

import asyncio
import glob
import logging
import os
import re
import time

import numpy as np

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

# SenseVoice 输出里有语言/情感/事件/ITN 标签：<|zh|><|NEUTRAL|><|Speech|><|woitn|>
_TAG_RE = re.compile(r"<\|[^|]*\|>")

DEFAULT_SENSEVOICE_DIR = (
    "/home/wangxuan/voicehub/models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
)


def _clean_text(text: str) -> str:
    """剥掉 <|lang|><|emotion|><|event|><|itn|> 标签并压缩空白。"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()


class SenseVoiceSTTService(STTBackend, FrameProcessor):
    """sherpa-onnx SenseVoiceSmall 离线识别器，按 VAD 停帧产出 final。

    与 StreamingWhisperSTTService 保持同构（同为 FrameProcessor + 同样的
    partial/final 语义），可直接替换进 pipeline。
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_SENSEVOICE_DIR,
        num_threads: int = 2,
        language: str = "zh",
        use_itn: bool = True,
        partial_interval: float = 0.5,
    ):
        super().__init__()
        import sherpa_onnx

        model_path = os.path.join(model_dir, "model.int8.onnx")
        if not os.path.exists(model_path):
            cands = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
            if not cands:
                raise FileNotFoundError(f"no *.onnx under {model_dir}")
            model_path = cands[0]
        tokens_path = os.path.join(model_dir, "tokens.txt")

        self._model_path = model_path
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            use_itn=use_itn,
            language=language,
            provider="cpu",
            debug=False,
        )
        self._audio_buffer = b""
        self._last_partial_text = ""
        self._last_partial_time: float = 0.0
        self._idle_flush_task: asyncio.Task | None = None
        self._partial_interval = partial_interval
        # 能量门控：实测 SenseVoice 在纯静音 / 2% 房间噪声上会输出“我。”等碎片
        # （虽不像 Whisper 那样编造“字幕by索兰娅”长水印，但仍会污染 Gate）。
        # 缓冲 RMS 低于该阈值直接跳过识别；用户实测语音平均 RMS ~2.2%，取 2%。
        self._silence_threshold = 0.02
        self._speaking = False
        _logger.warning(
            f'[PROBE-STT-INIT] SenseVoice ready model={os.path.basename(model_path)} '
            f'dir={model_dir} lang={language}'
        )

    # ---------- 识别 ----------
    async def _recognize(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        def _run() -> str:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)
            # 不同 sherpa-onnx 版本 API 略有差异，两种都兜住
            if hasattr(self._recognizer, "decode_streams"):
                self._recognizer.decode_streams([stream])
            else:
                self._recognizer.decode_stream(stream)
            return _clean_text(stream.result.text or "")

        return await asyncio.to_thread(_run)

    # ---------- 帧处理 ----------
    async def process_frame(self, frame, direction):
        if isinstance(frame, VADUserStartedSpeakingFrame):
            # VAD 确认“开始说话”：清空缓冲，只累积这一段语音
            self._speaking = True
            self._audio_buffer = b""
            return

        if isinstance(frame, InputAudioRawFrame):
            self._audio_buffer += frame.audio
            buf_dur = len(self._audio_buffer) / (2 * frame.sample_rate)
            if buf_dur >= self._partial_interval:
                samples = np.frombuffer(self._audio_buffer, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) / 32768.0
                # VAD 判定在说话 → 直接识别；否则要过能量门，挡掉静音/房间噪声
                if not (self._speaking or rms >= self._silence_threshold):
                    _logger.warning(
                        f'[PROBE-STT-skip] sv silent rms={rms:.4f} < {self._silence_threshold}'
                    )
                    return
                text = await self._recognize(self._audio_buffer, frame.sample_rate)
                _logger.warning(
                    f'[PROBE-STT-transcribe] sv rms={rms:.4f} speaking={self._speaking} '
                    f'buf_dur_ms={buf_dur * 1000:.0f} out_text="{text}"'
                )
                if text and text != self._last_partial_text:
                    pf = TranscriptionFrame(text, "", str(frame.pts or 0))
                    setattr(pf, "is_partial", True)
                    _logger.warning(f'[PROBE-STT-push] before-push partial text="{text[:40]}"')
                    await self.push_frame(pf)
                    _logger.warning(f'[PROBE-STT-push] after-push partial text="{text[:40]}"')
                    self._last_partial_text = text
                    self._last_partial_time = time.monotonic()
                    if self._idle_flush_task and not self._idle_flush_task.done():
                        self._idle_flush_task.cancel()
                    self._idle_flush_task = asyncio.create_task(self._idle_flush())
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            speaking = self._speaking
            self._speaking = False
            buf_dur = len(self._audio_buffer) / (2 * 16000)
            if (speaking or buf_dur >= 0.3) and self._audio_buffer:
                text = await self._recognize(self._audio_buffer)
                _logger.warning(
                    f'[PROBE-STT-vad] sv stop buf_dur_ms={buf_dur * 1000:.0f} text="{text}"'
                )
                if text:
                    self._last_partial_text = text
                await self._finalize_partial(frame, "vad-final")
                self._audio_buffer = b""
            return

        if isinstance(frame, EndFrame):
            await self._finalize_partial(frame, "end-final")
            return

        # 其余帧（StartFrame / CancelFrame 等）必须向下游透传
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def _finalize_partial(self, frame, tag: str) -> None:
        if self._last_partial_text:
            final_text = self._last_partial_text
            ff = TranscriptionFrame(final_text, "", str(getattr(frame, "pts", 0) or 0))
            setattr(ff, "is_partial", False)
            _logger.warning(f'[PROBE-STT-final] {tag} text="{final_text[:60]}"')
            await self.push_frame(ff)
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

    # ---------- STTBackend 接口 ----------
    async def warmup(self) -> None:
        pass

    async def transcribe(self, audio_chunk: bytes, final: bool = False):
        text = await self._recognize(audio_chunk)
        if text:
            yield text, final

    async def abort(self) -> None:
        self._audio_buffer = b""
        self._last_partial_text = ""

    async def close(self):
        if hasattr(self, "_recognizer"):
            del self._recognizer
