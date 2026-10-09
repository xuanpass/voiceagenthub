"""声纹向量提取器（sherpa-onnx ERes2Net，512 维）。

设计要点：
- 懒加载：构造时不碰模型文件，首次 embed() 才加载——服务即使没有
  声纹模型也能正常启动（降级 enabled=False，静默返回 None）。
- 异步：sherpa 计算是阻塞 CPU 操作（2s 音频实测 ~380ms），必须走
  asyncio.to_thread，绝不阻塞管线事件循环。
- 时长门控：<1s 声纹不稳定直接丢；>3s 只取最后 3s（最近语音最相关，
  且把计算量钉在 ~600ms 上限）。

模型：3D-Speaker ERes2Net base zh（HF: csukuangfj/speaker-embedding-models
的 3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx，39MB）。
实测区分度：同人重叠段 cos=0.96，不同段 cos=0.635，完全确定性。
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger("voicehub")

DEFAULT_MODEL_PATH = "/home/wangxuan/voicehub/models/3dspeaker-eres2net-base-zh/model.onnx"
SAMPLE_RATE = 16000
MIN_SECONDS = 1.0
MAX_SECONDS = 3.0


class SpeakerEmbedder:
    """把 16k/mono/int16 PCM 转成 512 维声纹向量。线程安全（sherpa 流每次新建）。"""

    def __init__(self, model_path: str | None = None, num_threads: int = 2):
        self._model_path = model_path or os.environ.get("VOICEPRINT_MODEL_PATH") or DEFAULT_MODEL_PATH
        self._num_threads = num_threads
        self._extractor = None
        self._load_attempted = False
        self.enabled = False  # 加载成功后才为 True

    def _ensure_loaded(self) -> None:
        """懒加载 sherpa SpeakerEmbeddingExtractor（只尝试一次）。"""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            import sherpa_onnx  # 延迟导入：没装 sherpa 的环境不应因此崩
            if not os.path.isfile(self._model_path):
                raise FileNotFoundError(self._model_path)
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self._model_path, num_threads=self._num_threads
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            self.enabled = True
            logger.warning(
                "[PROBE-VOICEPRINT] embedder ready path=%s dim=%s",
                self._model_path, self._extractor.dim,
            )
        except Exception as e:
            self.enabled = False
            logger.warning("[PROBE-VOICEPRINT] embedder disabled: %r", e)

    async def embed(self, pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> list[float] | None:
        """PCM(int16) → 512 维向量；门控不满足/未启用返回 None。"""
        self._ensure_loaded()
        if not self.enabled:
            return None
        # 对齐到完整 int16 样本，防止半截字节
        pcm = pcm_bytes[: len(pcm_bytes) // 2 * 2]
        if sample_rate != SAMPLE_RATE:
            logger.warning("[PROBE-VOICEPRINT] unexpected sample_rate=%s", sample_rate)
            return None
        n = len(pcm) // 2
        if n < int(MIN_SECONDS * sample_rate):
            return None
        if n > int(MAX_SECONDS * sample_rate):
            pcm = pcm[-int(MAX_SECONDS * sample_rate) * 2:]  # 取尾部 3s
        try:
            return await asyncio.to_thread(self._compute, pcm)
        except Exception as e:
            logger.warning("[PROBE-VOICEPRINT] embed err %r", e)
            return None

    def _compute(self, pcm: bytes) -> list[float]:
        """阻塞段：int16 → float32 归一化 → sherpa 流式提取。"""
        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio.tolist())
        emb = self._extractor.compute(stream)
        return [float(x) for x in emb]
