"""Edge TTS wrapper: per-agent voice, sentence-by-sentence streaming.

Edge TTS is free and Chinese-friendly. Each agent gets its own voice id
from agents.yaml (tts_voice). We split text into sentences so the first
sentence can be spoken before the rest finishes generating.

synth_pcm() returns 16-bit PCM (24 kHz mono) ready for Pipecat's
TTSAudioRawFrame; the mp3 from Edge TTS is decoded with miniaudio
(no ffmpeg required).
"""
from __future__ import annotations
import re
import edge_tts
import miniaudio
from typing import AsyncIterator, Optional
from .base import TTSBackend

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;,。.\n])\s*")

TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


class TTSEngine(TTSBackend):
    def __init__(self, voice_map: dict[str, str]):
        self.voice_map = voice_map  # agent_key -> edge voice id

    async def warmup(self) -> None:
        """Warmup TTS backend (no-op for Edge TTS)."""
        pass

    async def synthesize(self, text: str, agent_key: str) -> AsyncIterator[bytes]:
        """Synthesize text to audio stream, implements TTSBackend interface"""
        voice = self.voice_map.get(agent_key, "zh-CN-XiaoxiaoNeural")
        comm = edge_tts.Communicate(text, voice)
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def abort(self) -> None:
        """Abort current TTS synthesis (not supported by Edge TTS, but implemented for interface compliance)"""
        pass

    async def synth_pcm(self, text: str, agent_key: str) -> bytes:
        """Synthesize text to 16-bit PCM (24 kHz mono) via Edge TTS + miniaudio."""
        voice = self.voice_map.get(agent_key, "zh-CN-XiaoxiaoNeural")
        comm = edge_tts.Communicate(text, voice)
        mp3 = b""
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                mp3 += chunk["data"]
        if not mp3:
            return b""
        dec = miniaudio.decode(
            mp3,
            output_format=miniaudio.SampleFormat.SIGNED16,
            sample_rate=TTS_SAMPLE_RATE,
            nchannels=TTS_CHANNELS,
        )
        return bytes(dec.samples)
