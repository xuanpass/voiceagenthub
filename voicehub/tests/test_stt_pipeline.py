"""Integration test for the STT pipeline (StreamingWhisperSTTService).

This test exercises the real ``process_frame`` logic of the streaming STT
service using lightweight FAKE ``pipecat`` / ``faster_whisper`` modules.

IMPORTANT (test hygiene): the fake modules are injected ONLY for the duration
of each test (via the ``stt_env`` fixture) and removed afterwards by
``monkeypatch``.  This guarantees that running this file as part of a larger
pytest session does NOT pollute ``sys.modules`` and break other tests that
rely on the real ``pipecat`` / ``faster_whisper`` packages.

Conventions copied from ``test_backends.py``:
  * ``sys.path.insert(0, str(Path(__file__).parent.parent))`` so that
    ``import server.backends.xxx`` resolves (``voicehub/`` is the package root).
  * Plain ``def test_*`` functions that wrap async logic in ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import math
import struct
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# --------------------------------------------------------------------------
# Fake building blocks (pure objects; NO import side effects at load time)
# --------------------------------------------------------------------------

class _Seg:
    """Fake transcription segment mirroring faster-whisper's segment API."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.no_speech_prob = 0.0


class _FakeWhisperModel:
    """Drop-in replacement for ``faster_whisper.WhisperModel``.

    Never downloads or loads a model; ``transcribe`` (the streaming service's
    process_frame path) and ``transcribe_stream`` return a single segment
    carrying a fixed transcription so the pipeline runs end-to-end.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs

    def transcribe(self, audio, **kwargs):
        return ([_Seg("你好世界")], None)

    def transcribe_stream(self, audio, stream=True):  # noqa: D401 - API mirror
        return ([_Seg("你好世界")], None)


class InputAudioRawFrame:
    def __init__(self, audio, sample_rate, num_channels, num_frames, pts=None):
        self.audio = audio
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.pts = pts


class VADUserStartedSpeakingFrame:
    def __init__(self, pts=None):
        self.pts = pts


class VADUserStoppedSpeakingFrame:
    def __init__(self, pts=None):
        self.pts = pts


class EndFrame:
    def __init__(self, pts=None):
        self.pts = pts


class TranscriptionFrame:
    def __init__(self, text, user_id, pts):
        self.text = text
        self.user_id = user_id
        self.pts = pts


class FrameDirection:
    DOWNSTREAM = "DOWNSTREAM"
    UPSTREAM = "UPSTREAM"


class FrameProcessor:
    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)

    async def push_frame(self, frame, direction=None):
        pass


def _build_fake_modules():
    """Return fake module objects keyed by their ``sys.modules`` name."""
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = _FakeWhisperModel

    pc = types.ModuleType("pipecat")

    pc_frames_pkg = types.ModuleType("pipecat.frames")
    pc_frames = types.ModuleType("pipecat.frames.frames")
    pc_frames.InputAudioRawFrame = InputAudioRawFrame
    pc_frames.VADUserStartedSpeakingFrame = VADUserStartedSpeakingFrame
    pc_frames.VADUserStoppedSpeakingFrame = VADUserStoppedSpeakingFrame
    pc_frames.TranscriptionFrame = TranscriptionFrame
    pc_frames.EndFrame = EndFrame
    pc_frames_pkg.frames = pc_frames

    pc_proc_pkg = types.ModuleType("pipecat.processors")
    pc_fp = types.ModuleType("pipecat.processors.frame_processor")
    pc_fp.FrameDirection = FrameDirection
    pc_fp.FrameProcessor = FrameProcessor
    pc_proc_pkg.frame_processor = pc_fp

    return {
        "faster_whisper": fw,
        "pipecat": pc,
        "pipecat.frames": pc_frames_pkg,
        "pipecat.frames.frames": pc_frames,
        "pipecat.processors": pc_proc_pkg,
        "pipecat.processors.frame_processor": pc_fp,
    }


@pytest.fixture
def stt_env(monkeypatch):
    """Run one STT test against fake pipecat/faster_whisper, then auto-restore.

    Two layers of isolation, both reverted by ``monkeypatch`` after the test:

    1. Inject fake ``pipecat`` / ``faster_whisper`` into ``sys.modules`` so the
       module under test can be imported even where those packages are not
       installed (local CI), and so a *fresh* import binds the service's class
       references to our fakes.

    2. Rebind the class *name globals* inside ``server.backends.streaming_whisper``
       itself to our fakes.  This is the critical fix for running inside a larger
       suite: ``process_frame`` does ``isinstance(frame, InputAudioRawFrame)`` using
       the module-level ``InputAudioRawFrame``.  If another test already imported
       that module against the REAL pipecat (caching the real class), our locally
       constructed frame would fail the ``isinstance`` check and the audio branch
       would be silently skipped.  Overriding the name in the module namespace
       guarantees the check uses the exact class we build the frame from.

    Because everything is scoped to this fixture, no other test in the session is
    affected (``monkeypatch`` restores both ``sys.modules`` and the module globals).
    """
    fakes = _build_fake_modules()
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)

    import server.backends.streaming_whisper as sw_mod

    # Rebind the service's own class references to our fakes. This is what makes
    # the isinstance() checks line up regardless of how the module was first
    # imported (real pipecat vs fake pipecat).
    monkeypatch.setattr(sw_mod, "WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(sw_mod, "InputAudioRawFrame", InputAudioRawFrame)
    monkeypatch.setattr(sw_mod, "VADUserStartedSpeakingFrame", VADUserStartedSpeakingFrame)
    monkeypatch.setattr(sw_mod, "VADUserStoppedSpeakingFrame", VADUserStoppedSpeakingFrame)
    monkeypatch.setattr(sw_mod, "TranscriptionFrame", TranscriptionFrame)
    monkeypatch.setattr(sw_mod, "EndFrame", EndFrame)
    monkeypatch.setattr(sw_mod, "FrameProcessor", FrameProcessor)

    return sw_mod.StreamingWhisperSTTService, sw_mod


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _make_audio_frame(duration: float = 0.6, pts: int = 0) -> InputAudioRawFrame:
    """Build an ``InputAudioRawFrame`` of ``duration`` seconds at 16kHz mono.

    Bytes required for ``duration`` seconds of 16-bit mono audio:
        num_bytes = 2 * sample_rate * duration
    0.6s -> 19200 bytes (>= the 0.5s buffer threshold used by the service).

    Samples are a 440Hz sine wave (amplitude 0.3 FS), NOT silence: the service
    applies an RMS energy gate (``_silence_threshold=0.01``) before calling
    Whisper, so an all-zero buffer would be skipped and no frame emitted.
    """
    num_bytes = int(2 * 16000 * duration)
    amp = int(0.3 * 32768)
    samples = [
        int(amp * math.sin(2 * math.pi * 440 * i / 16000))
        for i in range(num_bytes // 2)
    ]
    audio = struct.pack(f"<{len(samples)}h", *samples)
    return InputAudioRawFrame(
        audio=audio,
        sample_rate=16000,
        num_channels=1,
        num_frames=int(16000 * duration),
        pts=pts,
    )


def _transcriptions(service):
    """Collect the transcription frames actually pushed downstream.

    Duck-types on the ``is_partial`` attribute instead of an ``isinstance``
    check against our fake ``TranscriptionFrame`` class.  This keeps the test
    correct whether ``server.backends.streaming_whisper`` was imported against
    the fake pipecat (this file, run alone) or against the REAL pipecat (when
    run as part of a larger suite where it was imported earlier by another
    test).  In the latter case the frames are real pipecat ``TranscriptionFrame``
    instances, which would not match our fake class.
    """
    out = []
    for f in service.pushed:
        if getattr(f, "is_partial", None) in (True, False):
            out.append(f)
    return out


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_stt_emits_partial_on_audio(stt_env):
    SvcBase, _ = stt_env

    class CapturingSTT(SvcBase):
        def __init__(self) -> None:
            super().__init__()
            self.pushed: list = []

        async def push_frame(self, frame, direction=None) -> None:
            self.pushed.append(frame)

    svc = CapturingSTT()
    asyncio.run(svc.process_frame(_make_audio_frame(), FrameDirection.DOWNSTREAM))

    transcriptions = _transcriptions(svc)
    assert len(transcriptions) == 1, "exactly one TranscriptionFrame expected"

    partial = transcriptions[0]
    assert getattr(partial, "is_partial", False) is True
    assert partial.text == "你好世界"


def test_stt_emits_final_on_vad_stop(stt_env):
    SvcBase, _ = stt_env

    class CapturingSTT(SvcBase):
        def __init__(self) -> None:
            super().__init__()
            self.pushed: list = []

        async def push_frame(self, frame, direction=None) -> None:
            self.pushed.append(frame)

    svc = CapturingSTT()
    asyncio.run(svc.process_frame(_make_audio_frame(), FrameDirection.DOWNSTREAM))
    asyncio.run(
        svc.process_frame(VADUserStoppedSpeakingFrame(pts=0), FrameDirection.DOWNSTREAM)
    )

    transcriptions = _transcriptions(svc)
    # One partial (from audio) + one final (from VAD stop).
    finals = [f for f in transcriptions if getattr(f, "is_partial", True) is False]
    assert len(finals) == 1, "exactly one final TranscriptionFrame expected"
    assert finals[0].text == "你好世界"
    # Partial must still be present and untouched.
    partials = [f for f in transcriptions if getattr(f, "is_partial", False) is True]
    assert len(partials) == 1
    assert partials[0].text == "你好世界"
