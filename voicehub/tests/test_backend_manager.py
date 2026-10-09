"""BackendManager 配置驱动测试: STT 引擎选型/参数 (env > agents.yaml > 默认) 与 TTS voice_map 构建。

不触碰真实 whisper/sensevoice 模型加载: 用受控 config 代替 agents.yaml,
假 STT 服务类记录构造参数, 并清除相关环境变量消除 .env 污染。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import server.backend_manager as bm_mod
from server.backend_manager import BackendManager, build_voice_map

# backend_manager / main 顶部会 load_dotenv(), 把根 .env 或 voicehub/.env 的变量
# 注入 os.environ (如 WHISPER_MODEL=small, ACTIVE_STT_BACKEND=sensevoice),
# 测试必须显式删除这些键, 才能验证 yaml 兜底路径而不是 .env 值。
_ENV_KEYS = [
    "ACTIVE_STT_BACKEND",
    "WHISPER_MODEL",
    "WHISPER_DEVICE",
    "WHISPER_COMPUTE_TYPE",
    "SENSEVOICE_MODEL_DIR",
    "TTS_VOICE_HERMES",
    "TTS_VOICE_OPENCLAW",
    "TTS_VOICE_CHERRY",
    "TTS_VOICE_CHERRYSTUDIO",
]


def _fake_config():
    return {
        "default_agent": "hermes",
        "stt": {
            "active": "whisper",
            "model": "small",
            "device": "cpu",
            "compute_type": "int8",
            "model_dir": "/yaml/models/sensevoice",
        },
        "agents": {
            "hermes": {"tts_voice": "zh-CN-YunyangNeural", "adapter": "openai_compat"},
            "openclaw": {"tts_voice": "zh-CN-XiaoxiaoNeural", "adapter": "openclaw"},
            "workbuddy": {
                "tts_voice": "zh-CN-YunxiNeural",
                "adapter": "workbuddy",
                "disabled": True,
            },
            "cherrystudio": {"tts_voice": "zh-CN-XiaoyiNeural", "adapter": "openai_compat"},
        },
    }


@pytest.fixture
def bm(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)

    captured = {}

    class FakeWhisper:
        def __init__(self, model, device, compute_type):
            captured["whisper"] = dict(model=model, device=device, compute_type=compute_type)

    class FakeSenseVoice:
        def __init__(self, model_dir):
            captured["sensevoice"] = dict(model_dir=model_dir)

    monkeypatch.setattr("server.config.load_config", _fake_config)
    # 不构造真实 LLM 后端 (openai_compat/openclaw), _init_llm_backends 接受空 dict
    monkeypatch.setattr(bm_mod, "build_backends", lambda config: {})
    monkeypatch.setattr(bm_mod, "StreamingWhisperSTTService", FakeWhisper)
    monkeypatch.setattr("server.backends.sensevoice_stt.SenseVoiceSTTService", FakeSenseVoice)

    BackendManager._instance = None
    yield captured
    BackendManager._instance = None


def test_stt_params_from_yaml_when_no_env(bm):
    BackendManager()
    # agents.yaml stt 段 (model/device/compute_type) 被消费, 而非硬编码默认
    assert bm["whisper"] == {"model": "small", "device": "cpu", "compute_type": "int8"}


def test_stt_env_overrides_yaml(bm, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("WHISPER_DEVICE", "gpu")
    BackendManager()
    # env 覆盖 yaml; 未设置的 compute_type 仍回落到 yaml
    assert bm["whisper"] == {"model": "tiny", "device": "gpu", "compute_type": "int8"}


def test_stt_active_from_env_sensevoice(bm, monkeypatch):
    monkeypatch.setenv("ACTIVE_STT_BACKEND", "sensevoice")
    manager = BackendManager()
    assert "sensevoice" in bm
    assert manager.active_stt_backend == "sensevoice"
    assert manager.get_stt_backend() is manager.stt_backends["sensevoice"]


def test_stt_sensevoice_failure_falls_back_to_whisper(bm, monkeypatch):
    class BrokenSenseVoice:
        def __init__(self, model_dir):
            raise RuntimeError("model dir missing")

    monkeypatch.setattr("server.backends.sensevoice_stt.SenseVoiceSTTService", BrokenSenseVoice)
    monkeypatch.setenv("ACTIVE_STT_BACKEND", "sensevoice")
    manager = BackendManager()
    assert "whisper" in bm
    assert manager.active_stt_backend == "whisper"
    # 回落必须改写 env, 否则 get_stt_backend() 会按 sensevoice 查 KeyError
    assert os.environ["ACTIVE_STT_BACKEND"] == "whisper"


def test_tts_voice_map_keys_are_agent_names(bm):
    manager = BackendManager()
    vm = manager.tts_backends["edge"].voice_map
    # 键 = agent 名; 旧硬编码别名键 "cherry" 消失; disabled 的 workbuddy 跳过
    assert set(vm) == {"hermes", "openclaw", "cherrystudio"}
    assert vm["cherrystudio"] == "zh-CN-XiaoyiNeural"
    assert vm["hermes"] == "zh-CN-YunyangNeural"
    assert vm["openclaw"] == "zh-CN-XiaoxiaoNeural"


def test_tts_env_overrides_yaml(bm, monkeypatch):
    monkeypatch.setenv("TTS_VOICE_HERMES", "zh-CN-XiaoxiaoNeural")
    manager = BackendManager()
    assert manager.tts_backends["edge"].voice_map["hermes"] == "zh-CN-XiaoxiaoNeural"


def test_tts_cherry_compat_env_key(bm, monkeypatch):
    # 部署机 .env 遗留的 TTS_VOICE_CHERRY 仍对 cherrystudio 生效
    monkeypatch.setenv("TTS_VOICE_CHERRY", "zh-CN-XiaoxiaoNeural")
    manager = BackendManager()
    assert manager.tts_backends["edge"].voice_map["cherrystudio"] == "zh-CN-XiaoxiaoNeural"


def test_stt_sensevoice_model_dir_from_yaml(bm, monkeypatch):
    monkeypatch.setenv("ACTIVE_STT_BACKEND", "sensevoice")
    BackendManager()
    # env 未设时 yaml stt.model_dir 生效, 而非硬编码绝对路径
    assert bm["sensevoice"]["model_dir"] == "/yaml/models/sensevoice"


def test_stt_sensevoice_model_dir_env_overrides_yaml(bm, monkeypatch):
    monkeypatch.setenv("ACTIVE_STT_BACKEND", "sensevoice")
    monkeypatch.setenv("SENSEVOICE_MODEL_DIR", "/custom/models/sensevoice")
    BackendManager()
    assert bm["sensevoice"]["model_dir"] == "/custom/models/sensevoice"


def test_build_voice_map_shared_with_backend_manager(bm):
    # VoiceRoom 与 BackendManager 共用 build_voice_map, 防两处音色逻辑漂移
    manager = BackendManager()
    assert build_voice_map(_fake_config()) == manager.tts_backends["edge"].voice_map