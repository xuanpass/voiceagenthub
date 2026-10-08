import os
import logging
from typing import Dict, Optional, Type, Union
from dotenv import load_dotenv
from .backends.base import LLMBackend, STTBackend, TTSBackend
from .backends.streaming_whisper import StreamingWhisperSTTService
from .tts import TTSEngine
from .backends.factory import build_backends


load_dotenv()


class BackendManager:
    """统一管理多后端实例的单例类"""
    
    _instance: Optional['BackendManager'] = None
    
    def __new__(cls) -> 'BackendManager':
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_backends()
        return cls._instance
    
    def _init_backends(self) -> None:
        """初始化所有后端实例"""
        # LLM 后端
        self.llm_backends: Dict[str, LLMBackend] = {}
        self._init_llm_backends()
        
        # STT 后端
        self.stt_backends: Dict[str, STTBackend] = {}
        self._init_stt_backends()
        
        # TTS 后端
        self.tts_backends: Dict[str, TTSBackend] = {}
        self._init_tts_backends()
        
        # 当前活跃后端: ACTIVE_LLM_BACKEND 未显式设置时, 回落到 agents.yaml 的
        # default_agent (hermes), 与路由默认一致; STT/TTS 维持原默认。
        self.active_llm_backend: str = os.getenv("ACTIVE_LLM_BACKEND") or getattr(
            self, "_default_agent", "hermes"
        )
        self.active_stt_backend: str = os.getenv("ACTIVE_STT_BACKEND", "whisper")
        self.active_tts_backend: str = os.getenv("ACTIVE_TTS_BACKEND", "edge")
    
    def _init_llm_backends(self) -> None:
        """从 agents.yaml 按 agent 名构建 LLM 后端。

        重要: 全链路 (Router / Orchestrator / Gate reviewer) 都用 *agent 名*
        (如 hermes / openclaw / cherrystudio) 去 llm_backends 取后端, 因此这里
        必须按 agent 名建 key。早期版本按 backend 类型 (openai/openclaw) 建 key,
        导致 hermes 等 openai_compat agent 因 key 不匹配而 KeyError。
        build_backends() 已正确按 agent 名建 key, 并读取各自 endpoint/key/model。
        """
        from .config import load_config
        config = load_config()
        self.llm_backends = build_backends(config)
        # active 默认指向 agents.yaml 的 default_agent, 供 _init_backends 回落使用
        self._default_agent = config.get("default_agent", "hermes")
    
    def _init_stt_backends(self) -> None:
        """初始化STT后端（只实例化真正启用的那个，避免两个模型同时占内存）"""
        active = os.getenv("ACTIVE_STT_BACKEND", "whisper")

        if active == "sensevoice":
            # SenseVoiceSmall (INT8 ONNX)：非自回归(CTC)，静音/噪声段不会像 Whisper
            # 那样自回归编造“字幕by索兰娅”等水印式幻觉；中文准确率优于 whisper-small。
            try:
                from .backends.sensevoice_stt import (
                    DEFAULT_SENSEVOICE_DIR,
                    SenseVoiceSTTService,
                )
                sv_dir = os.getenv("SENSEVOICE_MODEL_DIR") or DEFAULT_SENSEVOICE_DIR
                self.stt_backends["sensevoice"] = SenseVoiceSTTService(model_dir=sv_dir)
                logging.getLogger(__name__).warning(
                    '[PROBE-STT] sensevoice backend registered'
                )
                return
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f'[PROBE-STT] sensevoice unavailable, fallback whisper: {e!r}'
                )
                # 回落：把 active 改写成 whisper，后续 get_stt_backend() 才不会 KeyError
                os.environ["ACTIVE_STT_BACKEND"] = "whisper"

        # Faster Whisper后端
        self.stt_backends["whisper"] = StreamingWhisperSTTService(
            model=os.getenv("WHISPER_MODEL", "small"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        )
    
    def _init_tts_backends(self) -> None:
        """初始化TTS后端"""
        # Edge TTS后端
        voice_map = {
            "hermes": os.getenv("TTS_VOICE_HERMES", "zh-CN-YunxiNeural"),
            "cherry": os.getenv("TTS_VOICE_CHERRY", "zh-CN-XiaoxiaoNeural"),
            "openclaw": os.getenv("TTS_VOICE_OPENCLAW", "zh-CN-YunyangNeural"),
        }
        
        self.tts_backends["edge"] = TTSEngine(voice_map=voice_map)
    
    def get_llm_backend(self, name: Optional[str] = None) -> LLMBackend:
        """获取指定的LLM后端实例"""
        name = name or self.active_llm_backend
        if name not in self.llm_backends:
            raise ValueError(f"LLM后端 {name} 未找到")
        return self.llm_backends[name]
    
    def get_stt_backend(self, name: Optional[str] = None) -> STTBackend:
        """获取指定的STT后端实例"""
        name = name or self.active_stt_backend
        if name not in self.stt_backends:
            raise ValueError(f"STT后端 {name} 未找到")
        return self.stt_backends[name]
    
    def get_tts_backend(self, name: Optional[str] = None) -> TTSBackend:
        """获取指定的TTS后端实例"""
        name = name or self.active_tts_backend
        if name not in self.tts_backends:
            raise ValueError(f"TTS后端 {name} 未找到")
        return self.tts_backends[name]
    
    async def warmup_all(self) -> None:
        """预热所有后端"""
        # 预热LLM后端
        for backend in self.llm_backends.values():
            try:
                await backend.warmup()
            except Exception:
                pass
        
        # 预热STT后端
        for backend in self.stt_backends.values():
            try:
                await backend.warmup()
            except Exception:
                pass
        
        # 预热TTS后端
        for backend in self.tts_backends.values():
            try:
                await backend.warmup()
            except Exception:
                pass
    
    async def close_all(self) -> None:
        """关闭所有后端资源"""
        # 关闭LLM后端
        for backend in self.llm_backends.values():
            try:
                if hasattr(backend, 'close'):
                    await backend.close()
            except Exception:
                pass
        
        # 关闭STT后端
        for backend in self.stt_backends.values():
            try:
                if hasattr(backend, 'close'):
                    await backend.close()
            except Exception:
                pass
        
        # 关闭TTS后端
        for backend in self.tts_backends.values():
            try:
                if hasattr(backend, 'close'):
                    await backend.close()
            except Exception:
                pass