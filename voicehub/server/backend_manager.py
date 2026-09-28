import os
from typing import Dict, Optional, Type, Union
from dotenv import load_dotenv
from .backends.base import LLMBackend, STTBackend, TTSBackend
from .backends.openai_compat import OpenAICompatBackend
from .backends.openclaw import OpenClawBackend
from .backends.streaming_whisper import StreamingWhisperSTTService
from .tts import TTSEngine


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
        
        # 当前活跃后端
        self.active_llm_backend: str = os.getenv("ACTIVE_LLM_BACKEND", "openai")
        self.active_stt_backend: str = os.getenv("ACTIVE_STT_BACKEND", "whisper")
        self.active_tts_backend: str = os.getenv("ACTIVE_TTS_BACKEND", "edge")
    
    def _init_llm_backends(self) -> None:
        """初始化LLM后端"""
        # OpenAI兼容后端（本机部署实际指向 Hermes OpenAI Bridge :8642，需要 API_SERVER_KEY）
        openai_endpoint = os.getenv("OPENAI_API_BASE", "http://localhost:8642/v1/chat/completions")
        # api_key 优先 OPENAI_API_KEY，其次 HERMES_KEY（Hermes gateway 的 API_SERVER_KEY）；
        # 二者皆缺时回落占位值，会触发 401 以便尽早暴露配置缺失。
        # model 默认 auto（Hermes 按 auto 路由），不要用 gpt-3.5-turbo（Hermes 会拒）。
        openai_api_key = os.getenv("OPENAI_API_KEY", os.getenv("HERMES_KEY", "sk-123456"))
        openai_model = os.getenv("OPENAI_MODEL", "auto")
        
        self.llm_backends["openai"] = OpenAICompatBackend(
            name="openai",
            endpoint=openai_endpoint,
            api_key=openai_api_key,
            model=openai_model
        )
        
        # OpenClaw后端
        openclaw_bin = os.getenv("OPENCLAW_BIN", "~/.local/bin/openclaw")
        self.llm_backends["openclaw"] = OpenClawBackend(
            name="openclaw",
            bin_path=openclaw_bin
        )
    
    def _init_stt_backends(self) -> None:
        """初始化STT后端"""
        # Faster Whisper后端
        whisper_model = os.getenv("WHISPER_MODEL", "small")
        whisper_device = os.getenv("WHISPER_DEVICE", "cpu")
        whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        
        self.stt_backends["whisper"] = StreamingWhisperSTTService(
            model=whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type
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