from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from dataclasses import dataclass
import asyncio


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


class LLMBackend(ABC):
    """LLM后端抽象基类"""
    
    @abstractmethod
    async def warmup(self) -> None:
        """预加载模型/建立连接池"""
        pass

    @abstractmethod
    async def send(self, text: str, session_id: str, abort: Optional[asyncio.Event] = None) -> AsyncIterator[str]:
        """发送用户输入，返回流式响应片段
        
        Args:
            text: 用户输入文本
            session_id: 会话ID，用于隔离上下文
            abort: 异步取消事件，用于终止请求
        
        Returns:
            流式返回响应文本片段
        """
        pass

    @abstractmethod
    async def abort(self) -> None:
        """终止当前正在处理的请求"""
        pass


# Alias for backward compatibility
AgentBackend = LLMBackend


class STTBackend(ABC):
    """STT后端抽象基类"""
    
    @abstractmethod
    async def warmup(self) -> None:
        """预加载模型/初始化转录服务"""
        pass

    @abstractmethod
    async def transcribe(self, audio_chunk: bytes, final: bool = False) -> AsyncIterator[tuple[str, bool]]:
        """流式转录音频块
        
        Args:
            audio_chunk: 音频字节数据
            final: 是否为最后一个音频块
        
        Returns:
            流式返回(转录文本, 是否为最终结果)元组
        """
        pass

    @abstractmethod
    async def abort(self) -> None:
        """终止当前转录任务"""
        pass


class TTSBackend(ABC):
    """TTS后端抽象基类"""
    
    @abstractmethod
    async def warmup(self) -> None:
        """预加载TTS引擎/建立连接"""
        pass

    @abstractmethod
    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """将文本合成为音频流
        
        Args:
            text: 要合成的文本
        
        Returns:
            流式返回音频字节数据
        """
        pass

    @abstractmethod
    async def abort(self) -> None:
        """终止当前合成任务"""
        pass