from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from dataclasses import dataclass
import asyncio


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


# 语音友好输出规范：LLM 回复会经 TTS 朗读，必须避免 markdown/emoji/阿拉伯数字
# 等"读不出来"的符号。借鉴 langflow-twilio-voice 的 voice-optimized prompting 约定。
VOICE_FORMAT_SYSTEM_PROMPT = (
    "你是语音助手，正在与用户进行实时语音对话。你的回复会被语音合成（TTS）朗读"
    "给用户，因此必须严格遵守以下输出规范：\n"
    "1. 简洁直接：用一两句话回答，不铺垫、不啰嗦、不重复。\n"
    "2. 数字一律用中文文字拼读：如「二十」而不是「20」，「百分之五十」而不是「50%」"
    "，「负三度」而不是「-3°C」。\n"
    "3. 禁止任何 Markdown/格式化符号：不用 *、#、-、**、`、```、> 等。\n"
    "4. 禁止 emoji、表情符号、颜文字。\n"
    "5. 禁止项目符号/编号列表/表格/代码块；用自然连贯的句子表达。\n"
    "6. 保持口语化、适合朗读的语气。\n"
)


class LLMBackend(ABC):
    """LLM后端抽象基类"""
    
    @abstractmethod
    async def warmup(self) -> None:
        """预加载模型/建立连接池"""
        pass

    @abstractmethod
    async def send(
        self,
        text: str,
        session_id: str,
        context: Optional[list[Message]] = None,
        abort: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """发送用户输入，返回流式响应片段
        
        Args:
            text: 用户输入文本
            session_id: 会话ID，用于隔离上下文
            context: 显式上下文消息（非 None 时不用会话历史）
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