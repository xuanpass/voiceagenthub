"""声纹识别子系统：嵌入提取 + 在线聚类注册表 + 旁路处理器。

快速上手：
    from server.voiceprint import SpeakerEmbedder, VoiceprintRegistry, SpeakerIdProcessor
"""
from .embedder import SpeakerEmbedder
from .registry import VoiceprintRegistry, cosine
from .processor import SpeakerIdProcessor

__all__ = ["SpeakerEmbedder", "VoiceprintRegistry", "SpeakerIdProcessor", "cosine"]
