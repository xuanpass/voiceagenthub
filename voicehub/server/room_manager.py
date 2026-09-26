import asyncio
import uuid
from typing import Dict, Optional, Any
from .config import load_config
from .router import Router
from .orchestrator import Orchestrator
from .tts import TTSEngine
from .backend_manager import BackendManager


class VoiceRoom:
    """单个语音房间实例，包含独立的会话资源"""
    
    def __init__(self, room_id: str, config: dict):
        self.room_id = room_id
        self.config = config
        self.conv_ids: set[str] = set()
        self.active_agents: Dict[str, str] = {}  # connection_id -> active agent
        
        # 每个房间独立的路由、编排、TTS实例
        self.router = Router(config)
        self.tts = TTSEngine(
            {k: a["tts_voice"] for k, a in config["agents"].items() if not a.get("disabled")}
        )
        self.backend_manager = BackendManager()
        self.orchestrator = Orchestrator(
            self.backend_manager.llm_backends, 
            self.router
        )
        
        # 房间级别的会话历史
        self.session_histories: Dict[str, list[dict]] = {}
        
    def add_connection(self, connection_id: str) -> str:
        """添加连接到房间，生成唯一的conv_id"""
        conv_id = uuid.uuid4().hex[:12]
        self.conv_ids.add(conv_id)
        self.active_agents[connection_id] = self.router.default
        return conv_id
    
    def remove_connection(self, connection_id: str) -> None:
        """从房间移除连接"""
        if connection_id in self.active_agents:
            del self.active_agents[connection_id]
    
    def get_active_agent(self, connection_id: str) -> str:
        """获取连接当前活跃的agent"""
        return self.active_agents.get(connection_id, self.router.default)
    
    def set_active_agent(self, connection_id: str, agent: str) -> None:
        """设置连接当前活跃的agent"""
        self.active_agents[connection_id] = agent
    
    def get_session_history(self, conv_id: str) -> list[dict]:
        """获取会话历史"""
        return self.session_histories.setdefault(conv_id, [])
    
    def add_to_history(self, conv_id: str, role: str, content: str) -> None:
        """添加会话历史"""
        history = self.get_session_history(conv_id)
        history.append({"role": role, "content": content})
        # 限制历史长度
        max_history = self.config.get("server", {}).get("max_history", 24)
        if len(history) > max_history:
            del history[: len(history) - max_history]


class RoomManager:
    """全局房间管理器，管理所有语音房间"""
    
    _instance: Optional['RoomManager'] = None
    
    def __new__(cls) -> 'RoomManager':
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_rooms()
        return cls._instance
    
    def _init_rooms(self) -> None:
        self.config = load_config()
        self.rooms: Dict[str, VoiceRoom] = {}
        self.default_room_id = self.config.get("server", {}).get("default_room", "default")
        
    def get_room(self, room_id: Optional[str] = None) -> VoiceRoom:
        """获取指定房间，不存在则创建"""
        room_id = room_id or self.default_room_id
        if room_id not in self.rooms:
            self.rooms[room_id] = VoiceRoom(room_id, self.config)
        return self.rooms[room_id]
    
    def remove_room(self, room_id: str) -> None:
        """移除房间"""
        if room_id in self.rooms:
            del self.rooms[room_id]
    
    async def warmup_all(self) -> None:
        """预热所有房间的后端"""
        for room in self.rooms.values():
            await room.backend_manager.warmup_all()