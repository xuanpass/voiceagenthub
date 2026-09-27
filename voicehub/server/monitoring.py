import time
import asyncio
from typing import Dict, Optional, List
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from fastapi import Response
from .backend_manager import BackendManager


# Prometheus指标
REQUEST_DURATION = Histogram(
    "voicehub_request_duration_seconds",
    "Duration of requests processed by VoiceHub",
    ["endpoint", "method", "status_code"]
)

ACTIVE_CONNECTIONS = Gauge(
    "voicehub_active_connections",
    "Number of active WebRTC connections"
)

ACTIVE_ROOMS = Gauge(
    "voicehub_active_rooms",
    "Number of active voice rooms"
)

LLM_REQUESTS_TOTAL = Counter(
    "voicehub_llm_requests_total",
    "Total number of LLM requests processed",
    ["backend", "status"]
)

STT_REQUESTS_TOTAL = Counter(
    "voicehub_stt_requests_total",
    "Total number of STT requests processed",
    ["backend", "status"]
)

TTS_REQUESTS_TOTAL = Counter(
    "voicehub_tts_requests_total",
    "Total number of TTS requests processed",
    ["backend", "status"]
)

ERRORS_TOTAL = Counter(
    "voicehub_errors_total",
    "Total number of errors encountered",
    ["error_type", "endpoint"]
)


class SystemMonitor:
    """系统监控工具类"""
    
    _instance: Optional['SystemMonitor'] = None
    
    def __new__(cls) -> 'SystemMonitor':
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_monitor()
        return cls._instance
    
    def _init_monitor(self) -> None:
        self.start_time = time.time()
        self.request_timers: Dict[str, float] = {}
        self.backend_manager = BackendManager()
        
        # 初始化指标
        ACTIVE_ROOMS.set(0)
        ACTIVE_CONNECTIONS.set(0)
    
    def start_request(self, request_id: str, endpoint: str, method: str) -> None:
        """记录请求开始时间"""
        self.request_timers[request_id] = time.time()
    
    def end_request(self, request_id: str, status_code: int) -> float:
        """记录请求结束时间并返回耗时"""
        if request_id in self.request_timers:
            duration = time.time() - self.request_timers[request_id]
            del self.request_timers[request_id]
            
            # 更新请求耗时指标
            REQUEST_DURATION.labels(
                endpoint='unknown',
                method='unknown',
                status_code=str(status_code)
            ).observe(duration)
            
            return duration
        return 0.0
    
    def update_connection_count(self, count: int) -> None:
        """更新活跃连接数"""
        ACTIVE_CONNECTIONS.set(count)
    
    def update_room_count(self, count: int) -> None:
        """更新活跃房间数"""
        ACTIVE_ROOMS.set(count)
    
    def record_llm_request(self, backend: str, status: str = "success") -> None:
        """记录LLM请求"""
        LLM_REQUESTS_TOTAL.labels(backend=backend, status=status).inc()
    
    def record_stt_request(self, backend: str, status: str = "success") -> None:
        """记录STT请求"""
        STT_REQUESTS_TOTAL.labels(backend=backend, status=status).inc()
    
    def record_tts_request(self, backend: str, status: str = "success") -> None:
        """记录TTS请求"""
        TTS_REQUESTS_TOTAL.labels(backend=backend, status=status).inc()
    
    def record_error(self, error_type: str, endpoint: str) -> None:
        """记录错误"""
        ERRORS_TOTAL.labels(error_type=error_type, endpoint=endpoint).inc()
    
    def get_system_stats(self) -> Dict:
        """获取系统统计信息"""
        total_connections = ACTIVE_CONNECTIONS._value.get()
        total_rooms = ACTIVE_ROOMS._value.get()
        
        return {
            "uptime": time.time() - self.start_time,
            "active_connections": total_connections,
            "active_rooms": total_rooms,
            "backend_status": {
                "llm": list(self.backend_manager.llm_backends.keys()),
                "stt": list(self.backend_manager.stt_backends.keys()),
                "tts": list(self.backend_manager.tts_backends.keys()),
                "active": {
                    "llm": self.backend_manager.active_llm_backend,
                    "stt": self.backend_manager.active_stt_backend,
                    "tts": self.backend_manager.active_tts_backend
                }
            }
        }
    
    async def get_health_status(self) -> Dict:
        """获取完整健康状态"""
        stats = self.get_system_stats()
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "start_time": self.start_time,
            "version": "1.0.0",
            **stats
        }
        
        # 检查后端健康状态
        backend_health = {}
        
        # 检查LLM后端
        for name, backend in self.backend_manager.llm_backends.items():
            try:
                await backend.warmup()
                backend_health[name] = {"status": "healthy"}
            except Exception as e:
                backend_health[name] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
        
        # 检查STT后端
        for name, backend in self.backend_manager.stt_backends.items():
            try:
                await backend.warmup()
                backend_health[name] = {"status": "healthy"}
            except Exception as e:
                backend_health[name] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
        
        # 检查TTS后端
        for name, backend in self.backend_manager.tts_backends.items():
            try:
                await backend.warmup()
                backend_health[name] = {"status": "healthy"}
            except Exception as e:
                backend_health[name] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
        
        health["backend_health"] = backend_health
        return health


# 全局监控实例
monitor = SystemMonitor()


def get_metrics() -> Response:
    """获取Prometheus格式的监控指标"""
    return Response(
        generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


def get_health_check() -> Dict:
    """获取健康检查结果"""
    return monitor.get_system_stats()


async def get_full_health() -> Dict:
    """获取完整健康检查"""
    return await monitor.get_health_status()