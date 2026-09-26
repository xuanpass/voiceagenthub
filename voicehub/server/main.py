"""VoiceHub voice pipeline (Pipecat 1.11).

Audio path: SmallWebRTC transport <-> Pipecat pipeline
  mic -> SileroVAD -> WhisperSTT -> RouterProcessor (name routing + agent call)
       -> EdgeTTS (edge_tts + miniaudio decode) -> speaker

Run:  python -m server.main   (FastAPI + WebRTC signaling on :8765)

Note: Pipecat 1.11 split services into the core package; class paths differ
from older docs. See imports below for the verified 1.11 locations.
"""
from __future__ import annotations
import os

# .101 sits behind a network that cannot reach huggingface.co directly, so
# route model downloads (faster-whisper weights) through the HF China mirror.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
# The mirror doesn't support HF's xet (CAS) transport and it 401s; fall back
# to plain HTTP/LFS downloads so model loading works.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import asyncio
import logging
import uuid
import json
import time
from functools import wraps

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from collections import deque
from .monitoring import monitor, get_metrics, get_health_check, get_full_health

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TranscriptionFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from .backends.streaming_whisper import StreamingWhisperSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    ConnectionMode,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
    SmallWebRTCPatchRequest,
    IceCandidate,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from .config import load_config
from .backend_manager import BackendManager
from .room_manager import RoomManager
from .orchestrator import Orchestrator
from .router import Router
from .tts import TTSEngine, TTS_CHANNELS, TTS_SAMPLE_RATE, split_sentences

logger = logging.getLogger("voicehub")

# 日志缓存
log_buffer = deque(maxlen=100)
log_websockets: set[WebSocket] = set()

# 自定义日志处理器
class WebSocketLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_buffer.append(log_entry)
        # 发送给所有连接的管理客户端
        for ws in log_websockets:
            asyncio.create_task(ws.send_text(json.dumps({
                "type": "log",
                "message": log_entry
            })))


class RouterProcessor(FrameProcessor):
    """Intercepts the user transcript, routes to the right agent, and emits
    per-sentence TTS audio frames with that agent's voice."""

    def __init__(self, router: Router, backends: dict, tts: TTSEngine, orchestrator: Orchestrator, conv_id: str):
        super().__init__()
        self.router = router
        self.backends = backends
        self.tts = tts
        self.orchestrator = orchestrator
        # unique per WebRTC connection -> isolates each conversation's history
        self.conv_id = conv_id
        # per-connection active agent: avoids clobbering when several calls
        # share the single Router instance on the server.
        self.active = router.default
        # barge-in: set() when the user starts a new utterance while a previous
        # LLM call is still streaming -> adapters drop the in-flight result.
        self._active_abort: Optional[asyncio.Event] = None

    def _session(self, agent_key: str) -> str:
        # session_prefix comes from agents.yaml (per-agent); conv_id scopes it
        # to this conversation, so each agent keeps memory within a call and
        # stays isolated across calls.
        prefix = getattr(self.backends.get(agent_key), "session_prefix", "")
        return f"{prefix}{self.conv_id}"

    async def _speak(self, text: str, agent_key: str) -> None:
        for sentence in split_sentences(text):
            pcm = await self.tts.synth_pcm(sentence, agent_key)
            if pcm:
                await self.push_frame(
                    TTSAudioRawFrame(
                        audio=pcm,
                        sample_rate=TTS_SAMPLE_RATE,
                        num_channels=TTS_CHANNELS,
                    ),
                    FrameDirection.DOWNSTREAM,
                )

    async def process_frame(self, frame, direction: FrameDirection):
        # STT emits TranscriptionFrame; intercept it, do NOT forward downstream.
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return
            # barge-in: user started a new utterance -> abort the in-flight LLM
            # stream from the previous turn, then open a fresh one for this turn.
            if self._active_abort is not None:
                self._active_abort.set()
            turn_abort = asyncio.Event()
            self._active_abort = turn_abort
            res = self.router.route(text, active=self.active)
            agent_key = res.agent
            sess = self._session(agent_key)
            # P1: only speak when final transcription is received
            is_final = not hasattr(frame, "is_partial") or not frame.is_partial
            if is_final:
                self.active = res.agent
                if res.collab and len(res.collab_agents) >= 2:
                    agents = res.collab_agents
                    if res.collab_mode == "handoff":
                        async for who, piece in self.orchestrator.handoff(
                            agents, res.text, sess, abort=turn_abort
                        ):
                            if turn_abort.is_set():
                                break
                            await self._speak(piece, who)
                    elif res.collab_mode == "review":
                        # collab_agents ordered by utterance: "<reviewer> 评审 <target>"
                        # -> target answers first, reviewer critiques last.
                        target = agents[-1]
                        reviewer = agents[0]
                        async for who, piece in self.orchestrator.review(
                            target, reviewer, res.text, sess, abort=turn_abort
                        ):
                            if turn_abort.is_set():
                                break
                            await self._speak(piece, who)
                    else:
                        async for who, piece in self.orchestrator.roundtable(
                            agents, res.text, sess, abort=turn_abort
                        ):
                            if turn_abort.is_set():
                                break
                            await self._speak(piece, who)
                else:
                    async for piece in self.backends[agent_key].send(
                        res.text, sess, None, turn_abort
                    ):
                        if turn_abort.is_set():
                            break
                        await self._speak(piece, agent_key)
            return
        await super().process_frame(frame, direction)


def _build_stack():
    config = load_config()
    backend_manager = BackendManager()
    room_manager = RoomManager()
    stt_cfg = config.get("stt", {})
    srv = config.get("server", {})
    return config, backend_manager, room_manager, stt_cfg, srv


_CONFIG, _BACKEND_MANAGER, _ROOM_MANAGER, _STT, _SRV = _build_stack()


async def _warmup_backends() -> None:
    """Warm up LLM models on startup so the first real user turn is fast.

    Fires a 1-token request to every openai_compat agent in parallel, which
    loads model weights / establishes TCP connections. Runs in background so
    the WebRTC handshake is never blocked.
    """
    await _BACKEND_MANAGER.warmup_all()


async def _bot_background(webrtc_connection, room_id: Optional[str] = None):
    """Schedule the per-connection bot without blocking the SDP answer.

    SmallWebRTCRequestHandler awaits the callback, and _bot runs the pipeline
    until the peer disconnects. If we awaited it here the HTTP /offer response
    (carrying the answer SDP) would only return after the call ended - a
    deadlock, since the browser needs the answer to open the connection.
    """
    asyncio.create_task(_bot(webrtc_connection, room_id))


async def _bot(webrtc_connection: SmallWebRTCConnection, room_id: Optional[str] = None) -> None:
    """Per-connection bot: build pipeline and run until the peer disconnects."""
    # Get the target room
    room = _ROOM_MANAGER.get_room(room_id)
    
    # Generate connection ID and conv ID
    connection_id = uuid.uuid4().hex[:8]
    conv_id = room.add_connection(connection_id)
    
    # Get backend instances from room manager
    stt = _BACKEND_MANAGER.get_stt_backend()
    current_router = room.router
    current_orchestrator = room.orchestrator
    current_tts = room.tts
    
    # Get active agent for this connection
    active_agent = room.get_active_agent(connection_id)
    
    # Create router processor with room context
    router_proc = RouterProcessor(
        router=current_router,
        backends=_BACKEND_MANAGER.llm_backends,
        tts=current_tts,
        orchestrator=current_orchestrator,
        conv_id=conv_id
    )
    
    # Override active agent for this connection
    router_proc.active = active_agent
    
    params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    )
    transport = SmallWebRTCTransport(webrtc_connection, params)
    pipeline = Pipeline([transport.input(), stt, router_proc, transport.output()])
    task = PipelineTask(pipeline)
    
    try:
        await PipelineRunner().run(task)
    finally:
        # Clean up resources
        await stt.close()
        room.remove_connection(connection_id)


app = FastAPI(title="VoiceHub", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP请求监控中间件
@app.middleware("http")
async def request_monitoring_middleware(request: Request, call_next):
    request_id = f"{request.method}:{request.url.path}:{time.time()}"
    monitor.start_request(request_id, request.url.path, request.method)
    
    try:
        response = await call_next(request)
        monitor.end_request(request_id, response.status_code)
        return response
    except Exception as e:
        monitor.end_request(request_id, 500)
        raise e

_ICE = [IceServer(urls=["stun:stun.l.google.com:19302"])]
_HANDLER = SmallWebRTCRequestHandler(
    ice_servers=_ICE,
    host=_SRV.get("host", "0.0.0.0"),
    connection_mode=ConnectionMode.MULTIPLE,
)


@app.post("/offer")
async def offer(request: Request):
    req = SmallWebRTCRequest.from_dict(await request.json())
    answer = await _HANDLER.handle_web_request(req, _bot_background)
    return JSONResponse(answer)


@app.patch("/offer")
async def ice_candidate(request: Request):
    # Trickle ICE candidates from the browser to the active peer connection.
    # Coerce raw JSON into IceCandidate objects (handler reads .candidate/.sdp_mid).
    data = await request.json()
    cands = [
        IceCandidate(
            c.get("candidate"),
            c.get("sdp_mid"),
            c.get("sdp_mline_index"),
        )
        for c in data.get("candidates", [])
        if c.get("candidate")
    ]
    patch = SmallWebRTCPatchRequest(pc_id=data["pc_id"], candidates=cands)
    await _HANDLER.handle_patch_request(patch)
    return JSONResponse({"status": "success"})


@app.get("/room/{room_id}/offer")
async def offer_with_room(room_id: str, request: Request):
    """Create WebRTC offer for a specific room"""
    req = SmallWebRTCRequest.from_dict(await request.json())
    answer = await _HANDLER.handle_web_request(req, lambda conn: _bot_background(conn, room_id))
    return JSONResponse(answer)


@app.get("/admin")
async def admin_panel():
    """Serve the admin dashboard"""
    admin_html = Path(__file__).parent / "admin.html"
    if admin_html.exists():
        return FileResponse(admin_html)
    return JSONResponse({"error": "admin.html not found"}, status_code=404)


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    total_connections = 0
    for room in _ROOM_MANAGER.rooms.values():
        total_connections += len(room.active_agents)
    
    return JSONResponse({
        "rooms": len(_ROOM_MANAGER.rooms),
        "connections": total_connections,
        "llm_backends": list(_BACKEND_MANAGER.llm_backends.keys()),
        "active_llm_backend": _BACKEND_MANAGER.active_llm_backend,
        "stt_backends": list(_BACKEND_MANAGER.stt_backends.keys()),
        "active_stt_backend": _BACKEND_MANAGER.active_stt_backend,
        "tts_backends": list(_BACKEND_MANAGER.tts_backends.keys()),
        "active_tts_backend": _BACKEND_MANAGER.active_tts_backend
    })


@app.get("/api/logs")
async def get_logs():
    """Get recent logs"""
    return "\n".join(log_buffer)


@app.websocket("/ws/admin")
async def admin_websocket(websocket: WebSocket):
    """Admin WebSocket for real-time updates"""
    await websocket.accept()
    log_websockets.add(websocket)
    
    try:
        # 发送初始日志
        await websocket.send_text(json.dumps({
            "type": "initial_logs",
            "logs": list(log_buffer)
        }))
        
        # 定期发送统计数据
        while True:
            await asyncio.sleep(1)
            stats = await get_stats()
            await websocket.send_text(json.dumps({
                "type": "stats_update",
                "stats": await stats.json()
            }))
    except WebSocketDisconnect:
        log_websockets.remove(websocket)
    except Exception:
        if websocket in log_websockets:
            log_websockets.remove(websocket)


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return get_health_check()


@app.get("/health/full")
async def full_health_check():
    """Full detailed health check with backend status"""
    return await get_full_health()


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return get_metrics()


# 统一异常处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    monitor.record_error(f"http_{exc.status_code}", request.url.path)
    return JSONResponse(
        {"error": exc.detail, "status_code": exc.status_code},
        status_code=exc.status_code
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    monitor.record_error("internal_error", request.url.path)
    logger.exception(f"Unhandled exception at {request.url.path}")
    return JSONResponse(
        {"error": "Internal server error", "status_code": 500},
        status_code=500
    )


_CLIENT_HTML = Path(__file__).parent / "client.html"


@app.get("/")
async def index():
    # Serve the browser/phone WebRTC client.
    if _CLIENT_HTML.exists():
        return FileResponse(_CLIENT_HTML)
    return JSONResponse({"error": "client.html not found"}, status_code=404)


@app.get("/rooms")
async def list_rooms():
    """List all active voice rooms"""
    return JSONResponse({
        "rooms": list(_ROOM_MANAGER.rooms.keys()),
        "default_room": _ROOM_MANAGER.default_room_id
    })


@app.get("/room/{room_id}")
async def get_room(room_id: str):
    """Get details of a specific room"""
    room = _ROOM_MANAGER.get_room(room_id)
    return JSONResponse({
        "room_id": room.room_id,
        "active_connections": len(room.active_agents),
        "total_conversations": len(room.conv_ids)
    })


@app.post("/room/{room_id}/offer")
async def offer_with_room(room_id: str, request: Request):
    """Create WebRTC offer for a specific room"""
    req = SmallWebRTCRequest.from_dict(await request.json())
    answer = await _HANDLER.handle_web_request(req, lambda conn: _bot_background(conn, room_id))
    return JSONResponse(answer)


@app.post("/offer")
async def offer(request: Request):
    """Create WebRTC offer for the default room"""
    req = SmallWebRTCRequest.from_dict(await request.json())
    answer = await _HANDLER.handle_web_request(req, _bot_background)
    return JSONResponse(answer)


if __name__ == "__main__":
    import uvicorn

    ssl_kwargs = {}
    key = os.environ.get("VOICEHUB_SSL_KEY")
    cert = os.environ.get("VOICEHUB_SSL_CERT")
    if key and cert:
        ssl_kwargs = {"ssl_keyfile": key, "ssl_certfile": cert}
        print(f"[VoiceHub] HTTPS on :{_SRV.get('port', 8765)} (cert={cert})")
    else:
        print(f"[VoiceHub] HTTP on :{_SRV.get('port', 8765)} (set VOICEHUB_SSL_* for HTTPS)")
    # Schedule LLM warmup in a background task so the first user turn is fast.
    # uvicorn runs the event loop; create the task inside an on-startup hook
    # so it shares the loop the server actually runs on.
    @app.on_event("startup")
    async def _on_startup_warmup():
        asyncio.create_task(_warmup_backends())
        asyncio.create_task(_cleanup_zombie_tasks())


# 僵尸任务清理后台任务
async def _cleanup_zombie_tasks():
    """定期清理超时任务和僵尸连接"""
    while True:
        try:
            # 每30秒执行一次清理
            await asyncio.sleep(30)
            
            # 更新活跃连接和房间统计
            total_connections = 0
            for room in _ROOM_MANAGER.rooms.values():
                total_connections += len(room.active_agents)
            monitor.update_connection_count(total_connections)
            monitor.update_room_count(len(_ROOM_MANAGER.rooms))
            
            # 这里可以添加更多清理逻辑：
            # 1. 清理长时间空闲的会话
            # 2. 终止超时的LLM请求
            # 3. 清理无响应的连接
            
            logger.debug(f"Cleanup completed: {len(_ROOM_MANAGER.rooms)} rooms, {total_connections} active connections")
            
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(60)

    uvicorn.run(
        app,
        host=_SRV.get("host", "0.0.0.0"),
        port=_SRV.get("port", 8765),
        **ssl_kwargs,
    )
