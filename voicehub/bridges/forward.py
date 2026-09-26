"""Reverse-proxy bridge for LAN-only agents.

Some agents bind 127.0.0.1 only (e.g. CherryStudio :23333 on .109). The
VoiceHub on .101 cannot reach them directly, so we run this thin proxy ON
the agent's host (.109), listening on 0.0.0.0, forwarding to 127.0.0.1.

Run on .109:
    # CherryStudio
    TARGET=http://127.0.0.1:23333 PORT=8801 python bridges/forward.py
    # WorkBuddy (once its API/bridge exists)
    TARGET=http://127.0.0.1:8800 PORT=8800 python bridges/forward.py

Then VoiceHub (.101) calls http://192.168.123.109:8801/...
"""
from __future__ import annotations
import os
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

TARGET = os.environ.get("TARGET", "http://127.0.0.1:23333")
PORT = int(os.environ.get("PORT", "8801"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
async def proxy(request: Request, path: str):
    url = f"{TARGET}/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.request(
            request.method, url, headers=headers,
            params=request.query_params, content=body,
        )
        return Response(
            content=r.content, status_code=r.status_code,
            headers=dict(r.headers),
        )


if __name__ == "__main__":
    print(f"forwarding {TARGET} -> 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
