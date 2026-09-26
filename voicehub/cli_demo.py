"""Text-mode demo / smoke test for VoiceHub (no mic, no Pipecat needed).

Exercises the real agent backends: read a line from stdin, route by name,
stream the agent's reply to stdout. Use this to verify OpenClaw/Hermes/
CherryStudio connectivity on .101 before wiring the audio pipeline.

    python cli_demo.py            # text chat, routes by name
    python cli_demo.py --speak    # also speak replies via Edge TTS
"""
from __future__ import annotations
import argparse
import asyncio
import sys

from server.config import load_config
from server.backends.factory import build_backends
from server.router import Router
from server.tts import TTSEngine
from server.orchestrator import Orchestrator


async def run(speak: bool):
    config = load_config()
    backends = build_backends(config)
    router = Router(config)
    tts = TTSEngine({k: a["tts_voice"] for k, a in config["agents"].items()})
    orch = Orchestrator(backends, router)

    print("VoiceHub CLI demo. 叫名字唤醒 agent (小克/赫尔墨斯/小樱/小助); 空行退出。\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        res = router.route(line)
        print(f"[{res.agent}] ", end="", flush=True)
        buf = []
        if res.collab and len(res.collab_agents) >= 2:
            async for who, piece in orch.roundtable(res.collab_agents, res.text, "cli"):
                print(piece, end="", flush=True)
                buf.append(piece)
        else:
            async for piece in backends[res.agent].send(res.text, f"cli:{res.agent}"):
                print(piece, end="", flush=True)
                buf.append(piece)
        print()
        if speak and buf:
            audio = b"".join([c async for c in tts.synth("".join(buf), res.agent)])
            # play with ffplay/afplay if available; otherwise skip
            print(f"[audio {len(audio)} bytes for {res.agent}]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speak", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args.speak))
