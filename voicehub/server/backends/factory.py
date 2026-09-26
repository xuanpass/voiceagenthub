"""Build AgentBackend instances from agents.yaml config."""
from __future__ import annotations
from .base import AgentBackend
from .openai_compat import OpenAICompatBackend
from .openclaw import OpenClawBackend
from .workbuddy import WorkBuddyBackend


def build_backends(config: dict) -> dict[str, AgentBackend]:
    backends: dict[str, AgentBackend] = {}
    for key, cfg in config["agents"].items():
        if cfg.get("disabled"):
            continue
        adapter = cfg.get("adapter")
        if adapter == "openclaw":
            backends[key] = OpenClawBackend(
                name=key, bin_path=cfg.get("bin_path"), session_prefix=cfg.get("session_prefix", "")
            )
        elif adapter == "workbuddy":
            backends[key] = WorkBuddyBackend(
                name=key,
                bin_path=cfg.get("bin_path"),
                model=cfg.get("model"),
                session_prefix=cfg.get("session_prefix", ""),
            )
        elif adapter == "openai_compat":
            backends[key] = OpenAICompatBackend(
                name=key,
                endpoint=cfg["endpoint"],
                api_key=cfg.get("api_key") or None,
                model=cfg.get("model", "auto"),
                auth_header=cfg.get("auth_header", "Authorization"),
                auth_scheme=cfg.get("auth_scheme", "Bearer"),
                session_prefix=cfg.get("session_prefix", ""),
            )
        else:
            raise ValueError(f"未知 adapter: {adapter} (agent={key})")
    return backends
