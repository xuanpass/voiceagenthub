"""Load agents.yaml with ${ENV} substitution. Also loads a nearby .env file."""
from __future__ import annotations
import os
import re
import yaml
from pathlib import Path

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def _load_dotenv(path: Path) -> None:
    """Minimal .env parser (no external dependency)."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _sub_env(value):
    if not isinstance(value, str):
        return value
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


def _walk(node):
    if isinstance(node, dict):
        return {k: _walk(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v) for v in node]
    return _sub_env(node)


def load_config(path: str | os.PathLike = None) -> dict:
    cfg_path = Path(path or (Path(__file__).parent.parent / "agents.yaml"))
    dotenv_path = cfg_path.parent / ".env"
    if dotenv_path.exists():
        _load_dotenv(dotenv_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return _walk(data)
