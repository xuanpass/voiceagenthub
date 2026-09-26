#!/usr/bin/env bash
# VoiceHub 启动脚本（.101 部署用）
# 用途：加载环境变量 -> 激活 pyenv -> 前台拉起 uvicorn（被 systemd 托管）
set -eo pipefail
# Note: -u (nounset) intentionally omitted. .env may reference vars like
# ${GATEWAY_TOKEN} that are resolved at agent runtime, not shell source-time.
# Under -u, sourcing .env would abort on the first undefined reference.

cd "$(dirname "$0")"

# --- 环境变量（.env 优先级高于 .env.example）---
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# --- 激活虚拟环境（pipecat 等依赖装在 .venv）---
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
  echo "[start.sh] activated .venv: $(python --version 2>&1)"
fi

# --- HF 镜像（.101 无法直连 huggingface.co）---
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

# --- HTTPS 证书（自签，默认在项目目录内）---
export VOICEHUB_SSL_KEY="${VOICEHUB_SSL_KEY:-$(pwd)/key.pem}"
export VOICEHUB_SSL_CERT="${VOICEHUB_SSL_CERT:-$(pwd)/cert.pem}"

echo "[start.sh] launching VoiceHub on :${VOICEHUB_PORT:-8765}"
exec python -m server.main
