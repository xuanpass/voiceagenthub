#!/bin/bash

# VoiceHub 快速启动脚本

# 检查Python版本
PYTHON_REQUIRED="3.9"
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>/dev/null || echo "0.0.0")

if [ "$(printf "%s\n" "$PYTHON_REQUIRED" "$PYTHON_VERSION" | sort -V | head -n1)" != "$PYTHON_REQUIRED" ]; then
    echo "错误：需要Python >= $PYTHON_REQUIRED，当前版本：$PYTHON_VERSION"
    exit 1
fi

# 检查虚拟环境（优先使用已装好依赖的 .venv，否则回退到 venv）
if [ -d ".venv" ]; then
    VENV=".venv"
else
    VENV="venv"
    if [ ! -d "$VENV" ]; then
        echo "创建虚拟环境..."
        python3 -m venv "$VENV"
    fi
fi

# 激活虚拟环境
source "$VENV/bin/activate"

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 复制环境变量配置（实际生效文件为 voicehub/.env, 由 config.py 读取）
if [ ! -f "voicehub/.env" ]; then
    echo "复制环境变量配置文件..."
    cp voicehub/.env.example voicehub/.env
fi

# 启动服务（main.py 使用相对导入，必须以包方式运行：python -m server.main）
echo "启动VoiceHub服务..."
export PYTHONPATH="$PYTHONPATH:$(dirname "$0")/voicehub"
cd "$(dirname "$0")/voicehub" || exit 1
exec python -u -m server.main

# 暂停等待用户输入
read -p "服务已停止，按任意键退出..."