#!/bin/bash

# VoiceHub 快速启动脚本

# 检查Python版本
PYTHON_REQUIRED="3.9"
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>/dev/null || echo "0.0.0")

if [ "$(printf "%s\n" "$PYTHON_REQUIRED" "$PYTHON_VERSION" | sort -V | head -n1)" != "$PYTHON_REQUIRED" ]; then
    echo "错误：需要Python >= $PYTHON_REQUIRED，当前版本：$PYTHON_VERSION"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 复制配置文件
if [ ! -f ".env" ]; then
    echo "复制环境变量配置文件..."
    cp .env.example .env
fi

if [ ! -f "config.yaml" ]; then
    echo "复制配置文件..."
    cp config.yaml.example config.yaml
fi

# 启动服务
echo "启动VoiceHub服务..."
export PYTHONPATH="$PYTHONPATH:$(pwd)"
python3 voicehub/server/main.py

# 暂停等待用户输入
read -p "服务已停止，按任意键退出..."