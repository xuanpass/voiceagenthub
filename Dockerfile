FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建数据目录
RUN mkdir -p data logs

# 暴露端口
EXPOSE 8765

# 设置环境变量
ENV PYTHONPATH=/app
ENV VOICEHUB_SSL_KEY=""
ENV VOICEHUB_SSL_CERT=""

# 启动命令
CMD ["python", "voicehub/server/main.py"]