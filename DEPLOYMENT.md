# 🚀 VoiceHub 部署指南

## 目录

1. [环境要求](#环境要求)
2. [快速部署](#快速部署)
   - [方式一：手动部署](#方式一手动部署)
   - [方式二：Docker Compose部署](#方式二docker-compose部署)
3. [生产环境部署最佳实践](#生产环境部署最佳实践)
4. [配置详解](#配置详解)
5. [监控与运维](#监控与运维)
6. [常见问题](#常见问题)

## 环境要求

- **操作系统**: Linux/macOS/Windows（推荐Linux）
- **Python**: 3.9及以上
- **内存**: 至少2GB（推荐4GB）
- **存储**: 至少1GB空闲空间
- **网络**: 可访问外部AI服务（如OpenAI API、阿里云TTS等）

## 快速部署

### 方式一：手动部署

#### 1. 克隆项目
```bash
git clone https://github.com/your-repo/voicehub.git
cd voicehub
```

#### 2. 安装依赖
```bash
pip install -r requirements.txt
```

#### 3. 配置环境
```bash
# 复制环境变量配置
cp .env.example .env

# 复制配置文件
cp config.yaml.example config.yaml
```

#### 4. 编辑配置

编辑`.env`文件配置你的API密钥和参数：
```bash
nano .env
```

编辑`config.yaml`文件配置智能体和路由规则：
```bash
nano config.yaml
```

#### 5. 启动服务
```bash
python -m voicehub.server.main
```

#### 6. 访问服务
- 客户端：http://localhost:8765
- 管理面板：http://localhost:8765/admin

### 方式二：Docker Compose部署

#### 1. 安装Docker和Docker Compose

参考官方文档安装Docker：https://docs.docker.com/get-docker/

#### 2. 创建配置文件
```bash
cp .env.example .env
cp config.yaml.example config.yaml
```

#### 3. 启动服务
```bash
docker-compose up -d
```

#### 4. 查看日志
```bash
docker-compose logs -f
```

#### 5. 停止服务
```bash
docker-compose down
```

## 生产环境部署最佳实践

### 1. 使用反向代理

推荐使用Nginx作为反向代理，配置SSL证书：

```nginx
server {
    listen 80;
    server_name voicehub.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name voicehub.example.com;
    
    ssl_certificate /path/to/your/cert.pem;
    ssl_certificate_key /path/to/your/key.pem;
    
    location / {
        proxy_pass http://localhost:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. 使用Systemd服务

创建`/etc/systemd/system/voicehub.service`：

```ini
[Unit]
Description=VoiceHub AI Voice Hub
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/voicehub
Environment=PYTHONPATH=/path/to/voicehub
ExecStart=/usr/bin/python3 -m voicehub.server.main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable voicehub
sudo systemctl start voicehub
```

### 3. 监控配置

#### Prometheus监控

1. 创建`monitoring/prometheus.yml`配置文件：
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'voicehub'
    static_configs:
      - targets: ['voicehub:8765']
```

2. 启动Prometheus和Grafana：
```bash
docker-compose up prometheus grafana -d
```

#### 健康检查

服务内置了健康检查端点：
- `GET /health` - 简单健康检查
- `GET /health/full` - 完整健康检查

### 4. 安全配置

- 使用HTTPS协议
- 配置API密钥环境变量，不要硬编码
- 限制管理面板访问IP
- 定期更新依赖包

## 配置详解

### 环境变量配置

| 变量名 | 说明 | 默认值 | 是否必填 |
|------|------|------|------|
| ACTIVE_LLM_BACKEND | 活跃LLM后端 | openai | 否 |
| OPENAI_API_BASE | OpenAI API地址 | http://localhost:8642/v1/chat/completions | 否 |
| OPENAI_API_KEY | OpenAI API密钥 | sk-123456 | 是 |
| WHISPER_MODEL | Whisper模型 | small | 否 |
| WHISPER_DEVICE | 运行设备 | cpu | 否 |
| TTS_VOICE_HERMES | Hermes音色 | zh-CN-YunxiNeural | 否 |
| TTS_VOICE_CHERRY | Cherry音色 | zh-CN-XiaoxiaoNeural | 否 |
| PORT | 服务端口 | 8765 | 否 |
| VOICEHUB_SSL_KEY | SSL密钥路径 | 无 | 否 |
| VOICEHUB_SSL_CERT | SSL证书路径 | 无 | 否 |

### 配置文件（config.yaml）

```yaml
server:
  host: 0.0.0.0
  port: 8765
  max_history: 24
  default_room: default

llm_backends:
  openai:
    endpoint: http://localhost:8642/v1/chat/completions
    api_key: sk-123456
    model: gpt-3.5-turbo

stt:
  model: small
  device: cpu

tts:
  active_backend: edge
  voices:
    hermes: zh-CN-YunxiNeural

agents:
  hermes:
    enabled: true
    tts_voice: zh-CN-YunxiNeural
```

## 监控与运维

### 查看实时日志
```bash
# 手动部署
python -m voicehub.server.main

# Docker部署
docker-compose logs -f voicehub
```

### 查看系统状态
```bash
# 健康检查
curl http://localhost:8765/health

# 完整健康检查
curl http://localhost:8765/health/full

# Prometheus指标
curl http://localhost:8765/metrics
```

### 重启服务
```bash
# 手动部署
Ctrl+C 后重新启动

# Systemd部署
sudo systemctl restart voicehub

# Docker部署
docker-compose restart voicehub
```

## 常见问题

### Q: 麦克风权限被拒绝

A: 确保浏览器已授予麦克风权限，或者在命令行启动时授予权限：
```bash
# Linux
chromium --use-fake-ui-for-media-stream

# macOS
safari -allowAudioCapture
```

### Q: 端口被占用

A: 修改`PORT`环境变量或配置文件中的端口：
```bash
export PORT=8766
```

### Q: 无法连接到OpenAI API

A: 检查网络连接，确保API密钥正确，或者使用代理：
```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### Q: 如何添加新的智能体

A: 编辑`config.yaml`文件，添加新的智能体配置：

```yaml
agents:
  new_agent:
    enabled: true
    tts_voice: zh-CN-YunyangNeural
    session_prefix: new_agent_
```

## 升级指南

### 升级到最新版本

```bash
# 拉取最新代码
git pull

# 更新依赖
pip install -r requirements.txt --upgrade

# 重启服务
sudo systemctl restart voicehub
```

## 备份策略

### 需要备份的文件
- `config.yaml` - 配置文件
- `.env` - 环境变量
- `logs/` - 日志文件
- `data/` - 会话数据（如存在）

### 自动备份脚本

```bash
#!/bin/bash
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)

tar -czf $BACKUP_DIR/voicehub_backup_$DATE.tar.gz \
    /path/to/voicehub/config.yaml \
    /path/to/voicehub/.env \
    /path/to/voicehub/logs

# 保留最近7天的备份
find $BACKUP_DIR -type f -mtime +7 -delete
```