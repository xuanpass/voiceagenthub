# 🎙️ VoiceHub 局域网多智能体实时语音中枢

一个基于Pipecat 1.11构建的高性能实时语音协作系统，支持多智能体协同、多房间隔离、可视化管理。

## ✨ 核心特性

- 🚀 **实时语音管线**：基于Pipecat 1.11的低延迟语音交互
- 🧩 **多后端支持**：可切换STT/TTS/LLM多种AI服务提供商
- 🏠 **多房间隔离**：支持同时创建多个独立语音会话
- 🎨 **可视化管理**：内置Web管理面板，实时监控系统状态
- 🎯 **智能协作**：支持handoff/review/roundtable多种协作模式
- ⚡ **打断机制**：支持语音打断，实时中断当前响应
- 📊 **运维监控**：集成Prometheus监控和健康检查端点

## 🚀 快速开始

### 方式一：直接启动

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件配置你的API密钥和参数
```

3. **启动服务**
```bash
python -m voicehub.server.main
```

4. **访问服务**
- 客户端：http://localhost:8765
- 管理面板：http://localhost:8765/admin

### 方式二：Docker部署

```bash
# 构建镜像
docker build -t voicehub .

# 启动容器
docker run -d -p 8765:8765 --name voicehub voicehub
```

## 📁 项目结构

```
.
├── voicehub/
│   ├── server/
│   │   ├── backends/          # 后端实现
│   │   │   ├── base.py        # 抽象基类
│   │   │   ├── openai_compat.py # OpenAI兼容后端
│   │   │   ├── openclaw.py    # OpenClaw后端
│   │   │   └── streaming_whisper.py # 流式STT
│   │   ├── admin.html         # Web管理面板
│   │   ├── backend_manager.py # 后端管理
│   │   ├── client.html        # 语音客户端
│   │   ├── main.py            # 主服务
│   │   ├── monitoring.py      # 监控系统
│   │   ├── orchestrator.py    # 协作编排
│   │   ├── room_manager.py    # 房间管理
│   │   ├── router.py          # 路由逻辑
│   │   ├── tts.py             # TTS引擎
│   │   └── streaming_whisper.py # 流式STT
│   └── config.yaml.example    # 配置示例
├── .env.example               # 环境变量示例
├── config.yaml.example        # 配置文件示例
├── requirements.txt           # 依赖列表
└── README.md                 # 项目文档
```

## 🎯 使用指南

### 基本语音交互

访问客户端页面后，允许麦克风权限即可开始语音交互：
1. 说话即可触发语音识别
2. 实时语音转录显示在页面上
3. 智能体响应会自动语音播报
4. 说话可打断当前响应

### 管理面板使用

访问`/admin`路径进入管理面板：
- **系统概览**：查看实时连接数、房间数、后端状态
- **房间管理**：创建/删除房间，快速进入指定房间
- **日志监控**：实时查看系统运行日志
- **统计数据**：查看请求耗时、错误率等监控指标

### 多房间使用

通过管理面板创建房间后，可通过以下方式加入指定房间：
```
http://localhost:8765/?room=your-room-id
```

或者通过API创建房间：
```bash
curl http://localhost:8765/room/your-room-id
```

## 🔧 配置说明

### 环境变量配置

| 变量名 | 说明 | 默认值 |
|------|------|------|
| ACTIVE_LLM_BACKEND | 活跃LLM后端 | openai |
| OPENAI_API_BASE | OpenAI API地址 | http://localhost:8642/v1/chat/completions |
| OPENAI_API_KEY | OpenAI API密钥 | sk-123456 |
| WHISPER_MODEL | Whisper模型 | small |
| WHISPER_DEVICE | 运行设备 | cpu |
| TTS_VOICE_HERMES | Hermes音色 | zh-CN-YunxiNeural |

### 配置文件

可通过`config.yaml`文件配置更详细的系统参数，包括：
- 智能体配置
- 路由规则
- 后端列表
- 服务器参数

## 📡 API端点

### 基础端点
- `GET /` - 语音客户端页面
- `GET /admin` - 管理面板页面
- `POST /offer` - 创建WebRTC连接
- `POST /room/{room_id}/offer` - 为指定房间创建WebRTC连接

### 监控端点
- `GET /health` - 简单健康检查
- `GET /health/full` - 完整健康检查
- `GET /metrics` - Prometheus监控指标

### 管理API
- `GET /rooms` - 列出所有活跃房间
- `GET /room/{room_id}` - 获取房间详情
- `GET /api/stats` - 获取系统统计数据
- `GET /api/logs` - 获取系统日志

## 🧩 扩展开发

### 添加新的LLM后端

1. 在`backends/base.py`中实现`LLMBackend`抽象接口
2. 在`backend_manager.py`中注册新后端
3. 更新配置文件和环境变量

### 添加新的STT后端

1. 实现`STTBackend`抽象接口
2. 注册到后端管理器

### 添加新的TTS后端

1. 实现`TTSBackend`抽象接口
2. 注册到后端管理器

## 🐛 故障排除

### 常见问题

1. **麦克风权限问题**：确保浏览器已授予麦克风权限
2. **端口被占用**：修改`PORT`环境变量或配置文件中的端口
3. **依赖安装失败**：使用国内PyPI镜像加速：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`
4. **SSL证书错误**：如需HTTPS，配置`VOICEHUB_SSL_KEY`和`VOICEHUB_SSL_CERT`环境变量

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！