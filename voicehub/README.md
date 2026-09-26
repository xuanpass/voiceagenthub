# VoiceHub — 局域网多智能体实时语音中枢

叫不同名字唤醒不同智能体，实时语音交互，可协作。星型拓扑：VoiceHub 作中枢路由，agent 互不直连。

## 架构

```
浏览器/手机 (WebRTC 麦克风)  ──▶  Pipecat 语音层 (VAD + faster-whisper + Edge TTS)
                                        │ 转写文本
                                        ▼
                                  路由层 Router (名字匹配 + 协作意图识别)
                                        │ 统一接口 AgentBackend.send()
                                        ▼
                      适配层: OpenClaw(CLI) | Hermes/CherryStudio/WorkBuddy(openai_compat)
```

## 已验证的接入（2026-09-25）

| Agent | 接入 | 状态 |
|---|---|---|
| OpenClaw (.101) | `openclaw agent -m ... --json` CLI subprocess（走 Gateway 自动鉴权） | ✅ 实测 |
| Hermes (.101) | HTTP `:8642/v1/chat/completions` + `Authorization: Bearer <API_SERVER_KEY>` | ✅ 实测 |
| CherryStudio (.109) | HTTP OpenAI 兼容；经 .109 桥接 `:8801` 暴露给 .101 | ⏳ 需 key |
| WorkBuddy (.109) | 桌面 App 暂无 API；桥接 TBD | ⏳ |

## 目录

```
voicehub/
├── agents.yaml            # 路由表：别名/端点/音色
├── .env.example           # 凭据模板
├── server/
│   ├── config.py          # 加载 agents.yaml + .env
│   ├── router.py          # 名字路由 + 协作意图
│   ├── tts.py             # Edge TTS 多音色
│   ├── orchestrator.py     # 接力/圆桌/评审 DAG
│   ├── backends/          # openai_compat / openclaw / factory
│   └── main.py            # Pipecat 语音管线
├── bridges/forward.py     # .109 上把 127.0.0.1 服务暴露给 .101
├── cli_demo.py            # 文本模式冒烟测试（无需麦克风）
└── tests/test_router.py
```

## 本地验证（.101 上）

```bash
pip install pyyaml httpx edge-tts pytest
cp .env.example .env        # 填入 OPENCLAW_TOKEN / HERMES_KEY / CHERRY_KEY
pytest tests/               # 路由单元测试
python cli_demo.py          # 文本对话：叫"赫尔墨斯"切到 Hermes，叫"小克"切到 OpenClaw
python cli_demo.py --speak  # 加 TTS 朗读
```

## 语音管线（Pipecat）

```bash
pip install "pipecat-ai[webrtc,edge-tts]" faster-whisper
python -m server.main       # 浏览器打开 SmallWebRTC 提供的客户端页，说话即可
```
> Pipecat 服务类名随版本变化；首次运行若 import 报错，按报错修正 `server/main.py` 的 import 路径。

## 部署

- **VoiceHub 服务跑在 .101**（与 OpenClaw/Hermes 同机）。
- **.109 上的 agent**（CherryStudio/WorkBuddy）需桥接：`TARGET=http://127.0.0.1:23333 PORT=8801 python bridges/forward.py`。
- 手机访问需 HTTPS：`tailscale funnel <port>` 或 nginx + 域名证书。
- 安全：token/key 只存 .env（不进仓库）；对外只暴露 WebRTC 端口。
