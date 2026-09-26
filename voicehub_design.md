# VoiceHub — 局域网多智能体实时语音中枢 设计文档

> 目标：一个语音软件，叫不同名字，唤醒 LAN 内 4 个智能体（OpenClaw / Hermes / WorkBuddy / CherryStudio），实时语音交互，可打断，各 agent 独立音色与会话记忆。

## 1. 总体架构

```
浏览器 (WebRTC 麦克风/扬声器)
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  语音层 (Pipecat, Python)                            │
│  · WebRTC Transport (局域网/外网直连, 无需云)         │
│  · OpenWakeWord (常驻监听唤醒词, 本地 ONNX)          │
│  · Silero VAD (说话检测 + 打断 barge-in)             │
│  · 流式 STT: faster-whisper small (本地, 中文友好)    │
│  · 流式 TTS: Edge TTS (微软, 免费, 每 agent 一音色)   │
└──────────────────┬──────────────────────────────────┘
                   │ 转写文本流
                   ▼
┌─────────────────────────────────────────────────────┐
│  路由层 (Router)                                     │
│  · 名字匹配: 转写句首/句中匹配 agent 别名表            │
│  · 默认 agent (未叫名字时落到上次对话的 agent)         │
│  · 每 agent 独立 session_id (上下文互不污染)          │
└──────────────────┬──────────────────────────────────┘
                   │ 统一接口: AgentBackend.send(text, session) -> stream
                   ▼
┌─────────────────────────────────────────────────────┐
│  适配层 (Adapter Registry)  ←── 核心设计              │
│  OpenClawAdapter   │ HermesAdapter                  │
│  WorkBuddyAdapter  │ CherryStudioAdapter            │
└──────────────────┬──────────────────────────────────┘
                   ▼
        LAN 内 4 个智能体服务
```

**核心抽象：所有 agent 归一化为一个接口**（含流式返回），新接 agent 只写一个 Adapter，语音层和路由层零改动。

```python
class AgentBackend(Protocol):
    name: str
    async def send(self, text: str, session_id: str) -> AsyncIterator[str]: ...
    # 返回文本增量流, TTS 逐句合成播放, 首句即出声
```

## 2. 四个 agent 的接入方案

| Agent | 现状 | 接入方式 | 难度 |
|---|---|---|---|
| OpenClaw | Gateway (默认 18789), WebSocket v3 协议, delta 流式 | 直连 Gateway WebSocket；原生支持多 agent/persona | ★ 简单 |
| Hermes | Herald 版自带 OpenAI Bridge (OpenAI 兼容端点), SSE 流式 | HTTP + SSE 直连 | ★ 简单 |
| WorkBuddy | 桌面应用, 无公开局域网 API（需确认） | 三选一：① 若有本地网关/webhook 则直连 ② A2A 协议 ③ 桥接脚本（文件/HTTP 中转）| ★★ 需验证 |
| CherryStudio | 桌面客户端 | 三选一：① 新版内置 API 服务器（OpenAI 兼容, 需在设置里开启）② MCP 通道 ③ 桥接脚本 | ★★ 需验证 |

> WorkBuddy / CherryStudio 属于"桌面 App"而非"服务端"，共同兜底方案：**Bridge Adapter** —— 一个本地小程序把它们的输出转发成统一的流式接口，先跑通再优化。

## 3. "叫名字"路由设计

**agents.yaml（路由 + 音色 + 别名）：**

```yaml
agents:
  openclaw:
    aliases: ["小克", "openclaw", "克劳"]       # STT 转写后模糊匹配
    adapter: openclaw_gateway
    endpoint: ws://192.168.x.x:18789
    tts_voice: piper:zh_CN-huayan-medium
    session_prefix: voice-openclaw
  hermes:
    aliases: ["赫尔墨斯", "hermes", "爱马仕"]
    adapter: openai_compat
    endpoint: http://192.168.x.x:PORT/v1/chat/completions
    tts_voice: zh-CN-XiaoxiaoNeural
  workbuddy:
    aliases: ["工作搭档", "workbuddy", "小助"]
    adapter: bridge          # 待定
    tts_voice: zh-CN-YunxiNeural
  cherrystudio:
    aliases: ["樱桃", "cherry", "小樱"]
    adapter: openai_compat
    endpoint: http://127.0.0.1:23333/v1/chat/completions
    api_key: cs-sk-...        # 用户已开启 API 网关
    tts_voice: zh-CN-XiaoyiNeural
default_agent: openclaw        # 没叫名字时落到上次/默认 agent
```

**路由逻辑（转写完成后 ~10ms 内完成）：**
1. 提取转写文本，对别名表做模糊匹配（pinyin 归一化，容忍 STT 把"赫尔墨斯"写成"赫而莫斯"）
2. 命中 → 该句转发给对应 agent，并更新"当前活跃 agent"
3. 未命中 → 发给当前活跃 agent（连续对话不用每句都叫名字）
4. 切换 agent 时 TTS 播报该 agent 专属音色，听感上自然区分

### 3.3 唤醒词：OpenWakeWord 常驻监听（已锁定方案）

**设计**：麦克风常驻，Silero VAD 先低功耗判定"有人说话"，触发 OpenWakeWord 检测唤醒词 → 命中即走完整 STT+agent+TTS 链路；未命中静默丢弃（不消耗 STT/agent 算力）。

**唤醒词（4 个）：** Hey 小克 / Hey 赫尔墨 / Hey 小助 / Hey 樱桃（对应 4 agent）

**中文唤醒词的关键陷阱 + 对策（必须正视）：**
OpenWakeWord 官方模型全是英文训练的，"小克/赫尔墨"这类中文发音没有现成 .onnx，硬套会误唤醒率极高。两条路：

| 方案 | 做法 | 代价 |
|---|---|---|
| A. 英文唤醒词 | 用 OpenWakeWord 现成模型 + 英文词（"Hey Claw"/"Hey Hermes"） | 放弃中文叫名，但零训练、最稳 |
| B. 中文自定义训练 | 用 OpenWakeWord 训练脚本 + 中文录音集训 4 个 .onnx | 需造数据（几十~几百条/词），但实现"中文叫名" |
| C. 混合（推荐先上） | OpenWakeWord 做"有人喊唤醒前缀"粗检（英文 hey），命中后用 whisper-tiny 转写首句，再匹配中文别名 | 用 OpenWakeWord 的高灵敏度英文模型兜底常驻监听，中文名在转写层精确匹配，兼顾"免费+中文" |

**P1 落地选 C**：常驻监听用英文 "Hey" 类低误报模型触发，之后 whisper 转写识别具体叫谁——仍全免费、全本地，且支持中文叫名。B 作为后续优化项（真正的中文唤醒词直检）。

> 无论 A/B/C，"未命中唤醒词就丢弃"这条保证了常驻监听的隐私与算力开销可控。

## 4. 实时性设计（目标：首字响应 < 1.5s，局域网内）

| 环节 | 方案 | 延迟 |
|---|---|---|
| VAD 断句 | Silero, 静音 400ms 判定说完 | ~0ms |
| STT | faster-whisper small, 流式增量转写 | ~200-400ms |
| 路由 | 本地字符串匹配 | ~10ms |
| Agent 响应 | 各自流式返回（OpenClaw/Hermes 原生流式） | 取决于 agent |
| TTS | **逐句合成**：不等 agent 回完，首句文本到即合成出声 | ~150-300ms |
| 打断 | 说话时立即停 TTS + 取消未完成请求（Pipecat 内置） | ~0ms |

全链路局域网，无云依赖，隐私数据不出内网。

## 5. 项目结构

```
voicehub/
├── agents.yaml              # 路由表：别名/端点/音色
├── server/
│   ├── main.py              # Pipecat pipeline + FastAPI (WebRTC)
│   ├── router.py            # 名字匹配 + 活跃会话管理
│   ├── backends/
│   │   ├── base.py          # AgentBackend 协议
│   │   ├── openclaw.py      # Gateway WebSocket
│   │   ├── hermes.py        # OpenAI 兼容 SSE
│   │   ├── bridge.py        # 桌面 App 通用桥接（轮询/文件/WebSocket 中转）
│   │   └── cherrystudio.py
│   └── tts.py               # 多音色管理
├── web/                     # 浏览器前端 (Voice UI Kit)：按钮、波形、当前 agent 指示
└── bridges/                 # WorkBuddy / CherryStudio 桥接小程序
```

## 6. 实施路线

| 阶段 | 内容 | 验证标准 |
|---|---|---|
| P1 | Pipecat 骨架 + OpenClaw 适配 + 单音色 | 浏览器说话 → OpenClaw 语音回答，可打断 |
| P2 | 路由层 + 4 别名 + 多音色 + Hermes 接入 | 叫"赫尔墨斯"切到 Hermes，音色变化，上下文各自独立 |
| P3 | WorkBuddy / CherryStudio 适配（先验证它们暴露的接口） | 4 个 agent 全部可用 |
| P4 | 部署上线：服务器常驻 + Tailscale/CF 隧道 + 手机 HTTPS 访问；中文唤醒词 B 方案（自定义 .onnx 训练） | 手机浏览器叫名字 → 4 agent 实时应答 |

## 7. 多智能体协作设计（星型拓扑）

### 7.1 为什么走中枢而不是 agent 两两直连

- 4 个 agent 能力不对等：OpenClaw/Hermes 是"服务端"（可被调用也能调别人），WorkBuddy/CherryStudio 是"桌面 App"（只能被调用，难当调用方）
- 两两直连 = 6 条链路，每接一个新 agent +N 条；星型 = 每个只需 1 条 Adapter
- 语音场景下，用户通过 TTS 听所有结果——**所有流量本来就要汇聚到中枢做逐句合成**，协作结果顺路汇聚

```
          ┌─── OpenClaw ───┐
语音 ──▶ VoiceHub ── Hermes   │   星型：Hub 编排，agent 互不直连
          ├─── WorkBuddy ──┤   （例外见 8.4：OpenClaw 可被授权为子编排器）
          └─ CherryStudio ─┘
```

### 7.2 三种协作模式（路由层新增意图识别）

| 模式 | 语音指令示例 | 行为 |
|---|---|---|
| **接力 Handoff** | "让 OpenClaw 查下恒瑞的财报，然后让 Hermes 写个点评" | Hub 串行调度：A 的输出作为 B 的输入，最终 B 的回复播报 |
| **圆桌 Roundtable** | "叫上赫尔墨斯和小克一起聊聊这个问题" | Hub 并行 fan-out 给 N 个 agent，结果按需合成（可选让某一 agent 做汇总）后播报 |
| **评审 Review** | "让 Hermes 检查一下 OpenClaw 刚才的方案" | Hub 把上一轮对话上下文注入被调用方，请求批判性意见 |

**意图识别**：转写文本含协作关键词（"和/跟/一起/加上/然后让/检查/评审" + ≥2 个别名）→ 进入编排模式；否则单 agent 直答。误判兜底：播报前说一句"已让 X 和 Y 协作"给用户确认。

### 7.3 AgentBackend 接口扩展

```python
class AgentBackend(Protocol):
    name: str
    async def send(self, text: str, session_id: str,
                   context: list[Message] | None = None) -> AsyncIterator[str]: ...
    # context: 编排时注入的跨 agent 上下文（接力结果 / 评审材料）
```

编排器（`orchestrator.py`）维护协作 DAG：`[{agent: openclaw, depends_on: []}, {agent: hermes, depends_on: [openclaw]}]`，逐层执行，每步结果注入下一步 context。

### 7.4 Agent 主动互调（脱离语音编排时）

语音编排覆盖"人叫他们协作"的场景；无人值守时的互调用协议直连：

- **A2A 协议**：Hermes Herald 版已内置 A2A v1.0（agent 间发现/通信标准协议）。OpenClaw 若支持 A2A 则直入；不支持的话，给 OpenClaw 装一个"技能"封装 A2A 客户端即可
- **工具化封装**：OpenClaw 能力最强，给它注册 3 个 HTTP 工具（call_hermes / call_workbuddy / call_cherry），它就能在自己执行任务途中主动调用其他 agent —— 等效于把 OpenClaw 升级为**子编排器**
- WorkBuddy / CherryStudio 保持纯被调方

### 7.5 语音 UX：多人协作怎么"听"出来

- 每个 agent 保留专属音色 → 圆桌模式下逐个播放，像多人会议
- 播报前先报身份（"（小克）我查到…"），音色差异 + 名字双保险
- 打断规则：用户开口 = 全体停下；agent 播报中不互相打断（排队）

## 8. 需要你确认 / 提供的信息

1. 4 台 agent 服务在 LAN 的 IP 和端口（OpenClaw gateway token、Hermes bridge 端口）
2. WorkBuddy 是否有局域网 API / webhook？（决定走直连还是桥接）
3. ~~CherryStudio 版本是否带"API 服务器"功能？~~ ✅ 已确认：OpenAI 兼容网关 `http://127.0.0.1:23333`，需确认是否允许 LAN 访问（改 0.0.0.0）
4. ~~中文音色偏好~~ ✅ 选 Edge TTS（免费、中文自然）

## 9. 部署：服务器 + 手机访问（已锁定需求）

### 9.1 拓扑

```
手机/电脑浏览器 ──HTTPS──▶ 服务器 (VoiceHub, 常驻) ──LAN──▶ 4 agent 服务
                              │
                              └─ WebRTC (麦克风/扬声器) + TTS 音频流
```

### 9.2 手机访问的硬性要求：HTTPS

手机浏览器调麦克风**必须 HTTPS**（http 下 `getUserMedia` 被禁）。三条路任选：

| 方案 | 适用 | 做法 |
|---|---|---|
| **Tailscale Funnel** | 最省事、无公网 IP 也行 | `tailscale funnel 8765`，得 `https://机器名.ts.net`，内网+外网手机都能用 |
| **Cloudflare Tunnel** | 有域名、要稳定公网 | `cloudflared tunnel`，隐藏真实 IP |
| **nginx + 自签/Let's Encrypt** | 已在公网有服务器 | 反代 127.0.0.1:8765，配 SSL |

> 推荐先用 Tailscale Funnel 跑通（零配置），后续要长期稳定再换 nginx+域名。

### 9.3 服务器常驻

- 进程管理：`systemd` 或 `pm2`/`supervisor`，崩溃自启
- Pipecat 服务 + FastAPI(WebRTC) 同进程，监听 `0.0.0.0:<port>`
- 资源：faster-whisper small + Edge TTS 主要吃 CPU，无 GPU 也可跑；若上 medium/large 加 CUDA
- 安全：openclaw gateway token / CherryStudio api_key 存 `.env`（不进仓库），服务对外只暴露 WebRTC 端口

### 9.4 移动端 UX

- 复用 Web UI（Voice UI Kit），响应式即可，无需单独 App
- 横屏显示"当前活跃 agent + 波形 + 4 个 agent 快捷切换按钮"
- 弱网：WebRTC 自带抖动缓冲；若 4G 抖动大，TTS 改为完整句一次性下发而非逐句流
