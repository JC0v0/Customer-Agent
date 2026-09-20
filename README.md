# Agent-Customer

电商AI客服桌面应用程序，基于 PyQt6 构建，支持多平台渠道集成，集成 AI 大模型实现智能自动回复。

## 下载安装

> 普通用户无需配置 Python 环境，直接下载安装包即可使用。

**下载地址**：<https://github.com/JC0v0/Customer-Agent/releases/latest>

在页面的 **Assets** 中下载 `Agent-Customer-Setup-<版本号>.exe`（约 100 MB），双击运行即可安装。

- 安装到用户目录，**无需管理员权限**
- 双击安装向导（中文）→ 桌面快捷方式 → 装完即用
- 安装包已内置 Playwright 驱动，添加账号时自动调用本机已安装的 **Chrome 或 Edge** 浏览器完成登录，无需额外安装 Playwright 或 Chromium

## 功能特性

- **多渠道支持**：目前支持拼多多平台 WebSocket 实时消息接收
- **AI 智能回复**：基于自研 Agent 框架（不依赖 Agno），多轮工具调用 + 会话上下文管理
- **AI 主动推荐**：客服代理可主动获取商品列表、发送商品卡片给用户
- **双知识库体系**：产品知识库 + 客服知识库，分别检索商品信息与售后/物流/退款等政策
- **商品知识自动同步**：从拼多多 API 拉取商品列表，调用多模态 LLM 提取产品知识入库
- **关键词转人工**：自动识别用户意图，支持关键词触发转人工服务
- **消息队列处理**：异步消息队列 + 处理器链，支持高并发场景
- **自动重连机制**：WebSocket 连接支持断线自动重连和心跳检测

### AI Agent 可用工具

| 工具名称 | 功能描述 |
|----------|----------|
| `get_shop_products` | 获取店铺商品列表（支持价格区间、销量、库存、标签等） |
| `send_goods_link` | 向用户发送商品卡片链接 |
| `get_product_knowledge` | 查询指定商品的详细知识（成分、规格、用法、价格等） |
| `search_customer_service_knowledge` | 搜索客服知识库（售后、物流、退款等政策问答） |
| `transfer_conversation` | 转接会话给人工客服 |

## 环境要求

- Python >= 3.11
- Windows 操作系统（打包为 exe 后可在 Windows 上独立运行）

## 安装

```bash
# 安装依赖
uv sync
```

## 启动

```bash
python app.py
```

## 配置

首次运行后会生成 `config.json`。推荐在“系统设置 → LLM 模型配置”中填写 API Key；示例、日志和 CI 都不需要环境变量，也不包含真实密钥。

应用通过 LiteLLM 直接调用 Chat Completions，不需要单独启动 LiteLLM Proxy。设置页提供以下六个供应商：

| 设置页供应商 | LiteLLM 路由前缀 | 默认 Base URL | Base URL 规则 |
| --- | --- | --- | --- |
| DeepSeek | `deepseek/` | `https://api.deepseek.com` | 可留空使用默认值，或填写 HTTPS 覆盖 |
| 火山引擎 | `volcengine/` | `https://ark.cn-beijing.volces.com/api/v3` | 可留空使用默认值，或填写 HTTPS 覆盖 |
| OpenAI-compatible | `openai/` | 无 | 必须填写符合 Chat Completions 约定的 HTTPS Base URL；模型名可任意填写 |
| Kimi/Moonshot | `moonshot/` | `https://api.moonshot.cn/v1` | 可留空使用默认值，或填写 HTTPS 覆盖 |
| 智谱/Z.AI | `zai/` | `https://open.bigmodel.cn/api/paas/v4` | 可留空使用默认值，或填写 HTTPS 覆盖 |
| Qwen/DashScope | `dashscope/` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 可留空使用默认值，或填写 HTTPS 覆盖 |

每个供应商都需要手动填写模型标识和 API Key。OpenAI-compatible 不会套用内置供应商的模型命名规则；例如可以填写组织内部模型名和其 HTTPS 端点。

端点安全规则：远程端点必须使用 HTTPS 并保持证书校验；带账号密码的 URL、云实例元数据/链路本地地址和跨主机重定向会被拒绝。本地或私有端点（如本机或局域网内的 Ollama、LM Studio，可使用 HTTP）只有在设置页明确选择“允许本地或私有端点”后才可使用。使用自定义端点或能力未知的模型调用工具时，设置页会显示未验证状态并要求基于当前供应商、模型、端点和工具策略确认；明确不支持工具调用的模型不能启用，应用不会静默关闭工具。

旧版本没有 `llm.provider` 的配置会在加载时迁移：历史 Ark/火山引擎端点迁移为 `volcengine`，其他无法识别的自定义端点迁移为 `openai_compatible`，模型名、端点和 API Key 保持不变。API Key 仍使用 Windows DPAPI 保护后落盘。

保存成功只更新新的 profile。已经运行的账号继续使用自己的 profile 快照，需要按现有账号停止/重新启动流程后才会使用新配置。

产品知识同步会把必要的商品文本和图片发送给所选供应商。供应商侧的数据保留、训练、跨境传输和隐私政策由用户与供应商自行确认；应用不会声称可以控制供应商的留存策略，也不会发送 API Key、Cookie、授权头或内部 profile 字段。

## 开发规范

### 新增/修改接口前

1. **先用 curl 或 Python 脚本测试接口**，确认真实请求参数、请求头、响应结构
2. **根据实际响应结构修改解析代码**，不要凭猜测写字段名
3. **修改后用 mock 数据或真实调用验证解析逻辑**

> 例如：修改 `product_manager.py` 的商品列表接口时，先用 curl 测试接口，确认数据在 `result.onSaleGoods` 字段而非 `result.goodsList`，字段名为驼峰 `goodsId` 而非下划线 `goods_id`，价格单位是"分"需除以 100 转换为"元"

## 构建 Windows 安装包

在 Windows 上运行：

```bash
python scripts/build_win_exe.py --clean
```

该命令会依次完成：

1. 用 PyInstaller 打包应用（产物 `dist/AgentCustomer/`，含内置 Playwright 驱动）
2. 调用 Inno Setup 压缩为单个安装程序（产物 `dist/installer/Agent-Customer-Setup-<版本号>.exe`）

版本号自动从最近的 git tag 读取（去掉前导 `v`），也可用环境变量 `APP_VERSION` 指定。

> 若仅需 PyInstaller 的 onedir 目录、不生成安装程序，加 `--skip-installer`。

### CI 自动构建

推送 `v*` 格式的 tag 会触发 GitHub Actions 自动构建并发布 Release：

```bash
git tag v1.4
git push origin v1.4
```

构建产物（`Agent-Customer-Setup-<版本号>.exe`）会自动上传到该 tag 对应的 Release 页，即上文「下载安装」的来源。

## 项目结构

```
text
Agent-Customer/
├── Agent/                  # AI 代理模块（自研 Agent 框架）
│   └── CustomerAgent/
│       ├── custom/         # 自研实现：LLM 客户端、会话管理、工具执行器
│       └── tools/          # Agent 工具集（商品/知识/转人工）
├── Channel/                # 渠道集成
│   └── pinduoduo/
│       ├── core/           # 连接、生命周期、状态、消息处理拆分模块
│       └── utils/API/      # 拼多多 API 封装
├── Message/                # 消息处理（队列 + 处理器链）
│   ├── core/               # consumer / handlers / queue
│   └── handlers/           # 预处理器、AI、关键词处理器
├── bridge/                 # 桥接模块（Context/Reply/SendService）
├── core/                   # 核心服务（DI 容器、连接状态）
├── database/               # 数据库（SQLAlchemy + 知识服务 + 商品同步）
├── service/                # 服务层（账号/关键词服务薄封装）
├── ui/                     # PyQt6 用户界面
├── utils/                  # 工具模块（日志、路径等）
├── scripts/                # 构建与抓包脚本
└── app.py                  # 应用入口
```

## 技术栈

| 类别 | 技术 |
|------|------|
| UI 框架 | PyQt6 + pyqt6-fluent-widgets |
| AI 框架 | 自研 Agent 框架 + LiteLLM 直接异步调用 |
| 数据库 | SQLAlchemy + SQLite |
| 中文分词 | jieba（知识库检索） |
| Token 统计 | tiktoken |
| 异步通信 | asyncio + websockets |
| 文档解析 | pypdf + python-docx + openpyxl + xlrd |
| 日志 | Loguru |
| 配置 | Pydantic |

## License

MIT
