# Agent-Customer

电商 AI 客服桌面应用，基于 **Python + PyQt6** 构建，当前主要接入 **拼多多商家客服场景**。应用通过拼多多 WebSocket 接收客户消息，结合关键词规则、知识库和 OpenAI 兼容大模型，实现自动回复、商品推荐、知识检索和转人工。

> 当前项目是桌面客户端应用，不是 Flask/FastAPI 等传统后端服务，也不是 Web 前端项目。

## 功能特性

- **拼多多客服接入**：支持拼多多账号登录、Cookie 保存、WebSocket 实时消息接收。
- **AI 智能回复**：基于自研 Agent 框架，支持会话上下文、工具调用和多轮推理。
- **关键词转人工**：命中“转人工/退款/投诉/开发票”等关键词时，自动转接其他人工客服。
- **AI 工具转人工**：模型也可以在合适场景调用 `transfer_conversation` 工具发起转人工。
- **商品卡片推荐**：AI 可查询店铺商品，并向客户发送商品卡片。
- **产品知识库**：支持从拼多多商品接口同步商品，并调用多模态 LLM 提取商品知识。
- **客服知识库**：维护售后、物流、退款、发票等常见问答，可供 AI 回复时检索。
- **异步消息队列**：WebSocket 消息进入队列后，由处理器链依次执行关键词检测、AI 回复和兜底处理。
- **桌面管理界面**：提供账号管理、自动回复、关键词、知识库、日志和设置页面。
- **Windows 打包**：提供 PyInstaller 构建脚本，可打包为 Windows 桌面应用。

## AI Agent 可用工具

| 工具名称 | 功能描述 |
| --- | --- |
| `get_shop_products` | 获取店铺商品列表 |
| `send_goods_link` | 向用户发送商品卡片 |
| `get_product_knowledge` | 查询指定商品的产品知识 |
| `search_customer_service_knowledge` | 搜索客服知识库 |
| `transfer_conversation` | 将当前会话转接给人工客服 |

## 技术栈

| 类别 | 技术 |
| --- | --- |
| 语言 | Python >= 3.11 |
| 包管理 | uv |
| UI | PyQt6 + pyqt6-fluent-widgets |
| 异步通信 | asyncio + websockets |
| 浏览器自动化 | Playwright |
| HTTP 请求 | requests / aiohttp |
| AI 调用 | OpenAI 兼容 API |
| 数据库 | SQLAlchemy + SQLite |
| 知识检索 | SQL LIKE + jieba 分词 |
| Token 估算 | tiktoken |
| 文档/表格处理 | pandas、openpyxl、xlrd、pypdf、python-docx |
| 日志 | Loguru |
| 配置校验 | Pydantic |
| 打包 | PyInstaller |

## 环境要求

- Python >= 3.11
- uv
- 可访问配置的大模型 API 服务
- 拼多多商家账号
- 如需登录/刷新拼多多 Cookie，需要安装 Playwright 浏览器
- 如需构建 Windows EXE，需要在 Windows 系统上执行构建脚本

## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd Customer-Agent
```

### 2. 安装依赖

```bash
uv sync
```

### 3. 安装 Playwright 浏览器

首次使用拼多多登录能力前，需要安装浏览器二进制文件：

```bash
uv run python scripts/install_playwright.py
```

开发环境下浏览器默认安装到：

```text
.browsers/
```

### 4. 启动应用

建议始终在项目根目录运行：

```bash
uv run python app.py
```

不建议从其他目录直接运行，因为项目中部分配置、数据库、日志和浏览器路径依赖当前工作目录。

## 配置说明

首次运行时会在项目根目录自动生成：

```text
config.json
```

当前配置模型主要包含：

| 配置项 | 说明 |
| --- | --- |
| `llm.model_name` | 模型名称 |
| `llm.api_key` | LLM API Key |
| `llm.api_base` | OpenAI 兼容 API Base |
| `business_hours.start` | 人工客服开始时间，默认 `08:00` |
| `business_hours.end` | 人工客服结束时间，默认 `23:00` |
| `prompt.instructions` | AI 客服行为指令 |
| `db_path` | 数据库路径配置 |

默认配置示例：

```json
{
  "business_hours": {
    "start": "08:00",
    "end": "23:00"
  },
  "llm": {
    "model_name": "",
    "api_key": "",
    "api_base": ""
  },
  "prompt": {
    "instructions": [
      "1. 请用中文回复客户问题",
      "2. 当用户询问特定商品的信息、成分、使用方法、价格、规格等问题时，请优先使用 get_product_knowledge 工具获取商品详细知识，必须提供 goods_id（商品ID）和 shop_id（店铺ID）",
      "3. 当用户询问售后政策、物流信息、退换货规则、常见问题解答等非产品特定问题时，请使用 search_customer_service_knowledge 工具搜索客服知识，必须提供 query（搜索关键词）和 shop_id（店铺ID）",
      "4. 如果知识库中有相关信息，请根据知识库内容回答用户问题",
      "5. 如果知识库中没有相关信息，再根据已有知识回答或建议用户联系人工客服"
    ]
  }
}
```

也可以在应用的“设置”页面中维护 LLM、Prompt 和营业时间配置。

> 注意：`config.json` 中会保存 LLM API Key，请勿提交到代码仓库或发送给他人。

## 主要使用流程

### 1. 配置 LLM

1. 启动应用。
2. 进入“设置”页面。
3. 填写 API Base、API Key 和模型名称。
4. 根据业务需要调整 Prompt 和营业时间。
5. 保存配置。

### 2. 添加拼多多账号

1. 进入“账号管理”。
2. 点击添加账号。
3. 输入拼多多商家账号和密码。
4. 应用通过 Playwright 登录拼多多商家后台。
5. 登录成功后获取 Cookie、用户信息和店铺信息。
6. 写入本地 SQLite 数据库。

### 3. 启动自动回复

1. 进入“自动回复”。
2. 确认账号在线。
3. 点击“开始回复”。
4. 应用为该账号启动独立后台线程和 WebSocket 连接。
5. 收到客户消息后，消息进入队列并依次经过关键词处理器和 AI 回复处理器。

### 4. 维护知识库

- 产品知识：可从拼多多商品接口同步，并由 LLM 提取商品知识。
- 客服知识：可手动维护常见问答，也可通过表格批量导入。

## 转人工机制

转人工有两种触发方式：

1. **关键词转人工**
   - 用户文本消息命中关键词后触发。
   - 关键词来自数据库，加载失败时使用默认关键词。
   - 处理器会获取可用客服列表，过滤当前客服自己，然后选择第一个可用客服转接。

2. **AI 工具转人工**
   - AI 判断当前问题需要人工处理时，可以调用 `transfer_conversation` 工具。

核心调用链：

```text
客户消息
  -> PDDChannel
  -> Context
  -> MessageQueue
  -> KeywordDetectionHandler
  -> getAssignCsList
  -> move_conversation
```

相关文件：

```text
Message/handlers/keyword_handler.py
Agent/CustomerAgent/tools/move_conversation.py
Channel/pinduoduo/utils/API/send_message.py
```

## 架构概览

```text
app.py
  -> MainWindow
  -> AutoReplyUI / AutoReplyManager
  -> AutoReplyThread
  -> PDDChannel
  -> 拼多多 WebSocket
  -> Context
  -> MessageQueue
  -> KeywordDetectionHandler
  -> AIReplyHandler
  -> CustomerAgent
  -> LLMClient / ToolExecutor / KnowledgeService
  -> SendMessage API
```

## 项目结构

```text
Customer-Agent/
├── app.py                     # 应用入口
├── config.py                  # 配置管理
├── pyproject.toml             # 项目依赖与构建配置
├── uv.lock                    # uv 锁文件
├── Agent/                     # AI Agent 模块
│   └── CustomerAgent/
│       ├── custom/            # LLM 客户端、消息构造、会话管理、工具执行
│       └── tools/             # Agent 工具集
├── Channel/                   # 渠道集成
│   └── pinduoduo/             # 拼多多登录、API、WebSocket、消息处理
├── Message/                   # 消息队列与处理器链
├── bridge/                    # Context / Reply 桥接模型
├── core/                      # DI 容器、服务注册、连接状态等
├── database/                  # 数据库模型、数据库管理、知识库、商品同步
├── ui/                        # PyQt6 桌面界面
├── utils/                     # 日志、路径、运行时工具
├── scripts/                   # Playwright 安装和构建脚本
└── icon/                      # 图标资源
```

## 关键模块说明

| 模块 | 说明 |
| --- | --- |
| `ui/main_ui.py` | 主窗口和导航 |
| `ui/auto_reply/` | 自动回复页面、账号卡片、后台线程管理 |
| `ui/user_ui.py` | 拼多多账号管理 |
| `ui/keyword_ui.py` | 转人工关键词管理 |
| `ui/Knowledge_ui.py` | 产品知识库和客服知识库页面 |
| `ui/setting_ui.py` | LLM、Prompt、营业时间配置 |
| `Channel/pinduoduo/pdd_login.py` | 拼多多 Playwright 登录 |
| `Channel/pinduoduo/pdd_channel.py` | 拼多多 WebSocket 渠道入口 |
| `Channel/pinduoduo/utils/API/` | 拼多多接口封装 |
| `Message/__init__.py` | 消息系统入口和处理器链创建 |
| `Message/handlers/keyword_handler.py` | 关键词转人工处理器 |
| `Message/handlers/ai_handler.py` | AI 回复处理器 |
| `Agent/CustomerAgent/custom/customer_agent.py` | AI 客服 Agent 主实现 |
| `database/db_manager.py` | 账号、店铺、渠道、关键词等基础数据管理 |
| `database/knowledge_service.py` | 产品知识和客服知识 CRUD/检索 |
| `database/product_sync.py` | 拼多多商品同步和产品知识提取 |

## 运行时文件

应用运行过程中可能生成以下文件或目录：

```text
config.json                  # 本地配置，包含 LLM API Key
temp/channel_shop.db         # 主业务 SQLite 数据库
temp/agent.db                # Agent 会话历史数据库
logs/app.log                 # 应用日志
.browsers/                   # Playwright 浏览器文件
user_data/                   # 浏览器用户数据目录
```

> 这些文件通常不应提交到代码仓库。

## 构建 Windows 可执行文件

Windows EXE 构建必须在 Windows 系统上执行。

```bash
uv sync
python scripts/build_win_exe.py --clean
```

可选参数：

```bash
python scripts/build_win_exe.py --python 3.11
python scripts/build_win_exe.py --clean
```

构建产物位于：

```text
dist/AgentCustomer/
```

通用构建脚本：

```bash
python scripts/build_exe.py
python scripts/build_exe.py --clean
python scripts/build_exe.py --mode debug
python scripts/build_exe.py --installer
python scripts/build_exe.py --check-only
```

## 开发规范

### 修改外部接口前

拼多多接口字段可能变化，修改接口封装前请先确认真实请求和响应结构：

1. 先用浏览器开发者工具、curl 或 Python 脚本确认接口参数、请求头、响应结构。
2. 根据真实响应字段修改解析代码，不要凭猜测写字段名。
3. 修改后用 mock 数据或真实调用验证解析逻辑。

例如修改商品列表接口时，需要确认数据路径、字段命名和价格单位：

- 数据可能在 `result.onSaleGoods`，而不是 `result.goodsList`。
- 字段可能是 `goodsId`，而不是 `goods_id`。
- 价格单位可能是“分”，展示时需要转换为“元”。

### 当前工程化状态

当前项目已有：

- `pyproject.toml`
- `uv.lock`
- `.python-version`
- PyInstaller 构建脚本
- pytest 配置
- Ruff 配置
- 最小非侵入式元数据测试

安装开发依赖：

```bash
uv sync --extra dev
```

运行测试：

```bash
uv run pytest
```

运行 Ruff 检查：

```bash
uv run ruff check .
```

检查格式化：

```bash
uv run ruff format . --check
```

自动格式化：

```bash
uv run ruff format .
```

后续仍建议逐步补充：

- 业务单元测试和集成测试
- CI 配置
- pre-commit 配置
- 类型检查配置

## 安全注意事项

当前项目会在本地保存敏感信息：

- `config.json` 中保存 LLM API Key。
- SQLite 数据库中可能保存拼多多账号、密码和 Cookie。
- 日志中应避免输出完整 Cookie、Token、API Key 和账号密码。

如果用于真实商家或分发给客户，建议优先补充：

- API Key 加密存储或系统 Keychain 集成。
- 账号密码和 Cookie 加密存储。
- 日志脱敏检查。
- 数据库备份、恢复和迁移策略。

## 已知待完善项

- README 和部分脚本文档需要持续与代码保持同步。
- 当前未包含标准自动化测试体系。
- 当前未包含 lint / format / type-check 配置。
- 当前未包含数据库迁移体系。
- 部分运行时路径依赖当前工作目录，建议始终从项目根目录启动。
- 关键词更新后，运行中的关键词处理器缓存刷新逻辑需要重点验证。
- Windows 打包脚本和 hiddenimports 需要结合实际依赖继续清理。

## License

当前仓库未发现独立 `LICENSE` 文件。如需对外发布，请补充许可证文件并保持 README 声明一致。
