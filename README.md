# 🤖 拼多多智能客服系统

<div align="center">
  <img src="docs/设置.png" alt="系统配置界面" width="600">
  <p><em>拼多多智能客服系统 - 提升客服效率的智能化解决方案</em></p>
</div>

## 📖 项目简介

拼多多智能客服系统是一个专为电商平台设计的综合性客户服务管理工具。本系统通过AI技术和自动化流程，显著提高客服工作效率，实现智能回复的同时保留人工介入的灵活性，为商家提供完整的客服解决方案。

## ✨ 主要功能

### 🔐 账号管理
- 商家账号管理（支持多账号）
- 自动登录获取cookies
- 账号状态实时监控

<div align="center">
  <img src="docs/账号管理.png" alt="账号管理界面" width="500">
  <p><em>账号管理 - 管理您的拼多多商家账号</em></p>
</div>

### 💬 智能消息处理
- 实时消息监控与自动回复
- 集成 AI (Agno) 生成智能回复内容
- 支持知识库检索和自定义回复模板

<div align="center">
  <img src="docs/自动回复.png" alt="自动回复界面" width="500">
  <p><em>智能回复 - 自动回复客户消息</em></p>
</div>

### 🔄 智能转接系统
- 基于关键词智能识别客户需求
- 自动将复杂问题转接给人工客服
- 无缝衔接确保服务质量


### 📊 系统监控
- 实时日志记录
- 系统运行状态监控
- 详细的操作记录和统计


## 🚀 快速开始

### 环境要求
- Python 3.11+
- Windows 10/11 (推荐)
- 网络连接稳定

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/JC0v0/Customer-Agent.git
   cd Customer-Agent
   ```

2. **安装依赖**
   ```bash
   ##使用uv进行环境配置
   ##安装uv
   pip install uv

   uv venv
   uv sync
   ```

3. **安装浏览器驱动**
   ```bash
   uv run playwright install chrome
   ```


## 📱 使用指南

### 启动系统
```bash
python app.py
```

### 配置流程

1. **配置商家账号**
   - 在账号管理界面配置您的拼多多商家账号
   - 系统将自动获取并保存登录凭证

2. **设置关键词规则**
   - 配置需要人工转接的关键词
   - 设置自动回复的话术模板

3. **配置AI模型**
   - 在设置界面配置 AI 模型
   - 支持 OpenAI/DeepSeek/Gemini/Kimi/Claude 等

4. **启动系统**
   - 在账号管理界面启动系统
   - 系统将根据配置自动处理消息

5. **监控日志**
   - 在日志管理界面查看系统运行日志


## 🛠️ 技术架构

- **前端界面**: PyQt6 + PyQt-Fluent-Widgets
- **后端逻辑**: Python asyncio
- **AI集成**: Agno + OpenAI兼容API (支持 DeepSeek/Gemini/Kimi/Claude)
- **数据存储**: SQLAlchemy + SQLite + LanceDB (向量数据库)
- **浏览器自动化**: Playwright
- **日志系统**: Loguru
- **包管理**: UV

## 📁 项目结构

```
Customer-Agent/
├── Agent/              # AI智能代理模块
│   ├── bot.py             # 机器人基类
│   └── CustomerAgent/     # 客服代理
│       ├── agent.py           # 主要代理逻辑
│       ├── agent_knowledge.py # 知识库集成
│       ├── tools/             # 代理工具集
│       └── readers/           # 文档读取器
├── Channel/            # 渠道接口模块
│   ├── channel.py         # 渠道基类
│   └── pinduoduo/      # 拼多多渠道
│       ├── pdd_chnnel.py    # 渠道主类
│       ├── pdd_login.py     # 登录处理
│       └── utils/        # API工具
├── Message/            # 消息处理模块
│   ├── core/              # 核心组件
│   │   ├── consumer.py    # 消息消费者
│   │   ├── handlers.py    # 处理器管理
│   │   └── queue.py       # 消息队列
│   ├── handlers/          # 具体处理器
│   │   ├── ai_handler.py      # AI处理器
│   │   ├── keyword_handler.py # 关键词处理器
│   │   └── preprocessor.py    # 预处理器
│   └── models/           # 数据模型
├── core/               # 核心服务模块
│   ├── di_container.py    # 依赖注入容器
│   ├── cache.py           # 缓存服务
│   ├── connection_status.py # 连接状态管理
│   └── service_providers.py # 服务提供者
├── bridge/             # 桥接模块
│   ├── context.py         # 上下文管理
│   └── reply.py          # 回复处理
├── database/           # 数据库模块
│   ├── db_manager.py      # 数据库管理器
│   ├── models.py          # 数据模型
│   └── connection_pool.py # 连接池管理
├── ui/                 # 用户界面模块
│   ├── main_ui.py         # 主界面
│   ├── auto_reply_ui.py   # 自动回复界面
│   ├── Knowledge_ui.py    # 知识库界面
│   ├── keyword_ui.py      # 关键词管理界面
│   ├── log_ui.py          # 日志界面
│   ├── setting_ui.py      # 设置界面
│   └── user_ui.py         # 用户管理界面
├── utils/              # 工具函数
│   ├── logger_loguru.py   # Loguru日志系统
│   ├── logger_config.py   # 日志配置
│   ├── resource_manager.py # 资源管理
│   └── path_utils.py      # 路径工具
├── scripts/            # 构建脚本
├── docs/               # 文档和截图
├── app.py              # 应用程序入口
├── config.py           # 配置管理
├── pyproject.toml      # 项目配置
└── uv.lock             # 依赖锁定文件
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！如果您想参与项目开发：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

## 📞 联系我们

- **问题反馈**: [GitHub Issues](https://github.com/JC0v0/PDD-customer-bot/issues)
- **功能建议**: 欢迎通过 Issues 提出您的想法
- **技术交流**: 
<div align="center">
  <img src="icon/Customer-Agent-qr.png" alt="频道二维码" width="200">
  <p><em>频道二维码</em></p>
</div>

## 💖 支持项目

如果这个项目对您有帮助，您可以通过以下方式支持我们：

<div align="center">
  <img src="docs/赞赏码.png" alt="赞赏码" width="200">
  <p><em>您的支持是我们前进的动力</em></p>
</div>

---

<div align="center">
  <p>⭐ 如果这个项目对您有帮助，请给我们一个星标！</p>
  <p>Made with ❤️ by <a href="https://github.com/JC0v0">JC0v0</a></p>
</div>
