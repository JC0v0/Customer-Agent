# Agent-Customer 构建脚本使用指南

本文档介绍 `scripts/` 目录下各个构建和使用脚本的详细功能和使用方法。

## 📁 目录结构

```
scripts/
├── __pycache__/          # Python 缓存目录
├── build_exe.py          # 完整打包脚本（功能丰富）
├── build_win_exe.py      # 简化打包脚本（快速构建）
├── install_playwright.py # Playwright 浏览器安装脚本
├── capture_pdd_ws.py     # 拼多多客服 WebSocket 抓包（CDP 监听）
├── decode_pdd_ws.py      # 抓包离线解码，输出可读 JSON
├── version_info.txt      # Windows 可执行文件版本信息
└── README.md             # 本使用说明文档
```

## 🛠️ 脚本详细说明

### 1. build_exe.py - 完整打包脚本

**功能描述：**
使用 PyInstaller 将项目打包成独立的 Windows 可执行文件，支持完整的企业级打包功能。

**主要特性：**
- ✅ 自动设置 Python 虚拟环境
- ✅ 依赖检查和环境验证
- ✅ 支持发布和调试模式
- ✅ 自动创建分发包文件
- ✅ 生成 NSIS 安装包脚本
- ✅ 支持清理构建缓存
- ✅ 详细的错误处理和日志

**使用方法：**

```bash
# 基本用法
python scripts/build_exe.py

# 完整参数列表
python scripts/build_exe.py [选项]
```

**支持的参数：**
```bash
--python VERSION     # Python 版本，默认 3.11
--mode MODE          # 构建模式：release(默认) 或 debug
--clean              # 构建前清理缓存文件
--installer          # 生成 NSIS 安装包脚本
--check-only         # 仅检查依赖，不执行构建
```

**使用示例：**

```bash
# 标准构建
python scripts/build_exe.py

# 调试模式构建
python scripts/build_exe.py --mode debug

# 清理后重新构建
python scripts/build_exe.py --clean

# 生成安装包脚本
python scripts/build_exe.py --installer

# 仅检查依赖
python scripts/build_exe.py --check-only

# 完整构建流程（推荐）
python scripts/build_exe.py --clean --installer
```

**输出文件：**
- `dist/AgentCustomer.exe` - 主程序
- `dist/temp/` - 临时文件目录
- `dist/README.txt` - 用户说明
- `dist/run.bat` - Windows 批处理启动脚本
- `dist/config.json` - 配置文件模板
- `installer.nsi` - NSIS 安装包脚本（如果使用 --installer）

**前置要求：**
- 安装 `uv` 包管理器
- Windows 操作系统
- 项目根目录必须包含：
  - `app.py`
  - `config.json`
  - `icon/icon.ico`
  - `scripts/agent_customer.spec` (当前不存在，需要创建)
  - `requirements.txt`

---

### 2. build_win_exe.py - 简化打包脚本

**功能描述：**
快速构建 Windows 可执行文件的简化版本，适合快速测试和开发。

**主要特性：**
- ⚡ 快速构建，最小化配置
- 🔧 基础环境检查
- 📦 外置浏览器模式（减小包体积）

**使用方法：**

```bash
# 基本构建
python scripts/build_win_exe.py

# 指定 Python 版本
python scripts/build_win_exe.py --python 3.11
```

**输出文件：**
- `dist/AgentCustomer.exe` - 可执行文件

**前置要求：**
- 安装 `uv` 包管理器
- 项目基础依赖文件

---

### 3. install_playwright.py - Playwright 浏览器安装脚本

**功能描述：**
自动安装 Playwright 浏览器二进制文件和系统依赖，支持开发和打包环境。

**主要特性：**
- 🌐 自动安装 Chromium 和 Firefox
- 🔧 安装系统依赖
- ✅ 安装验证和测试
- 📂 智能路径检测

**使用方法：**

```bash
# 安装所有浏览器
python scripts/install_playwright.py
```

**安装内容：**
1. **Chromium 浏览器** - 主要用于 Web 自动化
2. **Firefox 浏览器** - 备用浏览器选择
3. **系统依赖** - 浏览器运行所需的系统库

**安装路径：**
- 开发环境：`.browsers/`
- 打包环境：可执行文件同目录的 `ms-playwright/`

**验证功能：**
脚本会自动测试浏览器是否能正常启动并访问网页。

---

### 4. version_info.txt - 版本信息配置

**功能描述：**
Windows 可执行文件的版本信息配置文件，用于 PyInstaller 打包。

**包含信息：**
- 公司名称：Agent-Customer
- 文件描述：电商AI客服助手
- 版本号：0.1.0.0
- 版权信息：Copyright (C) 2025 Agent-Customer Team
- 产品名称：Agent-Customer

**修改方法：**
如需修改版本信息，请编辑相应字段：
```python
StringStruct(u'FileVersion', u'0.1.0.0'),  # 修改版本号
StringStruct(u'CompanyName', u'你的公司名'),  # 修改公司名
```

---

### 5. capture_pdd_ws.py - 拼多多 WebSocket 抓包

**功能描述：**
复用项目已有的 Playwright 持久化登录态（`user_data/`），在真实浏览器会话里通过
CDP 监听 `wss://*-ws.pinduoduo.com/` 的全部帧，落盘为 JSONL。

**使用示例：**
```bash
# 列出数据库里的拼多多账号
python scripts/capture_pdd_ws.py --list

# 抓 10 分钟（窗口打开后如会话过期，手动登录一次即可）
python scripts/capture_pdd_ws.py --account <shop_id>:<user_id> --duration 600
```

脚本会先等 WebSocket 真正建立再开始计时，避免把抓包时长浪费在人工登录上。
抓包文件含买家昵称、商品信息等业务数据（`access_token` 会自动脱敏），分享前请自查。

### 6. decode_pdd_ws.py - 抓包离线解码

**功能描述：**
把抓包 JSONL 解码成可读格式：拆开 16 字节帧头 + protobuf 体，并解开
`titan.notifyDataLite` 的通知负载（大块 gzip / 小块明文两种变体），
还原内嵌 JSON，同时关联「通知 ↔ ACK」。

**使用示例：**
```bash
python scripts/decode_pdd_ws.py temp/pdd_ws_20260920_163006.jsonl
```

输出到 `temp/decoded/`：

- `<名字>.decoded.json` —— 逐帧完整解码（含帧头、方法名、序号）
- `<名字>.messages.json` —— 只含聊天消息，按时间排序

解码器不联网、不依赖 DI 容器，纯离线分析。

抓包得到的完整接口清单（WS 命令、HTTP 端点、错误码、消息类型）整理在
`docs/reference/pinduoduo-chat-protocol.md`。

---

## 🔄 构建流程推荐

### 开发阶段
```bash
# 1. 首次设置
python scripts/install_playwright.py

# 2. 快速测试构建
python scripts/build_win_exe.py
```

### 发布阶段
```bash
# 1. 检查依赖
python scripts/build_exe.py --check-only

# 2. 完整构建
python scripts/build_exe.py --clean --installer

# 3. 生成安装包（使用 NSIS）
# 打开 installer.nsi 文件，使用 NSIS 编译器生成安装包
```

### 调试阶段
```bash
# 调试模式构建
python scripts/build_exe.py --mode debug --clean
```

---

## ⚠️ 注意事项

### 依赖问题
- 确保所有必要的文件都存在于项目根目录
- `agent_customer.spec` 文件当前不存在，需要先创建
- 网络连接问题可能影响 Playwright 浏览器下载

### 构建问题
- 构建过程可能需要几分钟时间
- 如果构建失败，尝试使用 `--clean` 参数清理缓存
- 杀毒软件可能误报，需要添加白名单

### 平台支持
- 构建脚本主要针对 Windows 平台设计
- 在其他平台运行可能需要修改路径和命令

### 包大小
- 外置浏览器模式可显著减小可执行文件体积
- 用户需要单独安装浏览器或使用系统浏览器

---

## 🛠️ 故障排除

### 常见错误及解决方案

**1. uv 命令未找到**
```bash
# 解决方案：安装 uv
pip install uv
```

**2. 找不到 agent_customer.spec 文件**
```bash
# 需要先创建 PyInstaller spec 文件
# 可以使用以下命令生成模板：
pyi-makespec app.py --onefile --windowed --name AgentCustomer
```

**3. Playwright 浏览器安装失败**
```bash
# 检查网络连接
# 手动设置代理（如果需要）
set PLAYWRIGHT_DOWNLOAD_HOST=https://playwright.azureedge.net
python scripts/install_playwright.py
```

**4. 构建失败：找不到图标文件**
```bash
# 确保图标文件存在
icon/icon.ico
```

**5. 可执行文件运行报错**
```bash
# 检查依赖是否完整
# 使用调试模式查看详细错误
python scripts/build_exe.py --mode debug
```

---

## 📝 开发者信息

- **项目名称：** Agent-Customer
- **版本：** 0.1.0
- **描述：** 电商AI客服助手
- **主要技术栈：** Python, PyQt6, Playwright, OpenAI, LanceDB

---

## 📞 技术支持

如果遇到问题：
1. 查看本文档的故障排除部分
2. 检查项目的 GitHub Issues
3. 提交新的 Issue 并包含详细错误信息

---

*最后更新：2025年12月*