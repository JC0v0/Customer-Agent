# Agent-Customer 脚本说明

本文档说明 `scripts/` 目录下的辅助脚本。当前项目使用 `uv` 管理依赖，主要运行入口是根目录的 `app.py`。

## 脚本列表

```text
scripts/
├── agent_customer.spec       # PyInstaller spec 文件
├── build_exe.py              # 完整构建脚本
├── build_win_exe.py          # Windows 快速构建脚本
├── install_playwright.py     # Playwright 浏览器安装脚本
├── version_info.txt          # Windows 可执行文件版本信息
└── README.md                 # 当前说明文档
```

## install_playwright.py

用于安装 Playwright 浏览器二进制文件。拼多多登录和 Cookie 刷新依赖浏览器自动化能力，首次运行前建议执行。

```bash
uv run python scripts/install_playwright.py
```

开发环境下浏览器默认安装到：

```text
.browsers/
```

注意：浏览器下载依赖网络环境，如失败请检查代理或 Playwright 下载源配置。

## build_win_exe.py

Windows 快速构建脚本，适合日常构建和验证。

要求：

- Windows 系统
- 已安装 `uv`
- 项目根目录存在 `scripts/agent_customer.spec`

用法：

```bash
python scripts/build_win_exe.py
python scripts/build_win_exe.py --python 3.11
python scripts/build_win_exe.py --clean
```

脚本会执行：

1. 检查当前平台是否为 Windows。
2. 检查 `uv` 是否可用。
3. 如果不存在 `.venv`，创建 Python 3.11 虚拟环境。
4. 执行 `uv sync` 安装依赖。
5. 安装 PyInstaller。
6. 使用 `scripts/agent_customer.spec` 构建。

输出目录：

```text
dist/AgentCustomer/
```

## build_exe.py

完整构建脚本，包含依赖文件检查、构建、分发文件生成和可选 NSIS 安装脚本生成。

常用命令：

```bash
python scripts/build_exe.py
python scripts/build_exe.py --clean
python scripts/build_exe.py --mode debug
python scripts/build_exe.py --installer
python scripts/build_exe.py --check-only
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--python 3.11` | 指定 Python 版本 |
| `--mode release` | 默认发布模式 |
| `--mode debug` | 调试模式 |
| `--clean` | 构建前删除 `build/` 和 `dist/` |
| `--installer` | 生成 NSIS 安装脚本 |
| `--check-only` | 只检查必要文件，不执行构建 |

当前脚本检查的必要文件包括：

```text
app.py
config.json
icon/icon.ico
scripts/agent_customer.spec
pyproject.toml
```

其中 `config.json` 通常由首次运行应用自动生成。如果只做干净构建环境，可以先运行一次应用生成配置，或按根目录 README 的配置说明手动准备配置文件。

## agent_customer.spec

PyInstaller spec 文件，定义应用入口、资源收集、hiddenimports 和输出结构。

如构建失败，请优先检查：

- `pyproject.toml` 中依赖是否安装完整。
- spec 文件里的 hiddenimports 是否仍符合当前代码。
- Playwright 浏览器是否采用外置安装方式。
- `icon/icon.ico` 是否存在。

## version_info.txt

Windows 可执行文件版本信息。当前文件内版本信息可能与 `pyproject.toml` 的项目版本不同，发布前应统一：

```text
pyproject.toml: [project].version
scripts/version_info.txt: FileVersion / ProductVersion
构建脚本中的 APP_VERSION
```

## 推荐流程

### 开发运行

```bash
uv sync
uv run python scripts/install_playwright.py
uv run python app.py
```

### Windows 快速构建

```bash
python scripts/build_win_exe.py --clean
```

### Windows 完整构建

```bash
python scripts/build_exe.py --check-only
python scripts/build_exe.py --clean
```

如需要 NSIS 安装脚本：

```bash
python scripts/build_exe.py --clean --installer
```

## 注意事项

- Windows EXE 构建请在 Windows 系统执行。
- 当前项目没有 `requirements.txt`，依赖来源是 `pyproject.toml` 和 `uv.lock`。
- 构建产物、临时目录、日志、数据库和本地配置不应提交到仓库。
- `config.json` 可能包含 LLM API Key，不要放入公开发布包或代码仓库。
- 如需对外发布，请先补齐 License、版本号统一、安全存储和安装说明。
