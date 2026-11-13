"""
应用程序入口点

全局单例初始化顺序（重要）：
1. config           → 必须在最前面，其他模块都依赖配置
2. DI 容器           → 通过 configure_standard_services() 统一注册所有服务
3. db_manager       → 通过 DI 容器获取
4. logger           → 日志系统，依赖 config
5. queue_manager    → 通过 DI 容器获取
6. message_consumer_manager → 通过 DI 容器获取
7. status_manager   → 通过 DI 容器获取（ConnectionStatusManager 单例）

关键原则：
- config 必须最先初始化
- DI 容器通过 configure_standard_services() 统一管理所有服务的生命周期
- UI 模块在 main() 中通过延迟加载初始化
- 业务模块间通过延迟导入（lazy import）避免循环依赖
- PDDChannel 每个 AutoReplyThread 独立实例，共享 ConnectionStatusManager
"""
import sys
import ctypes
import os
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

# ============================================================================
# 全局单例预初始化（确保正确的初始化顺序）
# ============================================================================
# 1. 配置必须最先加载
from config import config as _app_config

# 2. 数据库管理器（通过 DI 代理，懒加载）
from database import db_manager as _app_db_manager

# 3. 日志系统（依赖配置）
from utils.logger_loguru import get_logger as _get_logger
from utils.runtime_path import get_data_path

# 4. 配置标准服务到 DI 容器（必须在其他业务模块导入前执行）
from core.di_container import configure_standard_services
configure_standard_services(_app_config)

# ============================================================================

import time

# 设置 Playwright 浏览器路径（支持打包后的 exe）
def get_project_root():
    """获取项目根目录（支持 PyInstaller 打包后的 exe）"""
    import sys
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的 exe
        return Path(sys._MEIPASS).parent
    return Path(__file__).resolve().parent

def setup_playwright_browsers_path():
    """设置 Playwright 浏览器安装路径"""
    project_root = get_project_root()
    browsers_path = project_root / ".browsers"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_path)
    return browsers_path


def kill_orphan_browsers():
    """清掉上次强退遗留的、占用自家 profile 的浏览器进程。

    本程序启动的 Edge/Chrome 都带 ``--user-data-dir=<数据目录>\\user_data\\...``。
    app 被强杀（任务管理器/崩溃）时这些浏览器会成为孤儿继续存活，而
    Chromium 单例锁会让下次同账号登录的浏览器秒退——pdd/抖音登录四条
    启动路径全部失败，用户只会看到「登录验证失败」。app 刚启动时自家
    profile 不可能有活着的浏览器，凡命令行带 user_data 根目录的都是孤儿，
    直接清掉（用户自己开的浏览器用的是 Edge 默认目录，不受影响）。

    注意：干净走 Win32 API 读命令行，不 spawn powershell——GUI 程序调控制台
    程序会弹黑框，用户看到的就是「一打开先闪一个命令行窗口」。
    """
    from utils.win_process import kill_by_commandline

    try:
        user_data_root = str(get_data_path() / "user_data")
    except Exception:
        return
    try:
        killed = kill_by_commandline(user_data_root, ("msedge.exe", "chrome.exe"))
    except Exception:
        return
    if killed > 0:
        _get_logger("App").info(f"已清理上次残留的孤儿浏览器进程 {killed} 个")


def main():
    """应用程序主函数"""
    # 设置 Playwright 浏览器路径
    browsers_path = setup_playwright_browsers_path()

    # 清理上次强退遗留的孤儿浏览器（占着 profile 锁会导致登录秒退）
    kill_orphan_browsers()

    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("Agent-Customer")

    # 创建主窗口
    logger = _get_logger("App")
    logger.info("应用程序启动...")

    t0 = time.perf_counter()
    t_import = time.perf_counter()
    from ui.main_ui import MainWindow  # noqa: F401
    logger.info(f"  MainWindow 模块导入耗时: {time.perf_counter() - t_import:.2f}s")
    t_window = time.perf_counter()
    window = MainWindow()
    window.show()
    logger.info(f"  MainWindow 实例化耗时: {time.perf_counter() - t_window:.2f}s")
    logger.info(f"窗口创建与显示总耗时: {time.perf_counter() - t0:.2f}s")

    # 将窗口设为应用级别的变量，防止被垃圾回收
    app.main_window = window

    # 运行 Qt 事件循环
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
