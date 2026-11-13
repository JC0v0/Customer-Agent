"""
抖音账号登录认证（结构与拼多多 pdd_login 对齐）

【2026-09-04 已实现】
抖店登录页 https://fxg.jinritemai.com/login/common 默认「手机登录」标签：
- 手机号 + 短信验证码（主路径；验证码需用户在浏览器内手动输入，无法自动读短信）
- 邮箱登录（备选；用户名含 @ 时走 邮箱 + 密码）

DouyinLogin.login() 流程：
  1. 打开登录页，按 URL 判断登录态（已登录访问 login 页会自动跳走）
  2. 未登录 → 手机号+验证码（或邮箱+密码）登录；滑块/短信验证需用户手动一次
  3. 登录成功 → 自动挂 WS 抓取 pigeon 会话凭据（token + pigeon_sign + device_id）
  4. 保存到按账号隔离的临时会话文件（pigeon 传输模块直接复用，一次登录管长期）
  5. 返回含 pigeon 会话 + 店铺信息的 JSON（键名与拼多多契约一致）

浏览器启动（本地 Chrome/Edge 探测、CDP 回退、profile 锁清理等）沿用原实现。
"""
import inspect
import os
import sqlite3
# 必须在导入 playwright 之前设置浏览器路径
from pathlib import Path
from utils.path_utils import get_app_dir
from utils.runtime_path import get_data_path
from utils.logger_loguru import get_logger
from Channel.douyin.session_paths import (
    legacy_pigeon_session_path,
    pigeon_capture_path,
    pigeon_session_path,
)

# 设置 Playwright 浏览器路径
app_dir = get_app_dir()
data_dir = get_data_path()
browsers_path = app_dir / ".browsers"
if browsers_path.exists():
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_path)
    logger_temp = get_logger("Douyin_login_init")
    logger_temp.info(f"设置 Playwright 浏览器路径: {browsers_path}")
else:
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.path.join(os.getenv("LOCALAPPDATA", ""), "ms-playwright")

# PyInstaller 打包（frozen）时，Playwright 默认用 inspect.getfile 定位驱动，
# 会指向构建机路径而失效。这里显式接管：node.exe 走环境变量，cli.js 走 monkeypatch。
import sys as _sys
if getattr(_sys, "frozen", False):
    _driver_dir = Path(_sys._MEIPASS) / "playwright" / "driver"
    _node = _driver_dir / "node.exe"
    _cli = _driver_dir / "package" / "cli.js"
    if _node.exists():
        os.environ["PLAYWRIGHT_NODEJS_PATH"] = str(_node)
    if _cli.exists():
        from playwright._impl import _driver as _pw_driver
        _pw_driver.compute_driver_executable = lambda: (
            os.environ.get("PLAYWRIGHT_NODEJS_PATH", str(_node)),
            str(_cli),
        )
del _sys

import json
import base64
import hashlib
import asyncio
import sys
import socket
import subprocess
import ctypes
import time
from ctypes import wintypes
import threading
import urllib.request
import re
import shutil
from urllib.parse import urlsplit, parse_qs
from typing import Optional, Dict, Any, List, Tuple
from database import db_manager
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


def _profile_scope_identity(profile_scope: Optional[str]):
    """Extract the expected channel/shop/user identity from a profile key."""
    if profile_scope is None:
        return None
    parts = str(profile_scope).split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[0], parts[1], parts[2]


def _profile_scope_matches(profile_scope: Optional[str], shop_id, user_id) -> bool:
    """Fail closed when a scoped browser session resolves to another account."""
    if profile_scope is None:
        return True
    expected = _profile_scope_identity(profile_scope)
    return bool(
        expected
        and expected[0] == "douyin"
        and str(expected[1]) == str(shop_id)
        and str(expected[2]) == str(user_id)
    )


def _profile_scope_matches_username(profile_scope: Optional[str], username) -> bool:
    """DB 中该 profile 对应的账号用户名 == 登录用户名 → 视为同一账号。

    场景：店铺名从登录手机号修正为真实店名后，shop_id 由
    douyin_<手机号> 变为 douyin_<店铺名>，严格 shop_id 匹配会误拒。
    同一用户名必然对应同一账号，用户名匹配是安全的身份放行。
    """
    if profile_scope is None:
        return False
    expected = _profile_scope_identity(profile_scope)
    if not expected or expected[0] != "douyin":
        return False
    try:
        from database import db_manager

        acct = db_manager.get_account(expected[0], expected[1], expected[2])
        if not acct:
            return False
        return str(acct.get("username") or "") == str(username or "")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 进程清理：全部走 Win32 API，不启动 taskkill / netstat 这类外部程序。
#
# 原因（真机实测）：打包后的程序把 _internal 目录塞进了 PATH，系统自带的
# taskkill.exe 启动时会加载到冲突 DLL，直接报 0xc0000142「应用程序无法正常
# 启动」；而且无控制台的 GUI 程序每 spawn 一次就弹一个黑窗。于是清理完全
# 失效、窗口越弹越多。改用 kernel32 / iphlpapi 的接口后既不会弹窗，也不受
# 打包环境的 PATH 影响。
# ---------------------------------------------------------------------------
_IS_WINDOWS = sys.platform == "win32"

_PROCESS_TERMINATE = 0x0001
_TH32CS_SNAPPROCESS = 0x00000002
_AF_INET = 2
_TCP_TABLE_OWNER_PID_LISTENER = 3

if _IS_WINDOWS:
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)

    # 64 位下句柄必须声明成 HANDLE，否则默认 c_int 会截断
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.TerminateProcess.restype = wintypes.BOOL
    _kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    class _MIB_TCPROW_OWNER_PID(ctypes.Structure):
        _fields_ = [
            ("dwState", wintypes.DWORD),
            ("dwLocalAddr", wintypes.DWORD),
            ("dwLocalPort", wintypes.DWORD),
            ("dwRemoteAddr", wintypes.DWORD),
            ("dwRemotePort", wintypes.DWORD),
            ("dwOwningPid", wintypes.DWORD),
        ]

    def _all_process_parent_pairs():
        """取一份 (pid, 父pid) 快照。"""
        pairs = []
        snapshot = _kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        invalid = ctypes.c_void_p(-1).value
        if not snapshot or snapshot == invalid:
            return pairs
        try:
            entry = _PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
            ok = _kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while ok:
                pairs.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID)))
                ok = _kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        except Exception:
            return pairs
        finally:
            _kernel32.CloseHandle(snapshot)
        return pairs

    def _child_pids(parent_pid):
        """递归收集所有子孙进程 pid。"""
        pairs = _all_process_parent_pairs()
        children = []
        frontier = [int(parent_pid)]
        while frontier:
            current = frontier.pop()
            for pid, ppid in pairs:
                if ppid == current and pid not in children:
                    children.append(pid)
                    frontier.append(pid)
        return children

    def _terminate_pid(pid):
        handle = _kernel32.OpenProcess(_PROCESS_TERMINATE, False, int(pid))
        if not handle:
            return
        try:
            _kernel32.TerminateProcess(handle, 1)
        finally:
            _kernel32.CloseHandle(handle)

else:
    def _child_pids(parent_pid):
        return []

    def _terminate_pid(pid):
        return None


def _kill_process_tree(pid):
    """强杀进程树（含子孙）。pid 已退出时静默忽略。"""
    if not pid or not _IS_WINDOWS:
        return
    try:
        targets = _child_pids(pid)
    except Exception:
        targets = []
    targets.append(int(pid))
    for target in targets:
        try:
            _terminate_pid(target)
        except Exception:
            pass


def _pid_listening_on_port(port):
    """从调试端口反查浏览器真身 pid。

    启动器进程退出后，唯一还和我们对得上的线索就是那个调试端口——
    它的监听者就是真正的浏览器进程。直接读系统 TCP 表，不调 netstat。
    """
    if not port or not _IS_WINDOWS:
        return None
    try:
        size = wintypes.DWORD(0)
        _iphlpapi.GetExtendedTcpTable(
            None, ctypes.byref(size), False, _AF_INET,
            _TCP_TABLE_OWNER_PID_LISTENER, 0,
        )
        if not size.value:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        result = _iphlpapi.GetExtendedTcpTable(
            buffer, ctypes.byref(size), False, _AF_INET,
            _TCP_TABLE_OWNER_PID_LISTENER, 0,
        )
        if result != 0:
            return None
        count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD)).contents.value
        first_row = ctypes.byref(buffer, ctypes.sizeof(wintypes.DWORD))
        rows = ctypes.cast(first_row, ctypes.POINTER(_MIB_TCPROW_OWNER_PID))
        wanted = int(port)
        for index in range(count):
            row = rows[index]
            # dwLocalPort 是网络字节序，低 16 位才是端口
            local_port = ((row.dwLocalPort & 0xFF) << 8) | ((row.dwLocalPort >> 8) & 0xFF)
            if local_port == wanted:
                return int(row.dwOwningPid)
    except Exception:
        return None
    return None


class _CdpContextHandle:
    """包装 CDP 连接的 context，close() 时同时终止手动启动的浏览器进程。

    两个坑导致关闭失败、窗口一直留在桌面上：

    1. connect_over_cdp 连上的浏览器属于「连接」而非「启动」，
       browser.close() 只断开连接，不会关窗口，必须发 CDP 的 Browser.close。
    2. 手动 Popen 出来的 msedge 启动器进程常常在移交工作后立刻退出，
       proc.poll() 不再是 None，原来的 `poll() is None` 判断把强杀整段跳过，
       真正的浏览器进程就成了没人管的孤儿。

    残留进程还会一直占着 user_data 目录，导致之后 migrate_profile 复制 profile
    长期失败（每次登录白等十几秒）。所以这里改成：先 CDP 关浏览器，再按 pid 强杀，
    最后按调试端口反查真身兜底。
    """

    def __init__(self, context, proc, browser=None, user_data_dir=None, port=None):
        self._context = context
        self._proc = proc
        self._browser = browser
        self._user_data_dir = user_data_dir
        self._port = port

    def __getattr__(self, name):
        return getattr(self._context, name)

    async def close(self):
        session = None
        pids = []
        if self._browser is not None:
            try:
                session = await self._browser.new_browser_cdp_session()
                info = await session.send("SystemInfo.getProcessInfo")
                processes = (info or {}).get("processInfo") or []
                for item in processes:
                    pid = item.get("id")
                    if pid:
                        pids.append(int(pid))
            except Exception:
                session = None

        # 让浏览器自己退出（Browser.close 会关掉所有窗口和子进程）
        if session is not None:
            try:
                await session.send("Browser.close")
            except Exception:
                pass
        try:
            await self._context.close()
        except Exception:
            pass

        # 兜底强杀：CDP 报出来的每个 pid
        # 启动器只在还活着时才按 pid 杀：它可能早就退出，pid 被系统回收后
        # 再 OpenProcess 就会误伤别的进程。
        if self._proc is not None and self._proc.poll() is None:
            await asyncio.to_thread(_kill_process_tree, self._proc.pid)
        for pid in pids:
            await asyncio.to_thread(_kill_process_tree, pid)

        # 最后一道：CDP 全程没连上时（pids 为空），按调试端口找真身再杀一次。
        # 这一步是「窗口一直不关」的最终防线。
        if not pids:
            pid_by_port = await asyncio.to_thread(_pid_listening_on_port, self._port)
            if pid_by_port:
                await asyncio.to_thread(_kill_process_tree, pid_by_port)


def _captured_frame_cmds(frames: List[Dict[str, Any]]) -> set:
    """解析录帧列表里出现过的命令号集合（顶层 f1 varint）。"""
    try:
        from tools.reverse_probe.douyin.pigeon_proto import parse_fields
    except Exception:
        return set()
    cmds = set()
    for item in frames:
        try:
            fields = parse_fields(base64.b64decode(item.get("b64") or ""))
            cmd = next((v for f, wt, v, _ in fields if f == 1 and wt == 0), None)
        except Exception:
            continue
        if cmd is not None:
            cmds.add(int(cmd))
    return cmds


class DouyinLogin():
    # 抖店登录页 / 飞鸽工作台（2026-09-04 实测定点）
    LOGIN_URL = "https://fxg.jinritemai.com/login/common"
    WORKSPACE_URL = "https://im.jinritemai.com/pc_seller_v2/main/workspace"
    DEFAULT_ACCESS_KEY = "975a9f0608893b4a902dbe8a8532f6c8"
    # pigeon 录帧合格线：握手帧命令号小于注册帧 10011。帧号随客户端版本
    # 变（10008/10009、10010、10005/10006 都实测出现过），不再写死。
    # 10011 注册帧不要求（shop_id 在加密 blob 里，无法跨会话合成；
    # 缺它只少 WS 实时推送，消息正文有 HTTP 拉取兜底）。
    # 抓 pigeon 凭据的最大轮数。首轮常因「登录刚完成、会话 cookie 未落地」
    # 被工作台踢回，抓不到 token；第二轮必成功（真机 2026-11-13）。
    CAPTURE_ATTEMPTS = 2

    def __init__(self, name, password, profile_scope=None):
        self.logger = get_logger("Douyin_login")
        self.channel_name = "douyin"  # 渠道名称固定为"douyin"
        self.base_url = DouyinLogin.LOGIN_URL
        self.name = name
        self.password = password
        self.profile_scope = profile_scope
        self.pigeon_session: Optional[Dict[str, Any]] = None
        self.shop_name: Optional[str] = None
        self.shop_id: Optional[str] = None
        self.shop_logo: Optional[str] = None

    @staticmethod
    def _dir_for_raw_name(raw_name: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._")[:48]
        safe_name = safe_name or "account"
        digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:12]
        return data_dir / "user_data" / f"{safe_name}-{digest}"

    def _legacy_profile_dir(self, profile_scope: Optional[str]) -> Path:
        """旧版按 douyin:shop:user 算出来的 profile 目录。"""
        return self._dir_for_raw_name(str(profile_scope or "").strip())

    def _profile_dir(self, profile_scope: Optional[str] = None) -> Path:
        """登录用的浏览器 profile 永远只按登录名算一个目录。

        2026-11-14 修复「每次启动都要重新发短信验证码」：旧实现把
        profile_scope 混进路径，添加账号时 login_douyin 不传 scope（写
        按用户名算的 A 目录），运行时刷新/重登传 scope（读按 shop:user
        算的 B 目录）。B 是空壳，打开登录页不会自动跳走，于是被判成
        未登录去要验证码；migrate_profile 又因为 B 已存在而整段跳过。
        浏览器登录态跟店铺维度无关，统一到一个目录即可根治。协议凭据
        dy_session_* / dy_send_capture_* 仍按 scope 隔离，互不影响。
        """
        return self._dir_for_raw_name(str(self.name or "").strip())

    def migrate_profile(self, profile_scope: str) -> None:
        """把旧版 scope 目录里的登录态迁到统一目录（只补关键文件）。

        Edge 的 cookie 值由 Local State 里的 DPAPI 密钥加密，所以这两个
        文件必须成对复制；只搬 Cookies 会解不出值。统一目录已经有
        Cookies 时不覆盖，避免用旧会话盖掉新会话。
        """
        if not profile_scope:
            return
        source = self._legacy_profile_dir(profile_scope)
        target = self._profile_dir()
        if source.resolve() == target.resolve():
            return
        source_cookies = source / "Default" / "Network" / "Cookies"
        target_cookies = target / "Default" / "Network" / "Cookies"
        if not source_cookies.exists() or target_cookies.exists():
            return
        rel_files = [
            Path("Local State"),
            Path("Default") / "Network" / "Cookies",
            Path("Default") / "Preferences",
        ]
        copied = 0
        try:
            for rel in rel_files:
                src = source / rel
                if not src.exists():
                    continue
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
        except OSError as exc:
            self.logger.warning(
                f"browser profile migration incomplete: error_type={type(exc).__name__}"
            )
            return
        if copied:
            self.logger.info(
                f"已迁移旧版浏览器登录态: {source.name} -> {target.name}"
            )

    def _saved_session_candidates(self):
        """按新到旧列出可能存着完整 cookies 的会话文件。"""
        paths = []
        if self.profile_scope:
            paths.append(pigeon_session_path(self.profile_scope, self.name))
        paths.append(pigeon_session_path(None, self.name))
        paths.append(legacy_pigeon_session_path())
        return paths

    # 判定「浏览器里还留着登录态」必须看关键 cookie，不能只看 Cookies 文件
    # 在不在。CDP 收尾会发 Browser.close 再强杀进程树，cookie 常来不及
    # 落盘：2026-11-14 03:36 真机 profile 里只剩 14 条基础项，文件依然
    # 存在且非空，但 sessionid / sid_guard / passport_auth_status 全没有，
    # 点「验证」就又停在登录页去发短信验证码。
    _LOGIN_COOKIE_MARKERS = ("sessionid", "sid_guard", "passport_auth_status")

    def _has_valid_login_cookies(self) -> bool:
        """统一 profile 里是否存着能免验证的关键登录 cookie。"""
        cookies = self._profile_dir() / "Default" / "Network" / "Cookies"
        if not cookies.exists() or cookies.stat().st_size == 0:
            return False
        uri = "file:" + str(cookies).replace("\\", "/") + "?mode=ro"
        try:
            con = sqlite3.connect(uri, uri=True, timeout=1.0)
            rows = con.execute("select name from cookies").fetchall()
            con.close()
        except Exception as exc:
            self.logger.debug(
                f"读取 profile cookies 失败，按无登录态处理: "
                f"error_type={type(exc).__name__}"
            )
            return False
        for row in rows:
            name = str(row[0]).lower()
            for marker in self._LOGIN_COOKIE_MARKERS:
                if marker in name:
                    return True
        return False

    async def _inject_saved_cookies(
        self, context, profile_has_login_cookies: bool = False
    ) -> int:
        """把上次登录抓到的完整 cookies 注入当前浏览器上下文。

        只在 profile 里没有关键登录 cookie 时注入：session 文件里的 cookies
        可能比浏览器里的旧，无条件覆盖会把有效会话盖坏。真机复现结论
        （2026-11-14 03:40）：profile 只剩 14 条基础 cookie 时访问登录页
        会停住，注入 dy_session_*.json 的 70 条后自动跳到
        ffa/mshop/homepage，也就是免掉了一次短信验证码。
        domain 沿用页面桥里真机验证过的 .jinritemai.com。
        """
        if profile_has_login_cookies:
            return 0
        cookies_full = None
        for path in self._saved_session_candidates():
            try:
                if not path.exists():
                    continue
                raw = json.loads(path.read_text(encoding="utf-8"))
                candidate = raw.get("cookies_full")
                if isinstance(candidate, dict) and candidate:
                    cookies_full = candidate
                    break
            except Exception as exc:
                self.logger.debug(
                    f"读取会话 cookies 跳过: error_type={type(exc).__name__}"
                )
        if not cookies_full:
            return 0
        payload = []
        for key in cookies_full:
            payload.append({
                "name": str(key),
                "value": str(cookies_full[key]),
                "domain": ".jinritemai.com",
                "path": "/",
                "secure": True,
                "httpOnly": False,
            })
        try:
            result = context.add_cookies(payload)
            if inspect.isawaitable(result):
                await result
            self.logger.info(f"已注入已保存的浏览器 cookies: {len(payload)} 个")
            return len(payload)
        except Exception as exc:
            self.logger.warning(
                f"注入浏览器 cookies 失败: error_type={type(exc).__name__}"
            )
            return 0

    # 本地浏览器启动参数（Chrome/Edge 通用）
    _LAUNCH_ARGS = [
        '--disable-blink-features=AutomationControlled',
        '--disable-notifications',
        '--disable-dev-shm-usage',
    ]

    _BROWSER_PATHS = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"{localappdata}\Google\Chrome\Application\chrome.exe",
    ]

    @staticmethod
    def _resolve_browser_path():
        """探测本地浏览器 exe 路径。返回 (path, channel_name) 或 (None, None)。"""
        localappdata = os.getenv("LOCALAPPDATA", "")
        for raw in DouyinLogin._BROWSER_PATHS:
            p = raw.replace("{localappdata}", localappdata)
            if os.path.isfile(p):
                channel = "msedge" if "Edge" in p else "chrome"
                return p, channel
        return None, None

    def _clean_profile_locks(self, user_data_dir: str) -> None:
        """启动前清理 user_data_dir 中 Chromium 的单例锁文件。"""
        if not user_data_dir:
            return
        profile_dir = Path(user_data_dir)
        removed = []
        for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
            try:
                (profile_dir / name).unlink(missing_ok=True)
                removed.append(name)
            except Exception:
                pass
        if removed:
            self.logger.debug(f"已清理 profile 锁文件 {removed}: {user_data_dir}")

    async def _launch_context(self, playwright, user_data_dir: str, headless: bool):
        """启动用户本地的浏览器，避免下载 Playwright 自带的 Chromium。"""
        exe_path, channel = self._resolve_browser_path()
        if exe_path:
            self.logger.info(f"探测到本地浏览器: {channel} -> {exe_path}")
        else:
            self.logger.warning("未探测到本地浏览器 exe，将依次尝试 channel / CDP / 自带 Chromium")
        attempts = []

        self._clean_profile_locks(user_data_dir)

        # The Feige micro-frontend can render blank in a Playwright-launched
        # Edge window. Launching normal Edge first and attaching over CDP keeps
        # the visible login flow usable for SMS and slider verification.
        if exe_path and not headless:
            handle = await self._launch_via_cdp(
                playwright, exe_path, user_data_dir, headless=False
            )
            if handle is not None:
                return handle
            attempts.append(f"{channel}(normal-cdp): unavailable")

        if exe_path:
            try:
                return await playwright.chromium.launch_persistent_context(
                    user_data_dir,
                    executable_path=exe_path,
                    headless=headless,
                    args=self._LAUNCH_ARGS,
                    timeout=30000,
                )
            except Exception as e:
                attempts.append(f"{channel}({exe_path}): {type(e).__name__}: {e}")
                self.logger.warning(f"启动本地浏览器 {channel} 失败: {type(e).__name__}: {e}")

                handle = await self._launch_via_cdp(playwright, exe_path, user_data_dir, headless)
                if handle is not None:
                    return handle
                attempts.append(f"{channel}(cdp-over-port): 启动后未就绪")

        if not exe_path:
            for ch in ("msedge", "chrome"):
                try:
                    return await playwright.chromium.launch_persistent_context(
                        user_data_dir,
                        channel=ch,
                        headless=headless,
                        args=self._LAUNCH_ARGS,
                        timeout=30000,
                    )
                except Exception as e:
                    attempts.append(f"{ch}(channel): {type(e).__name__}: {e}")
                    self.logger.warning(f"启动本地浏览器 {ch} 失败: {type(e).__name__}: {e}")

        if exe_path:
            handle = await self._launch_via_cdp(playwright, exe_path, user_data_dir, headless)
            if handle is not None:
                return handle
            attempts.append(f"{channel}(cdp-over-port): 启动后未就绪")

        try:
            return await playwright.chromium.launch_persistent_context(
                user_data_dir,
                headless=headless,
                args=self._LAUNCH_ARGS,
                timeout=30000,
            )
        except Exception as e:
            attempts.append(f"chromium(bundled): {type(e).__name__}: {e}")
            self.logger.warning(f"启动 Playwright Chromium 失败: {type(e).__name__}")

        raise RuntimeError(
            "未找到可用的浏览器。尝试过的方案：\n  - " +
            "\n  - ".join(attempts) +
            "\n请安装 Google Chrome 或 Microsoft Edge 后重试。"
        )

    async def _launch_via_cdp(self, playwright, exe_path, user_data_dir, headless):
        """手动启动浏览器（remote-debugging-port），再 connect_over_cdp 连接。"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        args = [
            exe_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            "--no-first-run", "--no-default-browser-check",
            "--disable-blink-features=AutomationControlled",
            "--disable-notifications",
        ]
        if headless:
            args.append("--headless=new")
        args.append("about:blank")

        stderr_chunks: list = []
        try:
            proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except Exception as e:
            self.logger.warning(f"手动启动浏览器失败: {type(e).__name__}")
            return None

        drain_thread = threading.Thread(
            target=self._drain_proc_stderr, args=(proc, stderr_chunks), daemon=True
        )
        drain_thread.start()

        self.logger.debug(f"CDP 回退：浏览器已启动 pid={proc.pid} port={port}")
        ready = False
        for _ in range(50):
            if proc.poll() is not None:
                # 启动器退出不等于浏览器没起来（Edge 会移交给子进程），
                # 只要端口还有人听就继续等，否则清理掉再放弃。
                if _pid_listening_on_port(port):
                    ready = True
                    break
                self._log_proc_stderr(
                    stderr_chunks, drain_thread,
                    f"CDP 回退：浏览器进程已退出 pid={proc.pid} port={port}",
                )
                return None
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/version", timeout=1
                )
                ready = True
                break
            except Exception:
                await asyncio.sleep(0.3)

        if not ready:
            if proc.poll() is None:
                proc.terminate()
            _kill_process_tree(_pid_listening_on_port(port))
            self._log_proc_stderr(
                stderr_chunks, drain_thread,
                f"CDP 回退：等待 CDP 端点超时 port={port}",
            )
            return None

        try:
            browser = await playwright.chromium.connect_over_cdp(
                f"http://127.0.0.1:{port}"
            )
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            self.logger.info(f"CDP 回退连接成功（端口 {port}）")
            return _CdpContextHandle(
                context,
                proc,
                browser=browser,
                user_data_dir=user_data_dir,
                port=port,
            )
        except Exception as e:
            if proc.poll() is None:
                proc.terminate()
            # 启动器可能已退出，浏览器还活着占着 profile：按端口把真身找出来清掉
            _kill_process_tree(_pid_listening_on_port(port))
            self._log_proc_stderr(
                stderr_chunks, drain_thread,
                f"CDP 回退连接失败: {type(e).__name__}",
            )
            return None

    @staticmethod
    def _drain_proc_stderr(proc, out_chunks: list) -> None:
        """后台持续读取子进程 stderr，防止管道缓冲区写满阻塞浏览器进程。"""
        try:
            stream = proc.stderr
            if stream is None:
                return
            while True:
                chunk = stream.read1(4096)
                if not chunk:
                    break
                out_chunks.append(chunk)
        except Exception:
            pass

    def _log_proc_stderr(self, stderr_chunks: list, drain_thread, headline: str) -> None:
        """浏览器启动失败后读取并打印捕获的 stderr，定位秒退原因。"""
        try:
            drain_thread.join(timeout=2.0)
        except Exception:
            pass
        raw = b"".join(stderr_chunks)
        if not raw:
            self.logger.warning(f"{headline}\n浏览器 stderr 为空（未输出任何错误信息）")
            return
        text = raw.decode("utf-8", errors="replace").strip()
        if len(text) > 2000:
            text = text[:2000] + "...(截断)"
        self.logger.warning(f"{headline}\n浏览器 stderr:\n{text}")

    async def login(self, headless=False):
        """使用账号登录抖音商家后台，登录成功后自动抓取 pigeon 会话凭据。

        流程（2026-09-04 已按真实登录页结构实现）：
          1. 打开抖店登录页 https://fxg.jinritemai.com/login/common
          2. 按 URL 判断登录态：仍在登录页 → 未登录；已跳走 → 已登录（跳过）
          3. 登录方式按账号形态自动选择：
             - 手机号（用户名不含 @）→ 手机号 + 短信验证码（默认标签）：
               自动填手机号并点「发送验证码」，验证码需用户在浏览器内输入并点「登录」
             - 邮箱（用户名含 @）→ 邮箱 + 密码
          4. 等待登录成功：滑块验证/短信验证需用户手动完成一次（风控无法全自动）
          5. 跳转飞鸽工作台 → 挂 WS 自动抓取 token + pigeon_sign（会话级凭据）
          6. 保存到按账号隔离的临时会话文件（pigeon 传输模块直接复用）
          7. 返回包含 pigeon 会话 + 店铺信息的 JSON 字符串（键名与拼多多契约一致）

        Returns:
            str（JSON）或 False。JSON 结构：
            {"pigeon_session": {...}, "shop_name": str, "user_name": str}
        """
        playwright = None
        context = None
        try:
            self.migrate_profile(self.profile_scope)
            playwright = await async_playwright().start()
            user_data_dir = str(self._profile_dir())
            self.logger.debug(f"使用用户数据目录: {user_data_dir}")
            had_login_cookies = self._has_valid_login_cookies()
            context = await self._launch_context(playwright, user_data_dir, headless=headless)
            await self._inject_saved_cookies(context, had_login_cookies)
            page = await context.new_page()

            # ---- 1. 打开登录页，按 URL 判断登录态（未登录访问 workspace 会空白卡死，
            #        故不再用 workspace 探测；已登录访问 login 页会自动跳走） ----
            await page.goto(DouyinLogin.LOGIN_URL, wait_until="domcontentloaded", timeout=60000)
            # 轮询等登录态判定：已登录访问 login 页会自动跳走，一旦离开
            # 登录页立即继续（原来死等 4 秒）；真未登录时等满 4 秒再判定。
            for _ in range(13):
                await page.wait_for_timeout(300)
                if not self._is_on_login_page(page):
                    break
            if self._is_on_login_page(page):
                self.logger.info("未登录，执行账号登录…")
                logged = await self._do_login(page)
                if not logged:
                    self.logger.error("登录失败或超时（可能滑块/验证码未完成）")
                    return False
            else:
                self.logger.info("检测到已登录状态，跳过登录步骤")

            # ---- 3. 登录成功 → 抓取 pigeon 会话凭据 ----
            self.logger.info("登录成功，正在获取会话凭据…")
            session = await self._capture_pigeon_session(page)
            if not session:
                self.logger.error("会话凭据获取失败（token/sign）")
                return False
            self.pigeon_session = session
            self._save_pigeon_session(session)
            self.logger.info(f"会话凭据已保存: token={session.get('token','')[:14]}...")

            # ---- 4. 店铺/账号信息 ----
            # 优先通过 backstage/currentuser API 获取真实店铺名（DOM 选择器经常失效导致显示手机号）
            api_shop = await self._fetch_shop_name_from_api(page)
            if api_shop:
                shop_name = api_shop["shop_name"]
                self.shop_id = str(api_shop.get("shop_id") or "")
                self.shop_logo = api_shop.get("shop_logo") or ""
                self.logger.info(f"通过 currentuser API 获取到真实店铺名: {shop_name}")
            else:
                # API 失败时回退 DOM 提取，再兜底登录名
                shop_name = await self._extract_shop_name(page) or self.name
                self.logger.warning(f"currentuser API 未获取到店铺名，回退: {shop_name}")
            self.shop_name = shop_name
            return json.dumps({
                "pigeon_session": session,
                "shop_name": shop_name,
                "user_name": self.name,
                # backstage API 鉴权/风控参数（登录时一并抓取，供运行时直连商品/订单等 API）
                "cookies_full": session.get("cookies_full"),
                "verifyFp": session.get("verifyFp"),
                "msToken": session.get("msToken"),
            }, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"登录失败: {type(e).__name__}")
            return False
        finally:
            if context:
                try:
                    await context.close()
                except Exception:
                    pass
            if playwright:
                try:
                    await playwright.stop()
                except Exception:
                    pass

    # ---- 登录页交互 ----

    @staticmethod
    def _is_on_login_page(page) -> bool:
        url = (page.url or "").lower()
        return "fxg.jinritemai.com/login" in url

    @staticmethod
    def _is_on_workspace(page) -> bool:
        url = (page.url or "").lower()
        return "im.jinritemai.com/pc_seller_v2" in url or "fxg.jinritemai.com" in url and "login" not in url

    async def _do_login(self, page) -> bool:
        """按用户名形态选择登录方式：含 @ → 邮箱+密码；否则 → 手机号+短信验证码。"""
        if "@" in (self.name or ""):
            self.logger.info(f"账号 '{self.name}' 使用邮箱+密码登录")
            return await self._do_email_login(page)
        self.logger.info(f"账号 '{self.name}' 使用手机号+短信验证码登录")
        return await self._do_phone_login(page)

    async def _do_phone_login(self, page) -> bool:
        """手机号 + 短信验证码登录（默认标签）。

        自动填手机号并点「发送验证码」；短信码需用户在可见浏览器内输入并点「登录」
        （短信验证码无法自动读取，为唯一手动步骤）。
        """
        try:
            await page.locator("input[placeholder='手机号码']").first.fill(self.name, timeout=6000)
        except Exception as e:
            self.logger.error(f"手机号输入框定位失败: {type(e).__name__}")
            return False
        await self._check_agree(page)
        try:
            await page.get_by_text("发送验证码").first.click(timeout=6000)
        except Exception as e:
            self.logger.warning(f"发送验证码按钮点击异常: {type(e).__name__}")
        self.logger.info(
            f"账号 '{self.name}' 已自动填入手机号并请求验证码，"
            f"请在浏览器窗口内输入短信验证码，程序会自动检测登录成功"
        )
        return await self._wait_login_success(page, timeout=180)

    async def _do_email_login(self, page) -> bool:
        """邮箱+密码登录。"""
        try:
            await page.get_by_text("邮箱登录", exact=True).first.click(timeout=6000)
            await page.wait_for_timeout(1500)
        except Exception:
            self.logger.info("未找到邮箱登录标签（可能默认即邮箱），继续")

        # 填邮箱+密码
        try:
            await page.locator("input[placeholder='请输入邮箱']").first.fill(self.name, timeout=6000)
        except Exception as e:
            self.logger.error(f"邮箱输入框定位失败: {type(e).__name__}")
            return False
        try:
            await page.locator("input[placeholder='密码']").first.fill(self.password, timeout=6000)
        except Exception as e:
            self.logger.error(f"密码输入框定位失败: {type(e).__name__}")
            return False

        # 勾选协议（若需要）
        await self._check_agree(page)

        # 点登录（SPA 表单可能重渲染，先 wait 再点）
        try:
            await page.wait_for_selector("button[type='submit']", timeout=10000)
            btn = page.locator("button[type='submit']", has_text="登录").first
            await btn.click(timeout=8000)
        except Exception as e:
            self.logger.warning(f"登录按钮点击异常（尝试直接回车）: {type(e).__name__}")
            try:
                await page.keyboard.press("Enter")
            except Exception:
                pass

        # 等待登录成功（可能需手动滑块/验证码，超时给足）
        self.logger.info(
            "已提交登录，等待登录成功。若浏览器弹出滑块验证/短信验证码，"
            "请在浏览器窗口内手动完成一次（风控滑块需人工）。"
        )
        return await self._wait_login_success(page, timeout=180)

    async def _check_agree(self, page) -> None:
        try:
            box = page.locator(".auxo-checkbox-input").first
            if await box.count() > 0 and not await box.is_checked():
                await box.check(timeout=3000, force=True)
        except Exception:
            try:
                await page.locator("input[type='checkbox']").first.check(timeout=2000, force=True)
            except Exception:
                pass

    async def _wait_login_success(self, page, timeout: int = 180) -> bool:
        """轮询等待登录成功：URL 离开登录页 / 工作台元素出现。"""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            if self._is_on_workspace(page):
                await page.wait_for_timeout(2500)
                return True
            # 兜底：出现工作台/首页特征元素
            try:
                if await page.locator("text=工作台").first.count() > 0:
                    return True
            except Exception:
                pass
            if asyncio.get_event_loop().time() > deadline:
                return False
            await page.wait_for_timeout(2000)

    async def _capture_pigeon_session(self, page) -> Optional[Dict[str, Any]]:
        """跳转飞鸽工作台，挂 WS 捕获 token/pigeon_sign/device_id + 录制 WS 帧。

        帧录制是「下载即用」的关键：握手帧和注册帧（10011）只在真实登录
        会话里出现，注册帧里带着店铺归属。录下来存盘后，运行时传输层直接
        复用，新机器不再需要手工抓包。

        录制用 CDP 的 ``Network.webSocketFrameSent`` 而不是 Playwright 的
        ``framesent`` 事件：framesent 要等 websocket 事件回调挂上监听才开始
        收，而握手帧在连接建立瞬间就发出去了，实测必漏（真机复盘
        2026-11-13：framesent 只录到 10009/10010，体检报缺握手/注册帧）。
        CDP 在导航前开 Network 域，从页面第一个帧起全量捕获，无竞态。
        """
        from tools.reverse_probe.douyin.pigeon_assets import (
            PIGEON_WS_HOST_MARK,
            is_handshake_cmd,
        )

        captured: Dict[str, Any] = {}
        ws_frames: List[Dict[str, Any]] = []
        # 命令号无法区分连接归属：工作台页面还会连 frontier.snssdk.com，
        # 那条连接的帧命令号同样落在 10000 段（实测掺进过 10012/10016）。
        # 这里用 webSocketCreated 建 requestId -> url 映射，只留 pigeon 连接。
        unattributed: List[Dict[str, Any]] = []

        async def _attach_cdp(target):
            """给任意 target（页面/Worker/ServiceWorker）挂帧录制。

            真机复盘 2026-11-13：只挂页面时每轮只录到一对帧
            （10010+10011 同毫秒发出），完整握手序列缺失——部分帧发自
            Worker 线程，它们有独立 CDP target，页面级会话看不到。
            """
            ws_urls: Dict[str, str] = {}

            def on_ws_created(params):
                rid = params.get("requestId")
                url = params.get("url") or ""
                if rid:
                    ws_urls[rid] = url
                self.logger.debug("CDP WS 建立: %s", url[:90])

            def on_frame(params):
                resp = params.get("response") or {}
                # opcode 2 = 二进制帧，CDP 的 payloadData 是 base64 串
                if resp.get("opcode") != 2 or not resp.get("payloadData"):
                    return
                item = {
                    "b64": resp["payloadData"],
                    "ts": int(time.time() * 1000),
                }
                url = ws_urls.get(params.get("requestId") or "", "")
                if PIGEON_WS_HOST_MARK in url:
                    ws_frames.append(item)
                elif url:
                    self.logger.debug("跳过非 pigeon 连接帧: %s", url[:70])
                else:
                    # 连接在 Network.enable 之前就建好了，拿不到 url。
                    # 先存着，最后只有一条 pigeon 帧都没录到时才用它兜底。
                    unattributed.append(item)

            try:
                s = await page.context.new_cdp_session(target)
                await s.send("Network.enable")
                s.on("Network.webSocketCreated", on_ws_created)
                s.on("Network.webSocketFrameSent", on_frame)
                self.logger.debug("CDP 录帧已挂到 %s", type(target).__name__)
            except Exception as e:
                self.logger.debug(
                    "CDP 录帧挂载失败(%s): %s", type(target).__name__,
                    type(e).__name__,
                )

        # 页面本体
        cdp_ok = False
        try:
            await _attach_cdp(page)
            cdp_ok = True
        except Exception as e:
            self.logger.warning(f"CDP 帧录制挂载失败: {type(e).__name__}")

        # 页面里的 Web Worker / Service Worker（可能承载 pigeon 连接）
        loop = asyncio.get_event_loop()

        def _attach_existing_workers():
            for w in list(page.workers):
                loop.create_task(_attach_cdp(w))
            try:
                for sw in list(page.context.service_workers):
                    loop.create_task(_attach_cdp(sw))
            except Exception:
                pass

        _attach_existing_workers()
        page.on("worker", lambda w: loop.create_task(_attach_cdp(w)))
        try:
            page.context.on(
                "serviceworker", lambda sw: loop.create_task(_attach_cdp(sw))
            )
        except Exception:
            pass

        def on_ws(ws):
            url = ws.url or ""
            if "ws.fxg.jinritemai.com" in url:
                q = parse_qs(urlsplit(url).query)
                captured["token"] = q.get("token", [""])[0]
                captured["pigeon_sign"] = q.get("pigeon_sign", [""])[0]
                captured["device_id"] = q.get("device_id", [""])[0]
                captured["aid"] = q.get("aid", ["1383"])[0]
                captured["access_key"] = q.get("access_key", [DouyinLogin.DEFAULT_ACCESS_KEY])[0]
                self.logger.debug(
                    f"WS 参数已捕获: token={captured['token'][:14]}... "
                    f"pigeon_sign={captured['pigeon_sign'][:14]}..."
                )

                # CDP 不可用时退回 framesent（有竞态，可能漏握手帧，
                # 但比完全不录好）。
                if not cdp_ok:
                    def on_frame(payload):
                        if isinstance(payload, (bytes, bytearray)) and payload:
                            ws_frames.append({
                                "b64": base64.b64encode(bytes(payload)).decode("ascii"),
                                "ts": int(time.time() * 1000),
                            })

                    try:
                        ws.on("framesent", on_frame)
                    except Exception as e:
                        self.logger.warning(f"WS 帧录制挂载失败: {type(e).__name__}")

        # 先落到空白页：登录跳转可能已经把页面带到工作台、WS 早已建好
        # （真机复盘 2026-11-13：CDP 开好后录到的只有旧连接的 10010/10011，
        # 握手帧在监听开始前就发完了）。导航离开会关闭本页旧连接，
        # 之后每轮 goto 都是一次全新的 WS 生命周期，CDP 全程可见。
        try:
            await page.goto("about:blank", wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(800)
        except Exception as e:
            self.logger.debug(f"about:blank 预跳转失败（继续）: {type(e).__name__}")

        page.on("websocket", on_ws)
        # 单轮录制：goto 工作台后等首个握手/认证帧（实测 1-2 秒内出现），
        # 再留 2 秒收尾凑后续帧，最多 10 秒。旧版 3 轮×12 秒的重试是在
        # 等一个永远等不到的 10011（注册帧只在部分客户端版本发出），
        # 白白让用户在「登录完成→浏览器关闭」之间多等 40 多秒。
        #
        # 但「抓不到凭据」必须重试一轮（2026-11-13 真机）：短信验证刚通过时
        # 会话 cookie 还没完全生效，此时导航到工作台会被踢回、页内显示
        # 「登录失败」，token 抓不到 → 判登录失败，用户得点第二次验证。
        # 第二轮 cookie 已落地，必进工作台。只有失败才多花时间。
        for attempt in range(DouyinLogin.CAPTURE_ATTEMPTS):
            if attempt > 0:
                self.logger.warning(
                    "首轮未抓到 pigeon 凭据（登录刚完成时工作台常提示「登录失败」），"
                    "等待 1.5 秒让会话落地后重试"
                )
                await page.wait_for_timeout(1500)
            try:
                await page.goto(
                    DouyinLogin.WORKSPACE_URL,
                    wait_until="domcontentloaded", timeout=60000,
                )
            except Exception as e:
                self.logger.warning(
                    f"工作台导航失败（第 {attempt + 1} 轮）: {type(e).__name__}"
                )
                continue
            round_start = asyncio.get_event_loop().time()
            round_deadline = round_start + 10
            first_frame_at = None
            while asyncio.get_event_loop().time() < round_deadline:
                await page.wait_for_timeout(300)
                got = _captured_frame_cmds(ws_frames)
                if got:
                    if first_frame_at is None:
                        first_frame_at = asyncio.get_event_loop().time()
                    elif asyncio.get_event_loop().time() - first_frame_at >= 2:
                        break
                # 被踢回登录页（会话未生效）时早退，不必白白等满 10 秒
                elif asyncio.get_event_loop().time() - round_start > 2 \
                        and not self._is_on_workspace(page):
                    self.logger.debug(f"页面未停留在工作台: {page.url}")
                    break
            if captured.get("token") and captured.get("pigeon_sign"):
                break
            self.logger.warning(
                f"第 {attempt + 1} 轮未拿到 token/pigeon_sign，"
                f"当前页面: {page.url}"
            )
        if not captured.get("token") or not captured.get("pigeon_sign"):
            return None
        if not ws_frames and unattributed:
            # 极端情况：pigeon 连接在 Network.enable 之前就建好了，拿不到
            # url。此时宁可带着可能掺了别的连接的帧，也好过完全没有帧。
            self.logger.warning(
                f"未按连接识别到 pigeon 帧（连接早于监听建立），"
                f"回退使用未归因帧 {len(unattributed)} 条"
            )
            ws_frames.extend(unattributed)
        got_cmds = _captured_frame_cmds(ws_frames)
        # 帧号不再是判定依据：小于注册帧算握手帧；连这条规则都失效时，
        # 只要有成对的帧就交给传输层按顺序发，别在这里就判死。
        if not any(is_handshake_cmd(c) for c in got_cmds) and len(got_cmds) < 2:
            self.logger.warning(
                f"录帧未捕获任何握手/认证帧（已录到 {sorted(got_cmds)}），"
                f"通道启动可能被体检拦下"
            )
        self._save_pigeon_capture(ws_frames, ws_host=PIGEON_WS_HOST_MARK)

        # ---- 完整浏览器 cookies（backstage API 鉴权关键；pigeon_sign 仅对 pigeon_im 有效）----
        try:
            raw_cookies = await page.context.cookies()
            cookies_full = {}
            for c in raw_cookies:
                name = c.get("name")
                value = c.get("value")
                if name and value is not None:
                    cookies_full[name] = str(value)
            if cookies_full:
                captured["cookies_full"] = cookies_full
                self.logger.info(
                    f"已抓取完整浏览器 cookies: {len(cookies_full)} 个 "
                    f"(含 sessionid/pigeon_session 等，backstage API 可用)"
                )
            else:
                self.logger.warning("完整浏览器 cookies 抓取为空")
        except Exception as e:
            self.logger.warning(f"抓取完整 cookies 失败: error_type={type(e).__name__}")

        # ---- verifyFp（backstage API 风控参数，从页面已加载资源提取）----
        try:
            verify_fp = await page.evaluate(
                """() => {
                    try {
                        const entries = performance.getEntriesByType('resource');
                        for (const e of entries) {
                            const m = (e.name || '').match(/verifyFp=([^&]+)/);
                            if (m) return decodeURIComponent(m[1]);
                        }
                    } catch (err) {}
                    return null;
                }"""
            )
            if verify_fp:
                captured["verifyFp"] = verify_fp
                self.logger.info(f"已从页面资源抓取 verifyFp: {str(verify_fp)[:24]}...")
        except Exception as e:
            self.logger.debug(f"verifyFp 抓取失败: {type(e).__name__}")

        # ---- msToken（backstage API 风控参数；cookie 优先，其次页面存储）----
        try:
            ms_token = None
            cookies_full = captured.get("cookies_full") or {}
            for k in ("msToken", "__msToken", "msToken_"):
                if cookies_full.get(k):
                    ms_token = cookies_full[k]
                    break
            if not ms_token:
                ms_token = await page.evaluate(
                    """() => {
                        try {
                            return localStorage.getItem('msToken')
                                || sessionStorage.getItem('msToken') || null;
                        } catch (err) { return null; }
                    }"""
                )
            if ms_token:
                captured["msToken"] = ms_token
                self.logger.info(f"已抓取 msToken: {str(ms_token)[:16]}...")
        except Exception as e:
            self.logger.debug(f"msToken 抓取失败: {type(e).__name__}")

        return captured

    def _save_pigeon_session(self, session: Dict[str, Any]) -> None:
        # 每个账号一份凭据，避免多开时互相覆盖。
        out = pigeon_session_path(self.profile_scope, self.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=1)
        self.logger.info(f"会话凭据已保存: {out}")

    def _save_pigeon_capture(self, frames: List[Dict[str, Any]],
                             ws_host: str = "") -> None:
        """把登录时录到的 WS 帧存成按账号隔离的抓包文件。

        文件落在 temp 目录（git 忽略），带真实凭据不入库。运行时
        ``PigeonTransport`` 优先读它：握手/注册帧只能来自真实会话；
        缺 10018 发送帧时传输层会用包内脱敏骨架兜底，所以录帧不全
        不影响文本发送。
        """
        if not frames:
            self.logger.warning(
                "未录到任何 WS 帧（浏览器可能未建立 pigeon 连接），"
                "发送帧模板将回退包内骨架，握手/注册帧可能缺失"
            )
            return
        out = pigeon_capture_path(self.profile_scope, self.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {"ws_sent": frames, "saved_at": int(time.time() * 1000)}
        if ws_host:
            # 记录来源连接，方便日后排查「录到的帧是不是 pigeon 的」
            data["ws_host"] = ws_host
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        self.logger.info(f"WS 帧录制完成: {len(frames)} 帧 -> {out}")

    async def _extract_shop_name(self, page) -> Optional[str]:
        """从飞鸽工作台提取店铺名。

        高置信度选择器 + 过滤提示文案，避免把"被拉黑消费者…请谨慎使用"等
        弹窗/提示文本误抓成店铺名。抓不到返回 None（调用方兜底用登录名）。
        """
        tip_hints = ("拉黑", "举报", "谨慎", "发送消息", "消费者", "店铺，", "邀请", "入驻")
        try:
            for sel in (
                "[class*='shop-name']", "[class*='shopName']",
                "[class*='shop_name']", "[class*='mallName']",
            ):
                el = page.locator(sel).first
                if await el.count() > 0:
                    t = (await el.inner_text(timeout=1200) or "").strip()
                    if (
                        t and len(t) <= 20
                        and not any(h in t for h in tip_hints)
                    ):
                        return t
        except Exception:
            pass
        return None

    # 飞鸽后台：当前登录用户/店铺信息（抓包实证返回 ShopName/ShopId/ShopLogo）
    CURRENT_USER_URL = (
        "https://pigeon.jinritemai.com/backstage/currentuser?biz_type=4&_pms=1"
    )

    @staticmethod
    async def _fetch_shop_name_from_api(page) -> Optional[Dict[str, Any]]:
        """通过飞鸽 backstage/currentuser API 获取真实店铺信息。

        使用 page.request.get()（Playwright APIRequestContext，与浏览器上下文共享 cookie），
        靠浏览器会话 cookie 自动鉴权，无需手动传递 cookie。

        Returns:
            成功返回 {"shop_name": str, "shop_id": str|int, "shop_logo": str}；
            失败（异常/超时/code!=0/data为空/无ShopName）返回 None。
        """
        headers = {
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
            ),
            "referer": "https://im.jinritemai.com/",
        }
        try:
            resp = await page.request.get(
                DouyinLogin.CURRENT_USER_URL,
                headers=headers,
                timeout=15000,
            )
            if not resp.ok:
                return None
            body = await resp.json()
        except Exception as e:
            # 网络异常 / 超时 / JSON 解析失败 → 兜底
            return None

        if not isinstance(body, dict):
            return None
        if body.get("code") != 0:
            return None
        data = body.get("data")
        if not isinstance(data, dict):
            return None
        shop_name = data.get("ShopName")
        if not shop_name or not str(shop_name).strip():
            return None

        return {
            "shop_name": str(shop_name).strip(),
            "shop_id": data.get("ShopId", ""),
            "shop_logo": data.get("ShopLogo", ""),
        }

    async def refresh_cookies(self):
        """静默刷新会话凭据，结构与拼多多 PDDLogin.refresh_cookies 一致。

        正常路径复用已保存的浏览器登录态，用无头浏览器读一次凭据，不弹窗口。
        登录态失效时立即返回 False，由 perform_relogin 回退到完整登录
        （可见浏览器，短信验证码/滑块需人工完成一次）。
        """
        playwright = None
        context = None
        try:
            self.migrate_profile(self.profile_scope)
            playwright = await async_playwright().start()
            user_data_dir = str(self._profile_dir())
            self.logger.debug(f"使用用户数据目录刷新cookies: {user_data_dir}")
            # 目录不存在不再直接放弃：_launch_context 会新建，登录态由
            # dy_session_*.json 注入回来，避免因为目录被清理就要短信验证码。
            had_login_cookies = self._has_valid_login_cookies()
            context = await self._launch_context(playwright, user_data_dir, headless=True)
            await self._inject_saved_cookies(context, had_login_cookies)
            page = await context.new_page()
            await page.goto(DouyinLogin.LOGIN_URL, wait_until="domcontentloaded", timeout=60000)
            # 同主流程：已登录会自动跳走，离开登录页即提前通过。
            for _ in range(10):
                await page.wait_for_timeout(300)
                if not self._is_on_login_page(page):
                    break
            if self._is_on_login_page(page):
                self.logger.warning("登录状态已失效，需要重新登录")
                return False

            self.logger.info("检测到已登录状态，正在刷新会话凭据…")
            session = await self._capture_pigeon_session(page)
            if not session:
                self.logger.error("会话凭据获取失败（token/sign）")
                return False
            self.pigeon_session = session
            self._save_pigeon_session(session)
            self.logger.info(f"成功刷新账号 '{self.name}' 的cookies")
            return json.dumps({
                "pigeon_session": session,
                "shop_name": self.shop_name or self.name,
                "user_name": self.name,
                "cookies_full": session.get("cookies_full"),
                "verifyFp": session.get("verifyFp"),
                "msToken": session.get("msToken"),
            }, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"刷新cookies失败: {type(e).__name__}")
            return False
        finally:
            if context:
                try:
                    await context.close()
                except Exception:
                    pass
            if playwright:
                try:
                    await playwright.stop()
                except Exception:
                    pass

    def Set_user_info(self, cookies_json):
        """从登录返回的 JSON（含 pigeon_session/shop_name/user_name）解析用户信息。

        Returns:
            (user_id, user_name, mall_id)。user_id 用登录名生成稳定本地标识。
        """
        data = self._parse_login_json(cookies_json)
        user_name = (data.get("user_name") or self.name or "douyin_user").strip()
        user_id = self._local_id(user_name, "user")
        return user_id, user_name, ""

    def Set_shop_info(self, cookies_json):
        """从登录返回的 JSON 解析店铺信息。

        Returns:
            (shop_id, shop_name, mallLogo)。
        """
        data = self._parse_login_json(cookies_json)
        shop_name = (data.get("shop_name") or self.shop_name or self.name or "douyin_shop").strip()
        shop_id = self._local_id(shop_name, "shop")
        mall_logo = self.shop_logo or ""
        return shop_id, shop_name, mall_logo

    @staticmethod
    def _parse_login_json(cookies_json) -> Dict[str, Any]:
        """解析 login() 返回的 JSON；兼容旧 cookie dict 形式。"""
        if isinstance(cookies_json, dict):
            return cookies_json
        try:
            data = json.loads(cookies_json or "{}")
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _local_id(name: str, prefix: str) -> str:
        """由名称生成稳定的本地 shop_id / user_id（douyin_xxx）。"""
        raw = str(name or "").strip()
        safe = re.sub(r"[^A-Za-z0-9_\u4e00-\u9fa5]+", "_", raw).strip("_")[:24]
        return f"douyin_{safe or prefix}"


async def login_douyin(name, password, headless=False, profile_scope=None):
    """
    使用账号密码登录并返回账号、店铺信息，不直接操作数据库。
    如果登录成功，返回包含详细信息的字典（键名与 login_pdd 完全一致）。
    如果登录失败，返回 False。

    :param name: 用户名
    :param password: 密码
    :param headless: 是否使用无头模式（默认 False，自动回退时传 True）
    :return: dict or bool
    """
    douyin_login = DouyinLogin(
        name=name, password=password, profile_scope=profile_scope
    )
    cookies_json = await douyin_login.login(headless=headless)
    if not cookies_json:
        douyin_login.logger.error(f"账号 '{name}' 登录失败，未能获取cookies")
        return False

    try:
        # 获取用户信息和店铺信息
        user_id, user_name, mall_id = douyin_login.Set_user_info(cookies_json)
        shop_id, shop_name, mallLogo = douyin_login.Set_shop_info(cookies_json)

        # 检查是否成功获取到必要信息
        if user_id is None or shop_id is None:
            douyin_login.logger.error(f"账号 '{name}' 登录成功，但获取用户信息或店铺信息失败")
            return False

        if not _profile_scope_matches(profile_scope, shop_id, user_id):
            # 店铺名修正（手机号→真实店名）后 shop_id 会变化，严格匹配误拒；
            # 用户名匹配放行（同一用户名必然对应同一账号）。
            if not _profile_scope_matches_username(profile_scope, name):
                douyin_login.logger.error("登录会话与指定账号不一致，已失败并不会应用 cookies")
                return False

        douyin_login.migrate_profile(
            f"{douyin_login.channel_name}:{shop_id}:{user_id}"
        )

        douyin_login.logger.info(f"账号 '{name}' 登录成功，获取到店铺: {shop_name}({shop_id})")

        # 登录成功，返回包含所有信息的字典（键名与拼多多一致）
        return {
            "channel_name": douyin_login.channel_name,
            "shop_id": shop_id,
            "shop_name": shop_name,
            "shop_logo": mallLogo,
            "user_id": user_id,
            "username": name,        # 使用传入的登录名
            "password": password,    # 使用传入的密码
            "cookies": cookies_json,
        }
    except Exception as e:
        douyin_login.logger.error(
            f"账号 '{name}' 登录成功，但在处理后续信息时出错: {type(e).__name__}"
        )
        return False


async def refresh_douyin_cookies(name, password=None, profile_scope=None):
    """
    刷新抖音账号的 cookies，使用已保存的浏览器登录态重新抓取 pigeon 会话。
    签名与 refresh_pdd_cookies 对齐，供 base_request._relogin_and_update_cookies 调用。

    :param name: 用户名
    :param password: 密码（可选，仅用于创建 DouyinLogin 实例）
    :return: dict or bool
    """
    douyin_login = DouyinLogin(
        name=name, password=password or "", profile_scope=profile_scope
    )
    cookies_json = await douyin_login.refresh_cookies()
    if not cookies_json:
        douyin_login.logger.error(f"账号 '{name}' cookies刷新失败")
        return False

    try:
        user_id, user_name, mall_id = douyin_login.Set_user_info(cookies_json)
        shop_id, shop_name, mallLogo = douyin_login.Set_shop_info(cookies_json)

        if user_id is None or shop_id is None:
            douyin_login.logger.error(f"账号 '{name}' cookies刷新成功，但获取用户信息或店铺信息失败")
            return False

        if not _profile_scope_matches(profile_scope, shop_id, user_id):
            if not _profile_scope_matches_username(profile_scope, name):
                douyin_login.logger.error("刷新会话与指定账号不一致，已失败并不会应用 cookies")
                return False

        if profile_scope and douyin_login.profile_scope is None:
            douyin_login.migrate_profile(profile_scope)

        douyin_login.logger.info(f"账号 '{name}' cookies刷新成功，店铺: {shop_name}({shop_id})")

        return {
            "channel_name": douyin_login.channel_name,
            "shop_id": shop_id,
            "shop_name": shop_name,
            "shop_logo": mallLogo,
            "user_id": user_id,
            "username": name,
            "password": password or "",
            "cookies": cookies_json,
        }
    except Exception as e:
        douyin_login.logger.error(
            f"账号 '{name}' cookies刷新成功，但在处理后续信息时出错: {type(e).__name__}"
        )
        return False
