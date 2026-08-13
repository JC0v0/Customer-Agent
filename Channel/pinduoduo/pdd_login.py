"""
拼多多账号异步登录认证
"""
import os
# 必须在导入 playwright 之前设置浏览器路径
from pathlib import Path
from utils.path_utils import get_app_dir
from utils.runtime_path import get_data_path
from utils.logger_loguru import get_logger

# 设置 Playwright 浏览器路径
app_dir = get_app_dir()
data_dir = get_data_path()
browsers_path = app_dir / ".browsers"
if browsers_path.exists():
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_path)
    logger_temp = get_logger("Pdd_login_init")
    logger_temp.info(f"设置 Playwright 浏览器路径: {browsers_path}")
else:
    # 回退到用户目录
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

from http import cookies
import requests
import json
import hashlib
import asyncio
import socket
import subprocess
import threading
import urllib.request
import re
import shutil
from urllib.parse import urlsplit
from typing import Optional, Dict, Any, Tuple
import sys
from database import db_manager
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from Channel.pinduoduo.utils.API.get_shop_info import GetShopInfo
from Channel.pinduoduo.utils.API.get_user_info import GetUserInfo


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
        and expected[0] == "pinduoduo"
        and str(expected[1]) == str(shop_id)
        and str(expected[2]) == str(user_id)
    )


class _CdpContextHandle:
    """包装 CDP 连接的 context，close() 时同时终止手动启动的浏览器进程。

    connect_over_cdp 返回的 BrowserContext.close() 只断开 Playwright 连接，
    不会关闭我们用 subprocess 拉起的浏览器进程，需在此兜底。

    Edge 是多进程的，proc.terminate() 只杀主进程，渲染/GPU/网络等子进程会
    残留为孤儿进程，浏览器窗口不关闭。改用 taskkill /T /F 杀整棵进程树。
    """

    def __init__(self, context, proc):
        self._context = context
        self._proc = proc

    def __getattr__(self, name):
        # new_page / cookies / pages 等方法透传给真实 context
        return getattr(self._context, name)

    async def close(self):
        try:
            await self._context.close()
        except Exception:
            pass
        if self._proc and self._proc.poll() is None:
            try:
                # 杀整棵进程树（主进程 + 所有 Edge 子进程），
                # 避免 terminate() 只杀主进程留下孤儿进程导致浏览器不关闭。
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(self._proc.pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass

class PDDLogin():
    def __init__(self, name, password, profile_scope=None):
        self.logger = get_logger("Pdd_login")
        self.channel_name = "pinduoduo"  # 渠道名称固定为"pinduoduo"
        self.base_url = "https://mms.pinduoduo.com/login"
        self.name = name
        self.password = password
        self.profile_scope = profile_scope

    def _profile_dir(self, profile_scope: Optional[str] = None) -> Path:
        """Return a filesystem-safe, stable profile path for this account."""
        raw_name = str(profile_scope or self.profile_scope or self.name or "").strip()
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._")[:48]
        safe_name = safe_name or "account"
        digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:12]
        return data_dir / "user_data" / f"{safe_name}-{digest}"

    def migrate_profile(self, profile_scope: str) -> None:
        """Copy a legacy username profile to its account-scoped location."""
        source = self._profile_dir()
        target = self._profile_dir(profile_scope)
        if source.resolve() == target.resolve() or not source.exists() or target.exists():
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        except OSError as exc:
            self.logger.warning(
                f"browser profile migration skipped: error_type={type(exc).__name__}"
            )

    # 本地浏览器启动参数（Chrome/Edge 通用）
    # 精简为对登录真正有用且兼容性好的参数；避免 --disable-web-security（登录用不到，
    # 且会触发部分 Edge 版本安全检查导致启动后秒退）和 --disable-features=
    # VizDisplayCompositor（新版浏览器无此特性，可能报错）。
    _LAUNCH_ARGS = [
        '--disable-blink-features=AutomationControlled',
        '--disable-notifications',
        '--disable-dev-shm-usage',
    ]

    # 本地浏览器可执行文件的常见安装位置，用于显式定位（channel 方式在部分安装
    # 下会找不到）。按优先级：Chrome > Edge。
    _BROWSER_PATHS = [
        # Google Chrome
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        # 用户级安装
        r"{localappdata}\Google\Chrome\Application\chrome.exe",
        # Microsoft Edge（Windows 自带）
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]

    @staticmethod
    def _resolve_browser_path():
        """探测本地浏览器 exe 路径。返回 (path, channel_name) 或 (None, None)。"""
        localappdata = os.getenv("LOCALAPPDATA", "")
        for raw in PDDLogin._BROWSER_PATHS:
            p = raw.replace("{localappdata}", localappdata)
            if os.path.isfile(p):
                # 从路径判断 channel：含 Edge 用 msedge，否则 chrome
                channel = "msedge" if "Edge" in p else "chrome"
                return p, channel
        return None, None

    def _clean_profile_locks(self, user_data_dir: str) -> None:
        """启动前清理 user_data_dir 中 Chromium 的单例锁文件。

        Edge/Chrome 异常退出后会残留 SingletonLock / SingletonCookie /
        SingletonSocket，下次启动时浏览器检测到 "profile 被占用" 会立即秒退。
        此处尽力清理；若文件被存活实例占用（删不掉）则静默跳过，不影响启动。
        """
        if not user_data_dir:
            return
        profile_dir = Path(user_data_dir)
        removed = []
        for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
            try:
                (profile_dir / name).unlink(missing_ok=True)
                removed.append(name)
            except Exception:
                # 文件不存在（missing_ok 已处理）或被占用，跳过
                pass
        if removed:
            self.logger.debug(f"已清理 profile 锁文件 {removed}: {user_data_dir}")

    async def _launch_context(self, playwright, user_data_dir: str, headless: bool):
        """启动用户本地的浏览器，避免下载 Playwright 自带的 Chromium。

        优先 Google Chrome，其次 Microsoft Edge（Windows 自带）；
        都不可用时回退到 Playwright 自带 Chromium（需曾运行 playwright install）；
        全部失败抛出明确错误，提示用户安装。
        """
        exe_path, channel = self._resolve_browser_path()
        attempts = []

        # 启动前清理上次崩溃可能残留的 profile 锁，避免浏览器检测到
        # "profile 被占用" 而秒退（本仓库历史上存在 Edge 秒退问题）
        self._clean_profile_locks(user_data_dir)

        # 1) 用探测到的真实路径启动本地浏览器（比 channel 更可靠）
        if exe_path:
            try:
                return await playwright.chromium.launch_persistent_context(
                    user_data_dir,
                    executable_path=exe_path,
                    headless=headless,
                    args=self._LAUNCH_ARGS,
                )
            except Exception as e:
                attempts.append(f"{channel}({exe_path}): error_type={type(e).__name__}")
                self.logger.warning(f"启动本地浏览器 {channel} 失败: {type(e).__name__}")

        # 2) 回退：channel 方式（路径探测未命中时的兜底）。
        # 级 1 已用 executable_path 试过同一浏览器（pipe 模式同样秒退），跳过。
        tried_channel = channel if exe_path else None
        for ch in ("chrome", "msedge"):
            if ch == tried_channel:
                continue
            try:
                return await playwright.chromium.launch_persistent_context(
                    user_data_dir,
                    channel=ch,
                    headless=headless,
                    args=self._LAUNCH_ARGS,
                )
            except Exception as e:
                attempts.append(f"{ch}(channel): error_type={type(e).__name__}")
                self.logger.warning(f"启动本地浏览器 {ch} 失败: {type(e).__name__}")

        # 3) 回退：手动启动浏览器用 remote-debugging-port，再 connect_over_cdp。
        # 部分电脑 Edge 在 Playwright 的 --remote-debugging-pipe 下启动后秒退
        # （Target page, context or browser has been closed），改用端口调试可绕过。
        if exe_path:
            handle = await self._launch_via_cdp(playwright, exe_path, user_data_dir, headless)
            if handle is not None:
                return handle
            attempts.append(f"{channel}(cdp-over-port): 启动后未就绪")

        # 4) 最后回退：Playwright 自带 Chromium（若已安装）
        try:
            return await playwright.chromium.launch_persistent_context(
                user_data_dir,
                headless=headless,
                args=self._LAUNCH_ARGS,
            )
        except Exception as e:
            attempts.append(f"chromium(bundled): error_type={type(e).__name__}")
            self.logger.warning(f"启动 Playwright Chromium 失败: {type(e).__name__}")

        raise RuntimeError(
            "未找到可用的浏览器。尝试过的方案：\n  - " +
            "\n  - ".join(attempts) +
            "\n请安装 Google Chrome 或 Microsoft Edge 后重试。"
        )

    async def _launch_via_cdp(self, playwright, exe_path, user_data_dir, headless):
        """手动启动浏览器（remote-debugging-port），再 connect_over_cdp 连接。

        绕开 Playwright 默认的 --remote-debugging-pipe，解决部分电脑 Edge
        在 pipe 模式下启动后秒退的问题。失败返回 None。
        """
        # 找一个空闲端口
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
        # about:blank 作为初始页，避免打开默认主页
        args.append("about:blank")

        # 用 PIPE 捕获 stderr，后台线程持续排空避免管道写满阻塞浏览器；
        # 进程退出后读取内容，用于诊断"秒退"的真正原因。
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
        # 等待 CDP 端点就绪（最多约 15 秒）
        ready = False
        for _ in range(50):
            if proc.poll() is not None:
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
            self._log_proc_stderr(
                stderr_chunks, drain_thread,
                f"CDP 回退：等待 CDP 端点超时 port={port}",
            )
            return None

        try:
            browser = await playwright.chromium.connect_over_cdp(
                f"http://127.0.0.1:{port}"
            )
            # 手动启动的浏览器带 --user-data-dir，第一个 context 即持久化 context
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            self.logger.info(f"CDP 回退连接成功（端口 {port}）")
            return _CdpContextHandle(context, proc)
        except Exception as e:
            if proc.poll() is None:
                proc.terminate()
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
            # read1 只要有一丁点数据就返回，适合持续排空；EOF 时返回 b''
            while True:
                chunk = stream.read1(4096)
                if not chunk:
                    break
                out_chunks.append(chunk)
        except Exception:
            pass

    def _log_proc_stderr(self, stderr_chunks: list, drain_thread, headline: str) -> None:
        """浏览器启动失败后读取并打印捕获的 stderr，定位秒退原因。"""
        # 进程已退出，等排空线程读完残余数据（read1 随即返回 b''）
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

    async def _try_click(self, page, selectors, timeout: float = 5000.0) -> bool:
        """逐个尝试点击候选选择器，任一成功返回 True，全部失败返回 False。"""
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                await locator.wait_for(state="visible", timeout=timeout)
                await locator.click(timeout=timeout)
                return True
            except Exception:
                continue
        return False

    async def _try_fill(self, page, selectors, value, timeout: float = 5000.0) -> bool:
        """逐个尝试向候选输入框填充内容，任一成功返回 True，全部失败返回 False。"""
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                await locator.wait_for(state="visible", timeout=timeout)
                await locator.fill(value, timeout=timeout)
                return True
            except Exception:
                continue
        return False

    async def login(self, headless=False):
        """使用账号密码登录

        Args:
            headless: 是否使用无头模式（自动回退时使用 True，避免弹出浏览器窗口）

        """
        playwright = None
        context = None
        try:
            # 启动Playwright
            playwright = await async_playwright().start()

            # 创建独立的用户数据目录，避免多实例冲突
            user_data_dir = str(self._profile_dir())
            self.logger.debug(f"使用用户数据目录: {user_data_dir}")

            # 使用本地浏览器（Chrome/Edge），无需下载 Playwright 自带 Chromium
            context = await self._launch_context(playwright, user_data_dir, headless=headless)
            
            page = await context.new_page()
            
            # 访问登录页面
            await page.goto(self.base_url)
            
            # 切换到「账号登录」标签（页面默认可能停留在扫码登录）。
            # 拼多多前端类名带 hash（如 Common_item__3diIn），易随版本变化，
            # 这里用多候选选择器逐个尝试，任一命中即可。
            await self._try_click(page, [
                "div.Common_item__3diIn:has-text('账号登录')",
                "text=账号登录",
                "a:has-text('账号登录')",
                "span:has-text('账号登录')",
                "div:has-text('账号登录')",
            ])

            # 等待账号输入框出现（多候选，任一出现即继续）
            try:
                await page.wait_for_selector(
                    "input[type='text'], input[placeholder*='账号'], "
                    "input[placeholder*='用户名'], input[placeholder*='店铺']",
                    timeout=15000,
                )
            except PlaywrightTimeoutError:
                self.logger.warning("等待账号输入框超时，尝试直接填充")

            # 输入店铺名
            if not await self._try_fill(page, [
                "input[type='text']",
                "input[placeholder*='账号']",
                "input[placeholder*='用户名']",
                "input[placeholder*='店铺']",
            ], self.name):
                raise RuntimeError("未找到账号输入框")

            # 输入密码
            if not await self._try_fill(page, [
                "input[type='password']",
                "input[placeholder*='密码']",
            ], self.password):
                raise RuntimeError("未找到密码输入框")

            # 点击登录按钮（多候选，兼容按钮文案空格/样式变化）
            if not await self._try_click(page, [
                "button:has-text('登录')",
                "button:has-text('登 录')",
                "button:has-text('立即登录')",
            ]):
                raise RuntimeError("未找到登录按钮")
            
            # 等待页面 title等于 拼多多 商家后台，首页或者订单查询
            await page.wait_for_function("() => document.title === '拼多多 商家后台' || document.title === '首页' || document.title === '订单查询'", timeout=30000)
            
            # 获取cookies并转换为字典格式
            cookies_list = await context.cookies()
            # 将playwright格式的cookies列表转换为字典格式，使用安全的get方法
            cookies_dict = {cookie.get('name', ''): cookie.get('value', '') for cookie in cookies_list if cookie.get('name')}
            cookies_json = json.dumps(cookies_dict)
            return cookies_json
            
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
        
    async def refresh_cookies(self):
        """重新获取cookies，使用已保存的用户数据，无需再次登录

        Returns:
            str: cookies的JSON字符串，如果失败返回False
        """
        playwright = None
        context = None
        try:
            # 启动Playwright
            playwright = await async_playwright().start()

            # 使用相同的用户数据目录（与login保持一致）
            user_data_dir = str(self._profile_dir())
            self.logger.debug(f"使用用户数据目录刷新cookies: {user_data_dir}")

            # 检查用户数据目录是否存在
            if not os.path.exists(user_data_dir):
                self.logger.error(f"用户数据目录不存在: {user_data_dir}，请先登录")
                return False

            # 使用本地浏览器（Chrome/Edge），无需下载 Playwright 自带 Chromium
            context = await self._launch_context(playwright, user_data_dir, headless=True)

            page = await context.new_page()

            # 访问拼多多商家后台首页，验证登录状态
            response = await page.goto("https://mms.pinduoduo.com/home/")

            # 只记录状态码和 URL 的路径，避免把 query 参数中的敏感信息写入日志。
            current_url = page.url or ""
            parsed_url = urlsplit(current_url)
            safe_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            response_status = response.status if response is not None else None
            self.logger.debug(
                f"刷新页面完成: status={response_status}, url={safe_url}"
            )

            # 有些情况下 goto 返回时已经落在登录页，直接判定登录态失效，
            # 避免再额外等待一次导航。
            if "login" in parsed_url.path.lower():
                self.logger.warning("登录状态已失效，需要重新登录")
                return False

            # 等待页面加载，检查是否需要重新登录
            try:
                # 如果页面跳转到登录页面，说明登录状态已失效
                await page.wait_for_url("**/login**", timeout=5000)
                self.logger.warning("登录状态已失效，需要重新登录")
                return False
            except PlaywrightTimeoutError:
                # 没有跳转到登录页面，说明登录状态有效
                # 再检查一次当前路径，覆盖“跳转发生但页面加载较慢”的情况。
                final_path = urlsplit(page.url or "").path.lower()
                if "login" in final_path:
                    self.logger.warning("登录状态已失效，需要重新登录")
                    return False
                self.logger.debug("5 秒内未跳转到登录页，继续读取 cookies")

            # 获取最新的cookies
            cookies_list = await context.cookies()
            cookies_dict = {cookie.get('name', ''): cookie.get('value', '') for cookie in cookies_list if cookie.get('name')}
            cookies_json = json.dumps(cookies_dict)

            self.logger.info(f"成功刷新账号 '{self.name}' 的cookies")
            return cookies_json

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

    def Set_user_info(self,cookies_json):
        user_info = GetUserInfo(cookies_json)
        result = user_info.get_user_info()
        if result is False:
            self.logger.error("获取用户信息失败")
            return None, None, None
        user_id, user_name, mall_id = result
        return user_id, user_name, mall_id

    def Set_shop_info(self,cookies_json):
        shop_info = GetShopInfo(cookies_json)
        result = shop_info.get_shop_info()
        if result is False:
            self.logger.error("获取店铺信息失败")
            return None, None, None
        shop_id, shop_name, mallLogo = result
        return shop_id, shop_name, mallLogo
    
async def login_pdd(name, password, headless=False, profile_scope=None):
    """
    使用账号密码登录并返回账号、店铺信息，不直接操作数据库。
    如果登录成功，返回包含详细信息的字典。
    如果登录失败，返回 False。

    :param name: 用户名
    :param password: 密码
    :param headless: 是否使用无头模式（默认 False，自动回退时传 True）
    :return: dict or bool
    """
    pdd_login = PDDLogin(
        name=name, password=password, profile_scope=profile_scope
    )
    cookies_json = await pdd_login.login(headless=headless)
    if not cookies_json:
        pdd_login.logger.error(f"账号 '{name}' 登录失败，未能获取cookies")
        return False

    try:
        # 获取用户信息和店铺信息
        user_id, user_name, mall_id = pdd_login.Set_user_info(cookies_json)
        shop_id, shop_name, mallLogo = pdd_login.Set_shop_info(cookies_json)
        
        # 检查是否成功获取到必要信息
        if user_id is None or shop_id is None:
            pdd_login.logger.error(f"账号 '{name}' 登录成功，但获取用户信息或店铺信息失败")
            return False

        if not _profile_scope_matches(profile_scope, shop_id, user_id):
            pdd_login.logger.error("登录会话与指定账号不一致，已失败并不会应用 cookies")
            return False

        # Initial UI login historically used the username as the profile key.
        # Preserve that data while creating the account-scoped profile used by
        # subsequent refresh/relogin operations.
        pdd_login.migrate_profile(
            f"{pdd_login.channel_name}:{shop_id}:{user_id}"
        )

        pdd_login.logger.info(f"账号 '{name}' 登录成功，获取到店铺: {shop_name}({shop_id})")

        # 登录成功，返回包含所有信息的字典
        return {
            "channel_name": pdd_login.channel_name,
            "shop_id": shop_id,
            "shop_name": shop_name,
            "shop_logo": mallLogo,
            "user_id": user_id,
            "username": name,  # 使用传入的登录名
            "password": password, # 使用传入的密码
            "cookies": cookies_json,
        }
    except Exception as e:
        pdd_login.logger.error(
            f"账号 '{name}' 登录成功，但在处理后续信息时出错: {type(e).__name__}"
        )
        return False

async def refresh_pdd_cookies(name, password=None, profile_scope=None):
    """
    刷新拼多多账号的cookies，使用已保存的用户数据，无需再次输入账号密码。
    如果刷新成功，返回包含最新cookies的字典。
    如果刷新失败（如登录状态已失效），返回 False。

    :param name: 用户名
    :param password: 密码（可选，仅用于创建PDDLogin实例）
    :return: dict or bool
    """
    pdd_login = PDDLogin(
        name=name, password=password or "", profile_scope=profile_scope
    )
    cookies_json = await pdd_login.refresh_cookies()
    if not cookies_json and profile_scope:
        # Existing installs may still have only the legacy username profile.
        # Read it once as a compatibility fallback; new logins migrate it to
        # the scoped path above.
        legacy_login = PDDLogin(name=name, password=password or "")
        cookies_json = await legacy_login.refresh_cookies()
        if cookies_json:
            pdd_login = legacy_login
    
    if not cookies_json:
        pdd_login.logger.error(f"账号 '{name}' cookies刷新失败")
        return False

    try:
        # 获取用户信息和店铺信息
        user_id, user_name, mall_id = pdd_login.Set_user_info(cookies_json)
        shop_id, shop_name, mallLogo = pdd_login.Set_shop_info(cookies_json)
        
        # 检查是否成功获取到必要信息
        if user_id is None or shop_id is None:
            pdd_login.logger.error(f"账号 '{name}' cookies刷新成功，但获取用户信息或店铺信息失败")
            return False

        if not _profile_scope_matches(profile_scope, shop_id, user_id):
            pdd_login.logger.error("刷新会话与指定账号不一致，已失败并不会应用 cookies")
            return False

        if profile_scope and pdd_login.profile_scope is None:
            # The identity check above makes this legacy-to-scoped migration
            # safe even when several usernames share a shop installation.
            pdd_login.migrate_profile(profile_scope)

        pdd_login.logger.info(f"账号 '{name}' cookies刷新成功，店铺: {shop_name}({shop_id})")

        # 刷新成功，返回包含最新信息的字典
        return {
            "channel_name": pdd_login.channel_name,
            "shop_id": shop_id,
            "shop_name": shop_name,
            "shop_logo": mallLogo,
            "user_id": user_id,
            "username": name,
            "password": password or "",
            "cookies": cookies_json,
        }
    except Exception as e:
        pdd_login.logger.error(
            f"账号 '{name}' cookies刷新成功，但在处理后续信息时出错: {type(e).__name__}"
        )
        return False

