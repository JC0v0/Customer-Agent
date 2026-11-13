"""抖音登录浏览器关闭逻辑的回归测试。

对应两次真实故障：

1. 登录完成后 Edge 窗口一直不关、profile 目录被占，导致 migrate_profile
   复制 profile 长期失败（每次登录白等十几秒）。根因是 close() 里
   `if self._proc.poll() is None:` 才 taskkill，而 msedge 启动器进程移交
   工作后会立刻退出，判断为假，强杀整段被跳过。
2. 改成调 taskkill 之后，打包版里 taskkill.exe 直接报 0xc0000142
   「应用程序无法正常启动」，而且每调一次就弹一个控制台黑窗。
   现在清理全部改走 Win32 API，不启动任何外部程序。
"""
import os
import ctypes
import json
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from ctypes import wintypes
from pathlib import Path
from unittest import mock

from Channel.douyin import douyin_login
from Channel.douyin.douyin_login import (
    _CdpContextHandle,
    _child_pids,
    _kill_process_tree,
    _pid_listening_on_port,
)

IS_WINDOWS = sys.platform == "win32"
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


@unittest.skipUnless(IS_WINDOWS, "Win32 API 只在 Windows 上可用")
class PortLookupTests(unittest.TestCase):
    def test_finds_owner_of_listening_socket(self):
        server = socket.socket()
        try:
            server.bind(("127.0.0.1", 0))
            server.listen(5)
            port = server.getsockname()[1]
            self.assertEqual(_pid_listening_on_port(port), os.getpid())
        finally:
            server.close()

    def test_returns_none_after_socket_closed(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(5)
        port = server.getsockname()[1]
        server.close()
        self.assertIsNone(_pid_listening_on_port(port))

    def test_returns_none_for_empty_port(self):
        self.assertIsNone(_pid_listening_on_port(None))
        self.assertIsNone(_pid_listening_on_port(0))


@unittest.skipUnless(IS_WINDOWS, "Win32 API 只在 Windows 上可用")
class KillProcessTreeTests(unittest.TestCase):
    def test_kills_parent_and_children(self):
        script = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)']); "
            "time.sleep(60)"
        )
        parent = subprocess.Popen([sys.executable, "-c", script])
        children = []
        try:
            deadline = time.time() + 15
            while time.time() < deadline:
                children = _child_pids(parent.pid)
                if children:
                    break
                time.sleep(0.3)

            _kill_process_tree(parent.pid)

            deadline = time.time() + 10
            while time.time() < deadline and parent.poll() is None:
                time.sleep(0.2)
            self.assertIsNotNone(parent.poll(), "父进程没有被杀掉")
            for pid in children:
                self.assertFalse(self._is_alive(pid), "子进程 %s 没有被杀掉" % pid)
        finally:
            if parent.poll() is None:
                parent.kill()

    @staticmethod
    def _is_alive(pid):
        """Windows 上 os.kill(pid, 0) 不可靠，用 Win32 查退出码。"""
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)


class FakeContext:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class BrowserCloseTests(unittest.IsolatedAsyncioTestCase):
    async def test_port_fallback_kills_orphan_browser(self):
        """CDP 会话不可用时，必须靠调试端口把真正的浏览器进程找出来杀掉。"""
        killed = []
        lookups = []

        def fake_lookup(port):
            lookups.append(port)
            return 33112

        def fake_kill(pid):
            killed.append(pid)

        launcher = mock.Mock()
        launcher.pid = 40924
        launcher.poll.return_value = 0  # 启动器已退出，旧逻辑在这里跳过了强杀

        context = FakeContext()
        handle = _CdpContextHandle(context, launcher, browser=None, port=50494)
        with mock.patch.object(douyin_login, "_pid_listening_on_port", fake_lookup), \
                mock.patch.object(douyin_login, "_kill_process_tree", fake_kill):
            await handle.close()

        self.assertTrue(context.closed)
        self.assertEqual(lookups, [50494])
        self.assertIn(33112, killed)
        # 启动器 pid 已失效，不能拿去杀（系统可能已回收给别的进程）
        self.assertNotIn(40924, killed)

    async def test_alive_launcher_is_killed(self):
        """启动器还活着（就是浏览器主进程）时，仍要按 pid 杀。"""
        killed = []

        launcher = mock.Mock()
        launcher.pid = 40924
        launcher.poll.return_value = None

        handle = _CdpContextHandle(FakeContext(), launcher, browser=None, port=None)
        with mock.patch.object(
            douyin_login, "_kill_process_tree", lambda pid: killed.append(pid)
        ):
            await handle.close()

        self.assertEqual(killed, [40924])

    async def test_cdp_session_sends_browser_close(self):
        """CDP 可用时应主动发 Browser.close，而不是只断开连接。"""
        methods = []

        class FakeSession:
            async def send(self, method, *args, **kwargs):
                methods.append(method)
                if method == "SystemInfo.getProcessInfo":
                    return {"processInfo": [{"id": 777, "type": "browser"}]}
                return None

        browser = mock.Mock()
        browser.new_browser_cdp_session = mock.AsyncMock(return_value=FakeSession())

        launcher = mock.Mock()
        launcher.pid = 100
        launcher.poll.return_value = None

        killed = []
        handle = _CdpContextHandle(
            FakeContext(), launcher, browser=browser, port=50494
        )
        with mock.patch.object(
            douyin_login, "_kill_process_tree", lambda pid: killed.append(pid)
        ):
            await handle.close()

        self.assertIn("SystemInfo.getProcessInfo", methods)
        self.assertIn("Browser.close", methods)
        self.assertIn(777, killed)
        self.assertIn(100, killed)

    async def test_close_survives_context_close_error(self):
        class BrokenContext:
            async def close(self):
                raise RuntimeError("target closed")

        launcher = mock.Mock()
        launcher.pid = 1
        launcher.poll.return_value = 0
        handle = _CdpContextHandle(BrokenContext(), launcher, browser=None, port=None)
        with mock.patch.object(douyin_login, "_kill_process_tree", lambda pid: None):
            await handle.close()


class ProfileScopeTests(unittest.TestCase):
    """回归：登录目录不能再随 profile_scope 漂移。

    真机现象（2026-11-14）：添加账号写按用户名算的目录，运行时刷新读
    按 shop:user 算的空目录，于是每次启动都判未登录、重新发短信验证码。
    """

    def test_profile_dir_is_identical_for_any_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(douyin_login, "data_dir", Path(tmp)):
                scoped = douyin_login.DouyinLogin(
                    "13800138000", "pw", profile_scope="douyin:shop1:user1"
                )
                plain = douyin_login.DouyinLogin("13800138000", "pw")
                self.assertEqual(scoped._profile_dir(), plain._profile_dir())
                self.assertEqual(
                    scoped._profile_dir(),
                    scoped._profile_dir("douyin:other:other"),
                )
                self.assertNotEqual(
                    scoped._profile_dir(),
                    scoped._legacy_profile_dir("douyin:shop1:user1"),
                )

    def test_migrate_profile_copies_cookie_key_and_cookies(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(douyin_login, "data_dir", Path(tmp)):
                scope = "douyin:shop1:user1"
                login = douyin_login.DouyinLogin(
                    "13800138000", "pw", profile_scope=scope
                )
                source = login._legacy_profile_dir(scope)
                target = login._profile_dir()
                (source / "Default" / "Network").mkdir(parents=True)
                (source / "Local State").write_text("key-v1", encoding="utf-8")
                source_cookies = source / "Default" / "Network" / "Cookies"
                source_cookies.write_text("cookies-v1", encoding="utf-8")

                login.migrate_profile(scope)
                self.assertEqual(
                    (target / "Local State").read_text(encoding="utf-8"), "key-v1"
                )
                target_cookies = target / "Default" / "Network" / "Cookies"
                self.assertEqual(
                    target_cookies.read_text(encoding="utf-8"), "cookies-v1"
                )

                # 统一目录已有会话时不能被旧目录盖掉
                source_cookies.write_text("cookies-v2", encoding="utf-8")
                login.migrate_profile(scope)
                self.assertEqual(
                    target_cookies.read_text(encoding="utf-8"), "cookies-v1"
                )


    def test_login_cookie_check_ignores_basic_only_profile(self):
        """回归：Cookies 文件非空不等于有登录态。

        真机 03:36 登录成功后 profile 只剩 14 条基础 cookie，旧判据
        「文件存在且非空」返回 True，注入被跳过，点验证又去发验证码。
        """
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(douyin_login, "data_dir", Path(tmp)):
                login = douyin_login.DouyinLogin("13800138000", "pw")
                cookies = login._profile_dir() / "Default" / "Network" / "Cookies"
                cookies.parent.mkdir(parents=True)
                con = sqlite3.connect(str(cookies))
                con.execute("create table cookies (name TEXT)")
                con.execute("insert into cookies values (?)", ("passport_csrf_token",))
                con.commit()
                con.close()

                self.assertFalse(login._has_valid_login_cookies())

                con = sqlite3.connect(str(cookies))
                con.execute("insert into cookies values (?)", ("sessionid",))
                con.commit()
                con.close()
                self.assertTrue(login._has_valid_login_cookies())


class SavedCookieInjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_injects_cookies_from_saved_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "dy_session.json"
            session.write_text(
                json.dumps({"cookies_full": {"sessionid": "abc", "msToken": "xyz"}}),
                encoding="utf-8",
            )
            login = douyin_login.DouyinLogin("13800138000", "pw")
            added = []

            class FakeContext:
                async def add_cookies(self, payload):
                    added.extend(payload)

            with mock.patch.object(
                login, "_saved_session_candidates", lambda: [session]
            ):
                count = await login._inject_saved_cookies(FakeContext())

            self.assertEqual(count, 2)
            self.assertEqual(added[0]["name"], "sessionid")
            self.assertEqual(added[0]["value"], "abc")
            self.assertEqual(added[0]["domain"], ".jinritemai.com")

    async def test_skips_injection_when_profile_has_cookies(self):
        """浏览器里已有较新会话时，不能用旧的 session 文件覆盖它。"""
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "dy_session.json"
            session.write_text(
                json.dumps({"cookies_full": {"sessionid": "old"}}),
                encoding="utf-8",
            )
            login = douyin_login.DouyinLogin("13800138000", "pw")
            added = []

            class FakeContext:
                async def add_cookies(self, payload):
                    added.extend(payload)

            with mock.patch.object(
                login, "_saved_session_candidates", lambda: [session]
            ):
                count = await login._inject_saved_cookies(
                    FakeContext(), profile_has_login_cookies=True
                )

            self.assertEqual(count, 0)
            self.assertEqual(added, [])


if __name__ == "__main__":
    unittest.main()
