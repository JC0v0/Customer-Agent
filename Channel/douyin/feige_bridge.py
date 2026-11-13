#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
飞鸽网页版浏览器桥（抖音方案A传输层）
=====================================
通过 Playwright 驱动本机 Edge（持久 profile 保登录）常驻抖店飞鸽客服工作台：
  - 接收：轮询会话列表 + 聊天区 DOM，提取买家新消息 -> 事件队列
  - 发送：切换到对应会话 -> 输入框填文本 -> 点发送按钮
  - 登录：复用 tools/reverse_probe/douyin/profile_v2 的登录态（首次扫码后长期有效）

线程模型：
  Playwright 同步 API 非线程安全，所有页面操作都在独立 worker 线程内完成。
  对外通过线程安全队列通信：
    - 事件队列（收到买家消息 -> 供 channel 消费）
    - 命令队列（send_text 等 -> 交给 worker 执行）

用法（独立自测）：
  python feige_bridge.py --self-test      # 打开飞鸽，扫描会话，向当前会话发一条测试消息

接入：
  channel（async）:
    bridge = FeigeBridge.instance()
    bridge.start()
    event = await asyncio.to_thread(bridge.get_event, timeout=3.0)
    bridge.send_text(buyer_name, reply)   # 阻塞到发送完成/超时
"""
import os
import sys
import time
import json
import queue
import logging
import threading
import concurrent.futures

# 路径常量（相对项目根）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

FEIGE_PROFILE = os.path.join(
    _PROJECT_ROOT, "tools", "reverse_probe", "douyin", "profile_v2"
)
WORKSPACE_URL = "https://im.jinritemai.com/pc_seller_v2/main/workspace"

_EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _find_edge():
    for c in _EDGE_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def account_profile(shop_id=None, user_id=None, username=None):
    """返回该账号登录时用过的浏览器 profile（与登录共用登录态）。

    登录侧目录规则是 ``user_data/{safe_name}-{sha256(raw)[:12]}``，
    这里按同样的规则生成候选并取第一个存在的；都没有则返回 None。
    """
    import hashlib
    import re

    from utils.runtime_path import get_data_path

    root = str(get_data_path() / "user_data")
    if not os.path.isdir(root):
        return None
    raws = []
    if shop_id and user_id:
        raws.append("douyin:%s:%s" % (shop_id, user_id))
    if username:
        raws.append(str(username))
    if shop_id:
        raws.append(str(shop_id))
    for raw in raws:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")[:48] or "account"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        path = os.path.join(root, "%s-%s" % (safe, digest))
        if os.path.isdir(path):
            return path
    return None


def account_sender_profile(shop_id=None, user_id=None, username=None):
    """页面发送专用 profile（与登录 profile 隔离）。

    登录流程和页面通道同时打开同一 user-data-dir 时，第二个实例会抛
    TargetClosedError（Edge 单实例锁），页面通道永远起不来。发送侧因此
    单独用一个目录存缓存，登录态靠注入 cookies。
    """
    import hashlib
    import re

    from utils.runtime_path import get_data_path

    raw = ""
    if shop_id and user_id:
        raw = "douyin:%s:%s" % (shop_id, user_id)
    elif username:
        raw = str(username)
    elif shop_id:
        raw = str(shop_id)
    if not raw:
        return None
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")[:48] or "account"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    root = str(get_data_path() / "user_data")
    path = os.path.join(root, "%s-%s-page" % (safe, digest))
    os.makedirs(path, exist_ok=True)
    return path


def _logger():
    lg = logging.getLogger("FeigeBridge")
    if not lg.handlers:
        from utils.logger_loguru import get_logger as _get_loguru

        _loguru = _get_loguru("FeigeBridge")

        class _LoguruSink(logging.Handler):
            def emit(self, record):
                try:
                    _loguru.log(record.levelname, record.getMessage())
                except Exception:
                    pass

        lg.addHandler(_LoguruSink())
        lg.setLevel(logging.INFO)
    return lg


# ---------------- 注入页面的 JS ----------------
JS_LIST_CONVS = """
() => {
  const out = [];
  const seen = new Set();
  for (const el of document.querySelectorAll('div[title]')) {
    const t = (el.getAttribute('title') || '').trim();
    if (!t || t.length > 20 || seen.has(t)) continue;
    // 买家名一般不含空格/换行；UI 标题一般含空格或更长
    if (/\\s/.test(t)) continue;
    let unread = 0;
    // 统计该列表项内未读角标数字
    const b = el.querySelector('.auxo-badge sup, .auxo-badge');
    if (b) {
      const m = (b.textContent || '').match(/\\d+/);
      if (m) unread = parseInt(m[0], 10);
    }
    const preview = (el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 60);
    seen.add(t);
    out.push({title: t, unread: unread, preview: preview});
  }
  return out;
}
"""

JS_READ_CHAT = """
() => {
  const out = [];
  for (const el of document.querySelectorAll('[class*="messageNotMe"],[class*="messageMe"]')) {
    const cls = el.className || '';
    const pre = el.querySelector('pre');
    const span = el.querySelector('span');
    let text = (pre ? pre.textContent : (span ? span.textContent : el.textContent)) || '';
    text = text.trim();
    if (!text) continue;
    const isMe = cls.indexOf('messageMe') >= 0;
    const notMe = cls.indexOf('messageNotMe') >= 0;
    out.push({from: isMe ? 'me' : (notMe ? 'buyer' : 'unknown'), text: text});
  }
  return out;
}
"""

JS_ACTIVE_TITLE = """
() => {
  // 1) 左栏活动会话项（data-btm-id 含 current）内的 [title]
  const cur = [...document.querySelectorAll('[data-btm-id*="current"]')];
  for (const el of cur) {
    const t = el.querySelector && el.querySelector('[title]');
    if (t && t.getAttribute('title')) return t.getAttribute('title').trim();
  }
  // 2) 兜底：输入框 placeholder "发送给 X，..."
  const ta = document.querySelector('textarea');
  if (ta && ta.placeholder) {
    const m = ta.placeholder.match(/发送给\\s*(.+?)[，,]/);
    if (m) return m[1].trim();
  }
  return null;
}
"""

JS_CLICK_CONV = """
(name) => {
  const hits = [...document.querySelectorAll('div[title]')]
    .filter(e => ((e.getAttribute('title') || '').trim()) === name);
  if (hits.length === 0) return 'none';
  if (hits.length > 1) return 'ambiguous:' + hits.length;
  hits[0].scrollIntoView({block: 'center'});
  hits[0].click();
  return 'ok';
}
"""

JS_SEND_BOX = """
() => {
  const tas = [...document.querySelectorAll('textarea')];
  const ta = tas.find(t => (t.placeholder || '').indexOf('发送给') >= 0) || tas[0];
  return {count: tas.length, placeholder: ta ? (ta.placeholder || '') : ''};
}
"""

JS_LAST_SENT = """
(title) => {
  for (const el of document.querySelectorAll('div[title]')) {
    if ((el.getAttribute('title') || '').trim() !== title) continue;
    let node = el;
    for (let i = 0; i < 4 && node.parentElement; i++) node = node.parentElement;
    return (node.textContent || '').replace(/\\s+/g, ' ').trim();
  }
  return '';
}
"""

JS_FILL_AND_SEND = """
() => {
  const ta = document.querySelector('textarea');
  return !!ta;
}
"""

JS_PAGE_STATE = """
() => {
  const ta = document.querySelector('textarea');
  const url = location.href;
  const title = document.title || '';
  const bodyText = (document.body ? document.body.innerText : '').slice(0, 300);
  const hasLoginHint = /扫码|登录|二维码|验证|短信|手机号/.test(bodyText);
  return {url: url, title: title, hasInput: !!ta, hasLoginHint: hasLoginHint, body: bodyText};
}
"""


class FeigeBridge:
    """飞鸽浏览器桥（按账号一个实例）。"""

    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def instance(cls, key=""):
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = FeigeBridge()
            return cls._instances[key]

    def __init__(self):
        self.log = _logger()
        self._stop = threading.Event()
        self._thread = None
        self._page = None
        self._profile = None
        self._cookies = {}
        self._ctx = None
        self._ready = threading.Event()
        self._send_ready = threading.Event()
        self._event_q = queue.Queue()
        self._cmd_q = queue.Queue()
        self._seen = {}          # title -> set(已处理买家消息文本)
        self._active_title = None

    # ---------- 生命周期 ----------
    def start(self, headless=False, poll=True, profile=None, cookies=None):
        """启动常驻浏览器。poll=False 时只供发送使用，不轮询会话列表。"""
        if self._thread and self._thread.is_alive():
            self.log.warning("bridge already running")
            return
        self._stop.clear()
        self._ready.clear()
        self._send_ready.clear()
        self._profile = profile or FEIGE_PROFILE
        self._cookies = dict(cookies or {})
        self._thread = threading.Thread(
            target=self._worker, args=(headless, poll), daemon=True, name="feige-bridge"
        )
        self._thread.start()
        self._ready.wait(timeout=120)
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._ready.clear()
        self._send_ready.clear()

    @property
    def is_ready(self):
        return self._ready.is_set() and not self._stop.is_set()

    @property
    def is_send_ready(self):
        return self._send_ready.is_set() and not self._stop.is_set()

    @property
    def is_running(self):
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    # ---------- 对外事件/命令 ----------
    def get_event(self, timeout=1.0):
        """取一条事件；超时返回 None。事件: {kind, session, content, msg_id, ts}"""
        try:
            return self._event_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_text(self, session, text, timeout=60):
        """发送文本消息到指定买家会话（阻塞到完成）。返回 True/False。"""
        if not self._send_ready.is_set():
            if not self.is_running:
                self.log.error("桥未运行，无法发送")
                return False
            # 浏览器正在启动：等会话列表渲染完再发，避免刚连上就丢回复。
            if not self._send_ready.wait(timeout=min(timeout, 40)):
                self.log.error("桥尚未就绪（会话列表未渲染），放弃本次发送")
                return False
        fut = concurrent.futures.Future()
        self._cmd_q.put(("send_text", (session, text), fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self.log.error("send_text 超时")
            return False

    # ---------- worker ----------
    def _worker(self, headless, poll=True):
        try:
            from playwright.sync_api import sync_playwright

            edge = _find_edge()
            self.log.info("启动飞鸽桥浏览器: %s", edge)
            with sync_playwright() as p:
                ctx = p.chromium.launch_persistent_context(
                    self._profile or FEIGE_PROFILE,
                    executable_path=edge,
                    headless=headless,
                    args=["--disable-blink-features=AutomationControlled", "--disable-notifications"],
                )
                self._ctx = ctx
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                self._page = page
                # 登录态来自 pigeon 抓到的整包 cookies：profile 里的网页会话
                # 可能早就过期，注入后才进得了工作台。
                if self._cookies:
                    try:
                        ctx.add_cookies([
                            {"name": str(k), "value": str(v), "domain": ".jinritemai.com",
                             "path": "/", "secure": True, "httpOnly": False}
                            for k, v in self._cookies.items()
                        ])
                    except Exception as e:
                        self.log.warning("注入 cookies 失败: %s", repr(e))
                try:
                    page.goto(WORKSPACE_URL, wait_until="domcontentloaded", timeout=60000)
                except Exception as e:
                    self.log.warning("goto 异常: %s", repr(e))
                # 等待工作台就绪（已登录；有无打开的具体会话都算就绪）
                ok = self._wait_ready(page, timeout=90)
                if ok:
                    self.log.info("飞鸽工作台就绪（已登录）")
                    self._ready.set()
                else:
                    self.log.error("未进入飞鸽工作台（可能未登录或页面异常），请检查浏览器窗口")
                    self._ready.set()  # 仍然标记 ready，让上层知道已启动但可能未登录
                if not poll:
                    # 只用于发送：等会话列表渲染出来，然后挂在命令队列上。
                    if self._wait_conversations(page, timeout=60):
                        self._send_ready.set()
                    self._command_loop(page)
                    return
                # 预热：把当前已显示的买家消息标记为已见，避免启动时追发历史消息
                try:
                    self._prime_seen(page)
                except Exception as e:
                    self.log.warning("预热失败: %s", repr(e))
                self._send_ready.set()
                self._poll_loop(page)
        except Exception as e:
            self.log.error("worker 异常退出: %s", repr(e))
        finally:
            self._ready.set()

    def _wait_conversations(self, page, timeout=60):
        """等左侧会话列表渲染完成，否则按标题点不到会话。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop.is_set():
                return False
            try:
                if page.evaluate(JS_LIST_CONVS):
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def _wait_ready(self, page, timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop.is_set():
                return False
            try:
                st = page.evaluate(JS_PAGE_STATE)
                if st:
                    url = st.get("url", "") or ""
                    title = st.get("title", "") or ""
                    # 进入工作台且非登录页（有无打开的具体会话都视为就绪）
                    if "workspace" in url and not st.get("hasLoginHint"):
                        if "飞鸽" in title or st.get("hasInput"):
                            return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def _poll_loop(self, page):
        while not self._stop.is_set():
            # 1. 先处理外部命令（发送等）
            self._drain_commands(page)
            # 2. 扫描新买家消息
            try:
                self._receive_cycle(page)
            except Exception as e:
                self.log.error("receive_cycle 异常: %s", repr(e))
            time.sleep(2.0)

    def scan_convs(self, timeout=30):
        """扫描会话列表（在工作线程内执行）。返回 [{title, unread, preview}]。"""
        fut = concurrent.futures.Future()
        self._cmd_q.put(("scan_convs", (), fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return []

    def read_chat(self, session, timeout=30):
        """读指定会话聊天区（工作线程内执行）。返回 [{from, text}, ...]。"""
        fut = concurrent.futures.Future()
        self._cmd_q.put(("read_chat", (session,), fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None

    def get_state(self, timeout=30):
        """读取页面登录态（工作线程内执行）。返回 {url, title, hasInput, hasLoginHint, body}。"""
        fut = concurrent.futures.Future()
        self._cmd_q.put(("get_state", (), fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None

    def js_eval(self, expr, timeout=30):
        """在工作线程内执行任意 JS 表达式（诊断/调试用）。返回求值结果。"""
        fut = concurrent.futures.Future()
        self._cmd_q.put(("js_eval", (expr,), fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None

    def _drain_commands(self, page):
        for _ in range(50):
            if self._stop.is_set():
                return
            try:
                cmd, args, fut = self._cmd_q.get_nowait()
            except queue.Empty:
                return
            self._run_command(page, cmd, args, fut)

    def _command_loop(self, page):
        """只用于发送的模式：阻塞等命令，不轮询会话列表。"""
        while not self._stop.is_set():
            try:
                cmd, args, fut = self._cmd_q.get(timeout=1.0)
            except queue.Empty:
                continue
            self._run_command(page, cmd, args, fut)

    def _run_command(self, page, cmd, args, fut):
        try:
            if cmd == "send_text":
                fut.set_result(self._do_send_text(page, *args))
            elif cmd == "scan_convs":
                fut.set_result(page.evaluate(JS_LIST_CONVS))
            elif cmd == "read_chat":
                session = args[0]
                page.evaluate(JS_CLICK_CONV, session)
                time.sleep(1.0)
                fut.set_result(page.evaluate(JS_READ_CHAT))
            elif cmd == "get_state":
                fut.set_result(page.evaluate(JS_PAGE_STATE))
            elif cmd == "js_eval":
                fut.set_result(page.evaluate(args[0]))
            else:
                fut.set_result(False)
        except Exception as e:
            self.log.error("命令 %s 执行异常: %s", cmd, repr(e))
            fut.set_result(False)

    # ---------- 接收 ----------
    def _prime_seen(self, page):
        """启动预热：把当前可见的买家消息全部标记为已见（不产生事件）。"""
        convs = page.evaluate(JS_LIST_CONVS)
        for c in convs:
            page.evaluate(JS_CLICK_CONV, c["title"])
            time.sleep(0.8)
            msgs = page.evaluate(JS_READ_CHAT)
            seen = self._seen.setdefault(c["title"], set())
            for m in msgs:
                if m["from"] == "buyer":
                    seen.add(m["text"])
            self.log.info("预热会话 %s: 已标记 %d 条历史消息", c["title"], len(seen))

    def _receive_cycle(self, page):
        convs = page.evaluate(JS_LIST_CONVS)
        if not convs:
            return
        # 活动会话
        try:
            self._active_title = page.evaluate(JS_ACTIVE_TITLE)
        except Exception:
            self._active_title = None
        # 优先切到有未读的会话（跳过正在处理的）
        for c in convs:
            if c["unread"] > 0 and c["title"] != self._active_title:
                self.log.info("切换到有未读的会话: %s", c["title"])
                page.evaluate(JS_CLICK_CONV, c["title"])
                time.sleep(1.0)
                self._read_and_emit(page, c["title"])
        # 读当前活动会话
        if self._active_title:
            self._read_and_emit(page, self._active_title)

    def _read_and_emit(self, page, title):
        try:
            msgs = page.evaluate(JS_READ_CHAT)
        except Exception as e:
            self.log.error("读聊天区异常: %s", repr(e))
            return
        seen = self._seen.setdefault(title, set())
        for m in msgs:
            if m["from"] != "buyer":
                continue
            if m["text"] in seen:
                continue
            seen.add(m["text"])
            ev = {
                "kind": "buyer_msg",
                "session": title,
                "content": m["text"],
                "msg_id": "dy_%s_%s" % (title, abs(hash((title, m["text"]))) % 10**10),
                "ts": int(time.time()),
            }
            self.log.info("新买家消息 [%s]: %s", title, m["text"])
            self._event_q.put(ev)

    # ---------- 发送 ----------
    def _do_send_text(self, page, session, text):
        try:
            # 1. 切到目标会话（昵称可能重名，重名时宁可不发）
            state = page.evaluate(JS_CLICK_CONV, session)
            if state == "none":
                self.log.error("未找到会话: %s", session)
                return False
            if isinstance(state, str) and state.startswith("ambiguous:"):
                self.log.error(
                    "昵称重复，拒绝发送以免发错人: %s (匹配 %s 个)",
                    session, state.split(":", 1)[1],
                )
                return False
            time.sleep(1.2)
            # 2. 确认输入框 placeholder 指向该买家（防止切错）
            box = page.evaluate(JS_SEND_BOX) or {}
            ph = box.get("placeholder", "") or ""
            if session not in ph:
                self.log.warning("输入框指向可能不符 (ph=%s, want=%s)，重试一次", ph, session)
                page.evaluate(JS_CLICK_CONV, session)
                time.sleep(1.2)
                box = page.evaluate(JS_SEND_BOX) or {}
                ph = box.get("placeholder", "") or ""
            if session not in ph:
                self.log.error("输入框仍未指向目标会话 (ph=%s, want=%s)", ph, session)
                return False
            # 3. 填文本 + 发送
            ta = page.locator("textarea").first
            ta.fill(str(text))
            time.sleep(0.3)
            btn = page.locator('[data-qa-id="qa-send-message-button"]')
            if btn.count() > 0:
                btn.first.click()
            else:
                ta.press("Enter")
            time.sleep(1.0)
            # 4. 会话列表里该会话的最后一条预览应是这条文本，否则视为没发出去
            head = str(text).replace("\n", " ")[:16]
            preview = ""
            for _ in range(6):
                preview = page.evaluate(JS_LAST_SENT, session) or ""
                if head and head in preview:
                    break
                time.sleep(0.5)
            if not head or head not in preview:
                self.log.error("发送后会话预览未见该消息 (preview=%r)", preview[:40])
                return False
            self.log.info("已发送 -> [%s]: %s", session, text)
            return True
        except Exception as e:
            self.log.error("发送失败: %s", repr(e))
            return False


def self_test():
    bridge = FeigeBridge.instance()
    bridge.start()
    if not bridge._ready.wait(timeout=120):
        print("[FAIL] 桥未就绪")
        return
    time.sleep(3)
    print("[OK] 桥已就绪")
    convs = bridge.scan_convs()
    print("[会话列表]", json.dumps(convs, ensure_ascii=False))
    if convs:
        target = convs[0]["title"]
        print("[发送测试] -> %s" % target)
        ok = bridge.send_text(target, "[桥自测] 自动回复链路测试")
        print("[发送结果]", ok)
    bridge.stop()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        bridge = FeigeBridge.instance()
        bridge.start()
        print("飞鸽桥已启动。按 Ctrl+C 停止。")
        try:
            while True:
                ev = bridge.get_event(timeout=1.0)
                if ev:
                    print("EVENT:", ev)
        except KeyboardInterrupt:
            bridge.stop()
            print("已停止")
