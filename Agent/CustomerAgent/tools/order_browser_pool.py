"""
常驻浏览器池：跨查询复用浏览器实例，把每次订单查询的「冷启动浏览器」开销
（约 3-8 秒）摊掉。

设计：
- 后台 daemon 线程 + 独立 asyncio 事件循环持有浏览器与持久化 context。
- 调用方（query_order_status 工具）通过同步 `fetch_customer_orders()`（按买家 uid
  调商家客服「客户订单」接口 userAllOrder）提交协程、阻塞等待结果；浏览器在该后台
  loop 中长期存活，下次查询直接复用。
- 登录态过期时自动用账号密码重登（PDDLogin.login 写入持久化 profile），
  随后重建 context 以加载新 cookie。
- 空闲超过 idle_close_seconds 后自动关闭浏览器，释放资源（下次查询再冷启动）。

注意：浏览器/context 与事件循环绑定，必须在同一 loop 内创建与销毁，因此不能
用每次工具调用里的 `asyncio.run()` 临时启动；本模块用独立常驻 loop 解决这一点。
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Dict, Optional

from utils.logger_loguru import get_logger

logger = get_logger("OrderBrowserPool")

ORDER_PAGE = "https://mms.pinduoduo.com/orders/list"


class OrderBrowserPool:
    def __init__(self, idle_close_seconds: int = 600, cache_ttl_seconds: int = 120):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()          # 同步状态保护 + 串行化查询提交
        self._pw = None                        # playwright 实例（常驻）
        self._context = None                   # 持久化浏览器 context（常驻）
        self._started_profile: Optional[str] = None
        self._started_name: Optional[str] = None
        # 注意：初始化为「当前时刻」而非 0（epoch）。若初始化为 0，则首个创建出的
        # context 在「任何一次未能成功更新 _last_active 的查询（如中途抛异常）」之后，
        # 下一次 fetch_orders 的 _maybe_idle_close 会因 time()-0 远超 idle_close 而误关
        # 一个仍存活的 context，导致后续 Page.goto 抛 TargetClosedError。
        self._last_active = time.time()
        self._idle_close = idle_close_seconds
        self._cache_ttl = cache_ttl_seconds
        # 查询结果缓存：同一 (账号,买家,uid/手机号/昵称,天数) 在 TTL 内直接返回，
        # 减少重复网页请求、降低后端压力；仅缓存成功结果。
        self._cache: Dict[tuple, tuple] = {}

    # ---- 后台线程 / 事件循环 ----
    def _ensure_thread(self):
        # 注意：调用方（fetch_orders）已持有 self._lock 做串行化，这里不再加锁，
        # 否则 threading.Lock 不可重入会死锁。
        if self._thread is None or not self._thread.is_alive():
            self._loop = None
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            # 等待后台线程把 self._loop 创建并跑起来（只由线程侧创建，
            # 避免主线程与线程各建一个 loop 导致提交到未运行的 loop 上）
            while self._loop is None or not self._loop.is_running():
                time.sleep(0.005)

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_forever()

    def _submit(self, coro, timeout: int = 120):
        self._ensure_thread()
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ---- 公开 API ----
    def fetch_orders(self, name, password="", buyer_uid="", mobile="", nick="",
                     days=90, headless=True) -> Dict:
        with self._lock:
            now = time.time()
            self._maybe_idle_close()
            key = (name, buyer_uid, mobile, nick, days)
            if key in self._cache:
                ts, cached = self._cache[key]
                if now - ts < self._cache_ttl:
                    logger.info("订单查询命中缓存（跳过浏览器请求）")
                    self._last_active = now
                    return cached
            res = self._submit(
                self._fetch_orders_async(name, password, buyer_uid, mobile, nick, days, headless),
                timeout=120,
            )
            # 仅缓存「真正成功且精确归因」的结果；filtered_failed 为安全拒绝（不缓存）。
            if res.get("success") and res.get("attribution") != "filtered_failed":
                self._cache[key] = (now, res)
            return res

    # ---- 客服「客户订单」接口（按买家 uid 拉单，替代 recentOrderList 全店爬取） ----
    def fetch_customer_orders(self, name, password="", uid="", headless=True) -> Dict:
        """按买家 uid 调 userAllOrder 拉取该买家在本店的订单（天然不泄漏他人）。

        与 fetch_orders（recentOrderList 全店 + buyerId 过滤，已证伪不可用）不同，
        这里走商家客服专用的客户订单接口，按会话买家 uid 精确返回其订单，无需
        让用户提供订单号，也不会把其他买家的订单泄漏给当前会话买家。
        """
        with self._lock:
            now = time.time()
            self._maybe_idle_close()
            key = ("customer", name, str(uid))
            if key in self._cache:
                ts, cached = self._cache[key]
                if now - ts < self._cache_ttl:
                    logger.info("客户订单查询命中缓存（跳过浏览器请求）")
                    self._last_active = now
                    return cached
            res = self._submit(
                self._fetch_customer_orders_async(name, password, uid, headless),
                timeout=60,
            )
            if res.get("success"):
                self._cache[key] = (now, res)
            return res

    def close(self):
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(self._close_async(), self._loop)
                try:
                    fut.result(timeout=10)
                except Exception:
                    pass
            self._context = None
            self._pw = None

    # ---- 异步实现（全部跑在后台 loop 内） ----
    async def _close_async(self):
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None

    def _maybe_idle_close(self):
        if self._context is not None and (time.time() - self._last_active) > self._idle_close:
            logger.info("浏览器池空闲超时，关闭以释放资源")
            if self._loop is not None and self._loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(self._close_async(), self._loop)
                try:
                    fut.result(timeout=10)
                except Exception:
                    pass

    async def _fetch_orders_async(self, name, password, buyer_uid, mobile, nick, days, headless):
        await self._ensure_context(name, password, headless)
        # 延迟导入，避免在池模块加载时拉起重依赖
        from .read_session_orders import _capture_and_replay
        try:
            result = await _capture_and_replay(
                self._context, buyer_uid=buyer_uid, mobile=mobile, nick=nick, days=days, headless=headless
            )
        except Exception as e:
            # 传输层异常（TargetClosedError 等）：当前 context 可能已损坏，置空，
            # 下次 fetch_orders 的 _ensure_context 会重建，避免复用坏 context 反复报错。
            logger.warning(f"订单捕获异常，丢弃当前 context 下次重建: {type(e).__name__}")
            self._context = None
            raise
        # 兜底：若仍检测到登录过期（重登也失效，如验证码），返回标记由调用方处理
        if result.get("login_expired"):
            await self._relogin(name, password, headless)
            result = await _capture_and_replay(
                self._context, buyer_uid=buyer_uid, mobile=mobile, nick=nick, days=days, headless=headless
            )
        self._last_active = time.time()
        return result

    async def _fetch_customer_orders_async(self, name, password, uid, headless):
        await self._ensure_context(name, password, headless)
        from .read_session_orders import post_customer_orders
        page = await self._context.new_page()
        try:
            result = await post_customer_orders(page, uid)
        finally:
            try:
                await page.close()
            except Exception:
                pass
        self._last_active = time.time()
        return result

    async def _ensure_context(self, name, password, headless):
        from Channel.pinduoduo.pdd_login import PDDLogin
        pdd = PDDLogin(name=name, password=password)
        profile_dir = str(pdd._profile_dir())
        if self._context is not None and self._started_profile == profile_dir:
            if await self._check_login():
                return
            await self._relogin(name, password, headless)
            return
        # 首次或切换账号：重建浏览器
        if self._context is not None:
            await self._close_async()
        if self._pw is None:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
        self._context = await pdd._launch_context(self._pw, profile_dir, headless=headless)
        self._started_profile = profile_dir
        self._started_name = name
        if not await self._check_login():
            await self._relogin(name, password, headless)

    async def _check_login(self):
        if self._context is None:
            return False
        page = await self._context.new_page()
        try:
            await page.goto(ORDER_PAGE, wait_until="domcontentloaded", timeout=30000)
            return "login" not in (page.url or "").lower()
        except Exception as e:
            logger.warning(f"登录态检测失败: {type(e).__name__}")
            return False
        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def _relogin(self, name, password, headless):
        logger.info("浏览器池检测到登录过期，自动用密码重新登录…")
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        from Channel.pinduoduo.pdd_login import PDDLogin
        pdd = PDDLogin(name=name, password=password)
        # 写入 profile 新 cookie（独立浏览器会话，用完即关）
        await pdd.login(headless=True)
        from playwright.async_api import async_playwright
        if self._pw is None:
            self._pw = await async_playwright().start()
        # 用同样的 profile_dir 重建 context 以加载新 cookie
        self._context = await pdd._launch_context(self._pw, str(pdd._profile_dir()), headless=headless)
        self._started_profile = str(pdd._profile_dir())
        self._started_name = name


_DEFAULT_POOL: Optional[OrderBrowserPool] = None


def get_order_browser_pool() -> OrderBrowserPool:
    """进程级单例：跨多次订单查询复用同一个浏览器。"""
    global _DEFAULT_POOL
    if _DEFAULT_POOL is None:
        _DEFAULT_POOL = OrderBrowserPool()
    return _DEFAULT_POOL
