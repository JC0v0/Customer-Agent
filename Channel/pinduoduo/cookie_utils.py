"""
Cookie 工具模块

提供:
- check_cookies_valid: 轻量级 cookie 有效性验证（HTTP 请求，非浏览器）
- ReloginGuard: 按账户的并发重登互斥锁 + 冷却期
- perform_relogin: 独立的 cookie 刷新/重登函数，可从任意上下文调用
"""

import json
import threading
import time
from typing import Dict, Optional

import requests

from database import db_manager
from utils.logger_loguru import get_logger
from utils.async_helper import run_async_in_thread

logger = get_logger("CookieUtils")

# 会话过期错误码（与 BaseRequest.SESSION_EXPIRED_ERROR_CODE 一致）
SESSION_EXPIRED_ERROR_CODE = 43001

# 验证请求的默认 headers
_VALIDATION_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
}


def _login_result_matches_account(result, shop_id: str, user_id: str) -> bool:
    """Reject cookies resolved from a browser session belonging to another account."""
    return (
        isinstance(result, dict)
        and str(result.get("shop_id")) == str(shop_id)
        and str(result.get("user_id")) == str(user_id)
    )


def check_cookies_valid(
    channel_name: str,
    shop_id: str,
    user_id: str,
    cookies: dict,
    timeout: float = 15.0,
) -> Optional[bool]:
    """
    通过轻量级 HTTP 请求验证 cookie 是否有效。

    向拼多多 getToken 端点发送 POST 请求，检查响应中是否包含
    error_code=43001（会话已过期）。

    Args:
        channel_name: 渠道名称
        shop_id: 店铺 ID
        user_id: 用户 ID
        cookies: cookie 字典
        timeout: 请求超时（秒）

    Returns:
        True: 服务端明确确认 cookie 有效
        False: 服务端明确确认 cookie 已过期
        None: 网络或响应异常，状态未知（由调用方决定是否重试）
    """
    url = "https://mms.pinduoduo.com/chats/getToken"
    payload = {'version': '3'}

    try:
        # 与 GetToken.get_token 保持一致：用 data= 表单编码发送 version=3。
        # （实测 GetToken 走 BaseRequest.post 的 data=dict + 显式 Content-Type
        #  header，实际 body 为 "version=3"；此前此处用 json= 会发送
        #  {"version": "3"}，导致验证请求与真实请求语义不一致，可能漏检过期。）
        response = requests.post(
            url,
            data=payload,
            headers=_VALIDATION_HEADERS,
            cookies=cookies,
            timeout=timeout,
        )

        if response.status_code == 200:
            try:
                data = response.json()
                if data.get('error_code') == SESSION_EXPIRED_ERROR_CODE and \
                   '会话已过期' in str(data.get('error_msg', '')):
                    logger.warning(f"Cookie 验证失败: 会话已过期 ({channel_name}:{shop_id}:{user_id})")
                    return False
                if isinstance(data, dict) and data.get('error_code') in (None, 0):
                    return True
                logger.debug("Cookie 验证: 服务端返回未知状态")
                return None
            except json.JSONDecodeError:
                logger.debug("Cookie 验证: 响应非 JSON，状态未知")
                return None
        else:
            logger.debug(f"Cookie 验证: HTTP {response.status_code}，状态未知")
            return None

    except requests.ConnectionError:
        logger.debug("Cookie 验证: 网络连接失败，状态未知")
        return None
    except requests.Timeout:
        logger.debug("Cookie 验证: 请求超时，状态未知")
        return None
    except Exception as e:
        logger.debug(f"Cookie 验证: 异常 {type(e).__name__}，状态未知")
        return None


class ReloginGuard:
    """按账户的并发重登互斥锁 + 冷却期，防止多个请求同时触发重登"""

    def __init__(self, cooldown_seconds: float = 60.0):
        self._lock = threading.Lock()
        self._account_locks: Dict[str, threading.Lock] = {}
        self._last_relogin: Dict[str, float] = {}
        self._cooldown = cooldown_seconds

    @staticmethod
    def _make_key(channel_name: str, shop_id: str, user_id: str) -> str:
        return f"{channel_name}:{shop_id}:{user_id}"

    def try_acquire(self, channel_name: str, shop_id: str, user_id: str) -> bool:
        """
        尝试获取重登权。

        若其他线程正在重登，或冷却期内刚完成重登，返回 False。
        否则获取锁并返回 True。
        """
        key = self._make_key(channel_name, shop_id, user_id)

        with self._lock:
            last_time = self._last_relogin.get(key, 0)
            if time.time() - last_time < self._cooldown:
                return False

            if key not in self._account_locks:
                self._account_locks[key] = threading.Lock()
            acct_lock = self._account_locks[key]

        return acct_lock.acquire(blocking=False)

    def is_in_cooldown(self, channel_name: str, shop_id: str, user_id: str) -> bool:
        """检查指定账户当前是否处于重登冷却期内。"""
        key = self._make_key(channel_name, shop_id, user_id)
        with self._lock:
            last_time = self._last_relogin.get(key, 0)
            return (time.time() - last_time) < self._cooldown

    def release(self, channel_name: str, shop_id: str, user_id: str, success: bool = False):
        """释放锁，成功时记录时间戳用于冷却期"""
        key = self._make_key(channel_name, shop_id, user_id)

        with self._lock:
            acct_lock = self._account_locks.get(key)

        if acct_lock:
            try:
                acct_lock.release()
            except RuntimeError:
                pass

        if success:
            with self._lock:
                self._last_relogin[key] = time.time()


# 模块级单例
relogin_guard = ReloginGuard(cooldown_seconds=60.0)


def notify_relogin_failure(
    channel_name: str,
    shop_id: str,
    user_id: str,
    username: str,
    reason: str,
) -> None:
    """重登失败后更新连接状态为 ERROR，提示用户在账号管理页手动重登。

    仅影响 UI 状态展示；任何异常都被吞掉，不阻断重登流程本身。
    """
    try:
        from core.di_container import container
        from core.connection_status import ConnectionStatusManager, ConnectionState
        status_manager = container.get(ConnectionStatusManager)
        status_manager.update_status(
            shop_id,
            user_id,
            username or user_id,
            ConnectionState.ERROR,
            f"Cookie 重新登录失败：{reason}，请在账号管理页手动重新登录",
        )
    except Exception:
        pass


def perform_relogin(
    channel_name: str,
    shop_id: str,
    user_id: str,
    username: str,
    password: str,
    headless_fallback: bool = True,
) -> bool:
    """
    执行 cookie 刷新/重登，优先刷新，失败时回退到完整登录。

    此函数可从任意上下文调用（同步/异步），内部处理线程隔离。

    Args:
        channel_name: 渠道名称
        shop_id: 店铺 ID
        user_id: 用户 ID
        username: 用户名
        password: 密码
        headless_fallback: 完整登录时是否使用 headless 模式

    Returns:
        是否成功获取新 cookie
    """
    if not relogin_guard.try_acquire(channel_name, shop_id, user_id):
        # 区分「冷却期」（刚重登失败过）与「锁被占用」（另一线程正在重登）：
        # 前者应如实返回失败，避免调用方把过期 cookie 误判为重登成功。
        if relogin_guard.is_in_cooldown(channel_name, shop_id, user_id):
            logger.info(f"重登处于冷却期内，跳过本次重登: {username}")
            return False
        logger.info(f"重登已在其他线程进行中: {username}")
        from Channel.pinduoduo.cookie_cache import cookie_cache
        return cookie_cache.get(channel_name, shop_id, user_id) is not None

    try:
        import importlib
        pdd_login_module = importlib.import_module('Channel.pinduoduo.pdd_login')

        # 阶段 1: 刷新 cookie（headless，利用浏览器持久化会话）
        logger.info(f"尝试刷新 cookie（无需重新登录）: {username}")
        try:
            profile_scope = f"{channel_name}:{shop_id}:{user_id}"
            refresh_result = run_async_in_thread(
                pdd_login_module.refresh_pdd_cookies(
                    username, password, profile_scope=profile_scope
                ),
                timeout=60.0,
            )
            if refresh_result and isinstance(refresh_result, dict):
                if not _login_result_matches_account(refresh_result, shop_id, user_id):
                    logger.error("Cookie 刷新返回的账号身份不匹配，已拒绝应用")
                    refresh_result = None
            if refresh_result and isinstance(refresh_result, dict):
                new_cookies = refresh_result.get('cookies')
                if new_cookies:
                    _apply_cookies_to_cache_and_db(
                        channel_name, shop_id, user_id, new_cookies
                    )
                    logger.info(f"Cookie 刷新成功: {username}")
                    relogin_guard.release(channel_name, shop_id, user_id, success=True)
                    return True
        except Exception as e:
            logger.warning(
                f"Cookie 刷新异常: {username}, error_type={type(e).__name__}"
            )

        # 阶段 2: 完整重新登录
        if not password:
            logger.error(f"缺少密码，无法完整重新登录: {username}")
            relogin_guard.release(channel_name, shop_id, user_id, success=False)
            notify_relogin_failure(
                channel_name, shop_id, user_id, username,
                "登录状态已失效且缺少密码，无法自动重登",
            )
            return False

        logger.info(f"回退到完整重新登录 (headless={headless_fallback}): {username}")
        try:
            login_result = run_async_in_thread(
                pdd_login_module.login_pdd(
                    username,
                    password,
                    headless_fallback,
                    profile_scope=f"{channel_name}:{shop_id}:{user_id}",
                ),
                timeout=60.0,
            )
            if login_result and isinstance(login_result, dict):
                if not _login_result_matches_account(login_result, shop_id, user_id):
                    logger.error("完整登录返回的账号身份不匹配，已拒绝应用")
                    login_result = None
            if login_result and isinstance(login_result, dict):
                new_cookies = login_result.get('cookies')
                if new_cookies:
                    _apply_cookies_to_cache_and_db(
                        channel_name, shop_id, user_id, new_cookies
                    )
                    logger.info(f"完整重新登录成功: {username}")
                    relogin_guard.release(channel_name, shop_id, user_id, success=True)
                    return True
        except Exception as e:
            logger.error(
                f"完整重新登录异常: {username}, error_type={type(e).__name__}"
            )

        logger.error(f"重新登录失败: {username}")
        relogin_guard.release(channel_name, shop_id, user_id, success=False)
        notify_relogin_failure(
            channel_name, shop_id, user_id, username,
            "Cookie 刷新与完整登录均失败",
        )
        return False

    except Exception as e:
        logger.error(
            f"重登流程异常: {username}, error_type={type(e).__name__}"
        )
        relogin_guard.release(channel_name, shop_id, user_id, success=False)
        notify_relogin_failure(
            channel_name, shop_id, user_id, username,
            f"重登流程异常（{type(e).__name__}）",
        )
        return False


def _apply_cookies_to_cache_and_db(
    channel_name: str, shop_id: str, user_id: str, new_cookies
):
    """将新 cookie 同步写入缓存和数据库"""
    from Channel.pinduoduo.cookie_cache import cookie_cache

    if isinstance(new_cookies, str):
        try:
            cookies_dict = json.loads(new_cookies)
        except json.JSONDecodeError:
            logger.error("应用 cookie 失败: JSON 解析错误")
            return
    elif isinstance(new_cookies, dict):
        cookies_dict = new_cookies
    else:
        logger.error(f"应用 cookie 失败: 不支持的类型 {type(new_cookies)}")
        return

    cookie_cache.set(channel_name, shop_id, user_id, cookies_dict)
    db_manager.update_account_cookies(channel_name, shop_id, user_id, new_cookies)
