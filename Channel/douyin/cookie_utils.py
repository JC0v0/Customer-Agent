"""
Cookie 工具模块（抖音渠道，结构与拼多多一致）

提供:
- check_cookies_valid: 轻量级 cookie 有效性验证（HTTP 请求，非浏览器）
- ReloginGuard: 按账户的并发重登互斥锁 + 冷却期
- perform_relogin: 独立的 cookie 刷新/重登函数，可从任意上下文调用

抖音的登录态保存在浏览器 cookies（cookies_full）中，验证走飞鸽
backstage/currentuser 端点；会话失效时复用 douyin_login 的刷新与完整登录
流程重新拉起浏览器、填入账号密码并等待登录验证。
"""

import json
import threading
import time
from typing import Dict, Optional

import requests

from database import db_manager
from utils.logger_loguru import get_logger
from utils.async_helper import run_async_in_thread

logger = get_logger("DouyinCookieUtils")

# 会话过期错误码（与 BaseRequest.SESSION_EXPIRED_ERROR_CODE 保持一致，按抖音补全）
SESSION_EXPIRED_ERROR_CODE = 43001

# 验证请求的默认 headers
_VALIDATION_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

# 飞鸽后台当前登录用户信息（登录时已用同一端点获取店铺名，code=0 表示会话有效）
_VALIDATION_URL = (
    "https://pigeon.jinritemai.com/backstage/currentuser?biz_type=4&_pms=1"
)

# 响应中可确认登录态失效的提示（只在这些情形返回 False，其余返回 None 保守降级）
_LOGIN_EXPIRED_HINTS = (
    "未登录",
    "请先登录",
    "登录失效",
    "登录状态失效",
    "会话过期",
    "会话已过期",
)


def _extract_browser_cookies(cookies: dict) -> dict:
    """从账号 cookies 中取出可直接发给飞鸽后台的浏览器 cookie 名值对。

    登录返回的 JSON 结构为 {pigeon_session, shop_name, user_name,
    cookies_full, verifyFp, msToken}，其中只有 cookies_full 是真实 cookie。
    兼容早期直接保存名值对的账号数据。

    Args:
        cookies: 数据库中保存的账号 cookies 字典

    Returns:
        dict: 浏览器 cookie 名值对；无可用 cookie 时返回空字典
    """
    if not isinstance(cookies, dict):
        return {}

    source = cookies.get("cookies_full")
    if not isinstance(source, dict):
        source = cookies

    browser_cookies = {}
    for name, value in source.items():
        if name is None or value is None:
            continue
        if not isinstance(value, (str, int, float, bool)):
            continue
        browser_cookies[str(name)] = str(value)
    return browser_cookies


def _login_result_matches_account(
    result, shop_id: str, user_id: str, username: str = ""
) -> bool:
    """校验重登结果是否属于目标账号。

    与拼多多一致按 shop_id/user_id 匹配；抖音店铺名修正后 shop_id 会变化，
    因此对同一登录名额外放行（login_douyin 内部已有相同的身份兜底）。
    """
    if not isinstance(result, dict):
        return False
    if username and str(result.get("username") or "") == str(username):
        return True
    return (
        str(result.get("shop_id")) == str(shop_id)
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

    向飞鸽 backstage/currentuser 端点发送 GET 请求，依据响应判定登录态：
    code=0 且能取到店铺信息即为有效；HTTP 401/403、被重定向到登录页、
    或响应里明确提示未登录/会话过期即为失效；其余情形返回 None（状态未知），
    由调用方按安全策略暂不重登。

    Args:
        channel_name: 渠道名称
        shop_id: 店铺 ID
        user_id: 用户 ID
        cookies: cookie 字典（登录 JSON 或早期名值对）
        timeout: 请求超时（秒）

    Returns:
        True: 服务端明确确认 cookie 有效
        False: 服务端明确确认 cookie 已过期
        None: 网络或响应异常，状态未知（由调用方决定是否重试）
    """
    browser_cookies = _extract_browser_cookies(cookies)
    if not browser_cookies:
        logger.debug("Cookie 验证: 无可用浏览器 cookie，状态未知")
        return None

    try:
        response = requests.get(
            _VALIDATION_URL,
            headers=_VALIDATION_HEADERS,
            cookies=browser_cookies,
            timeout=timeout,
            allow_redirects=False,
        )

        if response.status_code in (401, 403):
            logger.warning(f"Cookie 验证失败: 会话已过期 ({channel_name}:{shop_id}:{user_id})")
            return False

        location = response.headers.get("Location", "")
        if response.status_code in (301, 302, 303, 307, 308) and "login" in location.lower():
            logger.warning(f"Cookie 验证失败: 会话已过期 ({channel_name}:{shop_id}:{user_id})")
            return False

        if response.status_code != 200:
            logger.debug(f"Cookie 验证: HTTP {response.status_code}，状态未知")
            return None

        try:
            data = response.json()
        except json.JSONDecodeError:
            logger.debug("Cookie 验证: 响应非 JSON，状态未知")
            return None

        if not isinstance(data, dict):
            logger.debug("Cookie 验证: 响应格式未知，状态未知")
            return None

        code = data.get("code")
        if code == 0:
            return True

        message = str(
            data.get("message") or data.get("msg") or data.get("error_msg") or ""
        )
        for hint in _LOGIN_EXPIRED_HINTS:
            if hint in message:
                logger.warning(f"Cookie 验证失败: 会话已过期 ({channel_name}:{shop_id}:{user_id})")
                return False

        logger.debug(f"Cookie 验证: 服务端返回未知状态 code={code}")
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
        """尝试获取重登权。"""
        key = self._make_key(channel_name, shop_id, user_id)

        with self._lock:
            last_time = self._last_relogin.get(key, 0)
            if time.time() - last_time < self._cooldown:
                return False

            if key not in self._account_locks:
                self._account_locks[key] = threading.Lock()
            acct_lock = self._account_locks[key]

        return acct_lock.acquire(blocking=False)

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
    刷新与完整登录都会拉起本地浏览器：刷新复用持久化登录态，
    失效时自动填入账号密码并等待登录验证（短信/滑块需人工完成一次）。

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
        logger.info(f"重登已在其他线程进行中或冷却期内: {username}")
        from Channel.douyin.cookie_cache import cookie_cache
        return cookie_cache.get(channel_name, shop_id, user_id) is not None

    try:
        import importlib
        douyin_login_module = importlib.import_module('Channel.douyin.douyin_login')
        profile_scope = f"{channel_name}:{shop_id}:{user_id}"

        # 阶段 1: 刷新 cookie（复用浏览器持久化会话）
        logger.info(f"尝试刷新 cookie（无需重新登录）: {username}")
        try:
            refresh_result = run_async_in_thread(
                douyin_login_module.refresh_douyin_cookies(
                    username, password, profile_scope=profile_scope
                ),
                timeout=240.0,
            )
            if refresh_result and isinstance(refresh_result, dict):
                if not _login_result_matches_account(
                    refresh_result, shop_id, user_id, username
                ):
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
            return False

        logger.info(f"回退到完整重新登录 (headless={headless_fallback}): {username}")
        try:
            login_result = run_async_in_thread(
                douyin_login_module.login_douyin(
                    username,
                    password,
                    headless_fallback,
                    profile_scope=profile_scope,
                ),
                timeout=240.0,
            )
            if login_result and isinstance(login_result, dict):
                if not _login_result_matches_account(
                    login_result, shop_id, user_id, username
                ):
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
        return False

    except Exception as e:
        logger.error(
            f"重登流程异常: {username}, error_type={type(e).__name__}"
        )
        relogin_guard.release(channel_name, shop_id, user_id, success=False)
        return False


def _apply_cookies_to_cache_and_db(
    channel_name: str, shop_id: str, user_id: str, new_cookies
):
    """将新 cookie 同步写入缓存和数据库"""
    from Channel.douyin.cookie_cache import cookie_cache

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
