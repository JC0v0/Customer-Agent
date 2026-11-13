import requests
import json
import time
import random
from urllib.parse import urlsplit
from typing import Dict, Any, Optional, Union, Callable
from utils.logger_loguru import get_logger
from database import db_manager

# 延迟导入，避免循环导入
import importlib


class BaseRequest:
    """
    API请求基类，统一管理requests请求（抖音渠道，结构与拼多多一致）

    功能特性：
    - 统一的请求重试机制（指数退避+随机抖动）
    - 自动会话过期检测和重新登录
    - 统一的错误处理和日志记录
    - 灵活的请求头和cookies管理

    注意：抖音真实协议（取 token / 发送 / 重登）尚未逆向，相关传输缝位于
    Channel/douyin/utils/API/get_token.py、send_message.py 与 douyin_login.py。
    本基类仅提供通用请求骨架与 cookie 管理；当 GetToken/SendMessage 抛出
    NotImplementedError（或返回 None）时，调用方会按失败路径优雅降级。
    """

    # 会话过期错误码（按抖音实际响应补全）
    SESSION_EXPIRED_ERROR_CODE = 43001
    # 重试抖动范围（随机因子乘数）
    RETRY_JITTER_MIN = 0.1
    RETRY_JITTER_MAX = 0.3

    def __init__(self, shop_id: str = None, user_id: str = None, channel_name: str = "douyin",
                 max_retries: int = 3, retry_delay: float = 1.0, retry_backoff: float = 2.0):
        """
        初始化基类
        """
        self.logger = get_logger(self.__class__.__name__)
        self.shop_id = shop_id
        self.user_id = user_id
        self.channel_name = channel_name

        # 重试配置
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff

        # 默认请求头
        self.default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
            'Content-Type': 'application/json',
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'priority': 'u=1, i'
        }

        # 初始化账户信息和cookies
        self.cookies = {}
        self.account_name = "未知账号"
        # backstage API 风控参数（登录时随完整 cookies 一起保存）
        self.verify_fp = None
        self.ms_token = None

        if shop_id and user_id:
            self._init_account_info()

    @staticmethod
    def _extract_browser_cookies(cookies_data: Any) -> Dict[str, Any]:
        """从登录 JSON 中提取完整浏览器 cookies 与风控参数。

        登录返回 JSON 结构：
            {pigeon_session: {...}, shop_name, user_name,
             cookies_full: {name: value, ...}, verifyFp: ..., msToken: ...}

        Returns:
            (browser_cookies, verify_fp, ms_token) —— 均为 dict/str/None。
        """
        if not isinstance(cookies_data, dict):
            return {}, None, None
        cookies_full = cookies_data.get("cookies_full")
        verify_fp = cookies_data.get("verifyFp")
        ms_token = cookies_data.get("msToken")
        browser_cookies = {}
        if isinstance(cookies_full, dict):
            browser_cookies = {
                str(k): str(v)
                for k, v in cookies_full.items()
                if k is not None and v is not None
            }
        return browser_cookies, verify_fp, ms_token

    def _backstage_params(self) -> Dict[str, str]:
        """生成 backstage API 风控查询参数（verifyFp/fp/msToken，有则带上）。"""
        params = {}
        if self.verify_fp:
            params["verifyFp"] = self.verify_fp
            params["fp"] = self.verify_fp
        if self.ms_token:
            params["msToken"] = self.ms_token
        return params

    def _init_account_info(self):
        """初始化账户信息，优先从共享缓存读取 cookie"""
        from Channel.douyin.cookie_cache import cookie_cache

        try:
            # 优先从共享缓存获取 cookie
            cached = cookie_cache.get(self.channel_name, self.shop_id, self.user_id)
            if cached is not None:
                self.cookies = cached
                # 仍需从 DB 获取账户名
                account_info = db_manager.get_account(self.channel_name, self.shop_id, self.user_id)
                if account_info:
                    self.account_name = account_info.get('username', '未知账号')
                return

            # 缓存未命中，从 DB 加载
            account_info = db_manager.get_account(self.channel_name, self.shop_id, self.user_id)
            if account_info:
                self.account_name = account_info.get('username', '未知账号')
                cookies_data = account_info.get('cookies')

                # 处理cookies格式
                if isinstance(cookies_data, str):
                    try:
                        self.cookies = json.loads(cookies_data)
                    except json.JSONDecodeError:
                        self.logger.error(f"解析账号 {self.account_name} 的cookies失败")
                        self.cookies = {}
                elif isinstance(cookies_data, dict):
                    self.cookies = cookies_data
                else:
                    self.logger.warning(f"账号 {self.account_name} 的cookies为空")
                    self.cookies = {}

                # 登录 JSON 含 cookies_full 时，用完整浏览器 cookies 覆盖（backstage API 鉴权）
                browser_cookies, verify_fp, ms_token = self._extract_browser_cookies(self.cookies)
                if browser_cookies:
                    self.cookies = browser_cookies
                    self.logger.debug(
                        f"账号 {self.account_name} 使用完整浏览器 cookies: {len(browser_cookies)} 个"
                    )
                self.verify_fp = verify_fp or self.verify_fp
                self.ms_token = ms_token or self.ms_token

                # 写入共享缓存
                if self.cookies:
                    cookie_cache.set(self.channel_name, self.shop_id, self.user_id, self.cookies)
            else:
                self.logger.error(f"无法在数据库中找到账户: shop_id={self.shop_id}, user_id={self.user_id}")
        except Exception as e:
            self.logger.error(
                f"初始化账户信息失败: error_type={type(e).__name__}"
            )

    def _is_session_expired(self, response_data: Dict[str, Any]) -> bool:
        """
        检测会话是否过期（按抖音真实响应补全判定条件）
        """
        if not response_data:
            return False

        if (response_data.get('error_code') == self.SESSION_EXPIRED_ERROR_CODE and
            '会话已过期' in str(response_data.get('error_msg', ''))):
            self.logger.warning(f"检测到账号 {self.account_name} 会话过期")
            return True

        return False

    def _run_async_login_func(self, func: callable, *args, **kwargs) -> Any:
        """
        在线程中安全执行异步登录函数（避免事件循环冲突）
        """
        from utils.async_helper import run_async_in_thread

        async def _run_wrapper() -> Any:
            return await func(*args, **kwargs)

        return run_async_in_thread(_run_wrapper(), timeout=60.0)

    def _get_account_credentials(self) -> Optional[tuple]:
        """
        获取账号凭证（用户名和密码）
        """
        account_info = db_manager.get_account(self.channel_name, self.shop_id, self.user_id)
        if not account_info:
            self.logger.error(f"无法获取账号信息: shop_id={self.shop_id}, user_id={self.user_id}")
            return None

        username = account_info.get('username')
        password = account_info.get('password')

        if not username:
            self.logger.error(f"账号 {self.account_name} 缺少用户名")
            return None

        return username, password

    def _login_result_matches_account(self, result: Any) -> bool:
        """Ensure a browser session cannot replace another account's cookies."""
        if not isinstance(result, dict):
            return False
        return (
            str(result.get("channel_name") or self.channel_name) == str(self.channel_name)
            and str(result.get("shop_id")) == str(self.shop_id)
            and str(result.get("user_id")) == str(self.user_id)
        )

    def _relogin_and_update_cookies(self) -> bool:
        """
        重新获取cookies并更新
        优先使用refresh_cookies（无需重新输入密码），失败时回退到完整重新登录

        Returns:
            是否重新获取cookies成功
        """
        from Channel.douyin.cookie_utils import relogin_guard
        from Channel.douyin.cookie_cache import cookie_cache

        # 尝试获取重登权（防止并发重复重登）
        if not relogin_guard.try_acquire(self.channel_name, self.shop_id, self.user_id):
            self.logger.info(f"重登已在其他线程进行中或冷却期内: {self.account_name}，从缓存读取最新cookie")
            cached = cookie_cache.get(self.channel_name, self.shop_id, self.user_id)
            if cached:
                self.cookies = cached
            return cached is not None

        try:
            credentials = self._get_account_credentials()
            if not credentials:
                relogin_guard.release(self.channel_name, self.shop_id, self.user_id, success=False)
                return False
            username, password = credentials

            douyin_login_module = importlib.import_module('Channel.douyin.douyin_login')

            # 优先尝试刷新cookies
            self.logger.info(f"尝试为账号 {self.account_name} 刷新cookies（无需重新登录）...")

            try:
                refresh_result = self._run_async_login_func(
                    douyin_login_module.refresh_douyin_cookies,
                    username,
                    password,
                    profile_scope=f"{self.channel_name}:{self.shop_id}:{self.user_id}",
                )

                if refresh_result and isinstance(refresh_result, dict):
                    if not self._login_result_matches_account(refresh_result):
                        self.logger.error("cookies刷新返回的账号身份不匹配，已拒绝应用")
                        refresh_result = None
                if refresh_result and isinstance(refresh_result, dict):
                    new_cookies = refresh_result.get('cookies')
                    if new_cookies:
                        self._apply_new_cookies(new_cookies)
                        self.logger.info(f"账号 {self.account_name} cookies刷新成功")
                        relogin_guard.release(self.channel_name, self.shop_id, self.user_id, success=True)
                        return True
                    else:
                        self.logger.warning(f"账号 {self.account_name} cookies刷新返回无效数据")
                else:
                    self.logger.warning(f"账号 {self.account_name} cookies刷新失败，可能登录状态已失效")

            except Exception as refresh_error:
                self.logger.warning(
                    f"账号 {self.account_name} cookies刷新异常: "
                    f"error_type={type(refresh_error).__name__}"
                )

            # 回退到完整重新登录（headless 模式，避免弹出浏览器窗口）
            if not password:
                self.logger.error(f"账号 {self.account_name} 缺少密码，无法进行完整重新登录")
                relogin_guard.release(self.channel_name, self.shop_id, self.user_id, success=False)
                return False

            self.logger.info(f"回退到完整重新登录模式（headless，账号 {self.account_name}）...")

            try:
                login_result = self._run_async_login_func(
                    douyin_login_module.login_douyin,
                    username,
                    password,
                    True,
                    profile_scope=f"{self.channel_name}:{self.shop_id}:{self.user_id}",
                )

                if login_result and isinstance(login_result, dict):
                    if not self._login_result_matches_account(login_result):
                        self.logger.error("完整登录返回的账号身份不匹配，已拒绝应用")
                        login_result = None
                if login_result and isinstance(login_result, dict):
                    new_cookies = login_result.get('cookies')
                    if new_cookies:
                        self._apply_new_cookies(new_cookies)
                        self.logger.info(f"账号 {self.account_name} 完整重新登录成功，cookies已更新")
                        relogin_guard.release(self.channel_name, self.shop_id, self.user_id, success=True)
                        return True
                    else:
                        self.logger.error(f"账号 {self.account_name} 完整重新登录失败：未获取到有效cookies")
                        relogin_guard.release(self.channel_name, self.shop_id, self.user_id, success=False)
                        return False
                else:
                    self.logger.error(f"账号 {self.account_name} 完整重新登录失败")
                    relogin_guard.release(self.channel_name, self.shop_id, self.user_id, success=False)
                    return False

            except Exception as login_error:
                self.logger.error(
                    f"账号 {self.account_name} 完整重新登录异常: "
                    f"error_type={type(login_error).__name__}"
                )
                relogin_guard.release(self.channel_name, self.shop_id, self.user_id, success=False)
                return False

        except Exception as e:
            self.logger.error(
                f"账号 {self.account_name} 重新获取cookies过程中发生错误: "
                f"error_type={type(e).__name__}"
            )
            relogin_guard.release(self.channel_name, self.shop_id, self.user_id, success=False)
            return False

    def _should_retry(self, response: requests.Response = None, exception: Exception = None) -> bool:
        """判断是否应该重试"""
        if exception:
            if isinstance(exception, (requests.ConnectionError, requests.Timeout,
                                    requests.HTTPError, requests.TooManyRedirects)):
                return True

        if response:
            if response.status_code >= 500:
                return True
            if response.status_code in [429, 408, 502, 503, 504]:
                return True

        return False

    def _calculate_retry_delay(self, attempt: int) -> float:
        """计算重试延迟时间（指数退避+随机抖动）"""
        delay = self.retry_delay * (self.retry_backoff ** attempt)
        jitter = random.uniform(self.RETRY_JITTER_MIN, self.RETRY_JITTER_MAX) * delay
        return delay + jitter

    def _execute_with_retry(self, request_func: Callable, expect_json: bool = True) -> Optional[Dict[str, Any]]:
        """带重试机制执行请求"""
        last_exception = None
        last_response = None
        relogin_attempted = False

        for attempt in range(self.max_retries + 1):
            try:
                response = request_func()

                if response and response.status_code == 200:
                    response_data = self._handle_response(response, expect_json)

                    if (response_data and self._is_session_expired(response_data)
                        and not relogin_attempted and self.shop_id and self.user_id):

                        self.logger.info(f"检测到会话过期，尝试重新登录...")
                        relogin_attempted = True

                        if self._relogin_and_update_cookies():
                            self.logger.info(f"重新登录成功，重试请求...")
                            continue
                        else:
                            self.logger.error(f"重新登录失败，请求终止")
                            return response_data

                    return response_data

                last_response = response

                if attempt < self.max_retries and self._should_retry(response=response):
                    delay = self._calculate_retry_delay(attempt)
                    self.logger.warning(f"请求失败，状态码: {response.status_code}，"
                                      f"第 {attempt + 1} 次重试，延迟 {delay:.2f} 秒")
                    time.sleep(delay)
                    continue
                else:
                    return self._handle_response(response, expect_json)

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries and self._should_retry(exception=e):
                    delay = self._calculate_retry_delay(attempt)
                    self.logger.warning(f"请求异常: error_type={type(e).__name__}，"
                                      f"第 {attempt + 1} 次重试，延迟 {delay:.2f} 秒")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(
                        f"请求最终失败: error_type={type(e).__name__}"
                    )
                    return None

        if last_exception:
            self.logger.error(f"重试 {self.max_retries} 次后仍然失败，最后异常: {str(last_exception)}")
        elif last_response:
            self.logger.error(f"重试 {self.max_retries} 次后仍然失败，最后状态码: {last_response.status_code}")

        return None

    def _merge_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """合并请求头"""
        merged_headers = self.default_headers.copy()
        if headers:
            merged_headers.update(headers)
        return merged_headers

    def _handle_response(self, response: requests.Response, expect_json: bool = True) -> Optional[Dict[str, Any]]:
        """统一处理响应"""
        try:
            if response.status_code != 200:
                self.logger.error(
                    "请求失败，状态码: %s, 响应长度: %s",
                    response.status_code,
                    len(response.content or b""),
                )
                return None

            if expect_json:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    self.logger.error(
                        "解析JSON响应失败，状态码: %s, 响应长度: %s",
                        response.status_code,
                        len(response.content or b""),
                    )
                    return None
            else:
                return {"text": response.text, "status_code": response.status_code}

        except Exception as e:
            self.logger.error(f"处理响应时发生错误: {type(e).__name__}")
            return None

    _SENSITIVE_KEYS = {'password', 'cookies', 'token', 'api_key', 'access_token', 'anti-content', 'anti_content'}

    def _sanitize_for_log(self, data: Any) -> Any:
        """对日志中的敏感字段进行脱敏处理"""
        if not isinstance(data, dict):
            return data
        result = {}
        for k, v in data.items():
            if k.lower() in self._SENSITIVE_KEYS or any(s in k.lower() for s in self._SENSITIVE_KEYS):
                result[k] = '***'
            elif isinstance(v, dict):
                result[k] = self._sanitize_for_log(v)
            elif isinstance(v, list):
                result[k] = [self._sanitize_for_log(i) if isinstance(i, dict) else i for i in v]
            else:
                result[k] = v
        return result

    def _log_request(self, method: str, url: str, **kwargs):
        """记录请求日志（自动脱敏敏感字段）"""
        parsed = urlsplit(url)
        safe_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        self.logger.debug(f"发起{method}请求: {safe_url}")
        if 'data' in kwargs or 'json' in kwargs:
            params = kwargs.get('data') or kwargs.get('json')
            self.logger.debug(f"请求参数: {self._sanitize_for_log(params)}")

    def get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict[str, str]] = None,
            timeout: int = 30, expect_json: bool = True, **kwargs) -> Optional[Dict[str, Any]]:
        """发起GET请求"""
        merged_headers = self._merge_headers(headers)
        self._log_request("GET", url, params=params)

        def _make_request():
            return requests.get(
                url,
                params=params,
                headers=merged_headers,
                cookies=self.cookies,
                timeout=timeout,
                **kwargs
            )

        return self._execute_with_retry(_make_request, expect_json=expect_json)

    def post(self, url: str, data: Optional[Union[Dict, str]] = None, json_data: Optional[Dict] = None,
             headers: Optional[Dict[str, str]] = None, timeout: int = 30,
             expect_json: bool = True, **kwargs) -> Optional[Dict[str, Any]]:
        """发起POST请求"""
        merged_headers = self._merge_headers(headers)
        self._log_request("POST", url, data=data, json=json_data)

        def _make_request():
            return requests.post(
                url,
                data=data,
                json=json_data,
                headers=merged_headers,
                cookies=self.cookies,
                timeout=timeout,
                **kwargs
            )

        return self._execute_with_retry(_make_request, expect_json=expect_json)

    def generate_request_id(self) -> int:
        """生成请求ID"""
        return int(time.time() * 1000)

    def update_cookies(self, new_cookies: Union[Dict, str]) -> None:
        """更新cookies（登录 JSON 含 cookies_full 时自动提取完整浏览器 cookies 与风控参数）"""
        if isinstance(new_cookies, str):
            try:
                self.cookies = json.loads(new_cookies)
            except json.JSONDecodeError:
                self.logger.error("更新cookies失败: JSON解析错误")
        elif isinstance(new_cookies, dict):
            self.cookies = new_cookies
        else:
            self.logger.error("更新cookies失败: 不支持的数据类型")
            return

        # 登录 JSON 含 cookies_full 时，用完整浏览器 cookies 覆盖（backstage API 鉴权）
        browser_cookies, verify_fp, ms_token = self._extract_browser_cookies(self.cookies)
        if browser_cookies:
            self.cookies = browser_cookies
        if verify_fp:
            self.verify_fp = verify_fp
        if ms_token:
            self.ms_token = ms_token

    def set_default_header(self, key: str, value: str) -> None:
        """设置默认请求头"""
        self.default_headers[key] = value

    def remove_default_header(self, key: str) -> None:
        """移除默认请求头"""
        if key in self.default_headers:
            del self.default_headers[key]

    def set_retry_config(self, max_retries: int = None, retry_delay: float = None,
                        retry_backoff: float = None) -> None:
        """动态设置重试配置"""
        if max_retries is not None:
            self.max_retries = max_retries
        if retry_delay is not None:
            self.retry_delay = retry_delay
        if retry_backoff is not None:
            self.retry_backoff = retry_backoff

        self.logger.info(f"重试配置已更新: max_retries={self.max_retries}, "
                        f"retry_delay={self.retry_delay}, retry_backoff={self.retry_backoff}")

    def disable_retry(self) -> None:
        """禁用重试功能"""
        self.max_retries = 0
        self.logger.info("重试功能已禁用")

    def enable_retry(self, max_retries: int = 3) -> None:
        """启用重试功能"""
        self.max_retries = max_retries
        self.logger.info(f"重试功能已启用，最大重试次数: {max_retries}")

    def get_retry_config(self) -> Dict[str, Union[int, float]]:
        """获取当前重试配置"""
        return {
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'retry_backoff': self.retry_backoff
        }

    def _apply_new_cookies(self, new_cookies: Any) -> None:
        """应用新cookies到当前实例、共享缓存并持久化到数据库"""
        from Channel.douyin.cookie_cache import cookie_cache

        self.update_cookies(new_cookies)
        cookie_cache.set(self.channel_name, self.shop_id, self.user_id, self.cookies)
        db_manager.update_account_cookies(
            self.channel_name,
            self.shop_id,
            self.user_id,
            new_cookies
        )

    def force_relogin(self) -> bool:
        """强制重新获取cookies"""
        if not self.shop_id or not self.user_id:
            self.logger.error("无法强制重新获取cookies：缺少shop_id或user_id")
            return False

        self.logger.info(f"手动触发账号 {self.account_name} 重新获取cookies...")
        return self._relogin_and_update_cookies()

    def force_refresh_cookies(self) -> bool:
        """强制只刷新cookies（不进行完整重新登录）"""
        if not self.shop_id or not self.user_id:
            self.logger.error("无法强制刷新cookies：缺少shop_id或user_id")
            return False

        try:
            credentials = self._get_account_credentials()
            if not credentials:
                self.logger.error(f"无法获取账号信息进行cookies刷新")
                return False
            username, password = credentials

            self.logger.info(f"手动触发账号 {self.account_name} 刷新cookies（仅刷新模式）...")

            douyin_login_module = importlib.import_module('Channel.douyin.douyin_login')
            refresh_result = self._run_async_login_func(
                douyin_login_module.refresh_douyin_cookies,
                username,
                password,
                profile_scope=f"{self.channel_name}:{self.shop_id}:{self.user_id}",
            )

            if refresh_result and isinstance(refresh_result, dict):
                if not self._login_result_matches_account(refresh_result):
                    self.logger.error("cookies刷新返回的账号身份不匹配，已拒绝应用")
                    refresh_result = None
            if refresh_result and isinstance(refresh_result, dict):
                new_cookies = refresh_result.get('cookies')
                if new_cookies:
                    self._apply_new_cookies(new_cookies)
                    self.logger.info(f"账号 {self.account_name} cookies刷新成功（仅刷新模式）")
                    return True
                else:
                    self.logger.error(f"账号 {self.account_name} cookies刷新失败：未获取到有效cookies")
                    return False
            else:
                self.logger.error(f"账号 {self.account_name} cookies刷新失败")
                return False

        except Exception as e:
            self.logger.error(
                f"账号 {self.account_name} cookies刷新过程中发生错误: "
                f"error_type={type(e).__name__}"
            )
            return False
