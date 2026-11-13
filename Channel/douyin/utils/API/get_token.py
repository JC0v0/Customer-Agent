"""
抖音取 token（结构与拼多多 GetToken 对齐）。

【实现说明】
抖音纯协议（WS）模式当前不可用，实际运行走浏览器桥（feige_bridge）。
GetToken 仅在 douyin_lifecycle 的纯协议占位路径中被调用（当前不会到达）。
本实现作为兼容层：从登录时存入 DB 的 cookies 中提取 pigeon 会话 token，
返回字符串 token；无可用 token 时返回 None（与拼多多失败返回一致），
绝不抛出 NotImplementedError。

返回签名与拼多多 GetToken.get_token() 一致：
    str: 成功返回 token 字符串
    None: 获取失败返回 None
"""
from ..base_request import BaseRequest
from utils.logger_loguru import get_logger
from Channel.douyin.utils.log_throttle import throttled_warning, throttled_info

logger = get_logger("GetToken")

# cookies 中可能存储 pigeon token 的字段名（按优先级尝试）
_TOKEN_KEY_CANDIDATES = [
    "pigeon_session_token",
    "pigeon_token",
    "access_token",
    "token",
    "session_token",
]


class GetToken(BaseRequest):
    """抖音取 token（pigeon 会话 token 兼容层）。"""

    def __init__(self, shop_id, user_id, channel_name="douyin"):
        super().__init__(shop_id, user_id, channel_name)

    def get_token(self):
        """
        获取抖音访问 token（从登录 cookies 中提取 pigeon 会话 token）。

        Returns:
            str: 成功返回 token 字符串
            None: 获取失败返回 None（与拼多多接口签名一致）
        """
        # 1) 从 self.cookies（BaseRequest 从 DB 加载的登录 cookies）中提取 token
        if isinstance(self.cookies, dict):
            for key in _TOKEN_KEY_CANDIDATES:
                val = self.cookies.get(key)
                if val and isinstance(val, str) and len(val) > 0:
                    logger.debug(
                        f"抖音 token 提取成功（来源 cookies.{key}）: "
                        f"account={self.account_name}, token_len={len(val)}"
                    )
                    return val

            # 2) 兼容：cookies 中可能有嵌套的 pigeon_session dict
            pigeon_session = self.cookies.get("pigeon_session")
            if isinstance(pigeon_session, dict):
                for key in _TOKEN_KEY_CANDIDATES:
                    val = pigeon_session.get(key)
                    if val and isinstance(val, str) and len(val) > 0:
                        logger.debug(
                            f"抖音 token 提取成功（来源 cookies.pigeon_session.{key}）: "
                            f"account={self.account_name}"
                        )
                        return val

        # 3) 无可用 token：返回 None（不抛异常，调用方按失败路径处理）
        throttled_warning(logger, "token.missing",
            f"抖音 token 未找到：cookies 中无 pigeon 会话 token。"
            f"account={self.account_name}, cookies_keys={sorted(self.cookies.keys()) if isinstance(self.cookies, dict) else 'N/A'}"
        )
        return None


__all__ = ['GetToken']
