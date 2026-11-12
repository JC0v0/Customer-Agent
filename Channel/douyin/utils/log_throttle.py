"""抖音渠道重复日志限频。

拼多多的接口只在成功或明确失败时打一条日志，抖音商品/订单/客服等工具
会在 AI 每次回复时被调用，同样的降级 WARNING 会连打几十条。这里按日志
key 做简单冷却：相同文案在 interval 秒内只输出一次，不改变任何业务逻辑。

只用于「降级/兜底/异常回退」这类可预期的重复日志；真正的发送失败、
登录失败等错误日志不走这里，保持原有级别和可见性。
"""
import threading
import time

_lock = threading.Lock()
_last_logged_at = {}


def should_log(key: str, interval: float = 60.0) -> bool:
    """相同 key 在 interval 秒内只允许输出一次。"""
    now = time.monotonic()
    with _lock:
        last = _last_logged_at.get(key)
        if last is not None and now - last < interval:
            return False
        _last_logged_at[key] = now
        return True


def throttled_warning(logger, key: str, message, interval: float = 60.0) -> None:
    """替代 logger.warning(message)，相同 key 在冷却期内只输出一次。"""
    if should_log(key, interval):
        logger.warning(message)


def throttled_info(logger, key: str, message, interval: float = 60.0) -> None:
    """替代 logger.info(message)，相同 key 在冷却期内只输出一次。"""
    if should_log(key, interval):
        logger.info(message)
