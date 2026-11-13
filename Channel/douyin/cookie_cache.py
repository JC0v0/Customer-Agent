"""
共享内存级 Cookie 缓存（抖音渠道，与拼多多实现一致，按 (channel_name, shop_id, user_id) 索引）

所有 BaseRequest 实例通过此缓存共享 cookie，确保一处更新全局生效。
线程安全设计：使用 threading.RLock 保护所有读写操作。
"""

import threading
import time
from typing import Dict, Optional


class CookieCache:
    """线程安全的内存 Cookie 缓存，按 (channel_name, shop_id, user_id) 索引"""

    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Dict[str, dict] = {}
        self._timestamps: Dict[str, float] = {}

    @staticmethod
    def _make_key(channel_name: str, shop_id: str, user_id: str) -> str:
        return f"{channel_name}:{shop_id}:{user_id}"

    def get(self, channel_name: str, shop_id: str, user_id: str) -> Optional[dict]:
        """从缓存获取 cookie，未命中返回 None"""
        key = self._make_key(channel_name, shop_id, user_id)
        with self._lock:
            return self._cache.get(key)

    def set(self, channel_name: str, shop_id: str, user_id: str, cookies: dict):
        """更新缓存中的 cookie"""
        key = self._make_key(channel_name, shop_id, user_id)
        with self._lock:
            self._cache[key] = cookies
            self._timestamps[key] = time.time()

    def invalidate(self, channel_name: str, shop_id: str, user_id: str):
        """移除缓存条目"""
        key = self._make_key(channel_name, shop_id, user_id)
        with self._lock:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def get_age(self, channel_name: str, shop_id: str, user_id: str) -> Optional[float]:
        """获取缓存条目的年龄（秒），未命中返回 None"""
        key = self._make_key(channel_name, shop_id, user_id)
        with self._lock:
            ts = self._timestamps.get(key)
            if ts is None:
                return None
            return time.time() - ts


# 模块级单例
cookie_cache = CookieCache()
