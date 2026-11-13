"""
会话状态管理器

单例模式，用字典存储会话与过期时间戳，
用于记录"已转人工"状态，防止转人工后 AI 继续抢答。

session_key 格式：{channel}|{shop_id}|{uid}（由 bridge.context.make_handoff_key 统一生成，
必须带 channel 维度，避免同一买家 UID 跨平台（拼多多/淘宝/抖音）互相误判"已转人工"）。
默认有效期由 config.json 的 handoff.valid_hours 决定（单位：小时，默认 4 小时）。
改这个值即可把"转人工后 AI 静默多久"调整为 3 小时、5 小时或任意时长。

标记同时持久化到数据库（handoff_markers 表），
进程重启后内存缓存丢失时从数据库恢复，保证转人工状态不丢失。
"""
import time
import threading
from typing import Optional

from utils.logger_loguru import get_logger
from config import get_config


class SessionState:
    """会话状态管理器（线程安全单例）"""

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._handoffs = {}
                    obj._data_lock = threading.Lock()
                    obj._db = None  # 懒加载的 DatabaseManager
                    # 串行化数据库写操作，避免并发 merge 触发唯一约束冲突
                    obj._db_lock = threading.Lock()
                    obj._logger = get_logger("SessionState")
                    cls._instance = obj
        return cls._instance

    def mark_handoff(self, session_key: str, ttl_seconds: Optional[int] = None) -> None:
        """标记会话为已转人工，并持久化到数据库。

        有效期默认由 config.handoff.valid_hours（小时）决定，缺省 4 小时。
        也可显式传入 ttl_seconds（秒）临时覆盖（测试用）。
        """
        if ttl_seconds is None:
            try:
                hours = float(get_config("handoff.valid_hours", 4))
            except (TypeError, ValueError):
                hours = 4.0
            ttl_seconds = int(hours * 3600)
        expiry = time.time() + ttl_seconds
        with self._data_lock:
            self._handoffs[session_key] = expiry
        self._persist_marker(session_key, expiry)

    def is_handoff(self, session_key: str) -> bool:
        """判断会话是否处于转人工有效期内，过期自动清除"""
        now = time.time()
        expired = False
        # 内存缓存快路径
        with self._data_lock:
            expiry = self._handoffs.get(session_key)
            if expiry is not None:
                if now > expiry:
                    del self._handoffs[session_key]
                    expired = True
                else:
                    return True
        if expired:
            self._delete_marker(session_key)
            return False

        # 内存未命中：从数据库恢复（模拟进程重启后的场景）
        expiry = self._load_marker(session_key)
        if expiry is None:
            return False
        if now > expiry:
            self._delete_marker(session_key)
            return False
        with self._data_lock:
            self._handoffs[session_key] = expiry
        return True

    def clear_handoff(self, session_key: str) -> None:
        """手动清除转人工标记（内存与数据库）"""
        with self._data_lock:
            self._handoffs.pop(session_key, None)
        self._delete_marker(session_key)

    # ==================== 数据库持久化辅助方法 ====================

    def _get_db(self):
        """懒加载 DatabaseManager 单例。

        在调用时 import，确保数据库/模型导入顺序正确，
        也便于测试通过 mock 替换 get_db_manager。
        """
        if self._db is None:
            from database.db_manager import get_db_manager
            self._db = get_db_manager()
        return self._db

    def _persist_marker(self, session_key: str, expiry: float) -> None:
        """写入/更新数据库中的转人工标记（失败仅记录日志，不影响内存缓存）"""
        try:
            from database.models import HandoffMarker
            db = self._get_db()
            with self._db_lock:
                with db.session_scope() as session:
                    session.merge(
                        HandoffMarker(session_key=session_key, expiry=int(expiry))
                    )
        except Exception as e:
            self._logger.warning(
                f"持久化转人工标记失败（不影响内存缓存）: "
                f"session_key={session_key}, error_type={type(e).__name__}"
            )

    def _load_marker(self, session_key: str) -> Optional[float]:
        """从数据库读取转人工标记的过期时间，读取失败返回 None"""
        try:
            from database.models import HandoffMarker
            db = self._get_db()
            with self._db_lock:
                with db.session_scope() as session:
                    row = (
                        session.query(HandoffMarker)
                        .filter(HandoffMarker.session_key == session_key)
                        .first()
                    )
                    return float(row.expiry) if row is not None else None
        except Exception as e:
            self._logger.warning(
                f"读取转人工标记失败: session_key={session_key}, "
                f"error_type={type(e).__name__}"
            )
            return None

    def _delete_marker(self, session_key: str) -> None:
        """删除数据库中的转人工标记（失败仅记录日志）"""
        try:
            from database.models import HandoffMarker
            db = self._get_db()
            with self._db_lock:
                with db.session_scope() as session:
                    session.query(HandoffMarker).filter(
                        HandoffMarker.session_key == session_key
                    ).delete()
        except Exception as e:
            self._logger.warning(
                f"删除转人工标记失败: session_key={session_key}, "
                f"error_type={type(e).__name__}"
            )


# 全局单例实例（与 SessionState() 返回同一对象）
session_state = SessionState()
