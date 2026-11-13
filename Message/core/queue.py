"""
简化的消息队列实现
只支持FIFO队列，移除未使用的复杂功能
"""

import asyncio
import hashlib
import time
from typing import Optional, Dict, Set
from utils.logger_loguru import get_logger

from ..models.queue_models import MessageWrapper, QueueStats, QueueConfig
from bridge.context import Context, _context_value


logger = get_logger(__name__)


class SimpleMessageQueue:
    """简化的消息队列 - 只支持FIFO"""

    def __init__(self, name: str, config: QueueConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"Queue.{name}")

        # 基本队列
        self._queue = asyncio.Queue(maxsize=config.max_size)
        self._stats = QueueStats()
        self._closed = False

        # 去重缓存（可选）
        self._deduplication_cache: Set[str] = set() if config.enable_deduplication else None
        self._last_cleanup_time = time.time()

    async def put(self, context: Context) -> str:
        """放入消息"""
        if self._closed:
            raise RuntimeError("Queue is closed")

        if self._queue.full():
            self._stats.total_enqueued += 1  # 计入统计但拒绝
            raise RuntimeError("Queue is full")

        message_wrapper = MessageWrapper(
            message_id="",  # 将在__post_init__中生成
            context=context,
            timestamp=time.time()
        )

        # 检查去重
        if self._should_deduplicate(message_wrapper):
            self.logger.debug(f"Message deduplicated: {message_wrapper.message_id}")
            return message_wrapper.message_id

        try:
            await self._queue.put(message_wrapper)
            self._stats.enqueue()
            self.logger.debug(f"Message enqueued: {message_wrapper.message_id}")
            return message_wrapper.message_id

        except asyncio.QueueFull:
            raise RuntimeError("Queue is full")

    async def get(self, timeout: Optional[float] = None) -> Optional[MessageWrapper]:
        """获取消息"""
        if self._closed and self._queue.empty():
            return None

        try:
            if timeout is not None:
                wrapper = await asyncio.wait_for(self._queue.get(), timeout)
            else:
                wrapper = await self._queue.get()

            self._stats.dequeue()
            self.logger.debug(f"Message dequeued: {wrapper.message_id}")
            return wrapper

        except asyncio.TimeoutError:
            return None

    def size(self) -> int:
        """获取队列大小"""
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return self._queue.empty()

    def is_closed(self) -> bool:
        """Return whether no more messages may be enqueued."""
        return self._closed

    def get_stats(self) -> QueueStats:
        """获取统计信息"""
        stats = QueueStats(
            total_enqueued=self._stats.total_enqueued,
            total_dequeued=self._stats.total_dequeued,
            current_size=self.size(),
            last_activity=self._stats.last_activity
        )
        return stats

    def close(self):
        """关闭队列"""
        self._closed = True
        self.logger.info(f"Queue {self.name} closed")

    def _should_deduplicate(self, wrapper: MessageWrapper) -> bool:
        """检查是否应该去重

        去重键优先使用消息自身的 msg_id（用于防 WebSocket 重投递同一消息）；
        缺少 msg_id 时退回 from_uid + 渠道 + 内容，避免不同用户发送相同
        文本（如“你好”）被误判为重复而丢弃。
        """
        # dedup 关闭时缓存为 None；启用时空集合是正常状态，不能提前返回，
        # 否则首条消息永远进不了缓存、去重形同虚设。
        if self._deduplication_cache is None:
            return False

        msg_id = _context_value(wrapper.context, "msg_id")

        if msg_id:
            dedup_key = f"mid:{msg_id}"
        else:
            from_uid = _context_value(wrapper.context, "from_uid")
            channel = ""
            ct = getattr(wrapper.context, "channel_type", None)
            if ct is not None:
                channel = getattr(ct, "value", str(ct))
            content_str = str(wrapper.context.content) if wrapper.context.content is not None else ""
            dedup_key = f"u:{channel}:{from_uid}:{content_str}"

        key_hash = hashlib.md5(dedup_key.encode("utf-8")).hexdigest()
        if key_hash in self._deduplication_cache:
            return True

        # 添加到缓存，并定期清理 + 限制大小避免内存泄漏
        self._deduplication_cache.add(key_hash)
        self._cleanup_deduplication_cache()
        return False

    def _cleanup_deduplication_cache(self):
        """清理过期的去重缓存，限制最大条目数避免内存泄漏"""
        current_time = time.time()
        # 定期清空缓存
        if current_time - self._last_cleanup_time > self.config.deduplication_window:
            self._deduplication_cache.clear()
            self._last_cleanup_time = current_time
            self.logger.debug("Deduplication cache cleaned (time window)")
            return

        # 限制最大缓存条目数（高并发保护）
        MAX_DEDUP_ENTRIES = 10000
        if len(self._deduplication_cache) > MAX_DEDUP_ENTRIES:
            # 清空一半以平衡内存和去重效果
            self._deduplication_cache.clear()
            self._last_cleanup_time = current_time
            self.logger.warning(f"Deduplication cache exceeded {MAX_DEDUP_ENTRIES} entries, cleared")

    async def clear(self):
        """清空队列"""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.logger.info(f"Queue {self.name} cleared")


class QueueManager:
    """队列管理器 - 简化版"""

    def __init__(self):
        self._queues: Dict[str, SimpleMessageQueue] = {}
        self.logger = get_logger("QueueManager")

    def get_or_create_queue(self, name: str, config: Optional[QueueConfig] = None) -> SimpleMessageQueue:
        """获取或创建队列"""
        if name not in self._queues:
            if config is None:
                config = QueueConfig()
            queue = SimpleMessageQueue(name, config)
            self._queues[name] = queue
            self.logger.debug(f"Created queue: {name}")
        return self._queues[name]

    def get_queue(self, name: str) -> Optional[SimpleMessageQueue]:
        """获取队列"""
        return self._queues.get(name)

    def remove_queue(self, name: str) -> Optional[SimpleMessageQueue]:
        """Remove a closed queue from this loop/account registry."""
        queue = self._queues.pop(name, None)
        if queue is not None:
            queue.close()
        return queue

    def recreate_queue(self, name: str, config: Optional[QueueConfig] = None) -> SimpleMessageQueue:
        """重新创建队列以绑定当前事件循环"""
        try:
            old = self._queues.get(name)
            if old:
                old.close()
                self._queues.pop(name, None)
        except Exception:
            pass
        if config is None:
            config = QueueConfig()
        queue = SimpleMessageQueue(name, config)
        self._queues[name] = queue
        self.logger.info(f"Recreated queue: {name}")
        return queue

    def list_queues(self) -> Dict[str, QueueStats]:
        """列出所有队列及其统计信息"""
        return {name: queue.get_stats() for name, queue in self._queues.items()}

    async def close_all(self):
        """关闭所有队列"""
        for queue in list(self._queues.values()):
            queue.close()
        self._queues.clear()
        self.logger.info("All queues closed")


# 全局队列管理器实例
queue_manager = QueueManager()
