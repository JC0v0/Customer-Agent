"""消息队列（Message.core.queue）单元测试。

`SimpleMessageQueue` 是「消息队列处理 + 高并发」能力的基础组件，此前没有
专属测试。这里覆盖 FIFO 顺序、去重、背压（满队列拒绝）、超时读取与统计，
并回归修复 ``get(timeout=0)`` 因真值判断而永久阻塞的边界问题。
"""

import unittest

from bridge.context import ChannelType, Context, ContextType
from Message.core.queue import SimpleMessageQueue, QueueManager
from Message.models.queue_models import QueueConfig


def _context(from_uid: str, content: str, msg_id: str | None = None) -> Context:
    return Context.create_pinduoduo_context(
        content=content,
        msg_id=msg_id,
        from_uid=from_uid,
        shop_id="shop-1",
        user_id="account-1",
        user_msg_type=ContextType.TEXT,
        channel_type=ChannelType.PINDUODUO,
    )


class SimpleMessageQueueTests(unittest.IsolatedAsyncioTestCase):
    def _queue(self, **kwargs) -> SimpleMessageQueue:
        return SimpleMessageQueue("test", QueueConfig(**kwargs))

    async def test_fifo_order(self):
        q = self._queue()
        for i in range(3):
            await q.put(_context(f"u{i}", f"m{i}", msg_id=f"id-{i}"))

        got = []
        for _ in range(3):
            wrapper = await q.get()
            got.append(wrapper.context.content)
        self.assertEqual(got, ["m0", "m1", "m2"])

    async def test_get_with_timeout_on_empty_queue_returns_none(self):
        q = self._queue()
        self.assertIsNone(await q.get(timeout=0.01))

    async def test_get_with_zero_timeout_does_not_block(self):
        # 回归：原先 `if timeout:` 把 0 当作“无超时”而永久阻塞。
        q = self._queue()
        self.assertIsNone(await q.get(timeout=0))

    async def test_deduplicates_by_msg_id(self):
        q = self._queue(enable_deduplication=True)
        await q.put(_context("u", "hi", msg_id="dup"))
        await q.put(_context("u", "hi", msg_id="dup"))
        self.assertEqual(q.size(), 1)

        wrapper = await q.get()
        self.assertEqual(wrapper.context.content, "hi")
        self.assertIsNone(await q.get(timeout=0.01))

    async def test_deduplicates_by_content_when_no_msg_id(self):
        q = self._queue(enable_deduplication=True)
        await q.put(_context("u1", "你好"))
        await q.put(_context("u1", "你好"))
        self.assertEqual(q.size(), 1)

    async def test_same_content_from_different_users_is_not_deduplicated(self):
        q = self._queue(enable_deduplication=True)
        await q.put(_context("u1", "你好"))
        await q.put(_context("u2", "你好"))
        self.assertEqual(q.size(), 2)

    async def test_deduplication_can_be_disabled(self):
        q = self._queue(enable_deduplication=False)
        await q.put(_context("u", "hi", msg_id="dup"))
        await q.put(_context("u", "hi", msg_id="dup"))
        self.assertEqual(q.size(), 2)

    async def test_full_queue_rejects_with_error(self):
        q = self._queue(max_size=1)
        await q.put(_context("u", "m1", msg_id="1"))
        with self.assertRaises(RuntimeError):
            await q.put(_context("u", "m2", msg_id="2"))
        self.assertEqual(q.size(), 1)

    async def test_stats_track_enqueue_and_dequeue(self):
        q = self._queue()
        await q.put(_context("u", "m", msg_id="1"))
        await q.put(_context("u", "m2", msg_id="2"))
        await q.get()

        stats = q.get_stats()
        self.assertEqual(stats.total_enqueued, 2)
        self.assertEqual(stats.total_dequeued, 1)
        self.assertEqual(stats.current_size, 1)

    async def test_closed_queue_get_returns_none_when_empty(self):
        q = self._queue()
        q.close()
        self.assertIsNone(await q.get())

    async def test_put_to_closed_queue_raises(self):
        q = self._queue()
        q.close()
        with self.assertRaises(RuntimeError):
            await q.put(_context("u", "m", msg_id="1"))


class QueueManagerTests(unittest.TestCase):
    def test_get_or_create_returns_same_instance(self):
        manager = QueueManager()
        a = manager.get_or_create_queue("q")
        b = manager.get_or_create_queue("q")
        self.assertIs(a, b)
        self.assertIsNot(a, manager.get_or_create_queue("other"))

    def test_remove_queue_closes_and_returns(self):
        manager = QueueManager()
        q = manager.get_or_create_queue("q")
        removed = manager.remove_queue("q")
        self.assertIs(removed, q)
        self.assertTrue(q.is_closed())
        self.assertIsNone(manager.get_queue("q"))


if __name__ == "__main__":
    unittest.main()
