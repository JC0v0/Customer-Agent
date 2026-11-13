# -*- coding: utf-8 -*-
"""抖音接收回归：WS 立刻返回、只回新消息、同一条只派发一次。

接收路径是 v3 get_user_message 时间线接口（不是「当前会话列表」）：
会话被系统关闭后不在那个列表里，但买家仍会继续往该会话发消息，
以列表为主路径会彻底收不到。
"""
from __future__ import annotations

import asyncio
import time
import unittest
from unittest import mock

from Channel.douyin.core.douyin_lifecycle import LifecycleMixin
from Channel.douyin.message_rules import (
    Origin,
    is_agent_message,
    is_self_client_message,
    resolve_http_origin,
)
from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport


class FakeStop:
    """线程安全的停止标志，供 to_thread 里的假拉取函数置位。"""

    def __init__(self):
        self._flag = False

    def is_set(self):
        return self._flag

    def set(self):
        self._flag = True

    async def wait(self):
        await asyncio.sleep(0.05)
        return self._flag


def make_mixin(dispatch, stop):
    mixin = LifecycleMixin()
    mixin.logger = mock.MagicMock()
    mixin._stop_event = stop
    mixin._dispatch_pigeon_event = dispatch
    return mixin


def make_http(fetch):
    """桩 PigeonHttp：只提供 pull_messages(since_us)。"""
    http = mock.MagicMock()
    http.pull_messages = fetch
    return http


def make_transport():
    transport = mock.MagicMock()
    transport.conv_state = {}
    return transport


def buyer_message(**overrides):
    """v3 接口里真实买家消息的形态：biz=Buyer，sender_role 为空。"""
    message = {
        "type": 1000,
        "conv_id": "1000000000000000001",
        "msg_id": "buyer-msg-1",
        "text": "有人吗",
        "ts": int(time.time() * 1e6) - 240 * 10 ** 6,
        "sender_biz_role": "Buyer",
        "sender_role": "",
        "security_sender_id": "buyer-secret-1",
        "shop_id": "123456789",
        "talk_id": "7706578658005222691",
        "client_message_id": "buyer-client-1",
    }
    message.update(overrides)
    return message


class TestWsDrain(unittest.IsolatedAsyncioTestCase):
    async def test_first_event_returns_without_waiting_full_timeout(self):
        transport = PigeonTransport.__new__(PigeonTransport)
        transport._queue = asyncio.Queue()
        await transport._queue.put({"type": "message", "text": "你好"})
        await transport._queue.put({"type": "message", "text": "在吗"})

        started = time.monotonic()
        events = await transport._drain(1.0)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.2)
        self.assertEqual(len(events), 2)

    async def test_empty_queue_waits_for_timeout(self):
        transport = PigeonTransport.__new__(PigeonTransport)
        transport._queue = asyncio.Queue()

        started = time.monotonic()
        events = await transport._drain(0.2)
        elapsed = time.monotonic() - started

        self.assertEqual(events, [])
        self.assertGreaterEqual(elapsed, 0.15)


class TestDedupeKey(unittest.TestCase):
    def test_ws_and_http_same_message_share_one_key(self):
        ws_event = {
            "type": "message",
            "conv_id": "conv-1",
            "ts": 1794320000000000,
            "text": "这个有货吗",
            "sender_role": "3",
        }
        http_event = {
            "type": "text",
            "conv_id": "conv-1",
            "ts": 1794320000000000,
            "text": "这个有货吗",
            "sender_biz_role": "Buyer",
        }

        ws_key = LifecycleMixin._pigeon_dedupe_key(ws_event)
        http_key = LifecycleMixin._pigeon_dedupe_key(http_event)

        self.assertTrue(ws_key)
        self.assertEqual(ws_key, http_key)

    def test_ack_frame_has_no_key(self):
        self.assertEqual(
            LifecycleMixin._pigeon_dedupe_key({"type": "is_ack", "text": None}), ""
        )


class TestHttpOrigin(unittest.TestCase):
    """v3 接口的 role 语义与会话列表不同，必须单独判定。"""

    def test_real_buyer_message_is_customer(self):
        # 真实买家消息 sender_role 是空字符串（不是 1）。
        self.assertIs(
            resolve_http_origin({"sender_biz_role": "Buyer", "sender_role": ""}),
            Origin.CUSTOMER,
        )

    def test_service_allocated_hint_is_not_customer(self):
        # 「客服XX接入」也是 Buyer，但 role=1；当成买家会把系统提示交给 AI。
        self.assertIs(
            resolve_http_origin({"sender_biz_role": "Buyer", "sender_role": "1"}),
            Origin.SYSTEM,
        )

    def test_shop_reply_is_merchant(self):
        self.assertIs(
            resolve_http_origin({"sender_biz_role": "Shop", "sender_role": "4"}),
            Origin.MERCHANT,
        )

    def test_only_current_server_is_agent(self):
        self.assertTrue(
            is_agent_message({"sender_biz_role": "CurrentServer", "sender_role": "2"})
        )
        # 店铺机器人 / 本程序 AI 回复都是 role=4，不能算人工介入。
        self.assertFalse(
            is_agent_message({"sender_biz_role": "Shop", "sender_role": "4"})
        )

    def test_self_client_id_detection(self):
        self.assertTrue(is_self_client_message("llm_process_text_2_d0daf12b"))
        self.assertTrue(is_self_client_message("2_AI_WELCOME_7706578658005222691"))
        self.assertFalse(is_self_client_message("B89A3BEA-F6E4-42DE-9976-E8CA7BF6727F"))
        self.assertFalse(is_self_client_message(""))


class TestBaseline(unittest.IsolatedAsyncioTestCase):
    async def test_history_is_never_dispatched(self):
        """启动时已有的消息一律不回复，包括刚刚发过的。"""
        dispatched = []

        async def fake_dispatch(event, *args):
            dispatched.append(event)

        stop = FakeStop()
        mixin = make_mixin(fake_dispatch, stop)
        rounds = {"n": 0}

        def fetch(since_us):
            rounds["n"] += 1
            if rounds["n"] >= 2:
                stop.set()
                return []
            return [buyer_message(text="几天前的消息")]

        await mixin._pigeon_http_pull_loop(
            make_http(fetch), make_transport(), set(), "shop-1", "user-1",
            "name", "queue", None,
        )

        self.assertEqual(dispatched, [])

    async def test_message_newer_than_baseline_is_dispatched(self):
        """平台时间戳比本机慢时，新消息仍必须派发。

        回归点：原来用「本机启动时刻」过滤，平台时间戳慢 66~246 秒，
        刚发的消息会被当成历史消息丢掉，表现为「发了消息毫无反应」。
        """
        dispatched = []

        async def fake_dispatch(event, *args):
            dispatched.append(event)

        stop = FakeStop()
        mixin = make_mixin(fake_dispatch, stop)
        rounds = {"n": 0}
        now_us = int(time.time() * 1e6)
        history = buyer_message(ts=now_us - 600 * 10 ** 6, text="历史")
        # 新消息的时间戳比本机慢 240 秒，但仍晚于历史消息。
        fresh = buyer_message(ts=now_us - 240 * 10 ** 6, text="现在这条")

        def fetch(since_us):
            rounds["n"] += 1
            if rounds["n"] == 1:
                return [dict(history)]
            if rounds["n"] >= 3:
                stop.set()
            return [dict(fresh)]

        await mixin._pigeon_http_pull_loop(
            make_http(fetch), make_transport(), set(), "shop-1", "user-1",
            "name", "queue", None,
        )

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0]["text"], "现在这条")

    async def test_same_message_is_dispatched_once(self):
        dispatched = []

        async def fake_dispatch(event, *args):
            dispatched.append(event)

        stop = FakeStop()
        mixin = make_mixin(fake_dispatch, stop)
        rounds = {"n": 0}
        # 同一条消息：时间戳、文本、id 全部固定，三轮拉到的是它本身。
        message = buyer_message()

        def fetch(since_us):
            rounds["n"] += 1
            if rounds["n"] == 1:
                return []
            if rounds["n"] >= 4:
                stop.set()
            return [dict(message)]

        await mixin._pigeon_http_pull_loop(
            make_http(fetch), make_transport(), set(), "shop-1", "user-1",
            "name", "queue", None,
        )

        self.assertEqual(len(dispatched), 1)

    async def test_conversation_context_is_backfilled(self):
        """发送需要买家 ID / 店铺 ID / 子会话 ID，必须从消息里回填。"""
        stop = FakeStop()
        mixin = make_mixin(mock.AsyncMock(), stop)
        transport = make_transport()
        rounds = {"n": 0}
        message = buyer_message()

        def fetch(since_us):
            rounds["n"] += 1
            if rounds["n"] == 1:
                return []
            stop.set()
            return [dict(message)]

        await mixin._pigeon_http_pull_loop(
            make_http(fetch), transport, set(), "shop-1", "user-1", "name",
            "queue", None,
        )

        state = transport.conv_state["1000000000000000001"]
        self.assertEqual(state["security_receiver_id"], "buyer-secret-1")
        self.assertEqual(state["shop_id"], "123456789")
        self.assertEqual(state["talk_id"], "7706578658005222691")


class TestMerchantMessagesNotDispatched(unittest.IsolatedAsyncioTestCase):
    """商家侧消息不派发：不回复、不转人工（与拼多多一致）。"""

    async def _run_once(self, message):
        stop = FakeStop()
        dispatch = mock.AsyncMock()
        mixin = make_mixin(dispatch, stop)
        rounds = {"n": 0}

        def fetch(since_us):
            rounds["n"] += 1
            if rounds["n"] == 1:
                return []
            stop.set()
            return [dict(message)]

        await mixin._pigeon_http_pull_loop(
            make_http(fetch), make_transport(), set(), "shop-1", "user-1",
            "name", "queue", None,
        )
        return dispatch

    async def test_human_cs_message_is_not_dispatched(self):
        dispatch = await self._run_once({
            "type": 1000,
            "conv_id": "conv-1",
            "text": "我来帮您处理",
            "ts": int(time.time() * 1e6),
            "sender_biz_role": "CurrentServer",
            "sender_role": "2",
            "client_message_id": "human-uuid",
        })
        self.assertEqual(dispatch.call_count, 0)

    async def test_system_close_notice_is_not_dispatched(self):
        """系统关闭提示 f6=1005，角色字段和人工一样。"""
        dispatch = await self._run_once({
            "type": 1005,
            "conv_id": "conv-1",
            "text": "您超时未回复，系统关闭会话",
            "ts": int(time.time() * 1e6),
            "sender_biz_role": "CurrentServer",
            "sender_role": "2",
            "client_message_id": "close_non_process_7707670112412730662",
        })
        self.assertEqual(dispatch.call_count, 0)

    async def test_shop_robot_message_is_not_dispatched(self):
        dispatch = await self._run_once({
            "type": 1000,
            "conv_id": "conv-1",
            "text": "在的～您想了解哪方面呢",
            "ts": int(time.time() * 1e6),
            "sender_biz_role": "Shop",
            "sender_role": "4",
            "client_message_id": "llm_process_text_2_abc",
        })
        self.assertEqual(dispatch.call_count, 0)


class TestPullFailure(unittest.IsolatedAsyncioTestCase):
    async def test_consecutive_failures_trigger_one_refresh(self):
        """连续失败按凭据失效处理，冷却期内只触发一次刷新。"""
        stop = FakeStop()
        mixin = make_mixin(mock.AsyncMock(), stop)
        refresh = mock.AsyncMock(return_value=True)
        mixin._auto_refresh_douyin_credentials = refresh
        rounds = {"n": 0}

        def fetch(since_us):
            rounds["n"] += 1
            if rounds["n"] >= 8:
                stop.set()
            raise TimeoutError("read timed out")

        await mixin._pigeon_http_pull_loop(
            make_http(fetch), make_transport(), set(), "shop", "user", "name",
            "queue", None,
        )
        await asyncio.sleep(0.05)

        self.assertEqual(refresh.await_count, 1)
        # 失败日志必须被限频，不能每轮刷一条。
        self.assertLessEqual(mixin.logger.warning.call_count, 1)


class TestSendShortId(unittest.TestCase):
    def test_conversation_list_talk_id_wins_over_ack_short_id(self):
        """会话已关闭时 ACK 给的是别的会话短 ID，必须用会话列表里的 talk_id。"""
        selected = PigeonTransport._select_conversation_short_id(
            7706549914272023835, "7706540248934679808"
        )
        self.assertEqual(selected, 7706540248934679808)

    def test_falls_back_to_ack_short_id_without_talk_id(self):
        self.assertEqual(
            PigeonTransport._select_conversation_short_id(7706549914272023835, ""),
            7706549914272023835,
        )
        self.assertEqual(
            PigeonTransport._select_conversation_short_id(0, "not-a-number"), 0
        )
