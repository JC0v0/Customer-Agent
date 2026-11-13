"""Regression coverage for the verified Douyin Pigeon buyer-message path."""

from __future__ import annotations

import unittest
from unittest import mock

from bridge.context import ChannelType, Context, ContextType
from Channel.douyin.core.douyin_message_handler import MessageHandlerMixin


class TestDouyinPigeonMessageHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.handler = MessageHandlerMixin()
        self.handler.channel_name = "douyin"
        self.handler.logger = mock.MagicMock()
        self.queue = mock.MagicMock()
        self.queue.put = mock.AsyncMock(return_value="queued-message")
        self.handler.queue_manager = mock.MagicMock()
        self.handler.queue_manager.get_or_create_queue.return_value = self.queue

    async def test_buyer_text_is_converted_and_queued_for_auto_reply(self):
        converted = []

        def convert(message, shop_id, user_id, username):
            converted.append((message, shop_id, user_id, username))
            return Context.create_douyin_context(
                content=message.content,
                from_uid=message.from_uid,
                user_msg_type=message.user_msg_type,
                shop_id=shop_id,
                user_id=user_id,
                username=username,
                channel_type=ChannelType.DOUYIN,
            )

        self.handler._convert_to_context = convert
        await self.handler._process_pigeon_event(
            {
                "type": "text",
                "conv_id": "buyer-conversation",
                "msg_id": "message-1",
                "text": "请问什么时候发货",
                "ts": 1_700_000_000,
                "sender_biz_role": "Buyer",
            },
            "shop-1",
            "merchant-1",
            "customer-service",
            "douyin_queue",
        )

        self.assertEqual(len(converted), 1)
        message, shop_id, user_id, username = converted[0]
        self.assertEqual(message.from_uid, "buyer-conversation")
        self.assertEqual(message.content, "请问什么时候发货")
        self.assertEqual(message.user_msg_type, ContextType.TEXT)
        self.assertEqual((shop_id, user_id, username), ("shop-1", "merchant-1", "customer-service"))
        queued_context = self.queue.put.await_args.args[0]
        self.assertEqual(queued_context.channel_type, ChannelType.DOUYIN)
        self.assertEqual(queued_context.kwargs.from_uid, "buyer-conversation")

    async def test_non_buyer_or_control_message_is_not_queued(self):
        self.handler._convert_to_context = mock.MagicMock()
        base_event = {
            "type": "text",
            "conv_id": "buyer-conversation",
            "msg_id": "message-1",
            "text": "正常文本",
            "ts": 1_700_000_000,
        }

        # 最后一帧既没有 sender_biz_role 也没有 sender_role，角色不明时
        # 宁可交给 HTTP 补拉，不能冒险把商家回声当成买家消息。
        for update in (
            {"sender_biz_role": "Shop"},
            {"sender_role": "2"},
            {"text": '{"control": true}'},
            {"type": "message"},
        ):
            await self.handler._process_pigeon_event(
                {**base_event, **update}, "shop-1", "merchant-1", "customer-service", "douyin_queue"
            )

        self.handler._convert_to_context.assert_not_called()
        self.queue.put.assert_not_awaited()

    async def test_ws_push_with_message_type_is_dispatched_immediately(self):
        """飞鸽 WS 帧的 type 常为 message / 1000，不能只认 text 而丢掉正文。"""
        def convert(message, shop_id, user_id, username):
            return Context.create_douyin_context(
                content=message.content,
                from_uid=message.from_uid,
                user_msg_type=message.user_msg_type,
                shop_id=shop_id,
                user_id=user_id,
                username=username,
                channel_type=ChannelType.DOUYIN,
            )

        self.handler._convert_to_context = convert
        await self.handler._process_pigeon_event(
            {
                "type": "message",
                "conv_id": "buyer-conversation",
                "msg_id": "message-ws-1",
                "text": "这个有货吗",
                "ts": 1_700_000_000,
                "sender_role": "3",
            },
            "shop-1",
            "merchant-1",
            "customer-service",
            "douyin_queue",
        )

        self.assertEqual(self.queue.put.await_count, 1)
        queued_context = self.queue.put.await_args.args[0]
        self.assertEqual(queued_context.content, "这个有货吗")

    async def test_ws_merchant_echo_is_never_dispatched(self):
        """商家侧回声不能被当成买家消息 reply 出去。"""
        self.handler._convert_to_context = mock.MagicMock()
        await self.handler._process_pigeon_event(
            {
                "type": "message",
                "conv_id": "buyer-conversation",
                "msg_id": "echo-1",
                "text": "亲亲，很高兴为您服务",
                "ts": 1_700_000_000,
                "sender_role": "2",
                "sender_biz_role": "CurrentServer",
            },
            "shop-1",
            "merchant-1",
            "customer-service",
            "douyin_queue",
        )

        self.handler._convert_to_context.assert_not_called()
        self.queue.put.assert_not_awaited()

