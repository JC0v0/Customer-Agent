# -*- coding: utf-8 -*-
"""会话列表接收链路回归：新会话覆盖、买家判定、时间戳换算。"""
from __future__ import annotations

import unittest

from tools.reverse_probe.douyin.pigeon_conv_list import PigeonConvList


def make_conv(conv_id, talk_id, shop_id, msg_list):
    return {
        "parentConShortId": conv_id,
        "subConversationId": talk_id,
        "coreInfoMap": {"service_entity_id": shop_id},
        "msgList": msg_list,
    }


def make_msg(content, create_time, biz_role, sender_id="", logid=""):
    ext = {"s:sender_biz_role": biz_role}
    if sender_id:
        ext["security_sender_id"] = sender_id
    if logid:
        ext["message_logid"] = logid
    return {
        "serverMessageId": logid or "server-1",
        "messageBody": {
            "msgType": 1000,
            "content": content,
            "createTime": create_time,
            "ext": ext,
        },
    }


class TestConvListFilter(unittest.TestCase):
    def setUp(self):
        self.collector = PigeonConvList("not-used.json")
        self.conversations = [
            make_conv(
                "7680792019895125267",
                "7706549914272023835",
                "123456789",
                [
                    make_msg("你好呀", "1794321001935", "Buyer", "SECURE-BUYER-1", "logid-1"),
                    make_msg("客服示例店铺接入", "1794321015000", "Buyer", "", "logid-2"),
                    make_msg("人工客服 示例店铺 为您服务", "1794321015001", "Shop", "", "logid-3"),
                    make_msg("Hi，欢迎光临本店", "1794321016000", "Shop", "", "logid-4"),
                ],
            )
        ]

    def test_only_real_buyer_text_is_returned(self):
        self.collector.fetch_conversations = lambda: self.conversations
        messages = self.collector.fetch_buyer_messages(0)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["text"], "你好呀")
        self.assertEqual(message["conv_id"], "7680792019895125267")
        self.assertEqual(message["talk_id"], "7706549914272023835")
        self.assertEqual(message["security_sender_id"], "SECURE-BUYER-1")
        self.assertEqual(message["shop_id"], "123456789")
        self.assertEqual(message["msg_id"], "logid-1")

    def test_new_conversation_is_covered(self):
        """新买家接入会生成新的父会话，必须一起返回。"""
        self.collector.fetch_conversations = lambda: [
            make_conv("7680792019895125267", "7706549914272023835", "123456789",
                      [make_msg("你好呀", "1794321001935", "Buyer", "SECURE-1", "logid-1")]),
            make_conv("7700000000000000000", "7706540248934679808", "123456789",
                      [make_msg("有人吗", "1794318740095433", "Buyer", "SECURE-2", "logid-2")]),
        ]

        messages = self.collector.fetch_buyer_messages(0)

        convs = set()
        for message in messages:
            convs.add(message["conv_id"])
        self.assertEqual(convs, {"7680792019895125267", "7700000000000000000"})

    def test_cursor_filters_older_messages(self):
        self.collector.fetch_conversations = lambda: self.conversations
        # 22:30:01 的消息微秒时间戳是 1794321001935000
        messages = self.collector.fetch_buyer_messages(1794321001935000)
        self.assertEqual(messages, [])

    def test_millisecond_timestamp_is_converted_to_microsecond(self):
        self.assertEqual(PigeonConvList._to_us("1794321001935"), 1794321001935000)
        self.assertEqual(PigeonConvList._to_us(None), None)
        self.assertEqual(PigeonConvList._to_us("bad"), None)
