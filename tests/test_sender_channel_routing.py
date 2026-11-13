# -*- coding: utf-8 -*-
"""发送器渠道路由测试。

真实故障复盘（2026-11-13 14:56，打包版实测）：
抖音会话回复时 `get_sender(context.channel_type)` 拿不到 DOUYIN 的发送器，
静默回退成拼多多实现，于是用 `shop_id=douyin_示例店铺` 去查拼多多库：

    未找到店铺: douyin_示例店铺 (渠道: pinduoduo)
    无法在数据库中找到账户: shop_id=douyin_示例店铺, user_id=douyin_13800138000
    发送文本消息失败: response_keys=['error_code', 'error_msg', 'success']

抖音通道本身已连通（同一份日志里有「连接已建立」），只有发送这一步串了渠道。
"""
import unittest
from unittest import mock

from bridge.context import ChannelType
from bridge.sender import DouyinSender, PinduoduoSender, get_sender


class SenderRegistryTests(unittest.TestCase):
    """注册表必须按渠道返回对应实现，不能一串到底。"""

    def test_douyin_enum_returns_douyin_sender(self):
        self.assertIsInstance(get_sender(ChannelType.DOUYIN), DouyinSender)

    def test_douyin_string_returns_douyin_sender(self):
        self.assertIsInstance(get_sender("douyin"), DouyinSender)

    def test_pinduoduo_still_returns_pinduoduo_sender(self):
        self.assertIsInstance(get_sender(ChannelType.PINDUODUO), PinduoduoSender)
        self.assertIsInstance(get_sender("pinduoduo"), PinduoduoSender)

    def test_none_keeps_legacy_pinduoduo_default(self):
        """无参调用的老工具（历史行为）保持回退拼多多。"""
        self.assertIsInstance(get_sender(), PinduoduoSender)
        self.assertIsInstance(get_sender(None), PinduoduoSender)

    def test_explicit_unregistered_channel_does_not_fall_back(self):
        """明确传了未注册渠道时必须返回 None，绝不静默借拼多多发出去。"""
        self.assertIsNone(get_sender(ChannelType.TAOBAO))
        self.assertIsNone(get_sender("taobao"))

    def test_unknown_string_keeps_legacy_fallback(self):
        """无法识别的字符串仍按历史行为回退，避免影响既有调用方。"""
        self.assertIsInstance(get_sender("not_a_channel"), PinduoduoSender)

    def test_douyin_sender_delegates_to_douyin_send_message(self):
        """DouyinSender 必须调用抖音的 SendMessage，且参数顺序与基类一致。"""
        fake = mock.Mock()
        fake.send_text.return_value = {"success": True}
        with mock.patch(
            "Channel.douyin.utils.API.send_message.SendMessage", return_value=fake
        ) as ctor:
            result = DouyinSender().send_text("douyin_shop", "douyin_uid", "conv_1", "hi")

        ctor.assert_called_once_with("douyin_shop", "douyin_uid")
        fake.send_text.assert_called_once_with("conv_1", "hi")
        self.assertTrue(result.get("success"))


class TransferConversationChannelTests(unittest.TestCase):
    """转人工工具不能对抖音调用拼多多的客服列表 / 转移接口。"""

    def _params(self, channel_type):
        from Agent.CustomerAgent.tools.move_conversation import TransferConversationParams

        return TransferConversationParams(
            channel_type=channel_type,
            shop_id="douyin_示例店铺",
            user_id="douyin_13800138000",
            recipient_uid="1000000000000000001",
        )

    def test_douyin_routes_to_douyin_sender(self):
        """抖音必须拿 DouyinSender，不能落到拼多多实现。"""
        from Agent.CustomerAgent.tools import move_conversation

        sender = mock.Mock()
        sender.get_cs_list.return_value = {"cs_other": {"username": "x"}}
        sender.transfer_to_cs.return_value = {"success": True}
        with mock.patch.object(
            move_conversation, "get_sender", return_value=sender
        ) as get_sender_mock:
            result = move_conversation.transfer_conversation(self._params("douyin"))

        self.assertEqual(get_sender_mock.call_args[0][0], "douyin")
        self.assertEqual(result, "会话转接成功")

    def test_pinduoduo_path_still_uses_sender(self):
        from Agent.CustomerAgent.tools import move_conversation

        sender = mock.Mock()
        sender.get_cs_list.return_value = {"cs_other": {"username": "x"}}
        sender.transfer_to_cs.return_value = {"success": True}
        with mock.patch.object(
            move_conversation, "get_sender", return_value=sender
        ) as get_sender_mock:
            result = move_conversation.transfer_conversation(self._params("pinduoduo"))

        self.assertEqual(get_sender_mock.call_args[0][0], "pinduoduo")
        self.assertEqual(result, "会话转接成功")
        sender.transfer_to_cs.assert_called_once()


class ProductListChannelTests(unittest.TestCase):
    """预取商品列表也必须带渠道。

    真实故障复盘（2026-11-13 15:13，打包版实测）：
    `MessageBuilder.fetch_product_list_text(shop_id, user_id)` 只传了 id，
    `GetShopProductsParams.channel_type` 为空 → `normalize_channel(None)`
    回退 pinduoduo → 抖音会话用 `douyin_示例店铺` 去查拼多多库：

        未找到店铺: douyin_示例店铺 (渠道: pinduoduo)
        无法在数据库中找到账户: shop_id=douyin_示例店铺, user_id=douyin_13800138000
        获取商品列表失败: error_msg=None

    这段发生在 system prompt 预取阶段，不在 LLM 工具调用里，所以
    tool_decorator 的受信任注入管不到，必须由调用方显式传 channel_type。
    """

    def _builder(self):
        from Agent.CustomerAgent.custom.message_builder import MessageBuilder

        return MessageBuilder.__new__(MessageBuilder)

    def _fake_managers(self, hits):
        class FakePdd:
            def __init__(self, **kwargs):
                hits.append(("pinduoduo", kwargs))

            def get_product_list(self, page=1, size=10):
                return {"success": True, "products": [], "total": 0}

        class FakeDouyin:
            def __init__(self, **kwargs):
                hits.append(("douyin", kwargs))

            def get_product_list(self, page=1, size=10):
                return {"success": True, "products": [], "total": 0}

        return FakePdd, FakeDouyin

    def _run(self, channel_type, hits):
        import Agent.CustomerAgent.tools.get_product_list as gpl

        fake_pdd, fake_douyin = self._fake_managers(hits)
        with mock.patch.object(gpl, "PddProductManager", fake_pdd), mock.patch.object(
            gpl, "DouyinProductManager", fake_douyin
        ):
            self._builder().fetch_product_list_text(
                "douyin_示例店铺", "douyin_13800138000", channel_type
            )

    def test_douyin_prefetch_uses_douyin_manager(self):
        hits = []
        self._run("douyin", hits)
        self.assertEqual([name for name, _ in hits], ["douyin"])
        self.assertEqual(hits[0][1]["shop_id"], "douyin_示例店铺")

    def test_pinduoduo_prefetch_uses_pinduoduo_manager(self):
        hits = []
        self._run("pinduoduo", hits)
        self.assertEqual([name for name, _ in hits], ["pinduoduo"])

    def test_missing_channel_keeps_legacy_pinduoduo_default(self):
        """空渠道保持历史行为，与 normalize_channel 的默认一致。"""
        hits = []
        self._run("", hits)
        self.assertEqual([name for name, _ in hits], ["pinduoduo"])


if __name__ == "__main__":
    unittest.main()
