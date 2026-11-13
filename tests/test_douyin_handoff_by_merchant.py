# -*- coding: utf-8 -*-
"""抖音商家侧消息处理回归测试（对齐拼多多）。

商家侧消息（人工客服、系统提示、店铺机器人）不回复、不转人工。
"""
import json

from Channel.douyin import message_rules
from Channel.douyin.core.douyin_lifecycle import LifecycleMixin
from bridge.context import ChannelType, make_handoff_key


class _FakeConvList:
    """最小化会话列表桩：只返回预设的 payload。"""

    def __init__(self, payload):
        self.payload = payload

    def fetch_conversations(self):
        return self.payload


def _conv(conv_id, msg_list, shop_id="13800138000"):
    return {
        "parentConShortId": conv_id,
        "subConversationId": "7706540248934679808",
        "coreInfoMap": {"service_entity_id": shop_id},
        "msgList": msg_list,
    }


def _text_body(content, ext, create_time=1794325630058):
    return {
        "messageBody": {
            "msgType": 1000,
            "content": content,
            "createTime": create_time,
            "ext": ext,
        }
    }


def _merchant_ext(text_src="pc", biz_info='{"biz_role_type":"cs"}'):
    ext = {
        "s:sender_biz_role": "CurrentServer",
        "sender_role": "2",
        "type": "text",
        "message_logid": "logid-merchant-1",
        "security_receiver_id": "buyer-uid-1",
        "src": text_src,
        "shop_id": "13800138000",
    }
    if biz_info is not None:
        ext["biz_sender_info"] = biz_info
    return ext


def test_fetch_all_separates_merchant_from_buyer():
    from tools.reverse_probe.douyin.pigeon_conv_list import PigeonConvList

    client = PigeonConvList.__new__(PigeonConvList)
    client.session_path = ""
    client.last_status = None
    client.last_error = None
    client._conn = None
    client.fetch_conversations = lambda: [
        _conv(
            "conv-1",
            [
                _text_body(
                    "在吗",
                    {
                        "s:sender_biz_role": "Buyer",
                        "sender_role": "1",
                        "type": "text",
                        "security_sender_id": "buyer-uid-1",
                        "message_logid": "logid-buyer-1",
                    },
                    1794325631058,
                ),
                _text_body("您好，我是客服", _merchant_ext(), 1794325632058),
            ],
        )
    ]
    data = client.fetch_all(0)
    assert len(data["buyer"]) == 1
    assert data["buyer"][0]["text"] == "在吗"
    assert len(data["merchant"]) == 1
    assert data["merchant"][0]["text"] == "您好，我是客服"
    assert data["merchant"][0]["sender_kind"] == "human"
    assert data["merchant"][0]["conv_id"] == "conv-1"


def test_lifecycle_has_no_handoff_by_merchant_any_more():
    """商家侧消息不得再触发转人工（与拼多多一致）。"""
    assert not hasattr(LifecycleMixin, "_mark_handoff_by_merchant")


def test_system_close_notice_is_not_agent_message():
    """【系统关闭会话】是 role=4 的系统提示，不进商家侧分支，避免启动就刷日志。"""
    notice = {
        "type": 1005,
        "sender_biz_role": "Shop",
        "sender_role": "4",
        "client_message_id": "close_user_msg_7707676989560272143",
        "text": "【系统关闭会话】",
    }
    assert message_rules.is_agent_message(notice) is False


def test_handoff_key_matches_ai_handler_scope():
    """ai_handler 用 make_handoff_key(channel, shop_id, from_uid) 检查，
    from_uid 就是 conv_id。这里固化这个约定，防止以后改成买家 uid。"""
    from bridge.context import make_handoff_key as build

    expected = build("douyin", "shop-1", "conv-9")
    assert expected == "douyin|shop-1|conv-9"
    assert build(ChannelType.DOUYIN, "shop-1", "conv-9") == expected
    assert make_handoff_key(ChannelType.DOUYIN, "shop-1", "conv-9") == expected


def test_classify_merchant_sender():
    assert message_rules.classify_merchant_sender(_merchant_ext()) == "human"
    assert message_rules.classify_merchant_sender(
        {"biz_sender_info": json.dumps({"biz_role_type": "robot"})}
    ) == "robot"
    assert message_rules.classify_merchant_sender({}) == "unknown"
    assert message_rules.classify_merchant_sender(
        {"biz_sender_info": "not-json"}
    ) == "unknown"
