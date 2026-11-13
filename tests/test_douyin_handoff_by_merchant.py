# -*- coding: utf-8 -*-
"""商家侧消息 → 静默转人工 回归测试。

对齐拼多多：人工客服（或客服工作台）在会话里发出任何消息，就静默标记该会话
转人工，AI 不再回复、也不通知。抖音没有 from_uid 白名单，只能靠「本程序发出
记录」区分 AI 自己的回声，这里是那条边界的守门测试。
"""
import json

import pytest

import core.session_state as session_state_module
from Channel.douyin import message_rules
from Channel.douyin.core.douyin_lifecycle import LifecycleMixin
from bridge.context import ChannelType, make_handoff_key
from core.session_state import SessionState


@pytest.fixture(autouse=True)
def _fixed_handoff_ttl(monkeypatch):
    """本机 config.json 的 handoff.valid_hours 是 0（标记立即过期）。

    转人工时长属于部署配置，测试不该被它拖累，这里固定成 4 小时。
    """
    original = session_state_module.get_config

    def patched(key, default=None):
        if key == "handoff.valid_hours":
            return 4
        return original(key, default)

    monkeypatch.setattr(session_state_module, "get_config", patched)


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


class _RecordingLogger:
    """LifecycleMixin 的 logger 由具体渠道类注入，测试里用记录桩替代。"""

    def __init__(self):
        self.records = []

    def _push(self, level, message):
        self.records.append((level, str(message)))

    def debug(self, message):
        self._push("debug", message)

    def info(self, message):
        self._push("info", message)

    def warning(self, message):
        self._push("warning", message)

    def error(self, message):
        self._push("error", message)

    def messages(self, level):
        out = []
        for item_level, item_message in self.records:
            if item_level == level:
                out.append(item_message)
        return out


def _build_loop():
    """直接用 LifecycleMixin 的实例，避免拉起整条连接。"""
    loop = LifecycleMixin()
    loop.logger = _RecordingLogger()
    return loop


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


def test_merchant_message_marks_handoff_with_conv_id():
    """handoff key 的 uid 维度必须是 conv_id，否则 ai_handler 查不到。"""
    loop = _build_loop()
    shop_id = "13800138000"
    session_key = make_handoff_key(ChannelType.DOUYIN, shop_id, "conv-1")
    SessionState().clear_handoff(session_key)

    item = {
        "conv_id": "conv-1",
        "text": "您好，我是客服",
        "sender_kind": "human",
        "src": "pc",
    }
    loop._mark_handoff_by_merchant(item, shop_id, "13800138000")

    assert SessionState().is_handoff(session_key) is True
    SessionState().clear_handoff(session_key)


def test_self_sent_echo_does_not_mark_handoff():
    """AI 自己发出的回复回流成商家侧消息时，不能把自己判定为人工介入。"""
    loop = _build_loop()
    shop_id = "13800138000"
    conv_id = "conv-self"
    session_key = make_handoff_key(ChannelType.DOUYIN, shop_id, conv_id)
    SessionState().clear_handoff(session_key)

    message_rules.record_self_sent(shop_id, conv_id, "亲，这件是纯棉的")
    item = {
        "conv_id": conv_id,
        "text": "亲，这件是纯棉的",
        "sender_kind": "human",
        "src": "pc",
    }
    loop._mark_handoff_by_merchant(item, shop_id, "13800138000")
    assert SessionState().is_handoff(session_key) is False

    # 指纹消费掉之后，客服再手工发同一句话，应当算人工介入。
    loop._mark_handoff_by_merchant(item, shop_id, "13800138000")
    assert SessionState().is_handoff(session_key) is True
    SessionState().clear_handoff(session_key)


def test_robot_message_does_not_mark_handoff():
    loop = _build_loop()
    shop_id = "13800138000"
    conv_id = "conv-robot"
    session_key = make_handoff_key(ChannelType.DOUYIN, shop_id, conv_id)
    SessionState().clear_handoff(session_key)

    item = {
        "conv_id": conv_id,
        "text": "欢迎光临本店",
        "sender_kind": "robot",
        "src": "pc",
    }
    loop._mark_handoff_by_merchant(item, shop_id, "13800138000")
    assert SessionState().is_handoff(session_key) is False


def test_handoff_key_matches_ai_handler_scope():
    """ai_handler 用 make_handoff_key(channel, shop_id, from_uid) 检查，
    from_uid 就是 conv_id。这里固化这个约定，防止以后改成买家 uid。"""
    from bridge.context import make_handoff_key as build

    expected = build("douyin", "shop-1", "conv-9")
    assert expected == "douyin|shop-1|conv-9"
    assert build(ChannelType.DOUYIN, "shop-1", "conv-9") == expected


def test_classify_merchant_sender():
    assert message_rules.classify_merchant_sender(_merchant_ext()) == "human"
    assert message_rules.classify_merchant_sender(
        {"biz_sender_info": json.dumps({"biz_role_type": "robot"})}
    ) == "robot"
    assert message_rules.classify_merchant_sender({}) == "unknown"
    assert message_rules.classify_merchant_sender(
        {"biz_sender_info": "not-json"}
    ) == "unknown"
