# -*- coding: utf-8 -*-
"""抖音消息来源判定回归：买家 / 商家 / 平台机器人 三类必须分清。

真机样本取自 2026-11-10 22:50 的会话列表接口返回。
"""
from Channel.douyin.message_rules import Origin, is_customer_message, resolve_origin


def test_buyer_text_message_is_customer():
    ext = {
        "s:sender_biz_role": "Buyer",
        "sender_role": "1",
        "type": "text",
        "security_sender_id": "AQBXzhntHVYCI36WXL8r4WY6IjTDs2oYuYp5",
    }
    assert resolve_origin(ext) is Origin.CUSTOMER
    assert is_customer_message(ext) is True


def test_platform_robot_notice_is_system():
    """「当前会话已长时间未回复，平台可能主动介入处理」这类提示不能当买家消息。"""
    ext = {
        "biz_sender": "robot",
        "im_sender_role": "4",
        "s:sender_biz_role": "Shop",
        "sender_role": "4",
        "systemHint": "1",
        "type": "template_card",
        "visibility_type": "2",
    }
    assert resolve_origin(ext) is Origin.SYSTEM
    assert is_customer_message(ext) is False


def test_system_hint_without_robot_flag_is_system():
    assert resolve_origin({"systemHint": "1", "type": "text"}) is Origin.SYSTEM


def test_welcome_message_from_cs_is_merchant():
    ext = {
        "s:sender_biz_role": "CurrentServer",
        "im_sender_role": "2",
        "type": "text",
    }
    assert resolve_origin(ext) is Origin.MERCHANT


def test_allocated_service_notice_is_system():
    """"人工客服 XX 为您服务"是平台下发的接入提示，不是买家发言。"""
    ext = {
        "s:sender_biz_role": "Shop",
        "sender_role": "3",
        "type": "allocated_service",
    }
    assert resolve_origin(ext) is Origin.SYSTEM


def test_missing_role_is_unknown():
    assert resolve_origin({}) is Origin.UNKNOWN
    assert resolve_origin(None) is Origin.UNKNOWN
    assert is_customer_message({"type": "text"}) is False


# ---------------------------------------------------------------------------
# 平台自动接待消息 vs 真人打字（2026-11-11 01:12:58 真机样本）
#
# 买家一发消息，平台会自动转人工并推三条消息，角色字段全是
# CurrentServer / role=2 / biz_role_type=cs，和真人打字完全一样。
# 只要把其中任意一条当成人工介入，程序就会立刻静默，AI 一句话发不出去。
# ---------------------------------------------------------------------------
from Channel.douyin.message_rules import is_agent_message, is_platform_auto_message


def _merchant_ext(client_message_id, msg_type="text"):
    return {
        "sender_biz_role": "CurrentServer",
        "sender_role": "2",
        "biz_sender_info": '{"biz_role_type":"cs"}',
        "client_message_id": client_message_id,
        "msg_type": msg_type,
    }


def test_allocated_service_notice_is_not_agent():
    ext = {
        "sender_biz_role": "Shop",
        "sender_role": "3",
        "client_message_id": "7706590307919381821_ServiceForYou_",
        "msg_type": "allocated_service",
    }
    assert is_agent_message(ext) is False


def test_platform_transfer_welcome_is_not_agent():
    ext = _merchant_ext("7706590307919381821_Welcome_")
    assert is_platform_auto_message(ext) is True
    assert is_agent_message(ext) is False


def test_conversation_action_message_is_not_agent():
    ext = _merchant_ext("e2814707-625c-4dff-809d-11af896b72bb", "customer_enter_from_close")
    assert is_platform_auto_message(ext) is True
    assert is_agent_message(ext) is False


def test_platform_llm_message_is_not_agent():
    ext = {
        "sender_biz_role": "Shop",
        "sender_role": "4",
        "biz_sender_info": '{"biz_role_type":"llm"}',
        "client_message_id": "llm_process_text_2_abc",
        "msg_type": "text",
    }
    assert is_agent_message(ext) is False


def test_real_human_agent_text_is_agent():
    ext = _merchant_ext("a081a8ea-1111-2222-3333-444455556666")
    assert is_platform_auto_message(ext) is False
    assert is_agent_message(ext) is True
