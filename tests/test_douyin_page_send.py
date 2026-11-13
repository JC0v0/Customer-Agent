"""抖音文本发送：页面通道（点会话 → 填输入框 → 点发送）测试。

覆盖：页面优先、昵称解析、桥未就绪时的降级、账号 profile 定位。
全部 mock，不启动浏览器、不访问网络。
"""
from unittest.mock import MagicMock, patch

import pytest

from Channel.douyin.utils.API.send_message import SendMessage


@pytest.fixture
def sm():
    with patch.object(SendMessage, "__init__", lambda self, *a, **kw: None):
        instance = SendMessage.__new__(SendMessage)
        instance.logger = MagicMock()
        instance.shop_id = "100000001"
        instance.user_id = "13800138000"
        instance.account_name = "13800138000"
        instance.channel_name = "douyin"
        return instance


def _convs(parent, sub, nick):
    ext = {"nickname": nick} if nick else {}
    return {
        "parentConShortId": parent,
        "subConversationId": sub,
        "msgList": [{"messageBody": {"ext": ext}}],
    }


class TestBuyerNickname:
    def test_maps_conv_id_to_nickname(self, sm):
        convs = [_convs("111", "222", "买家甲"), _convs("333", "444", "买家乙")]
        with patch(
            "Channel.douyin.session_paths.existing_pigeon_session_path"
        ) as path_mock, patch(
            "tools.reverse_probe.douyin.pigeon_conv_list.PigeonConvList"
        ) as list_mock, patch(
            "Channel.douyin.utils.API.order_query.extract_buyer_nick_from_ext",
            side_effect=lambda ext: (ext or {}).get("nickname", ""),
        ):
            path_mock.return_value = "x.json"
            list_mock.return_value.fetch_conversations.return_value = convs
            assert sm._buyer_nickname("333") == "买家乙"

    def test_passes_username_when_resolving_session_file(self, sm):
        """会话文件按登录用户名落盘：只传 shop_id/user_id 会拼出不存在的路径。"""
        with patch(
            "Channel.douyin.session_paths.existing_pigeon_session_path"
        ) as path_mock, patch(
            "tools.reverse_probe.douyin.pigeon_conv_list.PigeonConvList"
        ) as list_mock:
            path_mock.return_value = "x.json"
            list_mock.return_value.fetch_conversations.return_value = []
            sm._buyer_nickname("333")
            args = path_mock.call_args[0]
            assert args[0] == "100000001"
            assert args[1] == "13800138000"
            assert args[2] == "13800138000"

    def test_returns_empty_when_conv_missing(self, sm):
        with patch(
            "Channel.douyin.session_paths.existing_pigeon_session_path"
        ) as path_mock, patch(
            "tools.reverse_probe.douyin.pigeon_conv_list.PigeonConvList"
        ) as list_mock:
            path_mock.return_value = "x.json"
            list_mock.return_value.fetch_conversations.return_value = [
                _convs("111", "222", "买家甲")
            ]
            assert sm._buyer_nickname("999") == ""


class TestSendByPage:
    def test_skips_when_bridge_not_ready(self, sm):
        with patch("Channel.douyin.feige_bridge.FeigeBridge") as bridge_cls:
            bridge_cls.instance.return_value.is_send_ready = False
            bridge_cls.instance.return_value.is_running = False
            assert sm._send_by_page("111", "hi") is False

    def test_waits_until_page_ready(self, sm):
        """浏览器还在加载时等它就绪，而不是直接放弃去走必然失败的 pigeon。"""

        class _Bridge:
            is_running = True
            send_text = MagicMock(return_value=True)

            def __init__(self):
                self._checks = 0

            @property
            def is_send_ready(self):
                self._checks += 1
                return self._checks > 2

        bridge = _Bridge()
        with patch("Channel.douyin.feige_bridge.FeigeBridge") as bridge_cls, patch.object(
            SendMessage, "_buyer_nickname", return_value="买家"
        ), patch("Channel.douyin.message_rules.record_self_sent"):
            bridge_cls.instance.return_value = bridge
            assert sm._send_by_page("333", "你好", wait_seconds=10) is True
        bridge.send_text.assert_called_once_with("买家", "你好")

    def test_sends_with_resolved_nickname(self, sm):
        bridge = MagicMock()
        bridge.is_send_ready = True
        bridge.send_text.return_value = True
        with patch("Channel.douyin.feige_bridge.FeigeBridge") as bridge_cls, patch.object(
            SendMessage, "_buyer_nickname", return_value="买家乙"
        ), patch("Channel.douyin.message_rules.record_self_sent") as recall:
            bridge_cls.instance.return_value = bridge
            assert sm._send_by_page("333", "你好") is True
            bridge.send_text.assert_called_once_with("买家乙", "你好")
            recall.assert_called_once_with("100000001", "333", "你好")

    def test_fails_when_nickname_missing(self, sm):
        bridge = MagicMock()
        bridge.is_send_ready = True
        with patch("Channel.douyin.feige_bridge.FeigeBridge") as bridge_cls, patch.object(
            SendMessage, "_buyer_nickname", return_value=""
        ):
            bridge_cls.instance.return_value = bridge
            assert sm._send_by_page("333", "你好") is False
            bridge.send_text.assert_not_called()


class TestSendTextOrder:
    def test_protocol_success_skips_page(self, sm):
        with patch(
            "tools.reverse_probe.douyin.pigeon_transport.PigeonTransport"
        ) as transport_cls, patch.object(SendMessage, "_send_by_page") as page_send, patch(
            "Channel.douyin.message_rules.record_self_sent"
        ):
            transport_cls.instance.return_value.send_text.return_value = True
            result = sm.send_text("333", "你好")
        assert result == {
            "success": True,
            "recipient_uid": "333",
            "transport": "pigeon",
        }
        page_send.assert_not_called()

    def test_falls_back_to_page_when_protocol_fails(self, sm):
        with patch(
            "tools.reverse_probe.douyin.pigeon_transport.PigeonTransport"
        ) as transport_cls, patch.object(
            SendMessage, "_send_by_page", return_value=True
        ) as page_send, patch(
            "Channel.douyin.message_rules.record_self_sent"
        ), patch(
            "Channel.douyin.message_rules.clear_self_sent"
        ):
            transport = transport_cls.instance.return_value
            transport.send_text.return_value = False
            transport.last_send_result = {"reason": "text_no_echo"}
            result = sm.send_text("333", "你好")
        assert result == {
            "success": True,
            "recipient_uid": "333",
            "transport": "page",
        }
        page_send.assert_called_once_with("333", "你好")


class TestAccountProfile:
    def test_finds_existing_profile_dir(self, tmp_path):
        import hashlib
        import re

        from Channel.douyin.feige_bridge import account_profile

        raw = "douyin:100000001:13800138000"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")[:48]
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        target = tmp_path / "user_data" / ("%s-%s" % (safe, digest))
        target.mkdir(parents=True)

        with patch("utils.runtime_path.get_data_path", return_value=tmp_path):
            found = account_profile("100000001", "13800138000", "13800138000")
        assert found == str(target)

    def test_returns_none_without_profile(self, tmp_path):
        from Channel.douyin.feige_bridge import account_profile

        (tmp_path / "user_data").mkdir()
        with patch("utils.runtime_path.get_data_path", return_value=tmp_path):
            assert account_profile("1", "2", "3") is None


class TestSenderProfile:
    """页面发送必须用独立 profile：与登录 profile 共用会被 Edge 单实例锁挡住。"""

    def test_sender_profile_has_page_suffix(self, tmp_path):
        import os

        from Channel.douyin.feige_bridge import account_sender_profile

        with patch("utils.runtime_path.get_data_path", return_value=tmp_path):
            path = account_sender_profile("100000001", "13800138000", "13800138000")
        assert path.endswith("-page")
        assert os.path.isdir(path)

    def test_sender_profile_differs_from_login_profile(self, tmp_path):
        import hashlib
        import re

        from Channel.douyin.feige_bridge import account_profile, account_sender_profile

        raw = "douyin:100000001:13800138000"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")[:48]
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        login = tmp_path / "user_data" / ("%s-%s" % (safe, digest))
        login.mkdir(parents=True)

        with patch("utils.runtime_path.get_data_path", return_value=tmp_path):
            assert account_profile("100000001", "13800138000", "13800138000") == str(login)
            sender = account_sender_profile("100000001", "13800138000", "13800138000")
        assert sender != str(login)
