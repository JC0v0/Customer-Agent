"""抖音 send_mallGoodsCard 商品卡发送协议测试。

覆盖：三级降级策略（pigeon→bridge→degraded）、渠道隔离、签名对齐。
全部 mock，不依赖真实网络。
"""
import pytest
from unittest.mock import patch, MagicMock

from Channel.douyin.utils.API.send_message import SendMessage


@pytest.fixture
def sm():
    with patch.object(SendMessage, "__init__", lambda self, *a, **kw: None):
        instance = SendMessage.__new__(SendMessage)
        instance.logger = MagicMock()
        instance.shop_id = "test_shop"
        instance.user_id = "test_user"
        instance.channel_name = "douyin"
        return instance


# ── 降级路径测试 ─────────────────────────────────────────

class TestDegradedPath:
    def test_degraded_returns_dict_not_none(self, sm):
        result = sm.send_mallGoodsCard("buyer_123", 1001)
        assert result is not None
        assert isinstance(result, dict)

    def test_degraded_success_false(self, sm):
        result = sm.send_mallGoodsCard("buyer_123", 1001)
        assert result["success"] is False

    def test_degraded_transport_label(self, sm):
        result = sm.send_mallGoodsCard("buyer_123", 1001)
        assert result["transport"] == "degraded"

    def test_degraded_reason(self, sm):
        result = sm.send_mallGoodsCard("buyer_123", 1001)
        assert "mall_goods_card_protocol_not_reverse_engineered" in result["reason"]

    def test_degraded_contains_fields(self, sm):
        result = sm.send_mallGoodsCard("buyer_123", 1001, biz_type=2)
        assert result["recipient_uid"] == "buyer_123"
        assert result["goods_id"] == 1001
        assert result["biz_type"] == 2

    def test_degraded_has_detail(self, sm):
        result = sm.send_mallGoodsCard("buyer_123", 1001)
        assert "detail" in result
        assert len(result["detail"]) > 0

    def test_degraded_logs_warning(self, sm):
        sm.send_mallGoodsCard("buyer_123", 1001)
        sm.logger.warning.assert_called_once()


# ── pigeon transport 路径测试 ────────────────────────────

class TestPigeonPath:
    def _mock_pigeon(self, has_method=True, return_value=None):
        mock_pt = MagicMock()
        if has_method:
            mock_pt.send_mallGoodsCard = MagicMock(return_value=return_value)
        else:
            del mock_pt.send_mallGoodsCard
        mock_pt.instance = MagicMock(return_value=mock_pt)
        return mock_pt

    def test_pigeon_path_used_when_available(self, sm):
        mock_pt = self._mock_pigeon(return_value={"ok": True})
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt):
            result = sm.send_mallGoodsCard("buyer_1", 2002, biz_type=2)
        assert result["success"] is True
        assert result["transport"] == "pigeon"
        mock_pt.send_mallGoodsCard.assert_called_once_with("buyer_1", 2002, biz_type=2)

    def test_pigeon_returns_none_falls_to_degraded(self, sm):
        mock_pt = self._mock_pigeon(return_value=None)
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt):
            result = sm.send_mallGoodsCard("buyer_1", 2002)
        assert result["transport"] == "degraded"

    def test_pigeon_no_method_falls_to_degraded(self, sm):
        mock_pt = MagicMock(spec=["instance"])
        mock_pt.instance = MagicMock(return_value=mock_pt)
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt):
            result = sm.send_mallGoodsCard("buyer_1", 2002)
        assert result["transport"] == "degraded"

    def test_pigeon_exception_falls_to_degraded(self, sm):
        mock_pt = MagicMock()
        mock_pt.instance.side_effect = RuntimeError("transport error")
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt):
            result = sm.send_mallGoodsCard("buyer_1", 2002)
        assert result["transport"] == "degraded"


# ── FeigeBridge 路径测试 ─────────────────────────────────

class TestBridgePath:
    def test_bridge_path_used_when_pigeon_unavailable(self, sm):
        mock_pt = MagicMock()
        mock_pt.instance = MagicMock(return_value=None)
        mock_bridge = MagicMock()
        mock_bridge.send_mallGoodsCard = MagicMock(return_value={"bridge_ok": True})
        mock_bridge.instance = MagicMock(return_value=mock_bridge)
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt), \
             patch("Channel.douyin.feige_bridge.FeigeBridge", mock_bridge):
            result = sm.send_mallGoodsCard("buyer_1", 2002)
        assert result["success"] is True
        assert result["transport"] == "bridge"

    def test_bridge_returns_none_falls_to_degraded(self, sm):
        mock_pt = MagicMock()
        mock_pt.instance = MagicMock(return_value=None)
        mock_bridge = MagicMock()
        mock_bridge.send_mallGoodsCard = MagicMock(return_value=None)
        mock_bridge.instance = MagicMock(return_value=mock_bridge)
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt), \
             patch("Channel.douyin.feige_bridge.FeigeBridge", mock_bridge):
            result = sm.send_mallGoodsCard("buyer_1", 2002)
        assert result["transport"] == "degraded"


# ── 渠道隔离测试 ─────────────────────────────────────────

class TestChannelIsolation:
    def test_degraded_no_http_call(self, sm):
        with patch.object(sm, "post") as mock_post, \
             patch.object(sm, "get") as mock_get:
            sm.send_mallGoodsCard("buyer_1", 2002)
        mock_post.assert_not_called()
        mock_get.assert_not_called()

    def test_uses_douyin_base_request_only(self, sm):
        from Channel.douyin.utils.base_request import BaseRequest as DouyinBase
        assert isinstance(sm, DouyinBase)


# ── 签名对齐测试 ─────────────────────────────────────────

class TestSignatureAlignment:
    def test_accepts_recipient_goods_id_biz_type(self, sm):
        result = sm.send_mallGoodsCard("buyer_1", 1001, biz_type=2)
        assert result is not None

    def test_biz_type_default(self, sm):
        result = sm.send_mallGoodsCard("buyer_1", 1001)
        assert result["biz_type"] == 2

    def test_class_constants_for_future_reverse(self, sm):
        assert hasattr(SendMessage, "MALL_GOODS_CARD_ENDPOINT")
        assert hasattr(SendMessage, "PIGEON_GOODS_CARD_MSG_TYPE")
        assert SendMessage.MALL_GOODS_CARD_ENDPOINT is None
        assert SendMessage.PIGEON_GOODS_CARD_MSG_TYPE is None
