"""抖音 send_image 图片发送协议测试。

覆盖：三级降级策略（pigeon→bridge→degraded）、渠道隔离、签名对齐。
全部 mock，不依赖真实网络。
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from Channel.douyin.utils.API.send_message import SendMessage


# ── fixtures ──────────────────────────────────────────────

@pytest.fixture
def sm():
    """构造 SendMessage 实例，mock 掉 BaseRequest 的 DB 初始化。"""
    with patch.object(SendMessage, "__init__", lambda self, *a, **kw: None):
        instance = SendMessage.__new__(SendMessage)
        instance.logger = MagicMock()
        instance.shop_id = "test_shop"
        instance.user_id = "test_user"
        instance.channel_name = "douyin"
        return instance


# ── 降级路径测试（8个） ───────────────────────────────────

class TestDegradedPath:
    """当前实际路径：pigeon 和 bridge 均无 send_image 方法 → 降级返回 dict。"""

    def test_degraded_returns_dict_not_none(self, sm):
        """降级路径必须返回 dict，绝不返回 None。"""
        result = sm.send_image("buyer_123", "https://example.com/img.jpg")
        assert result is not None
        assert isinstance(result, dict)

    def test_degraded_success_false(self, sm):
        """降级路径 success=False。"""
        result = sm.send_image("buyer_123", "https://example.com/img.jpg")
        assert result["success"] is False

    def test_degraded_transport_label(self, sm):
        """降级路径 transport='degraded'。"""
        result = sm.send_image("buyer_123", "https://example.com/img.jpg")
        assert result["transport"] == "degraded"

    def test_degraded_reason(self, sm):
        """降级路径 reason 明确标注协议未逆向。"""
        result = sm.send_image("buyer_123", "https://example.com/img.jpg")
        assert "image_protocol_not_reverse_engineered" in result["reason"]

    def test_degraded_contains_recipient(self, sm):
        """降级返回包含 recipient_uid。"""
        result = sm.send_image("buyer_123", "https://example.com/img.jpg")
        assert result["recipient_uid"] == "buyer_123"

    def test_degraded_contains_image_url(self, sm):
        """降级返回包含 image_url。"""
        url = "https://example.com/img.jpg"
        result = sm.send_image("buyer_123", url)
        assert result["image_url"] == url

    def test_degraded_has_detail(self, sm):
        """降级返回有 detail 说明。"""
        result = sm.send_image("buyer_123", "https://example.com/img.jpg")
        assert "detail" in result
        assert len(result["detail"]) > 0

    def test_degraded_logs_warning(self, sm):
        """降级路径记录 WARNING 日志。"""
        sm.send_image("buyer_123", "https://example.com/img.jpg")
        sm.logger.warning.assert_called_once()


# ── pigeon transport 路径测试（6个） ──────────────────────

class TestPigeonPath:
    """未来 PigeonTransport 添加 send_image 后自动启用。"""

    def _mock_pigeon(self, has_method=True, return_value=None, instance=None):
        """mock PigeonTransport.instance()。"""
        mock_pt = MagicMock()
        if has_method:
            mock_pt.send_image = MagicMock(return_value=return_value)
        else:
            del mock_pt.send_image
        mock_pt.instance = MagicMock(return_value=instance if instance is not None else mock_pt)
        return mock_pt

    def test_pigeon_path_used_when_available(self, sm):
        """PigeonTransport 有 send_image 时走 pigeon 路径。"""
        mock_pt = self._mock_pigeon(return_value={"ok": True})
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt):
            result = sm.send_image("buyer_1", "http://img.jpg")
        assert result["success"] is True
        assert result["transport"] == "pigeon"
        mock_pt.send_image.assert_called_once_with("buyer_1", "http://img.jpg", width=0, height=0)

    def test_pigeon_path_passes_width_height(self, sm):
        """pigeon 路径透传 width/height。"""
        mock_pt = self._mock_pigeon(return_value={"ok": True})
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt):
            sm.send_image("buyer_1", "http://img.jpg", width=800, height=600)
        mock_pt.send_image.assert_called_once_with("buyer_1", "http://img.jpg", width=800, height=600)

    def test_pigeon_returns_none_falls_to_bridge(self, sm):
        """pigeon send_image 返回 None 时继续尝试 bridge 路径。"""
        mock_pt = self._mock_pigeon(return_value=None)
        mock_bridge = MagicMock()
        mock_bridge.send_image = MagicMock(return_value={"bridge_ok": True})
        mock_bridge.instance = MagicMock(return_value=mock_bridge)
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt), \
             patch("Channel.douyin.feige_bridge.FeigeBridge", mock_bridge):
            result = sm.send_image("buyer_1", "http://img.jpg")
        assert result["transport"] == "bridge"

    def test_pigeon_no_method_falls_to_bridge(self, sm):
        """PigeonTransport 无 send_image 方法时跳过到 bridge。"""
        mock_pt = MagicMock(spec=["instance"])
        mock_pt.instance = MagicMock(return_value=mock_pt)
        # 不设置 send_image 属性
        mock_bridge = MagicMock()
        mock_bridge.send_image = MagicMock(return_value={"bridge_ok": True})
        mock_bridge.instance = MagicMock(return_value=mock_bridge)
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt), \
             patch("Channel.douyin.feige_bridge.FeigeBridge", mock_bridge):
            result = sm.send_image("buyer_1", "http://img.jpg")
        assert result["transport"] == "bridge"

    def test_pigeon_instance_none_falls_to_bridge(self, sm):
        """PigeonTransport.instance() 返回 None 时跳过到 bridge。"""
        mock_pt = MagicMock()
        mock_pt.instance = MagicMock(return_value=None)
        mock_bridge = MagicMock()
        mock_bridge.send_image = MagicMock(return_value={"bridge_ok": True})
        mock_bridge.instance = MagicMock(return_value=mock_bridge)
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt), \
             patch("Channel.douyin.feige_bridge.FeigeBridge", mock_bridge):
            result = sm.send_image("buyer_1", "http://img.jpg")
        assert result["transport"] == "bridge"

    def test_pigeon_exception_falls_to_bridge(self, sm):
        """pigeon 路径抛异常时优雅降级到 bridge。"""
        mock_pt = MagicMock()
        mock_pt.instance.side_effect = RuntimeError("transport error")
        mock_bridge = MagicMock()
        mock_bridge.send_image = MagicMock(return_value={"bridge_ok": True})
        mock_bridge.instance = MagicMock(return_value=mock_bridge)
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt), \
             patch("Channel.douyin.feige_bridge.FeigeBridge", mock_bridge):
            result = sm.send_image("buyer_1", "http://img.jpg")
        assert result["transport"] == "bridge"


# ── FeigeBridge 路径测试（2个） ───────────────────────────

class TestBridgePath:
    """FeigeBridge 有 send_image 时走 bridge 路径。"""

    def test_bridge_path_used_when_pigeon_unavailable(self, sm):
        """pigeon 不可用、bridge 有 send_image 时走 bridge。"""
        mock_pt = MagicMock()
        mock_pt.instance = MagicMock(return_value=None)
        mock_bridge = MagicMock()
        mock_bridge.send_image = MagicMock(return_value={"bridge_ok": True})
        mock_bridge.instance = MagicMock(return_value=mock_bridge)
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt), \
             patch("Channel.douyin.feige_bridge.FeigeBridge", mock_bridge):
            result = sm.send_image("buyer_1", "http://img.jpg")
        assert result["success"] is True
        assert result["transport"] == "bridge"
        mock_bridge.send_image.assert_called_once()

    def test_bridge_returns_none_falls_to_degraded(self, sm):
        """bridge send_image 返回 None 时降级到 degraded。"""
        mock_pt = MagicMock()
        mock_pt.instance = MagicMock(return_value=None)
        mock_bridge = MagicMock()
        mock_bridge.send_image = MagicMock(return_value=None)
        mock_bridge.instance = MagicMock(return_value=mock_bridge)
        with patch("tools.reverse_probe.douyin.pigeon_transport.PigeonTransport", mock_pt), \
             patch("Channel.douyin.feige_bridge.FeigeBridge", mock_bridge):
            result = sm.send_image("buyer_1", "http://img.jpg")
        assert result["transport"] == "degraded"


# ── 渠道隔离测试（3个） ───────────────────────────────────

class TestChannelIsolation:
    """抖音 send_image 不调用拼多多 API。"""

    def test_no_pinduoduo_import_in_send_image(self, sm):
        """send_image 执行路径中不导入拼多多模块。"""
        import sys
        pdd_modules = [m for m in sys.modules if "pinduoduo" in m.lower()]
        before = set(pdd_modules)
        sm.send_image("buyer_1", "http://img.jpg")
        after = set(m for m in sys.modules if "pinduoduo" in m.lower())
        # 不新增拼多多模块导入
        new_pdd = after - before
        assert len(new_pdd) == 0, f"send_image 引入了拼多多模块: {new_pdd}"

    def test_degraded_no_http_call(self, sm):
        """降级路径不发起任何 HTTP 请求。"""
        with patch.object(sm, "post") as mock_post, \
             patch.object(sm, "get") as mock_get:
            sm.send_image("buyer_1", "http://img.jpg")
        mock_post.assert_not_called()
        mock_get.assert_not_called()

    def test_uses_douyin_base_request_only(self, sm):
        """SendMessage 继承自抖音 BaseRequest，不继承拼多多。"""
        from Channel.douyin.utils.base_request import BaseRequest as DouyinBase
        assert isinstance(sm, DouyinBase)
        from Channel.pinduoduo.utils.base_request import BaseRequest as PddBase
        assert not isinstance(sm, PddBase)


# ── 签名对齐测试（3个） ───────────────────────────────────

class TestSignatureAlignment:
    """与拼多多 send_image 签名对齐。"""

    def test_accepts_recipient_and_image_url(self, sm):
        """基础签名：recipient_uid + image_url。"""
        result = sm.send_image("buyer_1", "http://img.jpg")
        assert result is not None

    def test_accepts_optional_width_height(self, sm):
        """扩展签名：可选 width/height（有默认值，兼容拼多多）。"""
        result = sm.send_image("buyer_1", "http://img.jpg", 800, 600)
        assert result is not None

    def test_class_constants_for_future_reverse(self, sm):
        """类常量标注待逆向端点和消息类型。"""
        assert hasattr(SendMessage, "IMAGE_UPLOAD_ENDPOINT")
        assert hasattr(SendMessage, "PIGEON_IMAGE_MSG_TYPE")
        # 当前为 None（待逆向）
        assert SendMessage.IMAGE_UPLOAD_ENDPOINT is None
        assert SendMessage.PIGEON_IMAGE_MSG_TYPE is None
