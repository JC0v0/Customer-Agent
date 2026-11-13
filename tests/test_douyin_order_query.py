"""
抖音 OrderQuery + query_order_status 工具测试

覆盖（对齐 2026-11-12 逆向后的 searchlist 真实 API 实现）：
1. OrderQuery.get_recent_orders：消息历史订单号 → 真实接口补详情 /
   无订单号返回空 / 店铺未登记失败
2. get_order_detail / get_buyer_orders（searchlist 真实 API 路径）
3. query_order_status 工具抖音路径：有订单→格式化返回；无订单→引导话术；
   不调拼多多 OrderQuery/浏览器
4. 全部 mock，不依赖真实 DB/网络
"""
import unittest
from unittest import mock

# 统一 mock BaseRequest._init_account_info，避免构造时触发真实 DB
_init_patch = mock.patch(
    "Channel.douyin.utils.base_request.BaseRequest._init_account_info",
    lambda self: None,
)
_init_patch.start()

from Channel.douyin.utils.API.order_query import OrderQuery as DouyinOrderQuery
from Agent.CustomerAgent.tools.query_order_status import (
    query_order_status,
    QueryOrderStatusParams,
)


# ============================================================================
# Mock 辅助
# ============================================================================

def _make_order(order_sn="123456789012345678", goods_name="测试商品"):
    """构造一个订单 dict（与拼多多对齐格式）"""
    return {
        "order_sn": order_sn,
        "order_sequence_no": "",
        "goods_name": goods_name,
        "goods_id": "",
        "spec": "",
        "quantity": 1,
        "order_status": None,
        "order_status_desc": "待核实",
        "shipping_status": None,
        "shipping_status_desc": "待核实",
        "logistics_company": "",
        "logistics_sn": "",
        "confirm_time": "",
        "pay_time": "",
        "after_sales_status": 0,
        "buyer_nick": "",
        "needs_shipping": True,
    }


# ============================================================================
# 1. OrderQuery.get_recent_orders（消息历史订单号 → 真实接口补详情）
# ============================================================================

class TestOrderQueryGetRecentOrders(unittest.TestCase):
    """抖音 OrderQuery.get_recent_orders"""

    def test_normal_returns_orders(self):
        """正常：消息历史有订单号 + 真实接口补详情 → 返回订单列表"""
        oq = DouyinOrderQuery(shop_id="douyin_test", user_id="user1")
        detail = _make_order()
        with mock.patch.object(
            oq, "_extract_order_ids", return_value=["123456789012345678"]
        ), mock.patch.object(
            oq, "get_order_detail", return_value=detail
        ):
            result = oq.get_recent_orders(buyer_uid="buyer1", days=7)

        self.assertTrue(result["success"])
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["orders"][0]["order_sn"], "123456789012345678")

    def test_empty_history_returns_empty_list(self):
        """空数据：消息历史无订单号 → success=True，orders=[]，不报错"""
        oq = DouyinOrderQuery(shop_id="douyin_test", user_id="user1")
        with mock.patch.object(oq, "_extract_order_ids", return_value=[]):
            result = oq.get_recent_orders(buyer_uid="buyer1")

        self.assertTrue(result["success"])
        self.assertEqual(result["orders"], [])
        self.assertEqual(result["total"], 0)

    def test_shop_unregistered_returns_failure(self):
        """店铺未登记：_extract_order_ids 返回 None → success=False，不抛异常"""
        oq = DouyinOrderQuery(shop_id="douyin_test", user_id="user1")
        with mock.patch.object(oq, "_extract_order_ids", return_value=None):
            result = oq.get_recent_orders(buyer_uid="buyer1")

        self.assertFalse(result["success"])
        self.assertEqual(result["orders"], [])
        self.assertIn("error_msg", result)

    def test_detail_failure_falls_back_to_skeleton(self):
        """真实接口失败 → 留订单号骨架，不抛异常"""
        oq = DouyinOrderQuery(shop_id="douyin_test", user_id="user1")
        with mock.patch.object(
            oq, "_extract_order_ids", return_value=["123456789012345678"]
        ), mock.patch.object(oq, "get_order_detail", return_value=None):
            result = oq.get_recent_orders(buyer_uid="buyer1")

        self.assertTrue(result["success"])
        self.assertEqual(len(result["orders"]), 1)
        o = result["orders"][0]
        self.assertEqual(o["order_sn"], "123456789012345678")

    def test_does_not_call_pinduoduo_api(self):
        """渠道隔离：不传订单号且消息历史为空时，不发任何 HTTP 请求"""
        oq = DouyinOrderQuery(shop_id="douyin_test", user_id="user1")
        with mock.patch.object(
            oq, "_extract_order_ids", return_value=[]
        ), mock.patch.object(oq, "post") as mock_post, \
             mock.patch.object(oq, "get") as mock_get:
            result = oq.get_recent_orders()
        mock_post.assert_not_called()
        mock_get.assert_not_called()
        self.assertTrue(result["success"])


# ============================================================================
# 2. searchlist 真实 API 路径（get_order_detail / get_buyer_orders）
# ============================================================================

class TestOrderQuerySearchlist(unittest.TestCase):
    """searchlist 主路径（mock self.get，不依赖真实网络）"""

    @staticmethod
    def _fake_search_response(row):
        return {"st": 0, "msg": "", "code": 0, "data": [row], "total": 1}

    @staticmethod
    def _fake_order_row(order_id="6912345678901234567"):
        return {
            "order_id": "",
            "shop_order_id": order_id,
            "order_status": 2,
            "order_status_info": {"order_status_text": "待发货"},
            "product_item": [{
                "product_name": "测试商品", "product_id": "p1",
                "img": "", "sku_spec": [{"name": "颜色", "value": "黑"}],
                "combo_num": 1, "pay_amount": 9900,
            }],
            "pay_amount": 9900,
            "user_nickname": "测*试",
        }

    def test_get_order_detail_by_order_id(self):
        oq = DouyinOrderQuery(shop_id="douyin_test", user_id="user1")
        resp = self._fake_search_response(self._fake_order_row())
        with mock.patch.object(
            oq, "get", return_value=resp
        ) as mock_get:
            result = oq.get_order_detail("6912345678901234567")

        self.assertEqual(result["order_sn"], "6912345678901234567")
        # 请求参数里应带 order_id 筛选
        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params.get("order_id"), "6912345678901234567")

    def test_get_buyer_orders_by_nick(self):
        oq = DouyinOrderQuery(shop_id="douyin_test", user_id="user1")
        resp = self._fake_search_response(self._fake_order_row())
        with mock.patch.object(
            oq, "get", return_value=resp
        ) as mock_get:
            result = oq.get_buyer_orders("测试买家昵称")

        self.assertTrue(result["success"])
        self.assertEqual(result["total"], 1)
        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params.get("buyer"), "测试买家昵称")

    def test_searchlist_st_error_returns_empty(self):
        """st != 0（如未登录）→ success=False 空列表，不抛异常"""
        oq = DouyinOrderQuery(shop_id="douyin_test", user_id="user1")
        resp = {"st": 10000, "msg": "invalid", "data": []}
        with mock.patch.object(oq, "get", return_value=resp):
            result = oq.get_buyer_orders("任意昵称")

        self.assertFalse(result["success"])
        self.assertEqual(result["orders"], [])


# ============================================================================
# 3. query_order_status 工具抖音路径
# ============================================================================

class TestQueryOrderStatusDouyin(unittest.TestCase):
    """query_order_status 工具抖音路径"""

    def _make_params(self, **kwargs):
        defaults = dict(
            channel_type="douyin",
            shop_id="douyin_test",
            user_id="user1",
            recipient_uid="buyer1",
            order_sn=None,
            days=90,
        )
        defaults.update(kwargs)
        return QueryOrderStatusParams(**defaults)

    def test_douyin_has_orders_returns_formatted(self):
        """抖音：有订单 → 格式化返回（含 [untrusted_order_data] 包裹）"""
        mock_orders = [_make_order(order_sn="123456789012345678", goods_name="抖音商品")]
        params = self._make_params()
        with mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_recent_orders",
            return_value={"success": True, "total": 1, "orders": mock_orders},
        ):
            result = query_order_status(params)

        self.assertIsInstance(result, str)
        self.assertIn("[untrusted_order_data]", result)
        self.assertIn("123456789012345678", result)
        self.assertIn("抖音商品", result)
        self.assertIn("[/untrusted_order_data]", result)

    def test_douyin_no_orders_returns_guidance(self):
        """抖音：无订单 → 返回引导话术（请提供订单号或到抖店App查看）"""
        params = self._make_params()
        with mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_recent_orders",
            return_value={"success": True, "total": 0, "orders": []},
        ):
            result = query_order_status(params)

        self.assertIsInstance(result, str)
        self.assertIn("没查到", result)
        self.assertIn("抖店", result)
        self.assertNotIn("[untrusted_order_data]", result)

    def test_douyin_query_failure_returns_guidance(self):
        """抖音：查询失败 → 返回引导话术，不暴露技术细节"""
        params = self._make_params()
        with mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_recent_orders",
            return_value={"success": False, "error_msg": "店铺未登记", "orders": [], "total": 0},
        ):
            result = query_order_status(params)

        self.assertIsInstance(result, str)
        self.assertIn("没查到", result)
        self.assertIn("抖店", result)
        # 不暴露技术细节（error_msg 内容不应出现在用户话术里）
        self.assertNotIn("店铺未登记", result)

    def test_douyin_does_not_call_pinduoduo_browser(self):
        """渠道隔离：抖音 query_order_status 走抖音 OrderQuery，返回格式化订单数据（不抛异常）"""
        params = self._make_params()
        mock_orders = [_make_order()]
        with mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_recent_orders",
            return_value={"success": True, "total": 1, "orders": mock_orders},
        ):
            result = query_order_status(params)
        # 抖音路径返回格式化订单数据（含 untrusted_order_data 包裹）
        self.assertIn("[untrusted_order_data]", result)
        self.assertIn("123456789012345678", result)

    def test_douyin_missing_shop_id_returns_error(self):
        """抖音：缺少 shop_id/user_id → 返回错误提示"""
        params = self._make_params(shop_id=None, user_id=None)
        result = query_order_status(params)
        self.assertIn("缺少必要的 shop_id 或 user_id", result)

    def test_no_direct_unsupported_message(self):
        """抖音不再直接返回'暂不支持'话术（有实际查询逻辑）"""
        params = self._make_params()
        with mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_recent_orders",
            return_value={"success": True, "total": 0, "orders": []},
        ):
            result = query_order_status(params)
        self.assertNotIn("暂未接入", result)
        self.assertNotIn("暂不支持", result)

    def test_nickname_only_does_not_return_order_details(self):
        """昵称重名风险：没有订单号时不能直接返回昵称命中的订单详情。"""
        params = self._make_params(recipient_nick="买家甲")
        candidate = _make_order(order_sn="123456789012345678")
        with mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_recent_orders",
            return_value={"success": True, "total": 0, "orders": []},
        ), mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_buyer_orders",
            return_value={"success": True, "total": 1, "orders": [candidate]},
        ):
            result = query_order_status(params)

        self.assertNotIn("[untrusted_order_data]", result)
        self.assertNotIn("123456789012345678", result)
        self.assertIn("订单号", result)

    def test_order_sn_must_match_buyer_candidates(self):
        """买家报的订单号只能在自己的候选订单里匹配，不能直接查全店。"""
        params = self._make_params(
            recipient_nick="买家甲",
            order_sn="999999999999999999",
        )
        with mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_recent_orders",
            return_value={"success": True, "total": 0, "orders": []},
        ), mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_buyer_orders",
            return_value={"success": True, "total": 1, "orders": [
                _make_order(order_sn="123456789012345678")
            ]},
        ), mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_order_detail"
        ) as get_detail:
            result = query_order_status(params)

        get_detail.assert_not_called()
        self.assertIn("没找到订单号", result)
        self.assertNotIn("[untrusted_order_data]", result)

    def test_refunded_orders_are_not_shown(self):
        """已退款/已关闭订单不应继续展示给买家。"""
        params = self._make_params()
        refunded = _make_order(order_sn="123456789012345678")
        refunded["needs_shipping"] = False
        refunded["refund_status_desc"] = "已退款"
        with mock.patch(
            "Channel.douyin.utils.API.order_query.OrderQuery.get_recent_orders",
            return_value={"success": True, "total": 1, "orders": [refunded]},
        ):
            result = query_order_status(params)

        self.assertNotIn("123456789012345678", result)
        self.assertIn("暂无有效在途订单", result)


if __name__ == "__main__":
    unittest.main()
