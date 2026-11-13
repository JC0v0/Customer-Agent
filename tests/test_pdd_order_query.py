"""拼多多 userAllOrder 解析回归：多商品、退款态、物流字段。"""
import unittest
from unittest import mock

_init_patch = mock.patch(
    "Channel.pinduoduo.utils.base_request.BaseRequest._init_account_info",
    lambda self: None,
)
_init_patch.start()

from Channel.pinduoduo.utils.API.order_query import OrderQuery


class PddOrderQueryTests(unittest.TestCase):
    def _query(self, order):
        oq = OrderQuery(shop_id="shop-1", user_id="user-1")
        response = {
            "success": True,
            "result": {"orders": [order], "total": 1},
        }
        with mock.patch.object(oq, "post", return_value=response):
            return oq.get_buyer_orders_by_uid("buyer-1")

    def test_refunded_order_is_marked_not_shippable(self):
        result = self._query({
            "orderSn": "260811-1",
            "orderStatus": 1,
            "orderStatusStr": "未发货，退款成功",
            "shippingStatus": 0,
            "orderAmount": 2800,
            "orderGoodsList": [{
                "goodsName": "保温杯",
                "goodsId": "g-1",
                "spec": "红色",
                "goodsNumber": 2,
            }],
            "trackingNumber": "",
            "uid": "buyer-1",
        })

        self.assertTrue(result["success"])
        order = result["orders"][0]
        self.assertFalse(order["needs_shipping"])
        self.assertEqual(order["refund_status_desc"], "已退款")
        self.assertIn("无需发货", order["shipping_status_desc"])
        self.assertEqual(order["goods_name"], "保温杯")
        self.assertEqual(order["quantity"], 2)
        self.assertEqual(order["pay_amount"], 28.0)

    def test_multiple_goods_mapping_is_joined(self):
        result = self._query({
            "orderSn": "260811-2",
            "orderStatus": 1,
            "orderStatusStr": "待发货",
            "shippingStatus": 0,
            "orderAmount": 500,
            "orderGoodsList": {
                "0": {"goodsName": "商品A", "goodsNumber": 1},
                "1": {"goodsName": "商品B", "goodsNumber": 1},
            },
            "uid": "buyer-1",
        })

        order = result["orders"][0]
        self.assertTrue(order["needs_shipping"])
        self.assertEqual(order["goods_name"], "商品A、商品B")
        self.assertEqual(order["quantity"], 2)

    def test_tracking_number_marks_order_shipped(self):
        result = self._query({
            "orderSn": "260811-3",
            "orderStatus": 2,
            "orderStatusStr": "待收货",
            "shippingStatus": 1,
            "orderAmount": 100,
            "orderGoodsList": {"goodsName": "商品", "goodsNumber": 1},
            "trackingNumber": "ZT123",
            "traceInfoList": [{"companyName": "中通快递"}],
            "uid": "buyer-1",
        })

        order = result["orders"][0]
        self.assertEqual(order["shipping_status_desc"], "已发货")
        self.assertEqual(order["logistics_sn"], "ZT123")
        self.assertEqual(order["logistics_company"], "中通快递")


if __name__ == "__main__":
    unittest.main()
