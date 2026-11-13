from ..base_request import BaseRequest
import time
from utils.logger_loguru import get_logger


class OrderQuery(BaseRequest):
    """
    拼多多订单查询API（mms.pinduoduo.com 商家后台内部接口）

    能力：
    - 按时间范围查询店铺近期订单列表
    - 返回订单号、商品名、发货状态、物流信息等

    认证：复用 BaseRequest 的 cookie 机制（含 anti-content）
    """

    def __init__(self, shop_id: str = None, user_id: str = None, cookies=None):
        super().__init__(shop_id=shop_id, user_id=user_id)
        if cookies:
            self.update_cookies(cookies)

    def get_recent_orders(self, days=7, page=1, page_size=20):
        """
        查询近期订单列表

        Args:
            days (int): 查询最近N天的订单，默认7天
            page (int): 页码，默认1
            page_size (int): 每页数量，默认20，最大50

        Returns:
            dict: {
                "success": bool,
                "total": int,
                "orders": [...],
                "error_msg": str or None
            }
        """
        import math

        now = int(time.time())
        end_ts = now
        start_ts = now - (days * 86400)

        url = "https://mms.pinduoduo.com/mangkhut/mms/recentOrderList"

        data = {
            "orderType": 1,
            "afterSaleType": 1,
            "remarkStatus": -1,
            "urgeShippingStatus": -1,
            "groupStartTime": start_ts,
            "groupEndTime": end_ts,
            "pageNumber": page,
            "pageSize": min(page_size, 50),
            "sortType": 10,
        }

        anti_content = (
            self.cookies.get("anti_content")
            or self.cookies.get("anti-content", "")
        )
        headers = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "anti-content": anti_content,
            "content-type": "application/json;charset=UTF-8",
            "origin": "https://mms.pinduoduo.com",
            "referer": "https://mms.pinduoduo.com/chat-merchant/index.html",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/146.0.0.0 Safari/537.36"
            ),
        }

        result = self.post(url, json_data=data, headers=headers)

        if result and (result.get("success") is True or "result" in result):
            return self._parse_order_list(result)
        else:
            # 真实响应字段为蛇形 error_code/error_msg（已对真实账号验证）
            error_msg = (
                result.get("error_msg")
                or result.get("errorMsg")
                or result.get("error_code")
                or "订单查询接口无响应"
                if result
                else "订单查询接口无响应"
            )
            self.logger.error(f"订单列表查询失败: {error_msg}")
            return {"success": False, "error_msg": error_msg, "orders": [], "total": 0}

    def get_buyer_orders_by_uid(self, uid, page=1, page_size=10, show_history=True):
        """
        按买家数字 uid 查询其在本店的全部订单（2026-11-12 逆向实测）。

        接口：POST https://mms.pinduoduo.com/latitude/order/userAllOrder
        请求体：{"pageNo", "pageSize", "showHistory", "uid"}
        实测：纯 cookie 即可（无 anti-content 校验）；
             uid 来自会话接口 orderCsGroupConvList 的 userInfo.uid，
             与客服 WS 消息里 from.uid 一致。

        Args:
            uid (str|int): 买家数字 uid（如 "4239748275"）
            page (int): 页码，从 1 开始
            page_size (int): 每页数量，默认 10
            show_history (bool): 是否含历史订单，默认 True

        Returns:
            dict: {"success": bool, "total": int, "orders": [...], "error_msg": str or None}
                  订单字段与 get_recent_orders 归一化格式一致
        """
        url = "https://mms.pinduoduo.com/latitude/order/userAllOrder"

        data = {
            "pageNo": page,
            "pageSize": min(page_size, 50),
            "showHistory": show_history,
            "uid": str(uid),
        }

        anti_content = (
            self.cookies.get("anti_content")
            or self.cookies.get("anti-content", "")
        )
        headers = {
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json;charset=UTF-8",
            "origin": "https://mms.pinduoduo.com",
            "referer": "https://mms.pinduoduo.com/chat-merchant/index.html",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/146.0.0.0 Safari/537.36"
            ),
        }
        if anti_content:
            headers["anti-content"] = anti_content

        result = self.post(url, json_data=data, headers=headers)

        if not result or (result.get("success") is not True):
            if result:
                error_msg = (
                    result.get("errorMsg")
                    or result.get("error_msg")
                    or "接口无响应"
                )
            else:
                error_msg = "接口无响应"
            self.logger.error(
                f"买家订单查询失败: uid={uid}, error={error_msg}"
            )
            return {"success": False, "error_msg": error_msg, "orders": [], "total": 0}

        result_data = result.get("result") or {}
        raw_items = result_data.get("orders") or []

        orders = []
        for item in raw_items:
            goods_items = _order_goods_items(item)
            names = [
                str(g.get("goodsName") or "")
                for g in goods_items
                if g.get("goodsName")
            ]
            goods_name = "、".join(names)
            spec = str((goods_items[0].get("spec") if goods_items else "") or "")
            try:
                quantity = sum(
                    int(g.get("goodsNumber") or g.get("goodsCount") or 0)
                    for g in goods_items
                ) or (goods_items[0].get("goodsNumber") if goods_items else 0) or 1
            except (TypeError, ValueError):
                quantity = (goods_items[0].get("goodsNumber") if goods_items else 0) or 1

            order_status_desc = item.get("orderStatusStr") or _map_order_status(
                item.get("orderStatus")
            )
            refund_status_desc = _classify_refund(order_status_desc)
            needs_shipping = refund_status_desc not in ("已退款", "已关闭", "售后已结束")
            tracking = str(item.get("trackingNumber") or item.get("tracking_number") or "")
            after_sales = item.get("afterSalesInfo") or {}
            order_amount = item.get("orderAmount")
            buyer_nick = (
                item.get("nickname")
                or item.get("buyerNick")
                or item.get("nickName")
                or ""
            )
            orders.append({
                "order_sn": item.get("orderSn", ""),
                "order_sequence_no": item.get("groupOrderId", "")
                or item.get("orderSequenceNo", ""),
                "goods_name": goods_name,
                "goods_id": (goods_items[0].get("goodsId", "") if goods_items else ""),
                "spec": spec,
                "quantity": quantity,
                "order_status": item.get("orderStatus"),
                "order_status_desc": order_status_desc,
                "refund_status_desc": refund_status_desc,
                "needs_shipping": needs_shipping,
                "shipping_status": item.get("shippingStatus"),
                "shipping_status_desc": (
                    "无需发货（已退款/已关闭）"
                    if not needs_shipping
                    else _map_shipping_status(item.get("shippingStatus"))
                ),
                "logistics_company": _extract_logistics_company(item),
                "logistics_sn": tracking,
                "pay_amount": (
                    round(float(order_amount) / 100.0, 2)
                    if order_amount is not None
                    else None
                ),
                "order_time": item.get("orderTime") or item.get("createdAt", ""),
                "confirm_time": item.get("confirmTime", ""),
                "pay_time": item.get("payTime", ""),
                "after_sales_status": (
                    1 if refund_status_desc or after_sales.get("status") else 0
                ),
                "buyer_nick": buyer_nick,
                "buyer_uid": str(item.get("uid") or uid),
            })

        return {
            "success": True,
            "total": result_data.get("total", len(orders)),
            "orders": orders,
        }

    def _parse_order_list(self, response_data):
        """解析 recentOrderList 响应"""
        try:
            result_data = response_data.get("result", {})
            total = result_data.get("totalItemNum", 0)
            raw_items = result_data.get("pageItems", [])

            orders = []
            for item in raw_items:
                order = {
                    "order_sn": item.get("orderSn", ""),
                    "order_sequence_no": item.get("orderSequenceNo", ""),
                    "goods_name": item.get("goodsName", ""),
                    "goods_id": item.get("goodsId", ""),
                    "spec": item.get("spec", ""),
                    "quantity": item.get("quantity", 0),
                    "order_status": item.get("orderStatus"),
                    "order_status_desc": _map_order_status(item.get("orderStatus")),
                    "shipping_status": item.get("shippingStatus"),
                    "shipping_status_desc": _map_shipping_status(
                        item.get("shippingStatus")
                    ),
                    "logistics_company": item.get("logisticsCompany", ""),
                    "logistics_sn": item.get("logisticsSn", ""),
                    "confirm_time": item.get("confirmTime", ""),
                    "pay_time": item.get("payTime", ""),
                    "after_sales_status": item.get("afterSalesStatus"),
                    "buyer_nick": item.get("buyerNick", "") or "",
                }
                orders.append(order)

            return {"success": True, "total": total, "orders": orders}

        except Exception as e:
            self.logger.error(f"解析订单列表失败: error_type={type(e).__name__}")
            return {"success": False, "error_msg": f"解析异常: {e}", "orders": [], "total": 0}


# ---- 状态映射（基于拼多多商家后台通用状态码）----

def _map_order_status(status_code):
    """订单成交状态码 → 中文描述"""
    _MAP = {
        0: "未支付",
        1: "已支付待成团",
        2: "已成交（已签收）",
        3: "已取消",
        5: "已结算",
        8: "非多多进宝商品",
    }
    return _MAP.get(status_code, f"未知状态({status_code})")


def _map_shipping_status(code):
    """发货状态码 → 中文描述"""
    _MAP = {
        0: "未发货",
        1: "已发货",
        2: "已揽收",
        3: "运输中",
        4: "派送中",
        5: "已签收",
        -1: "无需发货",
    }
    return _MAP.get(code, f"物流状态({code})")


def _classify_refund(status_desc):
    """从订单状态文案识别退款/关闭/售后状态。"""
    text = str(status_desc or "")
    if "退款成功" in text or "已退款" in text:
        return "已退款"
    if "退款中" in text or "退款" in text:
        return "退款中"
    if "售后成功" in text or "售后关闭" in text:
        return "售后已结束"
    if "售后中" in text or "售后" in text:
        return "售后处理中"
    if "已关闭" in text or "关闭" in text:
        return "已关闭"
    return ""


def _order_goods_items(item):
    """兼容 orderGoodsList 的 dict / list / 多商品映射三种形态。"""
    goods_list = item.get("orderGoodsList") or {}
    if isinstance(goods_list, dict):
        values = list(goods_list.values())
        if values and isinstance(values[0], dict):
            return [g for g in values if isinstance(g, dict)]
        return [goods_list] if goods_list else []
    if isinstance(goods_list, list):
        return [g for g in goods_list if isinstance(g, dict)]
    return []


def _extract_logistics_company(item):
    """从 traceInfoList / 顶层字段提取物流公司名。"""
    trace = item.get("traceInfoList")
    if isinstance(trace, list):
        for node in trace:
            if not isinstance(node, dict):
                continue
            for key in (
                "companyName",
                "logisticsName",
                "expressCompanyName",
                "logisticsCompany",
                "company",
            ):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    for key in ("logisticsCompany", "expressCompanyName", "express_delivery"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            name = value.get("name") or value.get("companyName")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return ""
