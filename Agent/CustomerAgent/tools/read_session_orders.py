"""
商家侧会话订单读取器（爬取当前会话买家在本店所购订单）

设计依据（已真机验证）：
- 拼多多商家后台订单接口（recentOrderList）对静态 anti-content 返回 40002 反爬；
  必须捕获 PDD 页面自身发出的 recentOrderList 请求（含每次动态生成的合法
  anti-content + 正确时间窗），再立即重放。
- ⚠️ 重要（2026-08-11 真机验证纠正）：recentOrderList **不认 `buyerId`**。
  原生请求字段为 ['orderType','afterSaleType','remarkStatus','urgeShippingStatus',
  'groupStartTime','groupEndTime','pageNumber','pageSize','sortType','mobile']，
  其中**没有 buyerId**。注入 buyerId（无论真假）均被忽略，始终返回全店订单
  （实测注入假 buyerId=111111111111 仍返回 totalItemNum=2=全店）。因此「服务端
  按 buyerId 精确过滤」从未生效，之前「buyerId=4239748275 精确返回 1 单」是伪验证
  （当时全店就 1 单，分不清过滤生效与否）。
- 原生请求里唯一的买家维度过滤字段是 `mobile`，但 PDD 已 deprecated 真实号查询
  （注入 mobile 返回 error="真实号查询已下线，请使用隐私号进行查询"），需每单的
  隐私号才能查——bot 仅有聊天里的 buyer user_id(from_uid)，拿不到手机号/隐私号。
- 订单项本身**不含任何买家 uid 字段**（已扫全部 ~110 个字段，仅有掩码 nickname
  如「1***」，且首字相同会碰撞，不可靠），无法在客户端按 uid 过滤。
- 结论：仅凭聊天买家 user_id，**无法**通过 recentOrderList 安全地把订单归因到该
  买家（要么返回全店=泄漏他人订单，要么只能用 order_sn 精确匹配该买家主动提供的
  订单号）。原「buyerId 服务端过滤 + 哨兵防误过滤」机制建立在错误前提上，实测
  哨兵会**恒定**触发 filtered_failed（buyerId 永远被忽略），导致订单查询永远拒答。
- 接口对时间窗有上限（约 90 天），重放时必须保留页面原始 groupStartTime/
  groupEndTime，否则返回 errorCode=1000「下单时间超出查询范围」。
- 商家后台登录态会过期（43001 会话已过期，页面跳登录）。捕获前先检测是否跳
  登录，若过期则返回 login_expired=True，由调用方（独立脚本或浏览器池）自动重登。

用法：
  python -m Agent.CustomerAgent.tools.read_session_orders --buyer-uid 4239748275   # 按买家读订单(服务端精确过滤)
  python -m Agent.CustomerAgent.tools.read_session_orders --days 90                 # 店铺近90天(全店)
  python -m Agent.CustomerAgent.tools.read_session_orders --self-test               # 解析器单测(无需浏览器)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List, Optional

# 项目内导入延迟到真正需要浏览器的函数内，避免纯函数/单测依赖重依赖链
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.logger_loguru import get_logger
    logger = get_logger("ReadSessionOrders")
except Exception:
    import logging
    logger = logging.getLogger("ReadSessionOrders")


ORDER_API = "https://mms.pinduoduo.com/mangkhut/mms/recentOrderList"
ORDER_PAGE = "https://mms.pinduoduo.com/orders/list"

# 买家身份候选字段（recentOrderList 的订单项不含买家 uid，仅掩码昵称；
# 真正按买家过滤走请求参数 buyerId，这里仅作为客户端兜底）
_BUYER_ID_KEYS = ("buyerId", "customerId", "userId", "buyer_id", "customer_id", "uid", "customerUid")
_BUYER_NICK_KEYS = ("buyerNick", "nickName", "nick", "buyerName", "nickname")


# ---------------------------------------------------------------------------
# 纯函数：解析 / 过滤（可单测，不依赖浏览器）
# ---------------------------------------------------------------------------

def normalize_order(item: Dict) -> Dict:
    """把 recentOrderList 返回的真实订单对象归一化为统一结构。"""
    order_amount = item.get("order_amount")
    pay_amount = None
    if order_amount is not None:
        try:
            # 接口以「分」为单位，转成「元」
            pay_amount = round(float(order_amount) / 100.0, 2)
        except (TypeError, ValueError):
            pay_amount = order_amount

    # 物流公司：优先 express_delivery（字符串或含 name 的对象），其次 waybill
    lc = item.get("express_delivery") or ""
    if isinstance(lc, dict):
        lc = lc.get("name") or lc.get("companyName") or ""
    logistics_sn = item.get("tracking_number") or item.get("logistics_sn") or ""

    order_status_str = item.get("order_status_str") or ""
    ship_status = item.get("shipping_status")

    return {
        "order_sn": item.get("order_sn", ""),
        "order_sequence_no": item.get("order_sequence_no", "") or item.get("orderSn", ""),
        "goods_name": item.get("goods_name", ""),
        "goods_id": item.get("goods_id", ""),
        "spec": item.get("spec", ""),
        "quantity": item.get("goods_number") or item.get("quantity", 0),
        "order_status": item.get("order_status"),
        "order_status_desc": order_status_str or _map_order_status(item.get("order_status")),
        "shipping_status": ship_status,
        "shipping_status_desc": _map_shipping_status(ship_status, logistics_sn, order_status_str),
        "logistics_company": lc,
        "logistics_sn": logistics_sn,
        "pay_amount": pay_amount,
        "order_time": item.get("order_time") or item.get("created_at", ""),
        "shipping_time": item.get("shipping_time", ""),
        "confirm_time": item.get("confirm_time", ""),
        "after_sales_status": item.get("after_sales_status"),
        "buyer_nick": item.get("nickname", "") or item.get("buyerNick", "") or item.get("nickName", ""),
        "buyer_id": _extract_buyer_id(item),
        "raw": item,
    }


def _extract_buyer_id(item: Dict) -> str:
    for k in _BUYER_ID_KEYS:
        v = item.get(k)
        if v not in (None, "", 0):
            return str(v)
    return ""


def filter_orders(items: List[Dict], buyer_uid: str = "", mobile: str = "", nick: str = "") -> List[Dict]:
    """按买家 uid / 手机号 / 昵称过滤订单（客户端兜底；优先走请求参数 buyerId）。

    注意：recentOrderList 的订单项通常不含买家 uid 字段（仅掩码昵称），
    此时 buyer_uid 无法匹配也不应把订单删掉——直接保留（交由上层判断）。
    只有订单确实带 uid 字段时才做严格 uid 过滤。
    """
    buyer_uid = str(buyer_uid or "").strip()
    mobile = str(mobile or "").strip()
    nick = str(nick or "").strip().lower()
    if not (buyer_uid or mobile or nick):
        return [normalize_order(it) for it in items]

    # 订单里是否真的存在买家 uid 字段（recentOrderList 通常不存在）
    has_uid = any(normalize_order(it).get("buyer_id") for it in items)

    out = []
    for it in items:
        o = normalize_order(it)
        matched = False
        if buyer_uid:
            if has_uid:
                if o["buyer_id"] and buyer_uid in o["buyer_id"]:
                    matched = True
            # 无 uid 字段：不按 uid 排除
        if mobile and (mobile in str(it.get("mobile", "")) or mobile in str(it.get("receiverPhone", ""))):
            matched = True
        if nick:
            bn = (o["buyer_nick"] or "").lower()
            plain = bn.replace("*", "")
            if plain and plain in nick:
                matched = True
            elif nick in bn:
                matched = True
        # 仅给了 buyer_uid 且数据无 uid 字段、也无其他条件 → 保留（无法排除）
        if not matched and buyer_uid and not has_uid and not mobile and not nick:
            matched = True
        if matched:
            out.append(o)
    return out


def _map_order_status(code) -> str:
    _MAP = {0: "未支付", 1: "已支付待成团", 2: "已成交（已签收）", 3: "已取消",
            5: "已结算", 8: "非多多进宝商品"}
    return _MAP.get(code, f"未知状态({code})")


def _map_shipping_status(code, logistics_sn: str = "", order_status_str: str = "") -> str:
    """物流状态映射。未发货且无运单号时明确为「未发货」。"""
    if (code is None or code == "") and not logistics_sn:
        # 待发货/未支付等场景：没有运单号即未发货
        if "待发货" in (order_status_str or ""):
            return "未发货"
        return "未发货" if code in (None, "", 0, -1) else f"物流状态({code})"
    _MAP = {0: "未发货", 1: "已发货", 2: "已揽收", 3: "运输中",
            4: "派送中", 5: "已签收", -1: "无需发货"}
    return _MAP.get(code, f"物流状态({code})")


# ---------------------------------------------------------------------------
# 客服「客户订单」接口（userAllOrder）—— 按买家 uid 维度拉单
# ---------------------------------------------------------------------------
# 背景：recentOrderList 是「店铺全量」接口，不认 buyerId、订单项也不含买家 uid，
# 无法安全地把订单归因到当前会话买家（要么返回全店=泄漏他人订单，要么只能让用户
# 提供订单号）。但真实商家客服面板能看到「这个客户在本店的订单」——它走的是 CS
# 专用接口 userAllOrder（按买家 uid 返回该买家在本店的订单，天然不泄漏其他买家）。
#   POST https://mms.pinduoduo.com/latitude/order/userAllOrder
#   body: {"uid": <买家uid>, "pageSize": 10}  ->  result.orders
# 真实响应字段（已抓包确认）：orderSn / orderStatus(int) / orderStatusStr(如
# 「未发货，退款成功」) / shippingStatus(int) / orderAmount(分) / uid(买家uid) /
# orderGoodsList(商品明细: goodsName/spec/goodsNumber/goodsPrice) / trackingNumber
# (运单号) / traceInfoList(物流轨迹，发货后才有，含公司名)。
CUSTOMER_ORDER_API = "https://mms.pinduoduo.com/latitude/order/userAllOrder"
CHAT_PAGE = "https://mms.pinduoduo.com/chat-merchant/index.html"


def _classify_refund(status_desc: str) -> str:
    """从订单状态串里识别退款/售后状态。

    PDD 的 userAllOrder 对「未发货但已退款」的订单，orderStatusStr 形如
    「未发货，退款成功」——字面含「未发货」却已无需发货。若只按 shipping_status
    判断会误报「未发货」，故单独抽退款/售后语义，供格式化与 LLM 如实转达。
    返回空串表示无退款/售后；否则返回易读标签。
    """
    d = (status_desc or "")
    if "退款成功" in d or "已退款" in d or "退款关闭" in d:
        return "已退款"
    if "退款中" in d or "退款" in d:
        return "退款中"
    if "售后成功" in d or "售后关闭" in d:
        return "售后已结束"
    if "售后中" in d or "售后" in d:
        return "售后处理中"
    return ""


def _extract_logistics_company(item: Dict) -> str:
    """从 traceInfoList / 顶层字段里尽量提取物流公司名（发货后才有）。"""
    ti = item.get("traceInfoList")
    if isinstance(ti, list):
        for t in ti:
            if isinstance(t, dict):
                for k in ("companyName", "logisticsName", "expressCompanyName",
                          "logisticsCompany", "company"):
                    v = t.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    for k in ("logisticsCompany", "expressCompanyName", "express_delivery"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            nv = v.get("name") or v.get("companyName")
            if isinstance(nv, str) and nv.strip():
                return nv.strip()
    return ""


def normalize_customer_order(item: Dict) -> Dict:
    """把 userAllOrder 返回的订单对象归一化为统一结构（与 normalize_order 对齐）。"""
    order_amount = item.get("orderAmount")
    pay_amount = None
    if order_amount is not None:
        try:
            pay_amount = round(float(order_amount) / 100.0, 2)   # 接口以「分」为单位
        except (TypeError, ValueError):
            pay_amount = order_amount

    # 商品明细：orderGoodsList 可能是「单个商品 dict」（值均为标量）或
    # 「key->商品dict」的映射（多商品）。统一抽成商品 dict 列表。
    gl = item.get("orderGoodsList") or {}
    if isinstance(gl, dict):
        vals = list(gl.values())
        goods_items = vals if vals and isinstance(vals[0], dict) else [gl]
    elif isinstance(gl, list):
        goods_items = [g for g in gl if isinstance(g, dict)]
    else:
        goods_items = []

    names = [g.get("goodsName", "") for g in goods_items if g.get("goodsName")]
    goods_name = "、".join(names) if names else ""
    spec = (goods_items[0].get("spec", "") if goods_items else "") or ""
    try:
        quantity = sum(int(g.get("goodsNumber") or g.get("goodsCount") or 0)
                       for g in goods_items) or (goods_items[0].get("goodsNumber") or 0) if goods_items else 0
    except (TypeError, ValueError):
        quantity = goods_items[0].get("goodsNumber") or 0 if goods_items else 0

    order_status = item.get("orderStatus")
    order_status_desc = item.get("orderStatusStr") or _map_order_status(order_status)
    shipping_status = item.get("shippingStatus")
    tracking = item.get("trackingNumber") or ""
    logistics_company = _extract_logistics_company(item)
    refund_status_desc = _classify_refund(order_status_desc)
    # 已退款/售后已结束的订单无需发货：即便 shipping_status=0 也不应报「未发货」
    needs_shipping = refund_status_desc not in ("已退款", "售后已结束")

    return {
        "order_sn": item.get("orderSn", ""),
        "order_sequence_no": "",
        "goods_name": goods_name,
        "goods_id": (goods_items[0].get("goodsId", "") if goods_items else ""),
        "spec": spec,
        "quantity": quantity,
        "order_status": order_status,
        "order_status_desc": order_status_desc,
        "shipping_status": shipping_status,
        "shipping_status_desc": (
            "无需发货（已退款/已关闭）"
            if not needs_shipping
            else _map_shipping_status(shipping_status, tracking, order_status_desc)
        ),
        "refund_status_desc": refund_status_desc,
        "needs_shipping": needs_shipping,
        "logistics_company": logistics_company,
        "logistics_sn": tracking,
        "pay_amount": pay_amount,
        "order_time": item.get("orderTime") or item.get("createdAt") or "",
        "shipping_time": item.get("shippingTime") or "",
        "buyer_id": str(item.get("uid") or ""),
        "raw": item,
    }


def parse_customer_orders(resp_json: Optional[Dict]) -> Dict:
    """把 userAllOrder 响应解析为统一结构。"""
    if not resp_json:
        return {"success": False, "error_msg": "无响应", "orders": [], "total": 0}
    if not resp_json.get("success"):
        return {"success": False,
                "error_msg": (resp_json.get("errorMsg") or resp_json.get("error_msg")
                              or "客户订单接口返回失败"),
                "orders": [], "total": 0}
    result = resp_json.get("result") or {}
    raw_orders = result.get("orders") or []
    orders = [normalize_customer_order(it) for it in raw_orders]
    total = result.get("total", len(orders))
    return {"success": True, "orders": orders, "total": total, "returned": len(orders)}


async def post_customer_orders(page, uid: str, pageSize: int = 10) -> Dict:
    """在已登录的商家浏览器 context 内，调用 userAllOrder 拉取该买家在本店的订单。

    page 必须处于带商家 cookie 的 context（page.request 自动携带 context cookie）。
    返回 parse_customer_orders 的结构。
    """
    try:
        resp = await page.request.post(
            CUSTOMER_ORDER_API,
            data={"uid": str(uid), "pageSize": pageSize},
        )
        try:
            j = await resp.json()
        except Exception:
            text = await resp.text()
            return {"success": False, "error_msg": "客户订单接口响应解析失败: " + text[:200],
                    "orders": []}
        return parse_customer_orders(j)
    except Exception as e:
        return {"success": False, "error_msg": f"客户订单请求异常: {type(e).__name__}", "orders": []}


async def fetch_customer_orders(name: str, password: str = "", uid: str = "",
                                headless: bool = True) -> Dict:
    """独立运行入口：自己启动/关闭浏览器（CLI 与单测用）。

    用商家账号登录态，按买家 uid 调 userAllOrder 拉取其在本店的订单。
    """
    from playwright.async_api import async_playwright
    from Channel.pinduoduo.pdd_login import PDDLogin

    if not uid:
        return {"success": True, "orders": [], "total": 0, "error_msg": "缺少买家 uid"}
    pdd = PDDLogin(name=name, password=password)
    ud = str(pdd._profile_dir())
    async with async_playwright() as pw:
        ctx = await pdd._launch_context(pw, ud, headless=headless)
        try:
            page = await ctx.new_page()
            await page.goto(CHAT_PAGE, wait_until="domcontentloaded", timeout=30000)
            if "login" in (page.url or "").lower():
                return {"success": False, "login_expired": True,
                        "error_msg": "会话已过期，需要重新登录", "orders": []}
            return await post_customer_orders(page, uid)
        finally:
            try:
                await ctx.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 浏览器捕获 + 重放（anti-content 来自 PDD 自己的请求）
# ---------------------------------------------------------------------------

# 注：订单查询改用 page.route 在请求发出前就地注入 buyerId（见 _capture_and_replay），
# 复用页面自身生成的合法 anti-content，不再需要 requests 重放，规避 40002 反爬。


def parse_response(resp_json: Optional[Dict], buyer_uid: str, mobile: str, nick: str) -> Dict:
    """把 recentOrderList 响应解析为统一结构。"""
    if not resp_json or not (resp_json.get("success") or "result" in resp_json):
        return {
            "success": False,
            "error_msg": (resp_json or {}).get("error_msg")
            or (resp_json or {}).get("errorMsg")
            or "订单接口返回失败",
            "orders": [],
            "total": 0,
        }
    raw_items = (resp_json.get("result") or {}).get("pageItems") or []
    total = (resp_json.get("result") or {}).get("totalItemNum", len(raw_items))
    orders = filter_orders(raw_items, buyer_uid=buyer_uid, mobile=mobile, nick=nick)
    return {"success": True, "total": total, "returned": len(orders), "orders": orders}


# 哨兵买家 id：一个必定不存在的买家，用于检测「限流导致 buyerId 注入被忽略、
# 返回全店而非过滤结果」的误过滤场景（避免把别的买家订单泄漏给当前会话买家）。
_SENTINEL_BUYER_ID = "999999999999"


async def _capture_and_replay(
    ctx,
    buyer_uid: str = "",
    mobile: str = "",
    nick: str = "",
    days: int = 90,
    headless: bool = True,
) -> Dict:
    """在已登录的浏览器 context 内捕获并解析订单（多买家精确归因，防误过滤）。

    返回结构额外带 `attribution` 字段：
      - "buyer"  ：经 buyerId 服务端精确过滤（多买家精确归因，已校验过滤生效）
      - "shop_wide"：未给定买家时返回全店订单
      - "filtered_failed"：忽略 buyerId 导致误过滤，已安全拒绝（不返回任何订单）
      - 若登录过期：返回 {"success": False, "login_expired": True, ...}

    多买家精确归因实现（⚠️ 已证伪，见文件头说明）：
      原设计用 page.route 在 orderType=1 请求「发出前」注入 buyerId，期望服务端
      按 buyerId 精确返回该买家订单。但 2026-08-11 真机验证：recentOrderList **不认
      buyerId**（原生请求无此字段，注入后被忽略，始终返回全店）。故该注入+哨兵机制
      建立在错误前提上——哨兵（假买家）必定也返回全店 → sentinel_total 恒 >0 →
      恒触发 filtered_failed → 订单查询永远安全拒答（拿不到任何订单）。

    防误过滤（安全）：设计意图是「PDD 忽略 buyerId 返回全店 → 哨兵检测后拒答
      防泄漏」。但实测 buyerId 在**任何**时候都被忽略（并非仅限流时），所以哨兵**恒定**
      触发；这意味着当前实现下，凡是给定 buyer_uid 的查询都会进入 filtered_failed，
      即「多买家店铺里订单查询永远不可用」。要真正可用，必须改用别的归因维度
      （例如让买家提供 order_sn 后在全店列表里精确匹配，或接入支持 buyer 维度的
      专用接口），而不能依赖 buyerId。
    """
    page = await ctx.new_page()
    buyer_uid = str(buyer_uid or "").strip()
    try:
        # 路由处理器：把「当前要注入的 buyerId」写进列表请求体（orderType=1）。
        # current_bid 在两次导航间切换（真实 id / 哨兵 id）。
        current_bid = {"v": buyer_uid}
        if buyer_uid:
            async def route_handler(route):
                req = route.request
                try:
                    body = json.loads(req.post_data or "{}")
                except Exception:
                    body = {}
                if body.get("orderType") == 1:
                    # [DEBUG] 打印 PDD 原始请求字段，确认是否存在买家过滤参数及其真实名称
                    logger.warning("[DEBUG] recentOrderList 原始请求字段=%s" % list(body.keys()))
                    body = dict(body)
                    if current_bid["v"]:
                        body["buyerId"] = current_bid["v"]
                    body["pageSize"] = 50
                    logger.warning(
                        "[DEBUG] 注入后含 buyerId=%s, 当前 bid=%s" % (("buyerId" in body), current_bid["v"])
                    )
                    await route.continue_(post_data=json.dumps(body, ensure_ascii=False))
                else:
                    await route.continue_()

            await page.route("**/recentOrderList", route_handler)

        # 导航次数：给定买家时做两次（真实 id + 哨兵 id），用于检测误过滤
        passes = [buyer_uid] if buyer_uid else [None]
        if buyer_uid:
            passes.append(_SENTINEL_BUYER_ID)

        pass_results: Dict[str, Optional[Dict]] = {}  # bid -> best list response
        for idx, bid in enumerate(passes):
            current_bid["v"] = bid
            captured: List[Dict] = []
            async def on_resp(r):
                if "recentOrderList" not in r.url:
                    return
                try:
                    j = await r.json()
                except Exception:
                    return
                # 仅把「列表」请求（orderType=1）的响应视为订单数据；orderType=0
                # 是计数请求，其 pageItems 仅含 1 条且未经 buyerId 过滤，会污染结果。
                try:
                    req_body = json.loads(r.request.post_data or "{}")
                    is_list = req_body.get("orderType") == 1
                except Exception:
                    is_list = False
                items = (j.get("result") or {}).get("pageItems") or [] if (j.get("success") and is_list) else []
                captured.append({"json": j, "items": items})
                # [DEBUG] 列表响应：总单数 + 订单项是否含买家 uid 字段（决定能否客户端兜底过滤）
                if is_list and items:
                    bid_keys = [k for k in ("buyerId", "customerId", "userId", "uid",
                                            "customerUid", "buyer_id", "customer_id")
                                if k in items[0]]
                    logger.warning(
                        "[DEBUG] 响应 totalItemNum=%s, 订单项含买家字段=%s" % ((j.get("result") or {}).get("totalItemNum"), bid_keys)
                    )
            page.on("response", on_resp)
            await page.goto(ORDER_PAGE, wait_until="domcontentloaded", timeout=30000)
            if "login" in (page.url or "").lower():
                return {"success": False, "error_msg": "会话已过期，需要重新登录",
                        "orders": [], "total": 0, "login_expired": True}
            for _ in range(40):
                if any(c["items"] for c in captured):
                    break
                await asyncio.sleep(0.5)
            best = None
            for c in captured:
                if c["items"]:
                    if best is None or len(c["items"]) > len(best["items"]):
                        best = c
            pass_results[bid] = best
            page.remove_listener("response", on_resp)
            # 两次导航间稍作退避，降低连发限流概率
            if idx < len(passes) - 1:
                await asyncio.sleep(2)

        # 无买家：直接返回全店（passes=[None]）
        if not buyer_uid:
            best = pass_results.get(None)
            if best:
                result = parse_response(best["json"], buyer_uid, mobile, nick)
                result["attribution"] = "shop_wide"
                return result
            return {"success": True, "total": 0, "returned": 0, "orders": [], "attribution": "shop_wide"}

        # 给定买家：校验哨兵（误过滤检测，限流时统一安全拒绝）
        real_best = pass_results.get(buyer_uid)
        sentinel_best = pass_results.get(_SENTINEL_BUYER_ID)
        return _decide_attribution(real_best, sentinel_best, buyer_uid, mobile, nick)

    finally:
        try:
            await page.close()
        except Exception:
            pass


def _decide_attribution(real_best, sentinel_best, buyer_uid, mobile, nick) -> Dict:
    """根据真实买家与哨兵的捕获结果决定归因（纯函数，可单测）。

    核心安全原则：只要哨兵（不存在的买家）也返回 >0 单，就说明 buyerId 过滤
    未生效（PDD 忽略 buyerId、返回全店）。此时无论本店是单订单
    还是多订单，一律安全拒绝（filtered_failed），绝不泄漏他人订单。不为「单订单
    店铺」开特例放行——那样既增加逻辑复杂度，也让正常过滤路径难以被单账号单订单测试覆盖。

    - 哨兵返回 0 → 过滤真实生效，返回真实买家订单（attribution=buyer）。
    - 真实买家过滤后 0 单 → 仍属精确归因（attribution=buyer，空结果）。
    """
    sentinel_total = (
        (sentinel_best["json"].get("result") or {}).get("totalItemNum", 0)
        if sentinel_best else 0
    )

    if sentinel_total > 0:
        # PDD 忽略 buyerId 导致过滤失效（误过滤）：宁可拒答也绝不泄漏他人订单。
        # 无论单订单 / 多订单店铺，统一安全拒绝，简化逻辑、降低误判风险。
        return {"success": True, "total": 0, "returned": 0, "orders": [],
                "attribution": "filtered_failed",
                "error_msg": "订单查询暂时不可用，请稍后再试"}

    # 哨兵返回 0 → 过滤真实生效，返回真实买家订单
    if real_best:
        result = parse_response(real_best["json"], buyer_uid, mobile, nick)
        result["attribution"] = "buyer"
        return result

    # 真实买家过滤后确实 0 单（该买家在本店无订单）：仍属精确归因
    return {"success": True, "total": 0, "returned": 0, "orders": [], "attribution": "buyer"}


async def read_session_orders(
    name: str,
    password: str = "",
    buyer_uid: str = "",
    mobile: str = "",
    nick: str = "",
    days: int = 90,
    headless: bool = True,
) -> Dict:
    """独立运行入口：自己启动/关闭浏览器（CLI 与单测用）。

    流程：加载订单页 → 捕获列表请求（合法 anti-content + 正确时间窗）→
    按 buyerId 精确过滤重放 → 解析；若检测到登录过期则自动重登后重试一次。
    """
    from playwright.async_api import async_playwright
    from Channel.pinduoduo.pdd_login import PDDLogin

    pdd = PDDLogin(name=name, password=password)
    ud = str(pdd._profile_dir())

    async with async_playwright() as pw:
        ctx = await pdd._launch_context(pw, ud, headless=headless)
        try:
            result = await _capture_and_replay(
                ctx, buyer_uid=buyer_uid, mobile=mobile, nick=nick, days=days, headless=headless
            )
            if result.get("login_expired"):
                print("[会话过期] 检测到跳登录页，尝试用密码重新登录…")
                await ctx.close()
                await pdd.login(headless=headless)
                ctx = await pdd._launch_context(pw, ud, headless=headless)
                result = await _capture_and_replay(
                    ctx, buyer_uid=buyer_uid, mobile=mobile, nick=nick, days=days, headless=headless
                )
            return result
        finally:
            try:
                await ctx.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _self_test() -> int:
    sample = [
        {"order_sn": "260811-330091805023251", "goods_name": "保温杯", "goods_number": 1,
         "spec": "红银", "order_status": 1, "order_status_str": "待发货", "shipping_status": 0,
         "nickname": "啊***", "order_amount": 2800, "tracking_number": ""},
        {"order_sn": "260812-xxx", "goods_name": "杯子", "order_status": 2, "order_status_str": "已发货",
         "shipping_status": 1, "nickname": "别人", "order_amount": 500, "express_delivery": "中通快递",
         "tracking_number": "ZT123456"},
    ]
    n0 = normalize_order(sample[0])
    assert n0["order_sn"] == "260811-330091805023251", n0
    assert n0["quantity"] == 1, n0
    assert n0["pay_amount"] == 28.0, n0  # 2800 分 = 28 元
    assert n0["order_status_desc"] == "待发货", n0
    assert n0["shipping_status_desc"] == "未发货", n0
    assert n0["buyer_nick"] == "啊***", n0

    n1 = normalize_order(sample[1])
    assert n1["shipping_status_desc"] == "已发货", n1
    assert n1["logistics_company"] == "中通快递", n1
    assert n1["logistics_sn"] == "ZT123456", n1

    f = filter_orders(sample, nick="啊")
    assert len(f) == 1 and f[0]["order_sn"] == "260811-330091805023251", f
    f2 = filter_orders(sample)
    assert len(f2) == 2, f2

    # --- 多买家归因决策（_decide_attribution）单测 ---
    order1 = {"order_sn": "260811-330091805023251", "goods_name": "保温杯", "order_status": 1,
              "order_status_str": "待发货", "shipping_status": 0, "nickname": "啊***",
              "order_amount": 2800}
    order2 = {"order_sn": "260812-OTHER", "goods_name": "别家杯子", "order_status": 2,
              "order_status_str": "已发货", "shipping_status": 1, "nickname": "别人",
              "order_amount": 500}

    def _best(*orders):
        return {"json": {"success": True, "result": {"pageItems": list(orders),
                                                      "totalItemNum": len(orders)}},
                "items": list(orders)}

    # 1) 正常：真实买家有 1 单、哨兵 0 单 → 精确归因返回该单
    r = _decide_attribution(_best(order1), _best(), "4239748275", "", "")
    assert r["attribution"] == "buyer" and len(r["orders"]) == 1, r
    assert r["orders"][0]["order_sn"] == "260811-330091805023251", r

    # 1b) 正常：真实买家自身买了 2 单、哨兵 0 单（无限流）→ 精确归因返回这 2 单。
    #     证明「买家买多单」在正常工作时不会被哨兵误杀（哨兵是假买家，正常恒为 0）。
    r1b = _decide_attribution(_best(order1, order2), _best(), "4239748275", "", "")
    assert r1b["attribution"] == "buyer" and len(r1b["orders"]) == 2, r1b
    assert {o["order_sn"] for o in r1b["orders"]} == {"260811-330091805023251", "260812-OTHER"}, r1b

    # 2) 单订单店 + 限流（真实与哨兵返回完全相同的同一单）→ 统一安全拒绝（不再特例放行）
    r2 = _decide_attribution(_best(order1), _best(order1), "4239748275", "", "")
    assert r2["attribution"] == "filtered_failed" and len(r2["orders"]) == 0, r2

    # 3) 多订单店 + 限流（真实与哨兵返回相同多单）→ 宁可拒答也不泄漏
    r3 = _decide_attribution(_best(order1, order2), _best(order1, order2), "4239748275", "", "")
    assert r3["attribution"] == "filtered_failed" and len(r3["orders"]) == 0, r3

    # 4) 真实买家与哨兵返回集合不同（理论限流异常）→ 安全拒绝
    r4 = _decide_attribution(_best(order1), _best(order2), "4239748275", "", "")
    assert r4["attribution"] == "filtered_failed", r4

    # 5) 该买家本店确实 0 单（哨兵 0）→ 精确归因空结果
    r5 = _decide_attribution(None, _best(), "4239748275", "", "")
    assert r5["attribution"] == "buyer" and len(r5["orders"]) == 0, r5

    print("[self-test] 解析/字段映射/过滤/归因决策 全部通过")
    return 0


def _format(orders: List[Dict]) -> str:
    lines = []
    for i, o in enumerate(orders, 1):
        lines.append(f"--- 订单 #{i} ---")
        lines.append(f"订单号: {o['order_sn']}")
        if o["buyer_nick"]:
            lines.append(f"买家: {o['buyer_nick']}")
        lines.append(f"商品: {o['goods_name']}")
        if o["spec"]:
            lines.append(f"规格: {o['spec']}")
        if o["quantity"]:
            lines.append(f"数量: {o['quantity']}")
        lines.append(f"订单状态: {o['order_status_desc']}")
        lines.append(f"物流状态: {o['shipping_status_desc']}")
        if o["logistics_company"]:
            lines.append(f"物流公司: {o['logistics_company']}")
        if o["logistics_sn"]:
            lines.append(f"运单号: {o['logistics_sn']}")
        if o["pay_amount"] is not None:
            lines.append(f"金额: ¥{o['pay_amount']}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="商家侧会话订单读取器")
    ap.add_argument("--name", default=os.environ.get("PDD_NAME", "小峰陪你聊聊"), help="商家账号")
    ap.add_argument("--password", default=os.environ.get("PDD_PWD", ""), help="商家密码（profile 已登录可留空）")
    ap.add_argument("--buyer-uid", default="", help="会话内买家 from_uid（服务端过滤）")
    ap.add_argument("--mobile", default="", help="买家手机号过滤")
    ap.add_argument("--nick", default="", help="买家昵称过滤")
    ap.add_argument("--days", type=int, default=90, help="查询最近 N 天（受接口上限约 90 天）")
    ap.add_argument("--out", default="", help="结果输出 JSON 路径")
    ap.add_argument("--no-headless", action="store_true", help="显示浏览器窗口（调试用）")
    ap.add_argument("--self-test", action="store_true", help="仅跑解析器单测")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    try:
        result = asyncio.run(read_session_orders(
            name=args.name,
            password=args.password,
            buyer_uid=args.buyer_uid,
            mobile=args.mobile,
            nick=args.nick,
            days=args.days,
            headless=not args.no_headless,
        ))
    except Exception as e:
        print(f"[错误] {type(e).__name__}: {e}")
        return 1

    if not result.get("success"):
        print(f"[失败] {result.get('error_msg')}")
        return 1

    print(f"[成功] 接口返回总数={result.get('total')} 命中(过滤后)={result.get('returned')} "
          f"归因={result.get('attribution')}")
    print(_format(result.get("orders", [])))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[已写出] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
