"""
查询订单状态/物流信息工具

当客户询问订单状态、物流进度、发货情况时使用。
实现说明（2026-08-11 真机纠正）：
- 拼多多订单接口（recentOrderList）**不认 buyerId、订单项也不含买家 uid**（已扫
  全部字段证实），无法按会话买家自动过滤。故本工具改为「拉全店订单 + 按 order_sn
  精确匹配」：既能安全定位订单（不泄漏他人订单），又真正可用。
- 调用方（LLM）应尽量从对话提取买家提供的订单号传入 order_sn；无订单号时工具会
  请买家提供，绝不会把全店列表甩给当前买家。
- 拼多多订单接口对静态 anti-content 返回 40002 反爬，必须用浏览器内「捕获 PDD
  自身请求（含合法 anti-content）后重放」的方式；复用商家侧会话订单读取器
  read_session_orders（见同目录 order_browser_pool / read_session_orders）。
"""
import os
import random
import sys
from typing import Optional, Union

from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom.tool_decorator import agent_tool
from utils.logger_loguru import get_logger

# 仓库根目录（read_session_orders.py / order_browser_pool.py 位于 Agent/CustomerAgent/tools/）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from .order_browser_pool import get_order_browser_pool  # noqa: E402

logger = get_logger("QueryOrderStatusTool")

# 订单查询失败时（接口未返回 / 平台暂时取不到数据 / 程序异常）统一使用的真人化兜底话术。
# 设计要点：
# - 全部为「口语化、带情绪、给可执行动作（去 App 看）」的真人客服语气，绝不出现
#   「异常 / 系统 / 服务器 / 人工客服 / 请稍后重试 / 暂时繁忙」这类机器破绽词；
# - 提供多个自然变体，每次随机挑一个且与上一条不同，避免同一模板被所有买家反复命中
#   （千人一面也是明显的非真人信号）；
# - 这些话术会经 ai_handler 的「原样转告」指令透传给 LLM，再由 LLM 转发给客户，
#   因此本身就要写成可直接对客户说的成品句。
_ORDER_FAILURE_VARIANTS = [
    "哎呀，我这会儿没刷出您的订单信息，后台可能有点卡。您方便的话先去拼多多App里"
    "「我的订单」看一眼物流哈，我这边也帮您留意着～",
    "不好意思哈，刚试着帮您查订单，结果页面一直没刷出来。您先到拼多多App"
    "「个人中心-我的订单」里看看发货和物流，我待会儿再帮您确认下",
    "刚帮您查了下，订单信息这会儿没同步出来，估计是后台慢了点。您先去App里"
    "「我的订单」核对一下，有啥不对的随时喊我就行",
    "亲，您订单信息我这会儿暂时没查出来呢，可能是后台稍微有点延迟。您去拼多多App"
    "「我的订单」看一下发货和物流进度，我随时都在哈",
    "抱歉哈，订单这会儿没查出来，页面一直转圈。您方便的话到拼多多App里"
    "「我的订单」看下物流，其他问题我也接着帮您～",
]
_failure_last_idx = -1


def get_order_failure_msg() -> str:
    """随机返回一条真人化兜底话术，且与上一次不同，避免千篇一律。"""
    global _failure_last_idx
    n = len(_ORDER_FAILURE_VARIANTS)
    if n <= 1:
        return _ORDER_FAILURE_VARIANTS[0]
    idx = random.randrange(n - 1)
    if idx >= _failure_last_idx:
        idx += 1
    _failure_last_idx = idx
    return _ORDER_FAILURE_VARIANTS[idx]


class QueryOrderStatusParams(BaseModel):
    """查询订单状态参数（身份字段由 dependencies 自动注入）"""
    channel_type: Optional[str] = Field(default=None, description="渠道类型（受信任注入）")
    shop_id: Optional[Union[str, int]] = Field(default=None, description="店铺ID")
    user_id: Optional[Union[str, int]] = Field(default=None, description="商家账号ID")
    recipient_uid: Optional[str] = Field(
        default=None,
        description="当前对话买家UID（来自会话上下文，即拼多多买家 uid）。本工具用它调用"
                    "商家客服「客户订单」接口(userAllOrder)直接拉取该买家在本店的订单——"
                    "如同真人客服在后台看到「这个客户的订单」，无需用户报订单号，也不会"
                    "把其他买家的订单泄漏给当前会话买家。"
    )
    order_sn: Optional[str] = Field(
        default=None,
        description="可选。买家主动提供的订单号（形如 260811-xxxx）。传入时仅在本买家"
                    "订单内精确匹配（多单时定位某一单）；不传则返回该买家在本店的全部订单。"
    )
    recipient_nick: Optional[str] = Field(
        default=None,
        description="当前对话买家在平台展示的昵称（受信任注入，来自会话上下文，"
                    "LLM 无需提供）。抖音渠道用它精确反查该买家在本店的订单，"
                    "无需买家报订单号。"
    )
    days: Optional[int] = Field(
        default=90,
        description="查询最近N天的订单，默认90天"
    )


def _get_merchant_creds(shop_id, user_id) -> tuple:
    """拿到商家账号名与密码，用于定位浏览器持久化 profile 并在会话过期时自动重登。"""
    try:
        from database import db_manager
        acct = db_manager.get_account("pinduoduo", str(shop_id), str(user_id))
        if acct and acct.get("username"):
            return acct["username"], acct.get("password") or ""
    except Exception:
        pass
    return os.environ.get("PDD_NAME", "小峰陪你聊聊"), os.environ.get("PDD_PWD", "")


def _query_douyin_orders(params) -> str:
    """
    抖音订单查询：真实抖店后台订单接口（2026-11-12 逆向实测）。

    数据源：fxg.jinritemai.com/api/order/searchlist（纯 HTTP + cookie，
    无需 a_bogus/msToken）。查询三级策略：
      1. order_sn 订单号精确查（买家主动报单号）
      2. buyer_nick 真实昵称反查（服务端精确匹配；掩码/前缀查不到，
         实测见 Channel/douyin/ORDER_API.md §3.1）
      3. 消息历史提取订单号 → 按单号补详情（兜底）
    - 有订单 → 格式化返回（订单号/状态/物流，格式与拼多多一致）
    - 无订单 → 返回引导话术
    - 不调拼多多 API/浏览器（渠道隔离）
    """
    try:
        from Channel.douyin.utils.API.order_query import OrderQuery as DouyinOrderQuery

        oq = DouyinOrderQuery(shop_id=params.shop_id, user_id=params.user_id)
        order_sn = str(params.order_sn or "").strip()
        buyer_nick = str(params.recipient_nick or "").strip()
        days = params.days or 90

        orders = []
        # 1) 买家主动报了订单号 → 单号精确查
        if order_sn:
            detail = oq.get_order_detail(order_sn)
            if detail is not None:
                orders = [detail]
        # 2) 有真实昵称 → 昵称反查（searchlist buyer= 精确匹配）
        elif buyer_nick:
            result = oq.get_buyer_orders(buyer_nick)
            if result.get("success"):
                orders = result.get("orders") or []
        # 3) 兜底：消息历史订单号 + 真实接口补详情
        if not orders:
            result = oq.get_recent_orders(
                buyer_uid=params.recipient_uid,
                days=days,
            )
            if result.get("success"):
                orders = result.get("orders") or []

        if not orders:
            logger.info(
                f"抖音订单查询: 无订单 shop_id={params.shop_id}, "
                f"buyer_nick={buyer_nick!r}, order_sn={order_sn!r}"
            )
            return (
                "暂时没查到您在本店的订单记录哦。如果您刚下单，可能稍有延迟，"
                "稍等片刻我再帮您看看；也可以到抖店App「我的订单」里核对一下，"
                "或者把订单号发我帮您查～"
            )

        # order_sn 指定时只返回匹配的那一单
        if order_sn and len(orders) > 1:
            matched = []
            for o in orders:
                if str(o.get("order_sn") or "") == order_sn:
                    matched.append(o)
            if matched:
                orders = matched

        logger.info(
            f"抖音订单查询成功: shop_id={params.shop_id}, "
            f"total={len(orders)}, source="
            f"{'order_sn' if order_sn else ('buyer_nick' if buyer_nick else 'history')}"
        )
        return _format_output(orders, is_filtered=True)

    except Exception as e:
        logger.error(
            f"抖音订单查询异常: error_type={type(e).__name__}, "
            f"shop_id={params.shop_id}"
        )
        return (
            "抱歉，我这会儿查订单时遇到点小问题。您可以到抖店App"
            "「我的订单」里查看订单和物流进度，或者把订单号发我，我帮您确认～"
        )


@agent_tool(
    name="query_order_status",
    description=(
        "查询订单状态与物流信息。当客户询问'我的订单在哪里''发货了吗''物流到哪了'"
        "'订单状态'等时使用。本工具依据会话买家的 uid 直接调取该买家在本店的订单"
        "（与真人客服在后台看到『这个客户的订单』一致），无需向用户索要订单号。"
        "若用户主动给了订单号，会在其订单内精确匹配定位某一单；否则返回其全部订单。"
        "返回订单号、商品名、订单状态、物流状态"
        "（未发货/已发货/已揽收/运输中/派送中/已签收）、物流公司及运单号。"
    ),
    param_model=QueryOrderStatusParams,
)
def query_order_status(params: QueryOrderStatusParams) -> str:
    """读取当前会话买家的订单与物流信息（按买家 uid 直接归因，无需订单号）。"""
    try:
        if not params.shop_id or not params.user_id:
            return "查询订单失败：缺少必要的 shop_id 或 user_id 参数"

        from Agent.CustomerAgent.tools.channel_router import normalize_channel

        # 抖音渠道：用抖音 OrderQuery 从消息历史提取订单号（不调拼多多浏览器抓单）
        if normalize_channel(params.channel_type) == "douyin":
            return _query_douyin_orders(params)

        # 真实商家客服面板：凭会话买家的 uid 直接看到「该客户在本店的订单」，
        # 无需用户报订单号，也不会把其他买家的订单泄漏给当前会话买家。
        uid = str(params.recipient_uid or "").strip()
        if not uid:
            logger.warning("query_order_status 缺少 recipient_uid，无法按买家归因")
            return ("抱歉，暂时无法定位您的订单呢。您可以在拼多多App"
                    "「个人中心-我的订单」里查看订单和物流进度，或把订单号发我帮您查。")

        # 主路径：纯 HTTP 接口（2026-11-12 逆向 latitude/order/userAllOrder，
        # 纯 cookie 即可，实测无 anti-content 校验）——不拉浏览器，秒级返回
        result = None
        try:
            from Channel.pinduoduo.utils.API.order_query import (
                OrderQuery as PddOrderQuery,
            )
            oq = PddOrderQuery(shop_id=params.shop_id, user_id=params.user_id)
            result = oq.get_buyer_orders_by_uid(uid)
        except Exception as api_exc:
            logger.warning(
                f"订单直连接口异常，降级浏览器池: "
                f"error_type={type(api_exc).__name__}"
            )
            result = {"success": False, "orders": [], "total": 0}

        # 兜底：直连接口失败（cookies 失效等）→ 老的浏览器池路径
        if not result.get("success"):
            name, pwd = _get_merchant_creds(params.shop_id, params.user_id)
            pool = get_order_browser_pool()
            result = pool.fetch_customer_orders(
                name=name, password=pwd, uid=uid, headless=True
            )

        if not result.get("success"):
            logger.error(f"客户订单查询失败: {result.get('error_msg')}")
            # 接口未返回订单数据时，用真人化话术转告，不暴露技术细节（error_msg）
            return get_order_failure_msg()

        orders = result.get("orders", [])  # 该买家在本店的订单（已按 uid 归因，不会含他人）

        if not orders:
            # 该买家在本店确实没有订单（如刚进店咨询、或订单在别的店）
            return ("未查询到您在本店的订单记录哦。如果您刚下单，可能稍有延迟，"
                    "稍等片刻我再帮您看看；也可以到拼多多App「个人中心-我的订单」里核对一下。")

        # 若买家主动给了订单号，仅在本买家的订单里精确匹配（多单时定位某一单）；
        # 不给则返回该买家全部订单（都是他自己的，不存在跨买家泄漏）。
        if params.order_sn:
            matched = _match_orders(orders, params.order_sn)
            if matched:
                logger.info(f"订单号匹配成功: {params.order_sn} → {len(matched)} 条")
                return _format_output(matched, is_filtered=True, query_sn=params.order_sn)
            logger.info(f"订单号未在本买家订单内匹配: {params.order_sn}")
            return (f"在您的订单里没找到订单号 '{params.order_sn}' 哦，"
                    f"您本店当前订单如下：\n" + _format_output(orders, is_filtered=True))

        return _format_output(orders, is_filtered=True)

    except Exception as e:
        logger.error(f"query_order_status 工具异常: error_type={type(e).__name__}, msg={e}")
        return get_order_failure_msg()


def _match_orders(orders, sn):
    """按订单号模糊匹配（避免空串被任何字符串包含导致的误匹配）"""
    sn_clean = str(sn).strip()
    if not sn_clean:
        return []
    matched = []
    for o in orders:
        osn = o.get("order_sn") or ""
        oseq = o.get("order_sequence_no") or ""
        if (
            (sn_clean and sn_clean in osn)
            or (sn_clean and sn_clean in oseq)
            or (osn and osn in sn_clean)
            or (oseq and oseq in sn_clean)
        ):
            matched.append(o)
    return matched


def _format_output(orders, *, is_filtered=False, query_sn=None):
    """格式化订单列表为 LLM 可读文本（含未发货/已发货/已揽收等状态）"""

    def _s(v, limit=200):
        t = str(v or "")
        t = "".join(ch if ord(ch) >= 32 else " " for ch in t)
        return (
            t.replace("<", "＜").replace(">", "＞")
            .replace("[", "［").replace("]", "］")[:limit]
        )

    lines = ["[untrusted_order_data]"]
    if query_sn and not is_filtered:
        lines.append(f"查询订单号: {_s(query_sn)}（以下为近期订单供参考）")
    elif is_filtered:
        # 已退款/已关闭的订单不展示给客户（客户不需要知道已退的单）
        active_orders = [o for o in orders if o.get("needs_shipping", True)]
        if not active_orders:
            # 该买家所有订单均已退款/关闭，如实告知即可
            lines.append("该客户在本店暂无有效在途订单（历史订单已退款/关闭）。")
        else:
            lines.append(f"您的订单（共 {len(active_orders)} 条）：")

    lines.append("")

    # 仅展示有效在途订单，已退款/已关闭的订单不输出
    display_orders = [o for o in orders if o.get("needs_shipping", True)]
    for i, o in enumerate(display_orders, 1):
        lines.append(f"--- 订单 #{i} ---")
        lines.append(f"订单号: {_s(o.get('order_sn'))}")
        seq = o.get("order_sequence_no")
        if seq:
            lines.append(f"交易序号: {_s(seq)}")
        lines.append(f"商品: {_s(o.get('goods_name'))}")
        spec = o.get("spec")
        if spec:
            lines.append(f"规格: {_s(spec)}")
        qty = o.get("quantity")
        if qty:
            lines.append(f"数量: {qty}")

        status_desc = o.get("order_status_desc", "")
        ship_desc = o.get("shipping_status_desc", "")
        if status_desc:
            lines.append(f"订单状态: {status_desc}")
        if ship_desc:
            lines.append(f"物流状态: {ship_desc}")

        refund_desc = o.get("refund_status_desc")
        if refund_desc:
            lines.append(f"退款/售后: {refund_desc}（该订单已无需发货，请勿劝用户等待物流）")

        lc = o.get("logistics_company")
        ls = o.get("logistics_sn")
        if lc:
            lines.append(f"物流公司: {_s(lc)}")
        if ls:
            lines.append(f"运单号: {_s(ls)}")

        as_status = o.get("after_sales_status")
        if as_status and int(as_status or 0) > 0:
            lines.append(f"售后状态: 有售后进行中")

        lines.append("")

    lines.append("[/untrusted_order_data]")
    return "\n".join(lines)
