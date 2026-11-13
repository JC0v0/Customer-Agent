"""
抖音订单查询 API（对齐拼多多 OrderQuery 接口签名）。

【数据源：抖店商家后台订单列表接口，2026-11-12 逆向并实测验证】

主接口：
    GET https://fxg.jinritemai.com/api/order/searchlist
    鉴权: fxg.jinritemai.com 域 cookie（BaseRequest 携带登录时保存的完整浏览器 cookies）
    已实测确认：纯 HTTP + cookie 即可调用，无需 a_bogus / msToken 风控签名。

    实测通过的参数：
        page(0起) / pageSize / order_by=create_time / order=desc / tab=all
        order_id=<订单号>          按订单号精确查询（实测返回 1 条）
        buyer=<买家真实昵称>       按买家筛选（服务端精确匹配，2026-11-12 实测：
                                   掩码昵称「不*」与前缀「不」均返回 total=0，
                                   仅完整真实昵称可查到订单）
        source=shop_order_view_upgrade / appid=1 / _bid=ffa_order / aid=4272

    响应外壳：{"st": 0, "msg": "", "code": 0, "data": [订单行...], "total": n}
        st 为数字 0 成功（与飞鸽 pigeon 域接口的字符串 "0" 不同）。

    订单行字段（真实订单实测）：
        shop_order_id           订单号（order_id 字段实测为空串，勿用）
        order_status            状态码 int（实测 2=待发货）
        order_status_info.order_status_text   状态文案（"待发货"）
        product_item[]          商品子单数组：
            product_id / product_name / img
            sku_spec: [{name, value}] 规格
            combo_num 数量 / pay_amount / total_amount / combo_amount（单位：分）
        pay_amount / actual_pay_amount / post_amount / total_price   金额，单位分
        create_time_str         "2026-11-12 09:05:45"（已格式化）
        pay_time / exp_ship_time / create_time   int 秒级时间戳
        order_status_info.dead_line_time       发货截止时间戳
        receiver_info.post_receiver / post_tel      收件人 / 电话（脱敏）
        receiver_info.post_addr {province, city, town, street}.name + detail
        user_nickname           买家昵称（平台脱敏后）
        buyer_words 买家备注 / remark 商家备注
        latest_logistic_info.tracking_no / company_code   物流（未发货为空）
        pay_type_desc           支付方式（"抖音支付"）

已知限制：
    1. 飞鸽域 pigeon.jinritemai.com/backstage/getOrder 实测返回
       {"code":"10001010A","msg":"当前环境存在风险"}，被风控拦截，
       需 a_bogus 签名，本实现不使用该接口。
    2. 飞鸽域 getuserbyorder 实测返回 11001「飞鸽版本过低」，同样不使用。
    3. cookie 需包含抖店后台域。登录流程抓取的是全浏览器 cookie，
       若抖店后台未登录过，订单接口返回未登录，走消息历史骨架兜底。

回退路径：
    真实接口失败（未登录 / 网络 / 订单不存在）时，回退到会话消息历史里
    正则提取订单号，仅返回订单号骨架，其余字段标注「待核实」。
"""
import re

from ..base_request import BaseRequest
from utils.logger_loguru import get_logger
from Channel.douyin.utils.log_throttle import throttled_warning, throttled_info

logger = get_logger("DouyinOrderQuery")

# 抖音订单号正则：15-25 位连续数字（抖店订单号通常 18-20 位）
_ORDER_ID_PATTERN = re.compile(r"\b(\d{15,25})\b")

# 订单状态码 → 中文（order_status_info.order_status_text 缺失时兜底）
# 实测 2=待发货；其余状态码按抖店通用枚举推断，未逐一实测
_ORDER_STATUS_MAP = {
    0: "待付款",
    1: "待发货",
    2: "待发货",
    3: "已发货",
    4: "已完成",
    5: "已关闭",
}

# 会话消息里昵称占位值（机器人/系统消息的 ext/uname 实测为「用户」）
_NICK_PLACEHOLDERS = {"用户", "", "None"}


def extract_buyer_nick_from_ext(ext):
    """
    从飞鸽消息 messageBody.ext 里提取买家真实昵称。

    2026-11-12 真实会话实测字段（买家「不喜欢说话」发「转人工」）：
        ext/nickname   买家发言消息里为真实昵称（机器人消息无此字段）
        ext/uname      买家消息同真实昵称；机器人/系统消息为「用户」

    Returns:
        str: 真实昵称（可直接传给 searchlist 的 buyer 参数）
        None: 无昵称 / 占位值（机器人、系统消息）
    """
    if not isinstance(ext, dict):
        return None
    nickname = str(ext.get("nickname") or "").strip()
    if nickname and nickname not in _NICK_PLACEHOLDERS:
        return nickname
    uname = str(ext.get("uname") or "").strip()
    if uname and uname not in _NICK_PLACEHOLDERS:
        return uname
    return None


def extract_card_order_ids(message_body):
    """
    从卡片消息里提取订单号（2026-11-12 真实会话实测）。

    实测：机器人「订单确认信息」卡片 messageBody.ext 里
        bury_order_id = 6919514772943875116（买家真实订单号）
        req_id 形如 <id>_<id>_ACK_ORDER_ADDRESS_<订单号>

    注意：req_id 里还混有卡片 id / 消息 id（同为 19 位数字），
    不能整串正则，只能按 ACK_ORDER_ADDRESS_ 标记截取。

    Returns:
        List[str]: 订单号（去重、保持出现顺序），无则空列表
    """
    order_ids = []
    if not isinstance(message_body, dict):
        return order_ids
    ext = message_body.get("ext")
    if not isinstance(ext, dict):
        return order_ids

    bury = str(ext.get("bury_order_id") or "").strip()
    if bury and bury.isdigit():
        order_ids.append(bury)

    req_id = str(ext.get("req_id") or "")
    marker = "ACK_ORDER_ADDRESS_"
    idx = req_id.find(marker)
    if idx >= 0:
        tail = req_id[idx + len(marker):]
        match = re.match(r"(\d+)", tail)
        if match:
            oid = match.group(1)
            if oid not in order_ids:
                order_ids.append(oid)
    return order_ids


_LIST_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "referer": "https://fxg.jinritemai.com/ffa/morder/order/list",
}

_FIXED_PARAMS = {
    "order_by": "create_time",
    "order": "desc",
    "tab": "all",
    "source": "shop_order_view_upgrade",
    "appid": "1",
    "_bid": "ffa_order",
    "aid": "4272",
}


def _fmt_ts(ts):
    """秒级时间戳 → 'YYYY-MM-DD HH:MM:SS'，无效返回空串。"""
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return ""
    if ts <= 0:
        return ""
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fen_to_yuan(v):
    """分 → 元（float）。无效返回 None。"""
    try:
        return round(int(v) / 100.0, 2)
    except (TypeError, ValueError):
        return None


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


class OrderQuery(BaseRequest):
    """抖音订单查询器（抖店后台 searchlist 主路径，消息历史兜底）。"""

    ORDER_LIST_URL = "https://fxg.jinritemai.com/api/order/searchlist"

    def __init__(self, shop_id: str = None, user_id: str = None, cookies=None):
        super().__init__(shop_id=shop_id, user_id=user_id, channel_name="douyin")
        if cookies:
            self.update_cookies(cookies)
        # 最近一次 _search_orders 失败原因（"" = 未失败/未调用）
        self.last_search_error = ""

    # ==================================================================
    # 对外接口
    # ==================================================================

    def get_order_detail(self, order_id):
        """
        按订单号查订单详情（真实接口实测可用）。

        Args:
            order_id (str): 店铺订单号（shop_order_id，18-20 位数字）

        Returns:
            dict: 归一化后的订单，字段与拼多多 OrderQuery 对齐
            None: 接口失败（未登录 / 网络 / 订单不存在）
        """
        order_id = str(order_id or "").strip()
        if not order_id:
            return None

        rows = self._search_orders({"order_id": order_id}, limit=1)
        if not rows:
            return None
        return self._normalize_order(rows[0])

    def get_buyer_orders(self, buyer_nick, page=0, page_size=20):
        """
        按买家昵称查订单列表（真实接口）。

        Args:
            buyer_nick (str): 买家昵称（searchlist 的 buyer 参数）

        Returns:
            dict: {"success", "total", "orders", "error_msg"}
        """
        buyer_nick = str(buyer_nick or "").strip()
        if not buyer_nick:
            return {"success": False, "error_msg": "买家昵称为空", "orders": [], "total": 0}

        rows = self._search_orders({"buyer": buyer_nick}, limit=page_size)
        if rows is None:
            # 接口失败（未登录/风控/网络），与「该买家无订单」区分开
            return {
                "success": False,
                "error_msg": self.last_search_error or "订单接口调用失败",
                "orders": [],
                "total": 0,
            }
        orders = []
        for row in rows:
            orders.append(self._normalize_order(row))
        return {
            "success": True,
            "total": len(orders),
            "orders": orders,
        }

    def get_recent_orders(self, buyer_uid=None, days=7, page=1, page_size=20):
        """
        查询近期订单列表。

        策略：先从会话消息历史提取订单号，再逐个调真实接口补全详情。
        真实接口不可用时保留订单号骨架，不返回空。

        Args:
            buyer_uid (str): 买家UID，传入时仅查该买家的消息提取订单号
            days (int): 查询最近N天的消息，默认7天（与拼多多对齐）
            page (int): 页码，默认1
            page_size (int): 每页数量，默认20

        Returns:
            dict: {"success", "total", "orders", "error_msg"}
        """
        try:
            orders = self._collect_orders(buyer_uid=buyer_uid, days=days)
            if orders is None:
                return {
                    "success": False,
                    "error_msg": "店铺未登记或消息历史查询失败",
                    "orders": [],
                    "total": 0,
                }

            total = len(orders)
            start = (page - 1) * page_size
            end = start + page_size
            page_orders = orders[start:end]

            throttled_info(
                logger, "order.ok",
                f"抖音订单查询完成: shop_id={self.shop_id}, buyer_uid={buyer_uid}, "
                f"total={total}, page={page}, returned={len(page_orders)}"
            )
            return {
                "success": True,
                "total": total,
                "orders": page_orders,
            }
        except Exception as e:
            logger.error(
                f"抖音订单查询异常: error_type={type(e).__name__}, shop_id={self.shop_id}"
            )
            return {
                "success": False,
                "error_msg": f"查询异常: {type(e).__name__}",
                "orders": [],
                "total": 0,
            }

    # ==================================================================
    # 内部实现
    # ==================================================================

    def _search_orders(self, extra_params, limit=20):
        """
        调 searchlist 拿原始订单行。

        Args:
            extra_params (dict): order_id=xxx 或 buyer=xxx 等筛选参数
            limit (int): pageSize

        Returns:
            List[dict]: 原始订单行列表（可能为空=确实无数据）
            None: 接口失败（未登录 / 网络异常 / st!=0），与「无订单」区分
        """
        params = dict(_FIXED_PARAMS)
        params["page"] = "0"
        params["pageSize"] = str(max(1, min(int(limit), 50)))
        for k, v in extra_params.items():
            if v:
                params[k] = str(v)

        try:
            result = self.get(
                self.ORDER_LIST_URL,
                params=params,
                headers=_LIST_HEADERS,
                timeout=15,
            )
        except Exception as e:
            logger.warning(
                f"抖音订单列表请求异常: params={list(extra_params.keys())}, "
                f"error_type={type(e).__name__}"
            )
            return None

        if not isinstance(result, dict):
            logger.warning("抖音订单列表无有效响应")
            return None

        # 实测外壳: {"st": 0, "data": [...], "total": n}
        if result.get("st") != 0:
            msg = result.get("msg") or result.get("message") or ""
            throttled_warning(
                logger, "order.list_fail",
                f"抖音订单列表失败: st={result.get('st')}, msg={msg}"
            )
            self.last_search_error = "st=%s %s" % (result.get("st"), msg)
            return None

        data = result.get("data")
        if not isinstance(data, list):
            # 有的分页外壳会把行放在 data.order_list，做一次兼容
            inner = data.get("order_list") if isinstance(data, dict) else None
            if isinstance(inner, list):
                return inner
            return []
        self.last_search_error = ""
        return data

    def _collect_orders(self, buyer_uid=None, days=7):
        """
        汇总订单：消息历史取订单号 -> 真实接口补详情 -> 失败留骨架。

        Returns:
            List[dict]: 订单列表，无订单返回空列表
            None: 店铺未登记等硬失败
        """
        order_ids = self._extract_order_ids(buyer_uid=buyer_uid, days=days)
        if order_ids is None:
            return None
        if not order_ids:
            logger.info(f"抖音订单查询: 消息历史中未匹配到订单号 shop_id={self.shop_id}")
            return []

        orders = []
        detail_ok = 0
        for order_id in order_ids:
            detail = self.get_order_detail(order_id)
            if detail is not None:
                orders.append(detail)
                detail_ok += 1
            else:
                orders.append(self._build_order_skeleton(order_id))

        logger.info(
            f"抖音订单查询: 订单号 {len(order_ids)} 个，"
            f"真实接口补全 {detail_ok} 个，骨架 {len(order_ids) - detail_ok} 个"
        )
        return orders

    def _extract_order_ids(self, buyer_uid=None, days=7):
        """
        从 Agent 会话历史里提取当前买家发过的订单号。

        说明：订单工具运行在 Agent 内部，买家消息在调用模型前已经写入
        ``agent_messages``。这里按同一套 conversation key 读取当前会话，
        不再依赖并不存在的 CapturedMessage 表。

        Returns:
            List[str]: 订单号（去重、保持出现顺序）
            None: 数据库不可用
        """
        try:
            from hashlib import sha256

            from Agent.CustomerAgent.custom.session_manager import AgentMessage
            from bridge.context import make_account_key
            from database import db_manager
            from sqlalchemy import select
            from datetime import datetime, timedelta

            buyer_uid = str(buyer_uid or "").strip()
            if not buyer_uid:
                return []

            account_key = make_account_key("douyin", self.shop_id, self.user_id)
            raw = f"{account_key}|customer|{buyer_uid}"
            session_id = f"conversation_{sha256(raw.encode('utf-8')).hexdigest()}"

            with db_manager.Session() as session:
                stmt = select(AgentMessage).where(
                    AgentMessage.session_id == session_id,
                    AgentMessage.role == "user",
                )
                if days and days > 0:
                    cutoff = datetime.now() - timedelta(days=days)
                    stmt = stmt.where(AgentMessage.timestamp >= cutoff)
                stmt = stmt.order_by(AgentMessage.timestamp.desc()).limit(500)

                messages = list(session.scalars(stmt))

            if not messages:
                logger.info(
                    f"抖音订单查询: 无相关消息历史 shop_id={self.shop_id}, "
                    f"buyer_uid={buyer_uid}"
                )
                return []

            seen = set()
            order_ids = []
            for msg in messages:
                content = msg.content or ""
                for match in _ORDER_ID_PATTERN.finditer(content):
                    order_id = match.group(1)
                    if order_id not in seen:
                        seen.add(order_id)
                        order_ids.append(order_id)
            return order_ids

        except ImportError as e:
            logger.warning(f"抖音订单查询: 数据库模型导入失败: {e}")
            return []
        except Exception as e:
            logger.error(
                f"抖音订单查询: 消息历史查询异常 error_type={type(e).__name__}"
            )
            return None

    def _resolve_db_shop_id(self, db_manager, Shop):
        """将本地 shop_id（如 douyin_<店铺名>）解析为 shops 表的 int id。"""
        try:
            with db_manager.Session() as session:
                shop = session.query(Shop).filter(
                    Shop.shop_id == str(self.shop_id or "")
                ).first()
                if shop:
                    return shop.id
                shop = session.query(Shop).filter(
                    Shop.shop_name.like(f"%{self.shop_id}%")
                ).first()
                if shop:
                    return shop.id
                return None
        except Exception:
            return None

    @staticmethod
    def _normalize_order(row):
        """
        把 searchlist 的订单行归一化成订单 dict。

        字段名与层级全部来自真实订单实测（见模块 docstring），
        归一化后与拼多多 OrderQuery 对齐，上层工具无需区分渠道。
        """
        status_info = row.get("order_status_info") or {}
        if not isinstance(status_info, dict):
            status_info = {}

        receiver_info = row.get("receiver_info") or {}
        if not isinstance(receiver_info, dict):
            receiver_info = {}
        addr = receiver_info.get("post_addr") or {}
        if not isinstance(addr, dict):
            addr = {}

        logistic = row.get("latest_logistic_info") or {}
        if not isinstance(logistic, dict):
            logistic = {}

        # 商品子单数组
        product_items = row.get("product_item")
        if not isinstance(product_items, list):
            product_items = []

        # 数量：优先 product_count，兜底所有子单 combo_num 之和
        quantity = row.get("product_count") or 0
        if not quantity:
            total = 0
            for it in product_items:
                try:
                    total = total + int(it.get("combo_num") or 0)
                except (TypeError, ValueError):
                    pass
            quantity = total or 1

        # 规格：sku_spec [{name, value}] → "颜色分类:米白色 / 尺码大小:L"
        first = product_items[0] if product_items else {}
        spec_parts = []
        sku_spec = first.get("sku_spec")
        if isinstance(sku_spec, list):
            for spec in sku_spec:
                if isinstance(spec, dict):
                    name = str(spec.get("name") or "")
                    value = str(spec.get("value") or "")
                    if name and value:
                        spec_parts.append(name + ":" + value)
                    elif value:
                        spec_parts.append(value)
        spec_text = " / ".join(spec_parts)

        # 状态文案
        order_status = row.get("order_status")
        status_desc = str(status_info.get("order_status_text") or "")
        if not status_desc:
            try:
                status_desc = _ORDER_STATUS_MAP.get(int(order_status), "未知状态")
            except (TypeError, ValueError):
                status_desc = "未知状态"

        refund_status_desc = _classify_refund(status_desc)
        needs_shipping = refund_status_desc not in ("已退款", "已关闭", "售后已结束")

        # 地址：{province:{name}, city:{name}, town:{name}, street:{name}, detail}
        addr_parts = []
        for key in ("province", "city", "town", "street"):
            node = addr.get(key)
            if isinstance(node, dict) and node.get("name"):
                addr_parts.append(str(node["name"]))
        detail = str(addr.get("detail") or "")
        if detail:
            addr_parts.append(detail)
        address = " ".join(addr_parts)

        order = {
            "order_sn": str(row.get("shop_order_id") or ""),
            "order_sequence_no": str(row.get("shop_order_id") or ""),
            "goods_name": str(first.get("product_name") or ""),
            "goods_id": str(first.get("product_id") or ""),
            "goods_img": str(first.get("img") or ""),
            "spec": spec_text,
            "quantity": quantity,
            "order_status": order_status,
            "order_status_desc": status_desc,
            "refund_status_desc": refund_status_desc,
            "needs_shipping": needs_shipping,
            "shipping_status_desc": (
                "无需发货（已退款/已关闭）"
                if not needs_shipping
                else (
                    "已签收"
                    if "已完成" in status_desc
                    else ("已发货" if str(logistic.get("tracking_no") or "").strip() else "未发货")
                )
            ),
            # 金额单位是分，统一转成元
            "total_amount": _fen_to_yuan(row.get("total_price") or row.get("pay_amount")),
            "pay_amount": _fen_to_yuan(row.get("pay_amount")),
            "goods_amount": _fen_to_yuan(row.get("total_goods_amount")),
            "post_amount": _fen_to_yuan(row.get("post_amount")),
            "coupon_amount": None,
            "create_time": str(row.get("create_time_str") or _fmt_ts(row.get("create_time"))),
            "pay_time": _fmt_ts(row.get("pay_time")),
            "logistics_time": _fmt_ts(row.get("logistics_time")),
            "exp_ship_time": _fmt_ts(status_info.get("dead_line_time") or row.get("exp_ship_time")),
            "receiver": str(receiver_info.get("post_receiver") or ""),
            "receiver_phone": str(receiver_info.get("post_tel") or ""),
            "address": address,
            "seller_words": str(row.get("remark") or ""),
            "buyer_words": str(row.get("buyer_words") or ""),
            "pay_type": str(row.get("pay_type_desc") or ""),
            "tracking_no": str(logistic.get("tracking_no") or ""),
            "logistics_company": str(logistic.get("company_code") or ""),
            "after_sales_status": 1 if refund_status_desc else 0,
            "buyer_nick": str(row.get("user_nickname") or ""),
            "source": "fxg.searchlist",
        }

        # 多商品订单：product_item 是子单数组，展开成一行一条便于 AI 复述
        sub_orders = []
        for it in product_items:
            if not isinstance(it, dict):
                continue
            sub_spec = []
            it_sku_spec = it.get("sku_spec")
            if isinstance(it_sku_spec, list):
                for spec in it_sku_spec:
                    if isinstance(spec, dict):
                        sub_spec.append(str(spec.get("value") or ""))
            sub_orders.append({
                "order_sn": str(it.get("item_order_id") or ""),
                "goods_name": str(it.get("product_name") or ""),
                "goods_id": str(it.get("product_id") or ""),
                "spec": " / ".join([s for s in sub_spec if s]),
                "quantity": it.get("combo_num") or 1,
                "pay_amount": _fen_to_yuan(it.get("pay_amount")),
                "order_status_desc": str(
                    (it.get("sku_order_status_info") or {}).get("order_status_text") or ""
                ),
            })
        order["sub_orders"] = sub_orders

        return order

    @staticmethod
    def _build_order_skeleton(order_id):
        """
        真实接口失败时的订单骨架。

        只保证订单号正确，其余字段标注「待核实」，
        让 AI 能说「我查到您有订单 xxx，详情请在 App 查看」而不是完全查不到。
        """
        return {
            "order_sn": order_id,
            "order_sequence_no": "",
            "goods_name": "（订单详情待核实）",
            "goods_id": "",
            "goods_img": "",
            "spec": "",
            "quantity": 1,
            "order_status": None,
            "order_status_desc": "待核实",
            "total_amount": None,
            "pay_amount": None,
            "goods_amount": None,
            "post_amount": None,
            "coupon_amount": None,
            "create_time": "",
            "pay_time": "",
            "logistics_time": "",
            "exp_ship_time": "",
            "receiver": "",
            "receiver_phone": "",
            "address": "",
            "seller_words": "",
            "buyer_words": "",
            "after_sales_status": 0,
            "buyer_nick": "",
            "needs_shipping": True,
            "sub_orders": [],
            "source": "message_history",
        }


__all__ = ["OrderQuery"]
