"""
抖音订单查询 API（对齐拼多多 OrderQuery 接口签名）。

【数据源一：飞鸽 IM 客服工作台的订单接口，2026-11-13 抓包确认】

    POST https://pigeon.jinritemai.com/backstage/cmpoent/order/query

    这是买家进线时客服工作台右侧「订单」面板实际调用的接口：鉴权用飞鸽域
    自己的登录 cookie，与消息通道同一套凭据，不需要另外登录抖店商家后台，
    因此买家一进线就能查，比绕到工作台后台快一步。

    响应（真实抓包）：
        {"check_spam_resp": {...}, "order_info": {
            "order_id": "69194...", "order_status": 3,
            "order_status_desc": "已发货", "order_type_desc": "普通订单",
            "create_time": 1794303361, "actual_pay_amount": 12980,
            "full_address": "贵州省贵阳市…", "pay_type_desc": "支付宝",
            "logistics_info": {"tracking_no": "...", "express_company": "申通快递"},
            "sku_order_list": [...], "buyer_words": "", "order_remark": "",
        }}

    两处容易「顺手改错」的地方：
    1. 路径里的 `cmpoent` 是服务端真实拼写（不是 component），改了会 404；
    2. 金额单位是分（actual_pay_amount=12980 即 129.80 元），与 searchlist 一致。

【数据源二：抖店商家后台订单列表接口，2026-11-12 逆向并实测验证】

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
       需 a_bogus 签名，本实现不使用该接口。注意它与上面的
       cmpoent/order/query 不是同一个接口，后者是 IM 页面在用的。
    2. 飞鸽域 getuserbyorder 实测返回 11001「飞鸽版本过低」，同样不使用。
    3. cookie 需包含抖店后台域。登录流程抓取的是全浏览器 cookie，
       若抖店后台未登录过，订单接口返回未登录，走消息历史骨架兜底。

回退路径：
    IM 接口失败（登录态 / 风控 / 网络）时回退到抖店后台 searchlist；
    两者都失败时，用 Agent 会话历史里买家发过的订单号返回骨架，
    其余字段标注「待核实」。
"""
import json
import re

import requests

from ..base_request import BaseRequest
from utils.logger_loguru import get_logger
from Channel.douyin.utils.log_throttle import throttled_warning, throttled_info

logger = get_logger("DouyinOrderQuery")

# IM 页面（飞鸽工作台）订单接口。`cmpoent` 是服务端真实拼写，不要「纠正」。
PIGEON_ORDER_QUERY_URL = (
    "https://pigeon.jinritemai.com/backstage/cmpoent/order/query"
)

# 请求体形态候选。该接口是页面内部接口，没有对外文档，抓包只留下响应，
# 字段只能按响应结构反推：先试最可能的，命中后记住形态，不再逐个试探。
IM_BODY_STYLES = (
    ("order_id", lambda oid, cid: {"order_id": oid}),
    ("order_id_biz", lambda oid, cid: {"order_id": oid, "biz_type": 4}),
    ("data_wrap", lambda oid, cid: {"data": {"order_id": oid}}),
    ("order_id_cid", lambda oid, cid: {"order_id": oid, "cid": cid or ""}),
)

# 所有形态都失败后的冷却时间：避免每条买家消息都白试 4 次请求
_IM_COOLDOWN_SECONDS = 600

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

# IM 接口要带 IM 页面自己的 referer：用 fxg 的会被服务端按跨站请求拒掉
_IM_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "referer": "https://im.jinritemai.com/pc_seller_v2/main/workspace",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
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


def _unwrap_cookies(raw):
    """把登录 JSON / 纯 cookie 字典统一成 {name: value}。

    登录结果里真正的 cookie 在 cookies_full 下，外层还有 verifyFp / msToken
    等风控参数。直接拿外层当 cookie 发出去，服务端只认到空登录态。
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    inner = raw.get("cookies_full")
    if isinstance(inner, dict) and inner:
        return {str(k): str(v) for k, v in inner.items()}
    cookies = {}
    for key, value in raw.items():
        if key in ("verifyFp", "msToken", "shop_name", "user_name"):
            continue
        if isinstance(value, (str, int, float)):
            cookies[str(key)] = str(value)
    return cookies


def _resolve_session_path(shop_id=None, user_id=None, name=None):
    """定位当前账号的飞鸽登录态文件。

    登录态文件从早期的全局 dy_session_live.json 改成了按账号隔离
    （见 Channel/douyin/session_paths.py），所以这里不能写死路径，
    必须走同一个解析函数；否则多账号时会读到别人的凭据，或者直接
    一个都读不到（IM 查单和会话订单号会静默失效）。

    Args:
        shop_id / user_id / name: 账号标识。都不给时取最近写入的那份
            （探针脚本、单账号场景用）。

    Returns:
        Path 或 None
    """
    try:
        from Channel.douyin.session_paths import (
            existing_pigeon_session_path,
            legacy_pigeon_session_path,
        )
    except Exception as exc:
        logger.info(f"飞鸽登录态模块不可用: error_type={type(exc).__name__}")
        return None

    if shop_id or user_id or name:
        return existing_pigeon_session_path(shop_id, user_id, name)

    legacy = legacy_pigeon_session_path()
    newest_path = None
    newest_mtime = 0.0
    try:
        entries = list(legacy.parent.glob("dy_session_*.json"))
    except OSError:
        entries = []
    for entry in entries:
        if entry.name == legacy.name:
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest_path = entry
    if newest_path is not None:
        return newest_path
    return legacy


def load_session_cookies(session_path=None):
    """读取飞鸽登录态文件里的 cookie（与消息通道同一份凭据）。

    Returns:
        dict: cookie 字典；文件不存在 / 结构异常时返回空字典（调用方自行降级）
    """
    if not session_path:
        session_path = _resolve_session_path()
    if not session_path:
        return {}
    try:
        with open(session_path, encoding="utf-8") as fp:
            session = json.load(fp)
    except (OSError, ValueError):
        return {}
    return _unwrap_cookies(session)


def _cookie_header(cookies):
    parts = []
    for name, value in cookies.items():
        parts.append("%s=%s" % (name, value))
    return "; ".join(parts)


def collect_session_order_ids(conv_id, session_path=None):
    """从飞鸽会话列表取该会话出现过的订单号（买家进线即可查的那一步）。

    数据来源与消息接收同一条链路（get_current_conversation_list）：
    买家发来的订单卡片 messageBody.ext.bury_order_id 就是真实订单号；
    买家直接打字发订单号的，从正文按 15-25 位数字提取。

    与 _extract_order_ids 的分工：那边读 Agent 会话库（历史，稳），
    这边读平台会话列表（实时，买家刚发就能拿到），两者互为补充。

    Args:
        conv_id: 父会话 ID（抖音渠道的 from_uid 就是它）
        session_path: 登录态文件路径，默认按账号自动解析（见 session_paths）

    Returns:
        List[str]: 订单号（去重，保持出现顺序）。失败返回空列表，不抛异常。
    """
    conv_id = str(conv_id or "").strip()
    if not conv_id:
        return []

    try:
        from tools.reverse_probe.douyin.pigeon_conv_list import PigeonConvList

        if not session_path:
            session_path = _resolve_session_path()
        if not session_path:
            return []
        conversations = PigeonConvList(str(session_path)).fetch_conversations() or []
    except Exception as exc:
        logger.info(
            f"会话订单号提取失败（跳过，不影响其他路径）: "
            f"conv={conv_id}, error_type={type(exc).__name__}"
        )
        return []

    order_ids = []
    seen = set()

    def _add(order_id):
        order_id = str(order_id or "").strip()
        if order_id and order_id.isdigit() and order_id not in seen:
            seen.add(order_id)
            order_ids.append(order_id)

    for conv in conversations:
        cid = str(conv.get("parentConShortId") or "")
        if not cid:
            cid = str(conv.get("subConversationId") or "")
        if cid != conv_id:
            continue
        for item in conv.get("msgList") or []:
            body = item.get("messageBody") or {}
            if not isinstance(body, dict):
                continue
            for order_id in extract_card_order_ids(body):
                _add(order_id)
            content = str(body.get("content") or "")
            if content:
                for match in _ORDER_ID_PATTERN.finditer(content):
                    _add(match.group(1))
    return order_ids


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
    """抖音订单查询器（IM 订单接口优先，抖店后台兜底，最后会话历史骨架）。"""

    ORDER_LIST_URL = "https://fxg.jinritemai.com/api/order/searchlist"
    IM_ORDER_QUERY_URL = PIGEON_ORDER_QUERY_URL

    # 类级状态：哪种请求体形态被接受（进程内记住，避免每次都试探）
    _im_style = None
    # 各账号的冷却截止时间（key = shop_id|user_id）。按账号分开：
    # 一个店铺登录态失效，不该让另一个店铺也跟着退到后台接口。
    _im_disabled_until = {}

    def __init__(self, shop_id: str = None, user_id: str = None, cookies=None):
        super().__init__(shop_id=shop_id, user_id=user_id, channel_name="douyin")
        if cookies:
            self.update_cookies(cookies)
        # 最近一次 _search_orders 失败原因（"" = 未失败/未调用）
        self.last_search_error = ""
        # 最近一次 IM 接口失败原因（诊断用）
        self.last_im_error = ""

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

        # 1) IM 页面同款接口：与消息通道共用飞鸽 cookie，买家进线即可查
        detail = self.get_order_detail_via_im(order_id)
        if detail is not None:
            return detail

        # 2) 抖店后台订单列表（需要 fxg 域 cookie）
        rows = self._search_orders({"order_id": order_id}, limit=1)
        if not rows:
            return None
        return self._normalize_order(rows[0])

    def get_order_detail_via_im(self, order_id, cid=None):
        """用飞鸽 IM 工作台的订单接口查单。

        Args:
            order_id (str): 订单号
            cid (str): 会话 ID（可选，部分请求体形态需要）

        Returns:
            dict: 归一化订单；接口不可用/查不到返回 None（不抛异常）
        """
        order_id = str(order_id or "").strip()
        if not order_id:
            return None

        import time

        cooldown_key = f"{self.shop_id}|{self.user_id}"
        if time.time() < self.__class__._im_disabled_until.get(cooldown_key, 0.0):
            self.last_im_error = "cooldown"
            return None

        cookies = self._im_cookies()
        if not cookies:
            self.last_im_error = "no_cookie"
            return None

        styles = IM_BODY_STYLES
        if self.__class__._im_style:
            for style in IM_BODY_STYLES:
                if style[0] == self.__class__._im_style:
                    styles = (style,)
                    break

        for name, build in styles:
            body = build(order_id, cid)
            info = self._im_post(body, cookies)
            if info:
                if self.__class__._im_style != name:
                    self.__class__._im_style = name
                    logger.info(f"抖音 IM 订单接口命中请求体形态: {name}")
                return self._normalize_im_order(info)
            # 登录失效 / 被风控时换别的请求体也没用，直接停，别让买家等四轮超时
            if self._im_error_is_fatal(self.last_im_error):
                break

        # 全部形态都失败：冷却一段时间，避免每条买家消息都白试 4 次
        self.__class__._im_disabled_until[cooldown_key] = (
            time.time() + _IM_COOLDOWN_SECONDS
        )
        if not self.last_im_error:
            self.last_im_error = "no_order_info"
        throttled_warning(
            logger, "order.im_fail",
            f"抖音 IM 订单接口不可用，{_IM_COOLDOWN_SECONDS} 秒内改用后台接口: "
            f"{self.last_im_error}"
        )
        return None

    def _session_path(self):
        """当前账号的飞鸽登录态文件（按账号隔离，与消息通道同一份）。"""
        name = self.account_name
        if not name or name == "未知账号":
            name = None
        return _resolve_session_path(self.shop_id, self.user_id, name)

    def _im_cookies(self):
        """IM 接口用的 cookie：优先登录态文件，其次账号表里的完整 cookie。"""
        cookies = load_session_cookies(self._session_path())
        if cookies:
            return cookies
        return _unwrap_cookies(self.cookies)

    @staticmethod
    def _im_error_is_fatal(error):
        """这个错误再换请求体也没意义。

        登录过期（10005）、风控、401/403 都属于「凭据或通道不可用」，
        逐个试四种请求体只会把一次失败放大成四次超时等待。
        """
        error = str(error or "")
        if not error:
            return False
        if error.startswith("http_401") or error.startswith("http_403"):
            return True
        if "10005" in error:
            return True
        for word in ("登录", "风控", "风险", "login"):
            if word in error:
                return True
        return False

    def _im_post(self, body, cookies):
        """POST 一次 IM 订单接口，返回 order_info 字典或 None。"""
        headers = dict(_IM_HEADERS)
        headers["cookie"] = _cookie_header(cookies)
        try:
            resp = requests.post(
                self.IM_ORDER_QUERY_URL,
                json=body,
                headers=headers,
                timeout=5,
            )
        except Exception as exc:
            self.last_im_error = f"{type(exc).__name__}"
            return None

        if resp.status_code != 200:
            self.last_im_error = f"http_{resp.status_code}"
            return None

        try:
            payload = resp.json()
        except ValueError:
            self.last_im_error = "not_json"
            return None

        if not isinstance(payload, dict):
            self.last_im_error = "not_dict"
            return None

        info = payload.get("order_info")
        if not isinstance(info, dict) or not info:
            # 有的网关会把业务数据再包一层 data
            data = payload.get("data")
            if isinstance(data, dict):
                info = data.get("order_info")
        if not isinstance(info, dict) or not info:
            code = payload.get("code")
            msg = payload.get("msg") or payload.get("message") or ""
            self.last_im_error = f"code={code} msg={str(msg)[:60]}"
            return None

        self.last_im_error = ""
        return info

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
        汇总订单：会话订单号 -> 接口补详情 -> 失败留骨架。

        订单号取两路：先读飞鸽会话列表（买家刚发的订单卡片立刻能拿到），
        取不到再读 Agent 会话历史（跨进程重启也在）。两路都不依赖
        CapturedMessage 表——那张表在代码里没有写入点。

        Returns:
            List[dict]: 订单列表，无订单返回空列表
            None: 数据库不可用等硬失败
        """
        order_ids = []
        if buyer_uid:
            order_ids = collect_session_order_ids(buyer_uid, self._session_path())
            if order_ids:
                logger.info(
                    f"抖音订单查询: 会话内取到订单号 {len(order_ids)} 个 "
                    f"conv={buyer_uid}"
                )
        if not order_ids:
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
    def _normalize_im_order(info):
        """把 IM 接口的 order_info 归一化成订单 dict。

        字段名来自 2026-11-13 真实抓包（见模块 docstring）。同一个订单
        在两个接口里命名完全不同，所以单独一个归一化函数；退款 / 发货
        状态的判定复用 _classify_refund，保证两条路径给模型的字段一致。
        """
        if not isinstance(info, dict):
            return None

        def _s(value):
            return str(value or "").strip()

        def _pick(source, *keys):
            for key in keys:
                value = _s(source.get(key))
                if value:
                    return value
            return ""

        logistics = info.get("logistics_info") or {}
        if not isinstance(logistics, dict):
            logistics = {}

        sku_items = info.get("sku_order_list") or []
        if not isinstance(sku_items, list):
            sku_items = []
        first_sku = sku_items[0] if sku_items and isinstance(sku_items[0], dict) else {}

        # 地址：优先整串 full_address，缺了再拼省市区
        address = _s(info.get("full_address"))
        if not address:
            parts = []
            for key in ("province", "city", "town", "street"):
                value = _s(info.get(key))
                if value and value not in parts:
                    parts.append(value)
            detail = _s(info.get("detail") or info.get("address_detail"))
            if detail:
                parts.append(detail)
            address = " ".join(parts)

        status_desc = _s(info.get("order_status_desc"))
        if not status_desc:
            try:
                status_desc = _ORDER_STATUS_MAP.get(int(info.get("order_status")), "未知状态")
            except (TypeError, ValueError):
                status_desc = "未知状态"

        refund_status_desc = _classify_refund(status_desc)
        needs_shipping = refund_status_desc not in ("已退款", "已关闭", "售后已结束")

        tracking_no = _s(logistics.get("tracking_no"))
        if not needs_shipping:
            shipping_desc = "无需发货（已退款/已关闭）"
        elif "已完成" in status_desc:
            shipping_desc = "已签收"
        elif tracking_no:
            shipping_desc = "已发货"
        else:
            shipping_desc = "未发货"

        try:
            quantity = int(first_sku.get("count") or first_sku.get("num") or 0)
        except (TypeError, ValueError):
            quantity = 0
        if not quantity:
            quantity = 1

        order = {
            "order_sn": _s(info.get("order_id")),
            "order_sequence_no": _s(info.get("order_id")),
            "goods_name": _pick(first_sku, "product_name", "goods_name", "title", "sku_name"),
            "goods_id": _pick(first_sku, "product_id", "goods_id"),
            "goods_img": _pick(first_sku, "img", "product_img", "goods_img"),
            "spec": _pick(first_sku, "spec", "sku_spec_desc", "sku_name"),
            "quantity": quantity,
            "order_status": info.get("order_status"),
            "order_status_desc": status_desc,
            "refund_status_desc": refund_status_desc,
            "needs_shipping": needs_shipping,
            "shipping_status_desc": shipping_desc,
            # 金额单位是分，统一转元
            "total_amount": _fen_to_yuan(info.get("order_amount")),
            "pay_amount": _fen_to_yuan(
                info.get("actual_pay_amount") or info.get("seller_price")
            ),
            "goods_amount": _fen_to_yuan(info.get("seller_price")),
            "post_amount": _fen_to_yuan(info.get("post_amount")),
            "coupon_amount": _fen_to_yuan(info.get("promotion_amount")),
            "create_time": _fmt_ts(info.get("create_time")),
            "pay_time": _fmt_ts(info.get("pay_time")),
            "logistics_time": _fmt_ts(
                logistics.get("logistics_time_sec") or logistics.get("logistics_time")
            ),
            "exp_ship_time": _fmt_ts(info.get("exp_ship_time") or info.get("ship_time")),
            "receiver": _s(info.get("post_receiver") or info.get("receiver")),
            "receiver_phone": _s(info.get("mobile") or info.get("receiver_phone")),
            "address": address,
            "seller_words": _s(info.get("order_remark")),
            "buyer_words": _s(info.get("buyer_words")),
            "pay_type": _s(info.get("pay_type_desc")),
            "tracking_no": tracking_no,
            # 上层工具（query_order_status._format_output）读的是 logistics_sn，
            # 只写 tracking_no 会让「运单号」这一行永远不出现。
            "logistics_sn": tracking_no,
            "logistics_company": _s(
                logistics.get("express_company") or logistics.get("company")
            ),
            "after_sales_status": 1 if refund_status_desc else 0,
            "buyer_nick": _s(info.get("buyer_nick") or info.get("user_nickname")),
            "source": "pigeon.im_order_query",
        }

        sub_orders = []
        for item in sku_items:
            if not isinstance(item, dict):
                continue
            sub_orders.append({
                "order_sn": _pick(item, "item_order_id", "order_id"),
                "goods_name": _pick(item, "product_name", "goods_name", "title"),
                "goods_id": _pick(item, "product_id", "goods_id"),
                "spec": _pick(item, "spec", "sku_spec_desc", "sku_name"),
                "quantity": item.get("count") or item.get("num") or 1,
                "pay_amount": _fen_to_yuan(item.get("pay_amount")),
                "order_status_desc": _s(item.get("order_status_desc")),
            })
        order["sub_orders"] = sub_orders
        return order

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
            # 与 IM 路径一致：工具输出读 logistics_sn
            "logistics_sn": str(logistic.get("tracking_no") or ""),
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


__all__ = [
    "OrderQuery",
    "PIGEON_ORDER_QUERY_URL",
    "collect_session_order_ids",
    "extract_buyer_nick_from_ext",
    "extract_card_order_ids",
    "load_session_cookies",
]
