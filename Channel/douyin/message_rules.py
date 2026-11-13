"""抖音飞鸽消息来源判定（对齐拼多多 ``Channel/pinduoduo/message_rules.py``）。

拼多多的判定依据是 ``from.role`` + ``response``；飞鸽对应的字段在消息的
``ext`` 扩展表里，实测（2026-11-10 22:50 真机会话列表）取值如下：

  买家文本
      s:sender_biz_role = Buyer
      sender_role = 1
      type = text
      有 security_sender_id

  平台/机器人提示（例如「当前会话已长时间未回复，平台可能主动介入处理。」）
      s:sender_biz_role = Shop
      sender_role = 4
      im_sender_role = 4
      biz_sender = robot
      systemHint = 1
      type = template_card
      无 security_sender_id

  人工客服/欢迎语
      s:sender_biz_role = CurrentServer
      im_sender_role = 2
      type = text

  客服接入提示
      s:sender_biz_role = Shop
      sender_role = 3
      type = allocated_service

所以只靠「有没有正文」判断是不够的：平台提示同样是 msgType=1000 且有正文。
本模块把判定收敛成一处，接收链路（会话列表 / WS 帧）统一引用，避免各写一套。

三条原则（与拼多多一致）：

1. 无法确认是买家时一律不交给 AI，宁可由另一条链路补齐
2. 商家与系统消息不参与自动回复
3. 判定依据是多个字段的组合，不是单一字段
"""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Dict, Optional


class Origin(str, Enum):
    """消息来源角色。"""

    CUSTOMER = "customer"
    MERCHANT = "merchant"
    SYSTEM = "system"
    UNKNOWN = "unknown"


# 飞鸽 ext 里的关键字段名
KV_BIZ_ROLE = "s:sender_biz_role"
KV_SENDER_ROLE = "sender_role"
KV_IM_SENDER_ROLE = "im_sender_role"
KV_MSG_TYPE = "type"
KV_SYSTEM_HINT = "systemHint"
KV_BIZ_SENDER = "biz_sender"

BIZ_ROLE_BUYER = "Buyer"
BIZ_ROLE_MERCHANT = "CurrentServer"
BIZ_ROLE_SHOP = "Shop"

# 商家 / 系统侧角色值：飞鸽 sender_role 2=商家，4=系统。
# 1 与会话列表里的买家消息一致；历史 WS 抓包中买家帧也出现过 3，
# 两种都按买家处理——平台提示另有 systemHint / biz_role / type 兜住。
ROLE_BUYER = {"1", "3"}
ROLE_MERCHANT = {"2", "Shop", "CurrentServer"}
ROLE_SYSTEM = {"4", "system", "robot"}


def _text(kv: Dict[str, str], key: str) -> str:
    value = kv.get(key)
    if value is None:
        return ""
    return str(value).strip()


def resolve_origin(kv: Optional[Dict[str, str]]) -> Origin:
    """由消息扩展表判定来源。

    顺序：平台标记 → 业务角色 → 发送方角色 → 未知。
    平台标记优先，因为机器人提示的 biz_role 是 Shop、sender_role 是 4，
    但它同时带 systemHint/biz_sender，必须先被识别为系统消息。
    """
    if not kv:
        return Origin.UNKNOWN

    biz_sender = _text(kv, KV_BIZ_SENDER).lower()
    system_hint = _text(kv, KV_SYSTEM_HINT)
    msg_type = _text(kv, KV_MSG_TYPE).lower()

    # 1) 平台 / 机器人系统消息
    if biz_sender in {"robot", "system"}:
        return Origin.SYSTEM
    if system_hint not in {"", "0", "false", "False"}:
        return Origin.SYSTEM
    if msg_type in {"template_card", "allocated_service", "system_hint"}:
        return Origin.SYSTEM

    # 2) 业务角色（最可靠）
    biz_role = _text(kv, KV_BIZ_ROLE)
    if biz_role == BIZ_ROLE_BUYER:
        return Origin.CUSTOMER
    if biz_role in {BIZ_ROLE_MERCHANT, BIZ_ROLE_SHOP}:
        return Origin.MERCHANT

    # 3) 发送方角色兜底
    sender_role = _text(kv, KV_SENDER_ROLE)
    if sender_role in ROLE_BUYER:
        return Origin.CUSTOMER
    if sender_role in ROLE_MERCHANT:
        return Origin.MERCHANT
    if sender_role in ROLE_SYSTEM:
        return Origin.SYSTEM

    # 4) 无任何角色信息：不确定，交给调用方按未知处理（不回复）
    return Origin.UNKNOWN


def is_customer_message(kv: Optional[Dict[str, str]]) -> bool:
    """只有明确的买家消息才返回 True。"""
    return resolve_origin(kv) is Origin.CUSTOMER


# ---------------------------------------------------------------------------
# v3 get_user_message 的来源判定
#
# 这条接口把角色字段放在消息顶层（不是 ext 表），而且 role 语义与
# 会话列表**不一致**，不能共用一套常量。真机实测（2026-11-11 00:26）：
#
#   真实买家文本      sender_biz_role=Buyer   sender_role 缺失
#   客服接入提示      sender_biz_role=Buyer   sender_role=1   ← 必须排除
#   商家坐席          sender_biz_role=CurrentServer  sender_role=2
#   店铺机器人/系统   sender_biz_role=Shop    sender_role=4
#
# 特别注意：真实买家消息的 sender_role 是空字符串，所以在会话列表里
# 表示「买家」的 role=1，在这条接口里恰恰是系统提示。共用一套判定会把
# 「客服XX接入」当成买家发言。
# ---------------------------------------------------------------------------
HTTP_BIZ_ROLE_AGENT = "CurrentServer"


def resolve_http_origin(message: Optional[Dict[str, str]]) -> Origin:
    """判定 v3 get_user_message 单条消息的来源。"""
    if not message:
        return Origin.UNKNOWN
    biz_role = str(message.get("sender_biz_role") or "").strip()
    role = str(message.get("sender_role") or "").strip()

    if biz_role == BIZ_ROLE_BUYER:
        # role=1 是「客服XX接入」这类系统提示，不是买家发言。
        if role == "1":
            return Origin.SYSTEM
        return Origin.CUSTOMER
    if biz_role in {BIZ_ROLE_MERCHANT, BIZ_ROLE_SHOP}:
        return Origin.MERCHANT
    if role in {"2", "4"}:
        return Origin.MERCHANT
    return Origin.UNKNOWN


def is_agent_message(message: Optional[Dict[str, str]]) -> bool:
    """是否是商家坐席消息（人工介入判定用）。

    只认 CurrentServer / role=2 —— 真人客服在飞鸽工作台发出的消息落在这里。
    role=4 的 Shop 消息是店铺机器人（含本程序的 AI 回复），不能算人工，
    否则 AI 每回一条就把自己转人工一次。

    2026-11-11 追加：抖音电商智能客服（``biz_role_type=llm``）哪怕日后
    换了角色值，也一律不算人工介入。店铺开了「智能客服自动接待」之后，
    买家每发一条它都会先回一句，若把它当成人工客服，程序就会在它回完
    第一句时立刻静默转人工，永远接不上话术。
    """
    if not message:
        return False
    if is_platform_ai_message(message):
        return False
    # 平台转人工欢迎语 / 接入提示 / 会话动作记录：角色字段和真身一样是
    # CurrentServer + role=2 + cs，但都不是人打的字，不能触发转人工。
    if is_platform_auto_message(message):
        return False
    biz_role = str(message.get("sender_biz_role") or "").strip()
    role = str(message.get("sender_role") or "").strip()
    if biz_role != HTTP_BIZ_ROLE_AGENT and role != "2":
        return False
    # 同一个账号也可能由机器人代发（biz_role_type=robot），一并排除。
    info = _parse_biz_sender_info(message.get("biz_sender_info"))
    biz_role_type = str(info.get("biz_role_type") or "").strip().lower()
    if biz_role_type in {BIZ_ROLE_TYPE_LLM, BIZ_ROLE_TYPE_ROBOT}:
        return False
    return True


# ---------------------------------------------------------------------------
# 抖音电商智能客服（平台自带、店铺开启自动接待后自动回复）
#
# 真机指纹（会话 <talk_id>）：
#   s:sender_biz_role = Shop      sender_role = 4      biz_sender = robot
#   biz_sender_info   = {"biz_role_type":"llm"}
#   sender_info       = {"msg_source":{"desc":"智能客服",
#                                      "source_key":"official_llm_cs",
#                                      "send_by":"抖音电商智能客服发送"}}
#   msg_foot          = {"b_foot":{"content":{"text":"抖音电商智能客服发送"}}}
#   robot_auto_answer = 1
#
# 它和「真人客服」的区别只在 biz_role_type：真人坐席是 cs。
# 所以判定不靠角色值（都是商家侧），靠 biz_role_type 与 source_key。
# ---------------------------------------------------------------------------
BIZ_ROLE_TYPE_LLM = "llm"
BIZ_ROLE_TYPE_ROBOT = "robot"
SOURCE_KEY_OFFICIAL_LLM_CS = "official_llm_cs"


def is_platform_ai_message(message: Optional[Dict[str, str]]) -> bool:
    """是否抖音电商智能客服（平台机器人）发的消息。

    返回 True 表示「机器自动接待」，不能当成人工介入，程序要继续回。
    """
    if not message:
        return False
    info = _parse_biz_sender_info(message.get("biz_sender_info"))
    if str(info.get("biz_role_type") or "").strip().lower() == BIZ_ROLE_TYPE_LLM:
        return True
    sender = _parse_biz_sender_info(message.get("sender_info"))
    source = sender.get("msg_source")
    if isinstance(source, dict):
        if str(source.get("source_key") or "").strip() == SOURCE_KEY_OFFICIAL_LLM_CS:
            return True
    return False


# 平台机器消息的 client_message_id 前缀（真机实测）。
# AI 回复 = llm_process_text_2_<uuid>；欢迎语 = 2_AI_WELCOME_<talk_id>。
#
# 注意：这**不是**本程序发出的消息。本程序走 pigeon 协议发的正文，落到
# 时间线上 client_message_id 是服务端生成的纯 UUID（如 a081a8ea-...），
# 和这里的前缀完全不同。本程序自己的回声靠 record_self_sent 的文本指纹识别。
# 这两个前缀只说明「这是机器自动生成的接待消息」，不能当人工看。
SELF_CLIENT_ID_MARKERS = ("llm_process_text", "AI_WELCOME")


# ---------------------------------------------------------------------------
# 平台自动接待消息（转人工欢迎语 / 接入提示 / 会话动作）
#
# 真机指纹（会话 <talk_id>）：
#
#   type=allocated_service  role=3  Shop
#             cid=<talk_id>_ServiceForYou_        「客服<店铺名>接入」
#   type=text  role=2  CurrentServer  biz_role_type=cs
#             cid=<talk_id>_Welcome_              「人工客服 <店铺名> 为您服务」
#   type=customer_enter_from_close  role=2  CurrentServer
#             cid=<uuid>                          「从历史会话发起会话」
#
# 这三条从角色字段看和真人客服一模一样（CurrentServer / role=2 / cs），
# 但都是平台或本程序自己的动作产物：
#
#   * `_Welcome_` 是平台「转人工」时自动发的欢迎语；
#   * `_ServiceForYou_` 是平台「客服接入」提示；
#   * `customer_enter_from_close` 是本程序 10016 帧在会话里留下的动作记录。
#
# 把它们当成人工介入，会出现「买家一发消息就静默转人工，AI 一句话都不发」。
# 所以必须显式排除，只让真人真正打字的消息触发转人工。
# ---------------------------------------------------------------------------
AUTO_CLIENT_ID_SUFFIXES = ("_Welcome_", "_ServiceForYou_")
AUTO_MSG_TYPES = {"allocated_service", "customer_enter_from_close"}


def is_platform_auto_message(message: Optional[Dict[str, str]]) -> bool:
    """是否平台自动产生的接待消息（不算人工介入）。"""
    if not message:
        return False
    cid = str(message.get("client_message_id") or "").strip()
    for suffix in AUTO_CLIENT_ID_SUFFIXES:
        if cid.endswith(suffix):
            return True
    msg_type = str(message.get("msg_type") or "").strip().lower()
    if not msg_type:
        # WS 事件的 ext 直接平铺在 message 上，键名就是 type。
        # HTTP 拉取条目的 type 是数字消息类型（1000），语义类型在 msg_type，
        # 所以只在 msg_type 缺失时才回退到 type。
        raw_type = message.get("type")
        if not isinstance(raw_type, int):
            msg_type = str(raw_type or "").strip().lower()
    if msg_type in AUTO_MSG_TYPES:
        return True
    return False



def is_self_client_message(client_message_id) -> bool:
    """client_message_id 是否属于本程序发出的消息。"""
    value = str(client_message_id or "")
    if not value:
        return False
    for marker in SELF_CLIENT_ID_MARKERS:
        if marker in value:
            return True
    return False


# 真人客服 / 店铺机器人的区分字段（对齐拼多多的 cs_uid vs robot 判据）。
# 飞鸽商家侧文本的 ext 里 biz_sender_info 形如 {"biz_role_type":"cs"}，
# cs = 客服坐席（真人或客服工作台）；缺这个字段或值里带 robot 的是店铺机器人。
KV_BIZ_SENDER_INFO = "biz_sender_info"
KV_SRC_USER_ID = "src_user_id"

BIZ_ROLE_TYPE_CS = "cs"


def _parse_biz_sender_info(raw) -> Dict[str, str]:
    """biz_sender_info 是 JSON 字符串，解析失败按空表处理。"""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except (ValueError, TypeError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def classify_merchant_sender(kv: Optional[Dict[str, str]]) -> str:
    """给商家侧消息分类：human（真人客服）/ robot（店铺机器人）/ unknown。

    拿不到 biz_sender_info 时返回 unknown，由调用方决定是否保守转人工。
    这里只负责分类，不做任何状态变更。
    """
    if not kv:
        return "unknown"
    info = _parse_biz_sender_info(kv.get(KV_BIZ_SENDER_INFO))
    biz_role_type = str(info.get("biz_role_type") or "").strip().lower()
    if biz_role_type == BIZ_ROLE_TYPE_CS:
        return "human"
    if biz_role_type in {BIZ_ROLE_TYPE_LLM, BIZ_ROLE_TYPE_ROBOT}:
        return "robot"
    if "robot" in json.dumps(info, ensure_ascii=False).lower():
        return "robot"
    return "unknown"


# ---------------------------------------------------------------------------
# 本程序自己发出的消息回声
#
# 抖音没有拼多多那种 from_uid 白名单可区分「本程序发的」和「人工客服发的」：
# 同一个店铺客服账号在飞鸽前台发的消息、和本程序通过 pigeon 协议发的消息，
# 落到会话列表里都是 sender_role=2 + CurrentServer，字段上没有可靠差别。
# 因此只能靠「发出记录」比对：发送成功后记下 (店铺, 会话, 文本)，
# 商家侧消息回来时若命中同一条记录，就认定是自己发的，不触发转人工。
#
# 不做这层过滤的后果很严重：AI 每发一条回复都会被自己判定为「人工介入」，
# 发一次就永久转人工一次，自动回复直接瘫痪。
# ---------------------------------------------------------------------------
SELF_SENT_WINDOW_SECONDS = 180

_SELF_SENT_AT: dict = {}


def _self_sent_key(conv_id, text) -> str:
    """回声指纹只用「会话 + 文本」。

    刻意不带 shop_id：发送方与拉取方的 shop_id 来自不同入口（构造参数 vs
    账号配置），一旦口径不一致，AI 自己的回复就会被判成人工介入，整条自动
    回复直接瘫掉。conv_id 是平台父会话 ID，本身已经唯一，加 shop_id 没有收益。
    """
    return "%s|%s" % (str(conv_id or ""), str(text or "").strip())


def record_self_sent(shop_id, conv_id, text) -> None:
    """发送成功后登记一条回声指纹。"""
    key = _self_sent_key(conv_id, text)
    if key == "|":
        return
    conv = str(conv_id or "")
    body = str(text or "").strip()
    if not conv or not body:
        return
    now = time.monotonic()
    _SELF_SENT_AT[key] = now
    if len(_SELF_SENT_AT) > 2000:
        stale = []
        for cached_key, cached_at in _SELF_SENT_AT.items():
            if now - cached_at > SELF_SENT_WINDOW_SECONDS:
                stale.append(cached_key)
        for stale_key in stale:
            _SELF_SENT_AT.pop(stale_key, None)


def is_self_sent(shop_id, conv_id, text) -> bool:
    """商家侧消息是否就是本程序刚发出的那一条。"""
    conv = str(conv_id or "")
    body = str(text or "").strip()
    if not conv or not body:
        return False
    key = _self_sent_key(conv, body)
    sent_at = _SELF_SENT_AT.get(key)
    if sent_at is None:
        return False
    if time.monotonic() - sent_at > SELF_SENT_WINDOW_SECONDS:
        _SELF_SENT_AT.pop(key, None)
        return False
    return True


def clear_self_sent(shop_id, conv_id, text) -> None:
    """命中一次后即失效：同一条文本客服再手工发一遍，应当算人工介入。"""
    _SELF_SENT_AT.pop(_self_sent_key(conv_id, text), None)


__all__ = [
    "Origin",
    "resolve_origin",
    "resolve_http_origin",
    "is_customer_message",
    "is_agent_message",
    "is_platform_ai_message",
    "is_platform_auto_message",
    "is_self_client_message",
    "classify_merchant_sender",
    "record_self_sent",
    "is_self_sent",
    "clear_self_sent",
]
