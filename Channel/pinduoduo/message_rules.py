"""拼多多消息分类规则：来源判定与动作表（单一事实来源）。

本模块由解析层（``pdd_message.py``）与路由层（``core/pdd_message_handler.py``）
共同引用，避免规则散落在两处。设计依据见
``docs/plans/2026-09-20-001-feat-pdd-message-type-handling-plan.md``。

三条原则：

1. 每条消息都必须落到一个 Action，不允许静默丢弃
2. 未识别一律降级为 CONTEXT_ONLY 并告警，而不是丢掉
3. 判定依据是 (origin, type, sub_type) 三元组，不是单一 type
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Optional

from bridge.context import ContextType


class PDDMsgType(IntEnum):
    """拼多多消息类型。

    数值取自服务端实际推送的抓包统计（见协议文档第 3.3 节）。
    新增类型时必须同步补进动作表，否则会被判为 UNKNOWN。
    """
    TEXT = 0
    IMAGE = 1
    EMOTION = 5
    ORDER_CARD = 8
    VIDEO = 14
    READ_SYNC = 20
    TRANSFER = 24
    SYSTEM_PUSH = 30
    ROBOT_HINT = 31
    USER_SOURCE = 41
    RICH_TEXT_FAQ = 56
    GOODS_SPEC = 64
    MATERIAL_UPLOAD = 97
    WITHDRAW = 1002


class PDDSubType(IntEnum):
    """type=0 时的子类型。"""
    GOODS_INQUIRY = 0
    ORDER_INFO = 1
    CS_GOODS_CARD = 2


class Origin(str, Enum):
    """消息来源角色。

    MERCHANT 涵盖机器人自动回复与人工客服回复——产品约束是
    「机器人接待期间人工不会介入」，因此不区分两者。
    """
    CUSTOMER = "customer"
    MERCHANT = "merchant"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class Action(str, Enum):
    """一条消息应当被执行的动作。"""
    REPLY = "reply"                  # 需要 AI 生成回复
    CONTEXT_ONLY = "context_only"    # 不回复，但注入会话上下文
    OBSERVE = "observe"              # 只记录 / 触发状态变更
    HANDOFF = "handoff"              # 触发流程动作（转接等）
    IGNORE = "ignore"                # 明确忽略
    UNKNOWN = "unknown"              # 未识别；使用时降级为 CONTEXT_ONLY


# 服务端下发的 from.role 取值
ROLE_CUSTOMER = "user"
ROLE_MERCHANT = "mall_cs"
ROLE_SYSTEM = "system"
ROLE_MALL = "mall"

# 系统消息的 response 取值
RESPONSE_PUSH = "push"
RESPONSE_AUTH = "auth"
RESPONSE_MALL_SYSTEM_MSG = "mall_system_msg"
RESPONSE_SYSTEM_PUSH = "system_push"


def resolve_origin(from_role: Optional[str], response: Optional[str] = None) -> Origin:
    """由 from.role 与 response 判定来源。

    无法判定时返回 Origin.UNKNOWN，交由调用方按未知处理，
    不抛异常（缺失 from 字段的报文实测存在）。
    """
    role = (from_role or "").strip().lower()
    if role == ROLE_CUSTOMER:
        return Origin.CUSTOMER
    if role == ROLE_MERCHANT:
        return Origin.MERCHANT
    if role in (ROLE_SYSTEM, ROLE_MALL):
        return Origin.SYSTEM
    # 部分系统推送报文没有 from，仅靠 response 判别
    if response == RESPONSE_SYSTEM_PUSH:
        return Origin.SYSTEM
    return Origin.UNKNOWN


# ---------------------------------------------------------------------------
# 动作表
# ---------------------------------------------------------------------------
# 买家侧：AI 回复是主干
_CUSTOMER_ACTIONS = {
    PDDMsgType.TEXT: Action.REPLY,
    PDDMsgType.IMAGE: Action.REPLY,
    PDDMsgType.EMOTION: Action.REPLY,
    PDDMsgType.VIDEO: Action.REPLY,
    PDDMsgType.GOODS_SPEC: Action.REPLY,
    PDDMsgType.ORDER_CARD: Action.CONTEXT_ONLY,
    PDDMsgType.USER_SOURCE: Action.CONTEXT_ONLY,
    PDDMsgType.TRANSFER: Action.HANDOFF,
    PDDMsgType.ROBOT_HINT: Action.OBSERVE,
    PDDMsgType.READ_SYNC: Action.OBSERVE,
    PDDMsgType.SYSTEM_PUSH: Action.OBSERVE,
    PDDMsgType.MATERIAL_UPLOAD: Action.OBSERVE,
    PDDMsgType.RICH_TEXT_FAQ: Action.IGNORE,
    # 撤回不处理：买家撤回后通常会补发新消息
    PDDMsgType.WITHDRAW: Action.IGNORE,
}

# type=0 时按 sub_type 细分；未列出的 sub_type 按纯文本处理
_CUSTOMER_TEXT_ACTIONS = {
    PDDSubType.GOODS_INQUIRY: Action.REPLY,
    PDDSubType.ORDER_INFO: Action.REPLY,
    # 我方卡片；正常应来自 MERCHANT，此处兜底
    PDDSubType.CS_GOODS_CARD: Action.CONTEXT_ONLY,
}

# 客服侧：不参与 AI 回复，只作上下文与日志
# 决策：注入文本、商品卡、规格卡与订单卡；图片 / 表情 / 视频仅记录。
# 订单卡（type=8）带着售后状态与订单号，买家往往紧接着追问「退货地址」，
# 不注入的话 AI 看不到我方刚推过的结论；卡片入历史前会先摘要成一行。
_MERCHANT_ACTIONS = {
    PDDMsgType.TEXT: Action.CONTEXT_ONLY,
    PDDMsgType.GOODS_SPEC: Action.CONTEXT_ONLY,
    PDDMsgType.ORDER_CARD: Action.CONTEXT_ONLY,
    PDDMsgType.IMAGE: Action.OBSERVE,
    PDDMsgType.EMOTION: Action.OBSERVE,
    PDDMsgType.VIDEO: Action.OBSERVE,
    PDDMsgType.TRANSFER: Action.OBSERVE,
    PDDMsgType.ROBOT_HINT: Action.OBSERVE,
    PDDMsgType.USER_SOURCE: Action.OBSERVE,
    PDDMsgType.RICH_TEXT_FAQ: Action.IGNORE,
    PDDMsgType.WITHDRAW: Action.IGNORE,
}

# 系统侧：只观察，不参与对话
_SYSTEM_ACTIONS = {
    PDDMsgType.READ_SYNC: Action.OBSERVE,
    PDDMsgType.SYSTEM_PUSH: Action.OBSERVE,
    PDDMsgType.MATERIAL_UPLOAD: Action.OBSERVE,
    PDDMsgType.RICH_TEXT_FAQ: Action.IGNORE,
    PDDMsgType.WITHDRAW: Action.IGNORE,
}


def classify(origin: Origin, msg_type: Optional[int], sub_type: Optional[int] = None) -> Action:
    """由 (origin, type, sub_type) 得出动作。

    未列入动作表的组合返回 Action.UNKNOWN，由调用方降级并告警，
    不返回可静默丢弃的结果。
    """
    if msg_type is None:
        return Action.UNKNOWN

    if origin is Origin.CUSTOMER:
        if msg_type == PDDMsgType.TEXT and sub_type is not None:
            return _CUSTOMER_TEXT_ACTIONS.get(sub_type, Action.REPLY)
        return _CUSTOMER_ACTIONS.get(msg_type, Action.UNKNOWN)

    if origin is Origin.MERCHANT:
        # 客服侧未列出的类型一律记录，不猜测
        return _MERCHANT_ACTIONS.get(msg_type, Action.OBSERVE)

    if origin is Origin.SYSTEM:
        return _SYSTEM_ACTIONS.get(msg_type, Action.OBSERVE)

    return Action.UNKNOWN


def effective_action(action: Action) -> Action:
    """把 UNKNOWN 降级为 CONTEXT_ONLY（不丢弃）。"""
    return Action.CONTEXT_ONLY if action is Action.UNKNOWN else action


# 买家侧可能触发 AI 回复的 ContextType 集合。
# 供 AIReplyHandler 作为默认白名单，确保它不会拦住动作表判定为 REPLY 的消息
# （此前 GOODS_CARD 就有此落差）。
CUSTOMER_REPLY_CONTEXT_TYPES = frozenset({
    ContextType.TEXT,
    ContextType.IMAGE,
    ContextType.VIDEO,
    ContextType.EMOTION,
    ContextType.GOODS_INQUIRY,
    ContextType.ORDER_INFO,
    ContextType.GOODS_SPEC,
})

# 客服侧需要注入会话上下文的 ContextType
# （决策：文本、商品卡、规格卡、订单卡；图片 / 表情 / 视频只记日志）
MERCHANT_CONTEXT_CONTEXT_TYPES = frozenset({
    ContextType.MALL_CS,
    ContextType.GOODS_CARD,
    ContextType.GOODS_SPEC,
    ContextType.ORDER_INFO,
})


def is_reply_action(action: Action) -> bool:
    return action is Action.REPLY


def is_immediate_action(action: Action) -> bool:
    """需要立刻处理、不进队列的动作。"""
    return action in (Action.HANDOFF,)


# 对外导出：解析层与路由层共同引用
__all__ = [
    "PDDMsgType",
    "PDDSubType",
    "Origin",
    "Action",
    "resolve_origin",
    "classify",
    "effective_action",
    "is_reply_action",
    "is_immediate_action",
    "CUSTOMER_REPLY_CONTEXT_TYPES",
    "MERCHANT_CONTEXT_CONTEXT_TYPES",
    "ROLE_CUSTOMER",
    "ROLE_MERCHANT",
    "ROLE_SYSTEM",
    "ROLE_MALL",
    "RESPONSE_PUSH",
    "RESPONSE_AUTH",
    "RESPONSE_MALL_SYSTEM_MSG",
    "RESPONSE_SYSTEM_PUSH",
]
