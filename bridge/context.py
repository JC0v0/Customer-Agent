"""
上下文类型枚举和Pydantic模型定义
"""
from enum import Enum
from hashlib import sha256
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class ChannelType(str, Enum):
    """渠道类型枚举"""
    PINDUODUO = "pinduoduo"
    JINGDONG = "jingdong"
    TAOBAO = "taobao"
    DOUYIN = "douyin"
    KUAISHOU = "kuaishou"

    def __str__(self):
        return self.value

class ContextType(str, Enum):
    """上下文类型枚举"""
    TEXT = "text"  # 文本
    IMAGE = "image"  # 图片
    VIDEO = "video"  # 视频
    EMOTION = "emotion"  # 表情
    GOODS_CARD = "goods_card"  # 商品卡片
    GOODS_INQUIRY = "goods_inquiry"  # 商品规格咨询
    GOODS_SPEC = "goods_spec"  # 商品规格
    ORDER_INFO = "order_info"  # 订单信息
    SYSTEM_STATUS = "system_status"  # 系统状态
    MALL_SYSTEM_MSG = "mall_system_msg"  # 商城消息
    SYSTEM_HINT = "system_hint"  # 系统提示
    SYSTEM_BIZ = "system_biz"  # 系统业务
    MALL_CS = "mall_cs"  # 商城客服
    WITHDRAW = "withdraw"  # 撤回
    AUTH = "auth"  # 认证
    TRANSFER = "transfer"  # 转接

    def __str__(self):
        return self.value

class PinduoduoKwargs(BaseModel):
    """拼多多消息专用kwargs类型定义"""
    msg_id: Optional[str] = None
    shop_name: Optional[str] = None
    from_user: Optional[str] = None
    from_uid: Optional[str] = None
    to_user: Optional[str] = None
    to_uid: Optional[str] = None
    nickname: Optional[str] = None
    timestamp: Optional[str] = None
    user_msg_type: Optional[ContextType] = None
    # 路由元信息（由 PDDChatMessage 产出，供 handler 决定动作）
    origin: Optional[str] = None
    action: Optional[str] = None
    pdd_type: Optional[int] = None
    pdd_sub_type: Optional[int] = None
    template_name: Optional[str] = None
    shop_id: Optional[str] = None
    user_id: Optional[str] = None
    username: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

class Context(BaseModel):
    """上下文模型，使用Pydantic进行数据验证"""
    type: ContextType = Field(..., description="上下文类型")
    content: Optional[str] = Field(None, description="内容")
    kwargs: Any = Field(default_factory=dict, description="渠道专用参数")
    channel_type: Optional[ChannelType] = Field(None, description="渠道类型")

    @classmethod
    def create_pinduoduo_context(cls, content=None, msg_id=None, from_user=None, from_uid=None,
                                to_user=None, to_uid=None, nickname=None, timestamp=None,
                                user_msg_type=None, shop_id=None, user_id=None, username=None, shop_name=None,
                                raw_data=None, channel_type=None,
                                origin=None, action=None, pdd_type=None, pdd_sub_type=None,
                                template_name=None):
        """创建拼多多上下文实例的便捷方法"""
        kwargs = PinduoduoKwargs(
            msg_id=msg_id,
            from_user=from_user,
            from_uid=from_uid,
            to_user=to_user,
            to_uid=to_uid,
            nickname=nickname,
            timestamp=timestamp,
            user_msg_type=user_msg_type,
            shop_id=shop_id,
            user_id=user_id,
            username=username,
            shop_name=shop_name,
            raw_data=raw_data,
            origin=origin,
            action=action,
            pdd_type=pdd_type,
            pdd_sub_type=pdd_sub_type,
            template_name=template_name,
        )

        return cls(
            type=user_msg_type or ContextType.TEXT,
            content=content,
            kwargs=kwargs,
            channel_type=channel_type
        )


def _context_value(context: Optional[Context], name: str, default: str = "") -> str:
    """Safely read a channel-specific value from a Context.

    Context.kwargs is intentionally typed as ``Any`` for channel extensibility,
    so all identity-sensitive code goes through this helper instead of making
    assumptions about Pydantic models versus dictionaries.
    """
    if context is None:
        return default
    kwargs = getattr(context, "kwargs", None)
    if kwargs is None:
        return default
    value = getattr(kwargs, name, None)
    if value is None and isinstance(kwargs, dict):
        value = kwargs.get(name)
    return default if value is None else str(value)


def channel_value(channel_type: Optional[Union[ChannelType, str]]) -> str:
    """Return a stable string for either an enum or a raw channel value."""
    if channel_type is None:
        return "unknown"
    return str(getattr(channel_type, "value", channel_type))


def make_account_key(
    channel_type: Optional[Union[ChannelType, str]],
    shop_id: Optional[Union[str, int]],
    user_id: Optional[Union[str, int]],
) -> str:
    """Build the canonical account scope key used by queues and handlers."""
    return "|".join(
        (
            channel_value(channel_type),
            str(shop_id or "unknown"),
            str(user_id or "unknown"),
        )
    )


def make_conversation_key(context: Optional[Context], customer_uid: Optional[str] = None) -> str:
    """Build an identity-safe conversation key.

    ``from_uid`` is the customer identity and must be part of the key; using
    only the merchant account would mix every customer into one history.

    For merchant-sent messages the counterpart is ``to_uid`` rather than
    ``from_uid``; callers pass ``customer_uid`` explicitly so that both
    directions of one conversation resolve to the same key.
    """
    account_key = make_account_key(
        getattr(context, "channel_type", None),
        _context_value(context, "shop_id", "unknown"),
        _context_value(context, "user_id", "unknown"),
    )
    customer_id = customer_uid or _context_value(context, "from_uid", "unknown")
    raw = f"{account_key}|customer|{customer_id}"
    return f"conversation_{sha256(raw.encode('utf-8')).hexdigest()}"


def make_queue_name(
    channel_type: Optional[Union[ChannelType, str]],
    shop_id: Optional[Union[str, int]],
    user_id: Optional[Union[str, int]],
) -> str:
    """Build a filesystem/log-safe queue name isolated to one account."""
    raw = make_account_key(channel_type, shop_id, user_id)
    return f"{channel_value(channel_type)}_{sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def context_scope(context: Optional[Context]) -> Dict[str, str]:
    """Return the trusted identity fields for downstream handlers/tools."""
    channel = channel_value(getattr(context, "channel_type", None))
    shop_id = _context_value(context, "shop_id", "")
    user_id = _context_value(context, "user_id", "")
    from_uid = _context_value(context, "from_uid", "")
    return {
        "channel_type": channel,
        "shop_id": shop_id,
        "user_id": user_id,
        "recipient_uid": from_uid,
        "account_key": make_account_key(channel, shop_id, user_id),
        "conversation_key": make_conversation_key(context),
    }

