"""回复发送抽象层。

Message/Agent 层通过本模块获取发送器，避免直接 import 具体渠道（如拼多多），
使处理器链与工具保持渠道无关。新增渠道时实现 ReplySender 并在 _SENDERS 注册。

各发送方法均为同步阻塞调用（HTTP），调用方应在工作线程中执行
（如 asyncio.to_thread）以避免阻塞事件循环。
"""
from __future__ import annotations

from typing import Any, Optional

from bridge.context import ChannelType


class ReplySender:
    """回复发送器抽象基类（传输层）。"""

    def send_text(self, shop_id, user_id, recipient_uid, text: str) -> Any:
        raise NotImplementedError

    def send_image(self, shop_id, user_id, recipient_uid, image_url: str) -> Any:
        raise NotImplementedError

    def send_product_card(self, shop_id, user_id, recipient_uid, goods_id, biz_type: int = 2) -> Any:
        raise NotImplementedError

    def get_cs_list(self, shop_id, user_id) -> Optional[dict]:
        raise NotImplementedError

    def transfer_to_cs(self, shop_id, user_id, recipient_uid, cs_uid) -> Any:
        raise NotImplementedError


class PinduoduoSender(ReplySender):
    """拼多多发送器，封装现有 SendMessage。"""

    def send_text(self, shop_id, user_id, recipient_uid, text):
        from Channel.pinduoduo.utils.API.send_message import SendMessage
        return SendMessage(str(shop_id), str(user_id)).send_text(recipient_uid, text)

    def send_image(self, shop_id, user_id, recipient_uid, image_url):
        from Channel.pinduoduo.utils.API.send_message import SendMessage
        return SendMessage(str(shop_id), str(user_id)).send_image(recipient_uid, image_url)

    def send_product_card(self, shop_id, user_id, recipient_uid, goods_id, biz_type=2):
        from Channel.pinduoduo.utils.API.send_message import SendMessage
        return SendMessage(str(shop_id), str(user_id)).send_mallGoodsCard(recipient_uid, goods_id, biz_type)

    def get_cs_list(self, shop_id, user_id):
        from Channel.pinduoduo.utils.API.send_message import SendMessage
        return SendMessage(str(shop_id), str(user_id)).getAssignCsList()

    def transfer_to_cs(self, shop_id, user_id, recipient_uid, cs_uid):
        from Channel.pinduoduo.utils.API.send_message import SendMessage
        return SendMessage(str(shop_id), str(user_id)).move_conversation(recipient_uid, cs_uid)


# 渠道 -> 发送器 注册表
_SENDERS = {
    ChannelType.PINDUODUO: PinduoduoSender(),
}


def get_sender(channel_type=None) -> Optional[ReplySender]:
    """获取指定渠道的发送器。

    channel_type 为 None 或未注册时，回退到默认渠道（拼多多，当前唯一渠道）。
    """
    if channel_type is not None:
        try:
            ct = channel_type if isinstance(channel_type, ChannelType) else ChannelType(str(channel_type))
        except ValueError:
            # 无法识别的渠道标识：按“未注册”处理，回退到默认渠道
            ct = None
        if ct in _SENDERS:
            return _SENDERS[ct]
    return _SENDERS.get(ChannelType.PINDUODUO)
