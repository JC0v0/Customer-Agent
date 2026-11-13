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


class DouyinSender(ReplySender):
    """抖音发送器，封装 douyin SendMessage。

    抖音的 SendMessage 内部自己做三级降级：pigeon 纯协议（HTTP → WS）→
    浏览器桥。这里只做转发，不重复实现降级，也不吞异常，让上层看到真实原因。
    """

    def send_text(self, shop_id, user_id, recipient_uid, text):
        from Channel.douyin.utils.API.send_message import SendMessage
        return SendMessage(str(shop_id), str(user_id)).send_text(recipient_uid, text)

    def send_image(self, shop_id, user_id, recipient_uid, image_url):
        from Channel.douyin.utils.API.send_message import SendMessage
        return SendMessage(str(shop_id), str(user_id)).send_image(recipient_uid, image_url)

    def send_product_card(self, shop_id, user_id, recipient_uid, goods_id, biz_type=2):
        from Channel.douyin.utils.API.send_message import SendMessage
        return SendMessage(str(shop_id), str(user_id)).send_mallGoodsCard(recipient_uid, goods_id, biz_type)

    def get_cs_list(self, shop_id, user_id):
        from Channel.douyin.utils.API.send_message import SendMessage
        return SendMessage(str(shop_id), str(user_id)).getAssignCsList()

    def transfer_to_cs(self, shop_id, user_id, recipient_uid, cs_uid):
        from Channel.douyin.utils.API.send_message import SendMessage
        return SendMessage(str(shop_id), str(user_id)).move_conversation(recipient_uid, cs_uid)


# 渠道 -> 发送器 注册表
_SENDERS = {
    ChannelType.PINDUODUO: PinduoduoSender(),
    ChannelType.DOUYIN: DouyinSender(),
}


def get_sender(channel_type=None) -> Optional[ReplySender]:
    """获取指定渠道的发送器。

    历史行为：channel_type 为 None 时回退默认渠道（拼多多）。这条对无参
    调用的老工具保持兼容。

    但**明确传入**的渠道绝不再静默回退拼多多：抖音会话曾因此把回复交给
    拼多多 SendMessage，日志报「未找到店铺: douyin_xxx (渠道: pinduoduo)」，
    消息直接发不出去。未注册的渠道改为返回 None，由调用方记录「无可用发送器」。
    """
    if channel_type is None:
        return _SENDERS.get(ChannelType.PINDUODUO)
    if isinstance(channel_type, ChannelType):
        return _SENDERS.get(channel_type)
    try:
        ct = ChannelType(str(channel_type))
    except ValueError:
        # 无法识别的字符串仍按历史行为回退，避免影响既有调用方
        return _SENDERS.get(ChannelType.PINDUODUO)
    return _SENDERS.get(ct)
