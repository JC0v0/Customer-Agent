"""
抖音消息处理类

结构与拼多多 pdd_message.py 完全一致，仅渠道化命名 + 抖音专属字段映射。
所有"抖音真实消息帧格式/类型码/role 常量"处均标注 TODO，需按抓包结果补全。

关键平台差异（人工客服消息识别）：
拼多多用 `from_user == "mall_cs"` 判定"这是人工客服发的消息"并归入 MALL_CS；
抖音判定人工客服的字段/常量未知，请按抖音实际消息的 role 字段填写下面的
`DOUYIN_CS_ROLE`。只有被正确归类为 MALL_CS 的消息，才会在
MessageHandlerMixin._handle_human_agent_reply 中触发静默转人工。
"""
from bridge.context import ContextType
from Message.message import ChatMessage
from enum import IntEnum
from typing import Any, Dict, Optional
from utils.logger_loguru import get_logger

# TODO(传输缝): 抖音"人工客服/小蜜"消息的 role 常量未知，请按实际抓包填写
# （例如抖音 IM 可能用 "cs" / "service" / 特定 role 字符串）。
# 当前占位值不会命中任何真实消息，意味着人工客服静默转人工逻辑暂不触发，
# 待你补全真实常量后自动生效，无需改动其它文件。
DOUYIN_CS_ROLE = "TODO_CS_ROLE"


class DouyinMsgType(IntEnum):
    """抖音消息类型枚举（TODO: 用真实类型码替换占位值）"""
    TEXT = 0
    IMAGE = 1
    VIDEO = 14
    WITHDRAW = 1002
    EMOTION = 5
    GOODS_SPEC = 64
    TRANSFER = 24


class DouyinSubType(IntEnum):
    """抖音消息子类型枚举（TODO: 用真实子类型码替换占位值）"""
    ORDER_INFO = 1
    GOODS_INQUIRY = 0


def _safe_get(data: Dict[str, Any], *keys, default=None) -> Any:
    """安全获取嵌套字典值，避免链式get()时中间值为None导致AttributeError"""
    result = data
    for key in keys:
        if not isinstance(result, dict):
            return default
        result = result.get(key)
        if result is None:
            return default
    return result


class BaseMessageHandler:
    def __init__(self, msg):
        self.msg = msg
        self.data = msg.get("message", {})

    def get_basic_info(self):
        """获取基础信息（抖音字段路径假设与拼多多一致，需按真实帧补全）"""
        return {
            "msg_id": self.data.get("msg_id"),
            "nickname": self.data.get("nickname"),
            "from_role": self.data.get("from", {}).get("role"),
            "from_uid": self.data.get("from", {}).get("uid"),
            "to_role": self.data.get("to", {}).get("role"),
            "to_uid": self.data.get("to", {}).get("uid"),
            "timestamp": self.data.get("time"),
        }


class MessageTypeHandler:
    """消息类型处理类（字段路径与拼多多同构，需按抖音真实帧补全）"""

    @staticmethod
    def _get_content(msg_data: Dict[str, Any], context_type: ContextType, path: tuple) -> tuple:
        return context_type, _safe_get(msg_data, *path)

    @staticmethod
    def handle_text(msg_data):
        return MessageTypeHandler._get_content(msg_data, ContextType.TEXT, ("message", "content"))

    @staticmethod
    def handle_image(msg_data):
        return MessageTypeHandler._get_content(msg_data, ContextType.IMAGE, ("message", "content"))

    @staticmethod
    def handle_video(msg_data):
        return MessageTypeHandler._get_content(msg_data, ContextType.VIDEO, ("message", "content"))

    @staticmethod
    def handle_emotion(msg_data):
        return MessageTypeHandler._get_content(msg_data, ContextType.EMOTION, ("info", "description"))

    @staticmethod
    def handle_withdraw(msg_data):
        return MessageTypeHandler._get_content(msg_data, ContextType.WITHDRAW, ("info", "withdraw_hint"))

    @staticmethod
    def handle_goods_inquiry(msg_data):
        goods_info = {
            "goods_id": _safe_get(msg_data, "message", "info", "goodsID"),
            "goods_name": _safe_get(msg_data, "message", "info", "goodsName"),
            "goods_price": _safe_get(msg_data, "message", "info", "goodsPrice"),
            "goods_thumb_url": _safe_get(msg_data, "message", "info", "goodsThumbUrl"),
            "link_url": _safe_get(msg_data, "message", "info", "linkUrl"),
        }
        return ContextType.GOODS_INQUIRY, goods_info

    @staticmethod
    def handle_goods_spec(msg_data):
        goods_info = {
            "goods_id": _safe_get(msg_data, "message", "info", "data", "goodsID"),
            "goods_name": _safe_get(msg_data, "message", "info", "data", "goodsName"),
            "goods_price": _safe_get(msg_data, "message", "info", "data", "goodsPrice"),
            "goods_spec": _safe_get(msg_data, "message", "info", "data", "spec"),
        }
        return ContextType.GOODS_SPEC, goods_info

    @staticmethod
    def handle_order_info(msg_data):
        order_info = {
            "order_id": _safe_get(msg_data, "message", "info", "orderSequenceNo"),
            "goods_id": _safe_get(msg_data, "message", "info", "goodsID"),
            "goods_name": _safe_get(msg_data, "message", "info", "goodsName"),
            "afterSalesStatus": _safe_get(msg_data, "message", "info", "afterSalesStatus"),
            "afterSalesType": _safe_get(msg_data, "message", "info", "afterSalesType"),
            "spec": _safe_get(msg_data, "message", "info", "spec"),
        }
        return ContextType.ORDER_INFO, order_info

    @staticmethod
    def handle_mall_system_msg(msg_data):
        system_msg = {
            "user_id": _safe_get(msg_data, "message", "data", "user_id"),
        }
        return ContextType.MALL_SYSTEM_MSG, system_msg

    @staticmethod
    def handle_auth(msg_data):
        auth_info = {
            "uid": _safe_get(msg_data, "uid"),
            "result": _safe_get(msg_data, "auth", "result"),
            "status": _safe_get(msg_data, "status"),
        }
        return ContextType.AUTH, auth_info

    @staticmethod
    def handle_transfer(msg_data):
        transfer_info = {
            "from_uid": _safe_get(msg_data, "message", "from", "uid"),
            "to_uid": _safe_get(msg_data, "message", "to", "uid")
        }
        return ContextType.TRANSFER, transfer_info


class DouyinChatMessage(ChatMessage):
    """抖音消息实现类

    暴露与 PDDChatMessage 完全相同的属性名：
    msg_id, nickname, from_user, from_uid, to_user, to_uid,
    user_msg_type(ContextType), content, raw_data, timestamp
    """

    def __init__(self, msg):
        super().__init__(msg)
        self.logger = get_logger("DouyinChatMessage")
        self.msg = msg
        self.raw_data = msg
        self.base_handler = BaseMessageHandler(msg)
        # 获取基本信息
        basic_info = self.base_handler.get_basic_info()
        self.msg_id = basic_info.get("msg_id")
        self.nickname = basic_info.get("nickname")
        self.from_user = basic_info.get("from_role")
        self.from_uid = basic_info.get("from_uid")
        self.to_user = basic_info.get("to_role")
        self.to_uid = basic_info.get("to_uid")
        self.timestamp = basic_info.get("timestamp")

        # 检查是否人工客服消息（平台差异，见模块顶部 DOUYIN_CS_ROLE 说明）
        if self.from_user == DOUYIN_CS_ROLE:
            self.user_msg_type = ContextType.MALL_CS
            self.content = self.msg.get("message", {}).get("content")
            return
        # 处理消息
        self._process_message()

    def _process_message(self):
        # TODO(传输缝): 抖音真实消息帧结构未知，以下解析为与拼多多同构的占位实现。
        # 请按抖音实际消息的 response/event/cmd/type 与 message.type 子类型补全映射。
        self.msg_type = (
            self.msg.get("response") or
            self.msg.get("event") or
            self.msg.get("cmd") or
            self.msg.get("type")
        )
        if self.msg_type == "push":
            message_obj = self.msg.get("message", {})
            user_msg_type = self.msg.get("message", {}).get("type")
            if user_msg_type == DouyinMsgType.TEXT:
                sub_type = self.msg.get("message", {}).get("sub_type")
                if sub_type == DouyinSubType.ORDER_INFO:
                    self.user_msg_type, self.content = MessageTypeHandler.handle_order_info(self.msg)
                elif sub_type == DouyinSubType.GOODS_INQUIRY:
                    self.user_msg_type, self.content = MessageTypeHandler.handle_goods_inquiry(self.msg)
                else:
                    self.user_msg_type, self.content = MessageTypeHandler.handle_text(self.msg)
            elif user_msg_type == DouyinMsgType.IMAGE:
                self.user_msg_type, self.content = MessageTypeHandler.handle_image(self.msg)
            elif user_msg_type == DouyinMsgType.VIDEO:
                self.user_msg_type, self.content = MessageTypeHandler.handle_video(self.msg)
            elif user_msg_type == DouyinMsgType.WITHDRAW:
                self.user_msg_type, self.content = MessageTypeHandler.handle_withdraw(self.msg)
            elif user_msg_type == DouyinMsgType.EMOTION:
                self.user_msg_type, self.content = MessageTypeHandler.handle_emotion(self.msg)
            elif user_msg_type == DouyinMsgType.GOODS_SPEC:
                self.user_msg_type, self.content = MessageTypeHandler.handle_goods_spec(self.msg)
            elif user_msg_type == DouyinMsgType.TRANSFER:
                self.user_msg_type, self.content = MessageTypeHandler.handle_transfer(self.msg)
            else:
                self.user_msg_type = ContextType.SYSTEM_STATUS
                self.content = f"不支持的消息类型: {user_msg_type}"
                self.logger.warning(self.content)
        elif self.msg_type == "auth":
            self.user_msg_type, self.content = MessageTypeHandler.handle_auth(self.msg)
        elif self.msg_type == "mall_system_msg":
            self.user_msg_type, self.content = MessageTypeHandler.handle_mall_system_msg(self.msg)
        else:
            self.user_msg_type = ContextType.SYSTEM_STATUS
            self.content = f"不支持的消息类型: {self.msg_type}"


__all__ = ['DouyinChatMessage', 'DouyinMsgType', 'DouyinSubType', 'DOUYIN_CS_ROLE']
