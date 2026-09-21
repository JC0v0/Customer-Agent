"""
拼多多消息处理类

本模块只负责「解析」：把服务端报文还原成 (ContextType, content) 与来源 / 动作。
类型枚举与动作表集中在 Channel/pinduoduo/message_rules.py，
路由决策由 pdd_message_handler.py 依据本模块产出的 action 执行。
"""
from bridge.context import ContextType
from Message.message import ChatMessage
from typing import Any, Dict, Optional

from Channel.pinduoduo.message_rules import (
    Action,
    Origin,
    PDDMsgType,
    PDDSubType,
    RESPONSE_PUSH,
    classify,
    resolve_origin,
)

# 客服侧这些类型统一收敛为「我方消息」语义（MALL_CS）；
# 卡片与转接保留细分类型，供上下文注入与日志使用。
# 订单卡（type=8）特意不在此列：它携带售后状态、订单号与商品名，
# 压成 MALL_CS 等于把这些字段丢掉，买家追问「退货地址」时就答不上来。
_MERCHANT_MALL_CS_TYPES = frozenset({
    PDDMsgType.TEXT,
    PDDMsgType.IMAGE,
    PDDMsgType.EMOTION,
    PDDMsgType.VIDEO,
})


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
        self.data = msg.get("message",{})
    def get_basic_info(self):
        """获取基础信息"""
        return {
            "msg_id": self.data.get("msg_id"),
            "nickname": self.data.get("nickname"),
            "from_role": self.data.get("from",{}).get("role"),
            "from_uid": self.data.get("from",{}).get("uid"),
            "to_role": self.data.get("to",{}).get("role"),
            "to_uid": self.data.get("to",{}).get("uid"),
            # 服务端实际下发 message.ts；time 为历史字段，保留兜底
            "timestamp": self.data.get("ts") or self.data.get("time"),
        }

        
class MessageTypeHandler:
    """消息类型处理类"""

    @staticmethod
    def _get_content(msg_data: Dict[str, Any], context_type: ContextType, path: tuple) -> tuple:
        """通用内容提取"""
        return context_type, _safe_get(msg_data, *path)

    @staticmethod
    def handle_text(msg_data):
        """处理文本消息"""
        return MessageTypeHandler._get_content(msg_data, ContextType.TEXT, ("message", "content"))

    @staticmethod
    def handle_image(msg_data):
        """处理图片消息"""
        return MessageTypeHandler._get_content(msg_data, ContextType.IMAGE, ("message", "content"))

    @staticmethod
    def handle_video(msg_data):
        """处理视频消息"""
        return MessageTypeHandler._get_content(msg_data, ContextType.VIDEO, ("message", "content"))

    @staticmethod
    def handle_emotion(msg_data):
        """处理表情消息

        表情详情在 message.info 下；此前路径缺少 message 前缀，
        导致买家表情的 content 恒为 None。
        """
        return MessageTypeHandler._get_content(
            msg_data, ContextType.EMOTION, ("message", "info", "description")
        )

    @staticmethod
    def handle_withdraw(msg_data):
        """处理撤回消息"""
        return MessageTypeHandler._get_content(msg_data, ContextType.WITHDRAW, ("info", "withdraw_hint"))

    @staticmethod
    def handle_goods_card(msg_data):
        """处理商品卡片（type=0 sub_type=2）

        商品卡片是我方发出的消息，字段与商品咨询一致，
        复用其字段提取，只改变 ContextType。
        """
        _, goods_info = MessageTypeHandler.handle_goods_inquiry(msg_data)
        return ContextType.GOODS_CARD, goods_info

    @staticmethod
    def handle_goods_inquiry(msg_data):
        """处理商品咨询消息"""
        goods_info = {
            "goods_id": _safe_get(msg_data, "message", "info", "goodsID"),
            "goods_name": _safe_get(msg_data, "message", "info", "goodsName"),
            "goods_price": _safe_get(msg_data, "message", "info", "goodsPrice"),
            "goods_thumb_url": _safe_get(msg_data, "message", "info", "goodsThumbUrl"),
            "link_url": _safe_get(msg_data, "message", "info", "linkUrl"),
        }
        return ContextType.GOODS_INQUIRY,goods_info

    @staticmethod
    def handle_goods_spec(msg_data):
        """处理商品规格 / 业务卡片（type=64）

        实测 type=64 有 6 种模板，字段结构并不统一，例如：

        - 旧结构：info.data 直接含 goodsID / goodsName（驼峰）
        - 说明书：info.data.goods_info 多嵌一层（下划线命名）

        因此按结构探测取值，而不是只认 template_name。
        所有字段都取不到时回退为可读摘要，不产出全 null 字段骗下游。
        """
        data = _safe_get(msg_data, "message", "info", "data", default={})
        if not isinstance(data, dict):
            data = {}
        nested = data.get("goods_info")
        if not isinstance(nested, dict):
            nested = {}

        content = _safe_get(msg_data, "message", "content")
        goods_info = {
            "goods_id": data.get("goodsID") or nested.get("goods_id"),
            "goods_name": data.get("goodsName") or nested.get("goods_name"),
            "goods_price": data.get("goodsPrice") or nested.get("goods_price"),
            "goods_spec": data.get("spec") or data.get("specStr") or nested.get("spec"),
            "title": data.get("title") or content,
            "sub_title": data.get("sub_title"),
            "template_name": _safe_get(msg_data, "message", "template_name"),
        }

        # 判据只看商品字段本身：title / sub_title 会回退到 content，
        # 用它们判断会让「未识别」永远为假，兜底就失效了。
        has_goods_field = any(
            goods_info[key]
            for key in ("goods_id", "goods_name", "goods_price", "goods_spec")
        )
        if not has_goods_field:
            # 未识别的模板：保留原始 info，避免产出全 null 字段骗下游
            goods_info["raw_info"] = _safe_get(msg_data, "message", "info")
        return ContextType.GOODS_SPEC, goods_info

    @staticmethod
    def handle_order_info(msg_data):
        """处理订单信息消息"""
        order_info = {
            "order_id": _safe_get(msg_data, "message", "info", "orderSequenceNo"),
            "goods_id": _safe_get(msg_data, "message", "info", "goodsID"),
            "goods_name": _safe_get(msg_data, "message", "info", "goodsName"),
            "afterSalesStatus": _safe_get(msg_data, "message", "info", "afterSalesStatus"),
            "afterSalesType": _safe_get(msg_data, "message", "info", "afterSalesType"),
            "spec": _safe_get(msg_data, "message", "info", "spec"),
        }
        return ContextType.ORDER_INFO,order_info

    @staticmethod
    def handle_mall_system_msg(msg_data):
        """处理商城消息"""
        system_msg = {
            "user_id": _safe_get(msg_data, "message", "data", "user_id"),
        }
        return ContextType.MALL_SYSTEM_MSG,system_msg


    @staticmethod
    def handle_auth(msg_data):
        """处理认证消息"""
        auth_info = {
            "uid": _safe_get(msg_data, "uid"),
            "result": _safe_get(msg_data, "auth", "result"),
            "status": _safe_get(msg_data, "status"),
        }
        return ContextType.AUTH,auth_info

    @staticmethod
    def handle_transfer(msg_data):
        """处理转接消息（type=24）

        实测报文里 content 才是可读说明（「A 将该会话转移给 B，并留言：C」），
        info 里是真正的转接目标（target_id / new_csid）。
        原实现只取 from/to uid，而这两个值 basic_info 已经有了，
        等于日志里看不到「转给了谁、为什么转」，排查时没有信息量。
        """
        info = _safe_get(msg_data, "message", "info", default={})
        if not isinstance(info, dict):
            info = {}
        transfer_info = {
            "from_uid": _safe_get(msg_data, "message", "from", "uid"),
            "to_uid": _safe_get(msg_data, "message", "to", "uid"),
            "description": _safe_get(msg_data, "message", "content"),
            "origin_cs_id": info.get("origin_id"),
            "target_cs_id": info.get("target_id"),
            "new_csid": info.get("new_csid"),
        }
        return ContextType.TRANSFER, transfer_info

    @staticmethod
    def handle_read_sync(msg_data):
        """处理已读同步（type=20），没有用户可见内容"""
        return MessageTypeHandler._get_content(
            msg_data, ContextType.SYSTEM_STATUS, ("message", "content")
        )

    @staticmethod
    def handle_system_push(msg_data):
        """处理系统推送（type=30），例如账号异地登录提示"""
        return MessageTypeHandler._get_content(
            msg_data, ContextType.SYSTEM_STATUS, ("message", "content")
        )

    @staticmethod
    def handle_robot_hint(msg_data):
        """处理机器人系统提示（type=31）"""
        return MessageTypeHandler._get_content(
            msg_data, ContextType.SYSTEM_HINT, ("message", "content")
        )

    @staticmethod
    def handle_user_source(msg_data):
        """处理用户来源提示（type=41）"""
        return MessageTypeHandler._get_content(
            msg_data, ContextType.SYSTEM_HINT, ("message", "content")
        )

    @staticmethod
    def handle_order_card(msg_data):
        """处理订单卡片自动回复（type=8）

        该类型的订单信息嵌在 info.goods_info 里，字段名与
        handle_order_info 使用的驼峰命名不同，因此单独解析。
        """
        goods = _safe_get(msg_data, "message", "info", "goods_info", default={})
        if not isinstance(goods, dict):
            goods = {}
        order_card = {
            "order_id": goods.get("order_sequence_no"),
            "goods_id": goods.get("goods_id"),
            "goods_name": goods.get("goods_name"),
            "title": _safe_get(msg_data, "message", "info", "title"),
            "text": _safe_get(msg_data, "message", "info", "text"),
        }
        return ContextType.ORDER_INFO, order_card

    @staticmethod
    def handle_rich_text_faq(msg_data):
        """处理富文本 FAQ 列表（type=56），属于系统下发的常见问题菜单"""
        return MessageTypeHandler._get_content(
            msg_data, ContextType.SYSTEM_HINT, ("message", "content")
        )

    @staticmethod
    def handle_material_upload(msg_data):
        """处理素材上传回执（type=97），含 file_id 等字段"""
        upload_info = _safe_get(msg_data, "message", "data", default={})
        return ContextType.SYSTEM_STATUS, upload_info


class PDDChatMessage(ChatMessage):
    """拼多多消息实现类"""
    
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg
        self.base_handler = BaseMessageHandler(msg)
        #获取基本信息
        basic_info = self.base_handler.get_basic_info()
        self.msg_id = basic_info.get("msg_id")
        self.nickname = basic_info.get("nickname")
        self.from_user = basic_info.get("from_role")
        self.from_uid = basic_info.get("from_uid")
        self.to_user = basic_info.get("to_role")
        self.to_uid = basic_info.get("to_uid")
        self.timestamp = basic_info.get("timestamp")

        # 路由元信息：来源、类型、子类型、模板、动作
        self.response = self.msg.get("response")
        envelope = self.msg.get("message")
        if not isinstance(envelope, dict):
            envelope = {}
        self.pdd_type = envelope.get("type")
        self.pdd_sub_type = envelope.get("sub_type")
        self.template_name = envelope.get("template_name")
        self.origin = resolve_origin(self.from_user, self.response)

        # 解析不再因来源是客服而跳过——差异只体现在动作上，
        # 否则 handle_transfer 等分支永远不可达（实测转接消息均来自客服）。
        self._process_message()

        # 非 push 响应（auth / mall_system_msg / system_push）都是系统级通知
        if self.response != RESPONSE_PUSH:
            self.action = Action.OBSERVE
        else:
            self.action = classify(self.origin, self.pdd_type, self.pdd_sub_type)
        
    def _process_message(self):
        """处理消息"""
        self.msg_type=self.msg.get("response")
        if self.msg_type == "push":
            user_msg_type=self.msg.get("message",{}).get("type")
            if user_msg_type == PDDMsgType.TEXT:
                sub_type=self.msg.get("message",{}).get("sub_type")
                if sub_type == PDDSubType.ORDER_INFO:
                    self.user_msg_type,self.content = MessageTypeHandler.handle_order_info(self.msg)
                elif sub_type == PDDSubType.GOODS_INQUIRY:
                    self.user_msg_type,self.content = MessageTypeHandler.handle_goods_inquiry(self.msg)
                elif sub_type == PDDSubType.CS_GOODS_CARD:
                    self.user_msg_type,self.content = MessageTypeHandler.handle_goods_card(self.msg)
                else:
                    self.user_msg_type,self.content = MessageTypeHandler.handle_text(self.msg)
            elif user_msg_type == PDDMsgType.IMAGE:
                self.user_msg_type,self.content = MessageTypeHandler.handle_image(self.msg)
            elif user_msg_type == PDDMsgType.VIDEO:
                self.user_msg_type,self.content = MessageTypeHandler.handle_video(self.msg)
            elif user_msg_type == PDDMsgType.WITHDRAW:
                self.user_msg_type,self.content = MessageTypeHandler.handle_withdraw(self.msg)
            elif user_msg_type == PDDMsgType.EMOTION:
                self.user_msg_type,self.content = MessageTypeHandler.handle_emotion(self.msg)
            elif user_msg_type == PDDMsgType.GOODS_SPEC:
                self.user_msg_type,self.content = MessageTypeHandler.handle_goods_spec(self.msg)
            elif user_msg_type == PDDMsgType.TRANSFER:
                self.user_msg_type,self.content = MessageTypeHandler.handle_transfer(self.msg)
            elif user_msg_type == PDDMsgType.READ_SYNC:
                self.user_msg_type,self.content = MessageTypeHandler.handle_read_sync(self.msg)
            elif user_msg_type == PDDMsgType.SYSTEM_PUSH:
                self.user_msg_type,self.content = MessageTypeHandler.handle_system_push(self.msg)
            elif user_msg_type == PDDMsgType.ROBOT_HINT:
                self.user_msg_type,self.content = MessageTypeHandler.handle_robot_hint(self.msg)
            elif user_msg_type == PDDMsgType.USER_SOURCE:
                self.user_msg_type,self.content = MessageTypeHandler.handle_user_source(self.msg)
            elif user_msg_type == PDDMsgType.MATERIAL_UPLOAD:
                self.user_msg_type,self.content = MessageTypeHandler.handle_material_upload(self.msg)
            elif user_msg_type == PDDMsgType.ORDER_CARD:
                self.user_msg_type,self.content = MessageTypeHandler.handle_order_card(self.msg)
            elif user_msg_type == PDDMsgType.RICH_TEXT_FAQ:
                self.user_msg_type,self.content = MessageTypeHandler.handle_rich_text_faq(self.msg)
            else:
                self.user_msg_type = ContextType.SYSTEM_STATUS
                self.content = f"不支持的消息类型: {user_msg_type}"
        elif self.msg_type == "auth":
            self.user_msg_type,self.content = MessageTypeHandler.handle_auth(self.msg)
        elif self.msg_type == "mall_system_msg":
            self.user_msg_type,self.content = MessageTypeHandler.handle_mall_system_msg(self.msg)
        elif self.msg_type == "system_push":
            self.user_msg_type,self.content = MessageTypeHandler.handle_system_push(self.msg)
        else:
            self.user_msg_type = ContextType.SYSTEM_STATUS
            self.content = f"不支持的消息类型: {self.msg_type}"

        # 客服侧消息统一收敛为「我方消息」（MALL_CS）。
        # 例外：商品卡片（sub_type=2）保留 GOODS_CARD、转接保留 TRANSFER，
        # 便于上下文注入与日志区分。
        is_goods_card = (
            self.pdd_type == PDDMsgType.TEXT
            and self.pdd_sub_type == PDDSubType.CS_GOODS_CARD
        )
        if (
            self.origin is Origin.MERCHANT
            and self.pdd_type in _MERCHANT_MALL_CS_TYPES
            and not is_goods_card
        ):
            self.user_msg_type = ContextType.MALL_CS


# 向后兼容：历史上 PDDMsgType / PDDSubType 定义在本模块
__all__ = ["PDDChatMessage", "PDDMsgType", "PDDSubType"]
