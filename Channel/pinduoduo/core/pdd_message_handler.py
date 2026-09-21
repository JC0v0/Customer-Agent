"""Pinduoduo message conversion and account-scoped dispatch."""

from __future__ import annotations

import asyncio
import json

from bridge.context import ChannelType, Context, ContextType
from Channel.pinduoduo.message_rules import Action, effective_action
from Channel.pinduoduo.pdd_message import PDDChatMessage
from database import db_manager
from utils.logger_loguru import get_logger


class MessageHandlerMixin:
    async def _setup_message_consumer(
        self,
        queue_name: str,
        shop_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Create one consumer and one Agent for this PDDChannel/account."""
        from Message import handler_chain
        from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
        from core.di_container import container, configure_standard_services

        try:
            existing_consumer = self.consumer_manager.get_consumer(queue_name)
            if existing_consumer is not None:
                await self.consumer_manager.stop_consumer(queue_name, remove=True)
                self.queue_manager.remove_queue(queue_name)

            consumer = self.consumer_manager.create_consumer(
                queue_name,
                max_concurrent=10,
            )
            if self._account_agent is None:
                if not container.is_registered(CustomerAgent):
                    configure_standard_services()
                self._account_agent = container.get(CustomerAgent)

            handlers = handler_chain(
                use_ai=True,
                business_hours=self.business_hours,
                bot=self._account_agent,
            )
            for handler in handlers:
                consumer.add_handler(handler)

            await self.consumer_manager.start_consumer(queue_name)
            self.logger.debug(f"message consumer started: {queue_name}")
        except Exception as exc:
            self.logger.error(
                f"message consumer setup failed: error_type={type(exc).__name__}"
            )
            raise

    async def _process_websocket_message(
        self,
        message: str,
        shop_id: str,
        user_id: str,
        username: str,
        queue_name: str,
    ) -> None:
        try:
            if not message or not message.strip():
                return

            message_data = json.loads(message)
            msg_type = message_data.get("message", {}).get("type", "unknown")
            self.logger.debug(
                f"received message: type={msg_type}, shop_id={shop_id}"
            )

            pdd_message = PDDChatMessage(message_data)
            context = await asyncio.to_thread(
                self._convert_to_context, pdd_message, shop_id, user_id, username
            )
            if not context:
                return

            await self._dispatch_by_action(
                context, shop_id, user_id, queue_name, pdd_message
            )
        except json.JSONDecodeError:
            self.logger.error("invalid websocket JSON message")
        except Exception as exc:
            self.logger.error(
                f"websocket message handling failed: error_type={type(exc).__name__}"
            )

    def _resolve_action(self, pdd_message: PDDChatMessage) -> Action:
        """取解析层判定出的动作；缺失或非法时降级为 UNKNOWN。"""
        action = getattr(pdd_message, "action", None)
        if isinstance(action, Action):
            return action
        try:
            return Action(action)
        except ValueError:
            return Action.UNKNOWN

    async def _dispatch_by_action(
        self,
        context: Context,
        shop_id: str,
        user_id: str,
        queue_name: str,
        pdd_message: PDDChatMessage,
    ) -> None:
        """按动作分派。每条消息都有归宿，不存在静默丢弃的分支。"""
        raw_action = self._resolve_action(pdd_message)
        action = effective_action(raw_action)
        origin = getattr(getattr(pdd_message, "origin", None), "value", "unknown")
        pdd_type = getattr(pdd_message, "pdd_type", None)
        sub_type = getattr(pdd_message, "pdd_sub_type", None)
        template = getattr(pdd_message, "template_name", None)

        # 铁律：未识别的组合降级为 CONTEXT_ONLY 的同时必须告警。
        # 只降级不告警，等于让动作表的缺项悄悄消失，后续无从补表。
        if raw_action is Action.UNKNOWN:
            self.logger.warning(
                f"unknown message action, degraded to context_only: "
                f"origin={origin}, type={pdd_type}, sub_type={sub_type}, "
                f"template={template}"
            )

        if action is Action.REPLY:
            msg_id = await self.queue_manager.get_or_create_queue(queue_name).put(context)
            self.logger.debug(
                f"message queued: {queue_name}, ID: {msg_id}, type: {context.type}"
            )
            return

        if action is Action.HANDOFF:
            await self._handle_immediate_message(context, shop_id, user_id)
            return

        if action is Action.CONTEXT_ONLY:
            await self._record_context(context, origin, pdd_type, template)
            return

        if action is Action.IGNORE:
            self.logger.debug(
                f"message ignored: origin={origin}, type={pdd_type}, template={template}"
            )
            return

        # OBSERVE 及任何未预期分支：只记录，绝不丢弃
        await self._handle_observe(context, origin, pdd_type, template)

    async def _record_context(
        self,
        context: Context,
        origin: str,
        pdd_type,
        template,
    ) -> None:
        """把 CONTEXT_ONLY 消息写入会话历史，不触发回复。

        客服侧文本 / 商品卡走这里，使 AI 知道刚刚推送过什么，避免重复推荐。
        """
        agent = self._account_agent
        record = getattr(agent, "record_context", None)
        if record is None:
            self.logger.debug(
                f"context message (agent 无 record_context): origin={origin}, "
                f"type={pdd_type}, template={template}"
            )
            return
        try:
            stored = await record(self._summarized(context))
        except Exception as exc:
            self.logger.warning(
                f"record context failed: error_type={type(exc).__name__}"
            )
            return
        self.logger.debug(
            f"context recorded: stored={bool(stored)}, origin={origin}, "
            f"type={pdd_type}, template={template}"
        )

    @staticmethod
    def _summarized(context: Context) -> Context:
        """把结构化内容压成一行可读文本，再写进会话历史。

        商品卡 / 规格卡 / 订单卡的 content 是 JSON，原样入库既浪费 token，
        又要模型自己解析。这里复用买家侧同一套 MessagePreprocessor，
        保证两个方向格式一致；纯文本消息不做任何改动。
        """
        content = context.content
        if not isinstance(content, str):
            return context
        stripped = content.strip()
        if not stripped.startswith("{"):
            return context

        # 延迟导入：Message 包在导入期会拉起 AIReplyHandler -> CustomerAgent，
        # 模块级导入会与 Channel 成环（本文件的 handler_chain 也是这么处理的）。
        from Message.handlers.preprocessor import MessagePreprocessor

        summary = MessagePreprocessor().process(stripped, context.type)
        if not summary or summary == stripped:
            return context
        return context.model_copy(update={"content": summary})

    @staticmethod
    def _extract_auth_result(content) -> str:
        """从 auth 消息内容里取出 result。

        内容可能是 dict，也可能已被归一化为 JSON 字符串，两种都要支持。
        """
        if isinstance(content, dict):
            return str(content.get("result"))
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return content[:60]
            if isinstance(parsed, dict):
                return str(parsed.get("result"))
        return "unknown"

    async def _handle_observe(
        self,
        context: Context,
        origin: str,
        pdd_type,
        template,
    ) -> None:
        """只观察，不参与对话。

        type=30（system_push）实测内容为「账户在别处登录 请刷新重登。」，
        属于需要运维可见的信号，因此提升到 WARNING。
        """
        # auth 的连接鉴权结果保留 INFO 可见性（排查登录问题需要）。
        # 注意 _convert_to_context 已把 dict 归一化成 JSON 字符串，
        # 因此这里必须解析字符串——原实现只判断 dict，日志实际从未打印。
        if context.type == ContextType.AUTH:
            self.logger.info(f"auth result: {self._extract_auth_result(context.content)}")
            return

        message = (
            f"observe: origin={origin}, type={pdd_type}, "
            f"template={template}, ctx={context.type}"
        )
        # type=30 实测为「账户在别处登录 请刷新重登。」，需运维可见
        if pdd_type == 30 and context.content:
            self.logger.warning(f"{message}, content={str(context.content)[:120]}")
        elif context.type == ContextType.TRANSFER:
            # 默认日志级别是 INFO（utils/logger_loguru.DEFAULT_LOG_LEVEL），
            # 转接若记在 debug 等于没记；而「转接通知可排查」正是
            # 客服侧仍然要解析消息的理由之一，必须留在 INFO。
            self.logger.info(f"{message}, content={str(context.content)[:120]}")
        else:
            self.logger.debug(message)

    async def _handle_immediate_message(
        self,
        context: Context,
        shop_id: str,
        user_id: str,
    ) -> None:
        """处理 HANDOFF 动作：目前只有买家侧会话转接。

        AUTH 归入 OBSERVE、WITHDRAW 归入 IGNORE，都不再到这条路径，
        因此这里不再保留它们的分支，避免出现永不执行的死代码。
        """
        kwargs = context.kwargs
        recipient_uid = getattr(kwargs, "from_uid", None)
        if isinstance(kwargs, dict):
            recipient_uid = recipient_uid or kwargs.get("from_uid")
        recipient_uid = recipient_uid or ""
        try:
            from Channel.pinduoduo.utils.API.send_message import SendMessage

            def _send_notice() -> None:
                # 构造也要放进工作线程：SendMessage -> BaseRequest.__init__ ->
                # _init_account_info() 会同步读 cookie 缓存与数据库，
                # 在事件循环线程里构造会阻塞整个连接的收发。
                SendMessage(shop_id, user_id).send_text(recipient_uid, "[玫瑰]")

            if context.type == ContextType.TRANSFER:
                await asyncio.to_thread(_send_notice)
            else:
                self.logger.debug(f"handoff: unhandled type {context.type}")
        except Exception as exc:
            self.logger.error(
                f"immediate message handling failed: error_type={type(exc).__name__}"
            )

    def _convert_to_context(
        self,
        pdd_message: PDDChatMessage,
        shop_id: str,
        user_id: str,
        username: str,
    ) -> Context:
        shop_info = db_manager.get_shop(self.channel_name, shop_id) or {}
        shop_name = shop_info.get("shop_name", "")
        content = pdd_message.content
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        else:
            content = str(content)

        return Context.create_pinduoduo_context(
            content=content,
            msg_id=str(pdd_message.msg_id) if pdd_message.msg_id is not None else "",
            from_user=str(pdd_message.from_user or ""),
            from_uid=str(pdd_message.from_uid or ""),
            to_user=str(pdd_message.to_user or ""),
            to_uid=str(pdd_message.to_uid or ""),
            nickname=str(pdd_message.nickname or ""),
            timestamp=pdd_message.timestamp,
            user_msg_type=pdd_message.user_msg_type,
            shop_id=str(shop_id),
            user_id=str(user_id),
            username=str(username),
            shop_name=str(shop_name),
            raw_data=pdd_message.raw_data,
            channel_type=ChannelType.PINDUODUO,
            origin=getattr(getattr(pdd_message, "origin", None), "value", None),
            action=getattr(getattr(pdd_message, "action", None), "value", None),
            pdd_type=getattr(pdd_message, "pdd_type", None),
            pdd_sub_type=getattr(pdd_message, "pdd_sub_type", None),
            template_name=getattr(pdd_message, "template_name", None),
        )


__all__ = ["MessageHandlerMixin"]
