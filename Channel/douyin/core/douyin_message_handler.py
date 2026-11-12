"""Douyin message conversion and account-scoped dispatch.

结构与拼多多 pdd_message_handler 完全一致，仅替换渠道专属引用：
- 消息模型用 DouyinChatMessage
- Context 构造用 Context.create_douyin_context
- handoff key 用 ChannelType.DOUYIN
- SendMessage 引用 Channel.douyin.utils.API.send_message
"""

from __future__ import annotations

import asyncio
import json
import time

from bridge.context import ChannelType, Context, ContextType, make_handoff_key
from Channel.douyin.douyin_message import DouyinChatMessage
from Channel.douyin.message_rules import Origin, resolve_origin
from core.session_state import SessionState
from database import db_manager
from utils.config_updater import update_config_with_uid
from utils.logger_loguru import get_logger


# ===== 买家真实昵称解析（会话列表接口，带 TTL 缓存）=====
# v3 get_user_message 时间线条目不带买家昵称（实测响应里的 nickname 全是
# 卖家/系统标签），而抖店订单接口 searchlist 的 buyer= 参数要求「买家真实
# 昵称」精确匹配（见 Channel/douyin/ORDER_API.md §3.1）。此前
# _process_pigeon_event 硬编码 "pigeon_" + conv_id[-6:] 占位昵称，导致订单
# 昵称反查在生产路径永远查不到订单。这里从飞鸽会话列表（msgList 的
# ext/nickname 是买家真实昵称）解析并缓存，失败时退回占位昵称不阻塞消息处理。
_NICK_CACHE: dict = {}          # conv_id -> {"nick": str, "ts": float}
_NICK_TTL_SECONDS = 600.0       # 昵称缓存 10 分钟（买家改昵称低频）
_NICK_FETCH_MIN_INTERVAL = 30.0 # 同一会话两次拉取的最小间隔（限频防刷）


def _get_cached_nickname(conv_id: str):
    """读昵称缓存，未过期且有值时返回昵称，否则 None。"""
    import time as _time
    entry = _NICK_CACHE.get(conv_id)
    if entry is None:
        return None
    if _time.time() - entry["ts"] > _NICK_TTL_SECONDS:
        return None
    nick = str(entry.get("nick") or "")
    return nick or None


def _fetch_buyer_nickname(conv_id: str) -> str:
    """从飞鸽会话列表解析买家真实昵称（同步阻塞，调用方放 to_thread）。

    查找顺序：内存缓存 → 会话列表接口（msgList 中买家消息的
    ext/nickname，经 extract_buyer_nick_from_ext 过滤平台占位值）。
    拉取失败/找不到时返回空串；限频避免每条消息都拉一次列表。
    """
    import time as _time
    cached = _get_cached_nickname(conv_id)
    if cached:
        return cached

    now = _time.time()
    last_fetch = _NICK_CACHE.get(conv_id, {}).get("ts") or 0.0
    if now - last_fetch < _NICK_FETCH_MIN_INTERVAL:
        # 刚拉过但没拿到（买家还没在会话里发过消息），限频内直接放弃
        return ""

    try:
        from tools.reverse_probe.douyin.pigeon_conv_list import PigeonConvList
        from Channel.douyin.utils.API.order_query import extract_buyer_nick_from_ext
        from Channel.douyin.core.douyin_lifecycle import PIGEON_SESSION_PATH

        conv_list = PigeonConvList(PIGEON_SESSION_PATH)
        convs = conv_list.fetch_conversations() or []
    except Exception as exc:
        get_logger("DouyinNick").warning(
            f"会话列表拉取失败，昵称反查降级为占位: conv={conv_id}, "
            f"error_type={type(exc).__name__}"
        )
        _NICK_CACHE[conv_id] = {"nick": "", "ts": now}
        return ""

    nick = ""
    for conv in convs:
        # v3 的 conv_id 与会话列表的 parentConShortId 同一命名空间
        # （发送即用 parentConShortId，实测 ACK 成功）。
        cid = str(conv.get("parentConShortId") or "")
        if not cid:
            cid = str(conv.get("subConversationId") or "")
        if cid != conv_id:
            continue
        # 昵称在消息级 ext 里，取第一条能解析出真实昵称的买家消息
        for item in conv.get("msgList") or []:
            body = item.get("messageBody") or {}
            candidate = extract_buyer_nick_from_ext(body.get("ext"))
            if candidate:
                nick = candidate
                break
        break

    _NICK_CACHE[conv_id] = {"nick": nick, "ts": now}
    if nick:
        get_logger("DouyinNick").info(
            f"买家真实昵称解析成功: conv={conv_id}, nick={nick!r}"
        )
    return nick


class MessageHandlerMixin:
    async def _setup_message_consumer(
        self,
        queue_name: str,
        shop_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Create one consumer and one Agent for this DouyinChannel/account."""
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

            # TODO(传输缝): 抖音真实消息帧格式未知，DouyinChatMessage 的字段解析
            # 与类型映射需按抓包结果补全。此处沿用与拼多多相同的解析入口。
            douyin_message = DouyinChatMessage(message_data)
            context = await asyncio.to_thread(
                self._convert_to_context, douyin_message, shop_id, user_id, username
            )
            if not context:
                return

            if self._should_process_immediately(context):
                await self._handle_immediate_message(context, shop_id, user_id)
            elif self._should_queue_message(context):
                msg_id = await self.queue_manager.get_or_create_queue(queue_name).put(
                    context
                )
                self.logger.debug(
                    f"message queued: {queue_name}, ID: {msg_id}, type: {context.type}"
                )
        except json.JSONDecodeError:
            self.logger.error("invalid websocket JSON message")
        except Exception as exc:
            self.logger.error(
                f"websocket message handling failed: error_type={type(exc).__name__}"
            )

    async def _process_bridge_event(
        self,
        event: dict,
        shop_id: str,
        user_id: str,
        username: str,
        queue_name: str,
    ) -> None:
        """处理飞鸽浏览器桥事件（方案A 接收入口）。

        event 形如 {"kind": "buyer_msg", "session": <买家名>, "content": <文本>,
                     "msg_id": ..., "ts": ...}
        将其构造成 DouyinChatMessage 兼容的消息字典，复用与 WS 相同的
        _convert_to_context + 队列/立即处理 管线。
        """
        if not event or event.get("kind") != "buyer_msg":
            return
        session = str(event.get("session") or "")
        content = str(event.get("content") or "")
        if not session or not content.strip():
            return
        msg_dict = {
            "type": "push",
            "message": {
                "msg_id": str(event.get("msg_id") or ""),
                "nickname": session,
                "from": {"role": "user", "uid": session},
                "to": {"role": "mall_cs", "uid": ""},
                "content": content,
                "type": 0,  # DouyinMsgType.TEXT
                # PinduoduoKwargs.timestamp 为 str；桥事件 ts 是数值，需转字符串
                "time": str(int(event.get("ts") or time.time())),
            },
        }
        try:
            douyin_message = DouyinChatMessage(msg_dict)
            context = await asyncio.to_thread(
                self._convert_to_context, douyin_message, shop_id, user_id, username
            )
            if not context:
                return
            self.logger.debug(
                f"飞鸽桥消息 -> context: session={session}, content={content!r}"
            )
            if self._should_process_immediately(context):
                await self._handle_immediate_message(context, shop_id, user_id)
            elif self._should_queue_message(context):
                msg_id = await self.queue_manager.get_or_create_queue(queue_name).put(context)
                self.logger.debug(
                    f"飞鸽桥消息入队: {queue_name}, msg_id={msg_id}, type={context.type}"
                )
        except Exception as exc:
            self.logger.error(
                f"飞鸽桥消息处理失败: error_type={type(exc).__name__}, exc={exc}"
            )

    async def _process_pigeon_event(
        self,
        event: dict,
        shop_id: str,
        user_id: str,
        username: str,
        queue_name: str,
    ) -> None:
        """处理 pigeon 纯协议事件（方案B 接收入口，与拼多多同构）。

        event 形如 decode_inbound 输出：
          {type, conv_id, msg_id, text, ts, security_receiver_id, shop_id,
           sender_role, client_message_id, raw_kv}
        只处理买家文本消息：type=text 且 sender_role != 商家(2) 且非控制 JSON。
        conv_id 作为 from_uid，AI 回复经 SendMessage.send_text(conv_id) 回发，
        由 PigeonTransport.conv_state 回填 security_receiver_id/shop_id。
        """
        try:
            # 与拼多多一致：WS 推过来就立刻派发，不额外等 HTTP 轮询。
            # 飞鸽 WS 帧的 type 取值不固定（message / 1000 / text 都出现过），
            # 原来只认 "text" 会把 WS 正文全部丢掉，只能靠 HTTP 轮询兜底，
            # 这正是抖音比拼多多慢一截的原因之一。
            if not event or not event.get("text"):
                return
            event_type = str(event.get("type") or "").lower()
            if event_type in ("is_ack", "ack", "ping", "pong", "heartbeat"):
                return
            # 过滤商家自身回声 / 平台提示 / 控制 JSON / 空内容。
            # 来源判定收敛到 Channel/douyin/message_rules（与拼多多
            # resolve_origin 同构）：平台机器人提示的正文同样是文本，
            # 只看 sender_role 不在黑名单会把「平台可能主动介入」交给 AI
            # 去回复一条不该回复的消息。
            origin = resolve_origin({
                "s:sender_biz_role": event.get("sender_biz_role"),
                "sender_role": event.get("sender_role"),
                "systemHint": event.get("system_hint"),
                "biz_sender": event.get("biz_sender"),
                "type": event.get("type"),
            })
            if origin is not Origin.CUSTOMER:
                if origin is Origin.SYSTEM:
                    self.logger.debug(
                        f"已忽略平台/系统消息: conv={event.get('conv_id')}, "
                        f"text={str(event.get('text'))[:30]!r}"
                    )
                return
            content = str(event.get("text") or "").strip()
            if not content or content.startswith("{"):
                return
            conv_id = str(event.get("conv_id") or "")
            if not conv_id:
                return
            # 买家真实昵称（订单昵称反查的关联键，见 ORDER_API.md §3.1）。
            # 解析失败时退回占位昵称，不阻塞消息处理。
            buyer_nick = await asyncio.to_thread(_fetch_buyer_nickname, conv_id)
            if not buyer_nick:
                buyer_nick = "pigeon_" + conv_id[-6:]
            msg_dict = {
                "type": "push",
                "message": {
                    "msg_id": str(event.get("msg_id") or ""),
                    "nickname": buyer_nick,
                    "from": {"role": "user", "uid": conv_id},
                    "to": {"role": "mall_cs", "uid": ""},
                    "content": content,
                    "type": 0,  # DouyinMsgType.TEXT
                    "time": str(int(event.get("ts") or time.time())),
                },
            }
            douyin_message = DouyinChatMessage(msg_dict)
            context = await asyncio.to_thread(
                self._convert_to_context, douyin_message, shop_id, user_id, username
            )
            if not context:
                return
            self.logger.debug(
                f"pigeon 消息 -> context: conv={conv_id}, content={content!r}"
            )
            if self._should_process_immediately(context):
                await self._handle_immediate_message(context, shop_id, user_id)
            elif self._should_queue_message(context):
                msg_id = await self.queue_manager.get_or_create_queue(queue_name).put(context)
                self.logger.debug(
                    f"pigeon 消息入队: {queue_name}, msg_id={msg_id}, type={context.type}"
                )
        except Exception as exc:
            self.logger.error(
                f"pigeon 消息处理失败: error_type={type(exc).__name__}, exc={exc}"
            )

    def _should_process_immediately(self, context: Context) -> bool:
        return context.type in {
            ContextType.SYSTEM_STATUS,
            ContextType.AUTH,
            ContextType.WITHDRAW,
            ContextType.SYSTEM_HINT,
            ContextType.MALL_CS,
            ContextType.TRANSFER,
        }

    def _should_queue_message(self, context: Context) -> bool:
        return context.type in {
            ContextType.TEXT,
            ContextType.IMAGE,
            ContextType.VIDEO,
            ContextType.EMOTION,
            ContextType.GOODS_INQUIRY,
            ContextType.ORDER_INFO,
            ContextType.GOODS_CARD,
            ContextType.GOODS_SPEC,
        }

    async def _handle_immediate_message(
        self,
        context: Context,
        shop_id: str,
        user_id: str,
    ) -> None:
        """立即处理系统消息（认证/撤回/转接等，不进入 AI 队列）"""
        kwargs = context.kwargs
        username = getattr(kwargs, "username", None)
        recipient_uid = getattr(kwargs, "from_uid", None)
        if isinstance(kwargs, dict):
            username = username or kwargs.get("username")
            recipient_uid = recipient_uid or kwargs.get("from_uid")
        username = username or ""
        recipient_uid = recipient_uid or ""
        try:
            # 认证消息：捕获完整 UID 写入配置，无需创建 SendMessage
            if context.type == ContextType.AUTH:
                self._capture_auth_uid(context)
                return

            from Channel.douyin.utils.API.send_message import SendMessage
            send_message = SendMessage(shop_id, user_id)
            if context.type == ContextType.WITHDRAW:
                self.logger.info(f"收到撤回消息: {context.content}")
                send_message.send_text(recipient_uid, "[玫瑰]")

            elif context.type == ContextType.SYSTEM_STATUS:
                self.logger.debug(f"系统状态消息: {context.content}")

            elif context.type == ContextType.SYSTEM_HINT:
                self.logger.info(f"系统提示: {context.content}")

            elif context.type == ContextType.MALL_CS:
                self.logger.debug(f"收到客服消息: {context.content}")
                # 人工客服（非本程序账号）主动回复买家 → 静默转人工：
                # 标记该买家会话在 config.handoff.valid_hours 时长内 AI 完全静默、
                # 不回复、也不发任何企业微信通知；过期后自动恢复自动回复。
                self._handle_human_agent_reply(context, shop_id, user_id)

            elif context.type == ContextType.SYSTEM_BIZ:
                self.logger.info(f"系统业务消息: {context.content}")

            elif context.type == ContextType.MALL_SYSTEM_MSG:
                self.logger.info(f"商城系统消息: {context.content}")

            elif context.type == ContextType.TRANSFER:
                self.logger.info(f"转接消息: {context.content}")
                send_message.send_text(recipient_uid, "[玫瑰]")

        except Exception as e:
            self.logger.error(f"立即处理消息失败: {e}")

    def _handle_human_agent_reply(
        self, context: Context, shop_id: str, user_id: str
    ) -> None:
        """检测人工客服（非本程序账号）主动回复买家，静默标记该会话转人工。

        【平台差异 / TODO】
        拼多多靠 `DouyinChatMessage.from_user == "mall_cs"` 把"这是人工客服发的消息"
        标记为 MALL_CS 角色并进入本分支；抖音判定人工客服消息的字段/常量可能不同，
        请到 Channel/douyin/douyin_message.py 的 DouyinChatMessage 中，按抖音实际
        消息的 role 字段填写 `DOUYIN_CS_ROLE`（或对应判定函数）。只有当消息被正确
        归类为 MALL_CS 时，才会进入本分支执行静默转人工。

        命中本分支后（即平台已确认这是人工客服消息），逻辑与拼多多完全一致：
        过滤本程序自身账号发出的消息，避免 AI 被自己的回复误判为"人工介入"而停手；
        命中后调用 SessionState.mark_handoff 静默转人工。
        """
        kwargs = context.kwargs
        from_uid = getattr(kwargs, "from_uid", None) or ""
        # 抖音派发时 from_uid 就是会话 ID（conv_id），ai_handler 检查 is_handoff
        # 用的也是它。取 to_uid 会拿到空串（to.uid 一直是空），判定直接失效。
        conv_id = str(from_uid)

        if not conv_id:
            self.logger.debug("无法识别会话 ID，跳过人工介入静默标记")
            return

        session_key = make_handoff_key(ChannelType.DOUYIN, shop_id, conv_id)
        try:
            if SessionState().is_handoff(session_key):
                return
            SessionState().mark_handoff(session_key)
            self.logger.info(
                f"检测到人工客服主动回复，会话静默转人工（不再回复且不通知）: "
                f"session_key={session_key}"
            )
        except Exception as e:
            self.logger.error(f"人工介入静默标记失败（不影响主流程）: {e}")

    def _capture_auth_uid(self, context: Context) -> None:
        """认证成功后捕获完整 UID 并写入 config.json（与拼多多逻辑一致）"""
        try:
            auth_info = json.loads(context.content or "")
        except (json.JSONDecodeError, TypeError):
            self.logger.warning("认证消息格式异常，跳过 UID 捕获")
            return
        if not isinstance(auth_info, dict):
            self.logger.warning("认证消息格式异常，跳过 UID 捕获")
            return

        result = auth_info.get('result')
        if result != 'ok':
            self.logger.warning(f"{context.kwargs.username}认证失败")
            return
        self.logger.info(f"{context.kwargs.username}认证成功")

        uid = auth_info.get('uid')
        if not uid:
            self.logger.warning("认证响应缺少 uid，跳过 UID 捕获")
            return
        update_config_with_uid(str(uid))

    def _convert_to_context(self, douyin_message: DouyinChatMessage, shop_id: str, user_id: str, username: str) -> Context:
        """将抖音消息转换为Context格式"""
        shop_info = db_manager.get_shop(self.channel_name, shop_id) or {}
        shop_name = shop_info.get("shop_name", "")
        content = douyin_message.content
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        else:
            content = str(content)

        return Context.create_douyin_context(
            content=content,
            msg_id=str(douyin_message.msg_id) if douyin_message.msg_id is not None else "",
            from_user=str(douyin_message.from_user or ""),
            from_uid=str(douyin_message.from_uid or ""),
            to_user=str(douyin_message.to_user or ""),
            to_uid=str(douyin_message.to_uid or ""),
            nickname=str(douyin_message.nickname or ""),
            timestamp=douyin_message.timestamp,
            user_msg_type=douyin_message.user_msg_type,
            shop_id=str(shop_id),
            user_id=str(user_id),
            username=str(username),
            shop_name=str(shop_name),
            raw_data=douyin_message.raw_data,
            channel_type=ChannelType.DOUYIN,
        )


__all__ = ["MessageHandlerMixin"]
