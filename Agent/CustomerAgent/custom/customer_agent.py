"""
自定义 CustomerAgent 实现

完全自主实现，不依赖 Agno 框架。

本模块已重构，职责分离为：
- agent_config.py: 配置管理
- llm_client.py: LLM 客户端封装
- message_builder.py: 消息和 Prompt 构建
- tool_executor.py: 工具执行器
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from Agent.bot import Bot

# 导入工具模块，触发 @agent_tool 装饰器注册
from Agent.CustomerAgent.tools import (
    send_goods_link,                  # noqa: F401  — 注册 send_goods_link 工具
    move_conversation,                 # noqa: F401  — 注册 transfer_conversation 工具
    get_product_list,                 # noqa: F401  — 注册 get_shop_products 工具
    get_product_knowledge,             # noqa: F401  — 注册 get_product_knowledge 工具
    search_customer_service_knowledge,  # noqa: F401  — 注册 search_customer_service_knowledge 工具
)
from bridge.context import (
    Context,
    _context_value,
    context_scope,
    make_conversation_key,
)
from bridge.reply import Reply, ReplyType
from Agent.CustomerAgent.custom.session_manager import SessionManager
from Agent.CustomerAgent.custom.tool_decorator import get_tools_for_llm
from utils.logger_loguru import get_logger

# 导入重构后的模块
from Agent.CustomerAgent.custom.agent_config import (
    AgentConfig,
    DEFAULT_DB_PATH,
    DEFAULT_TOKEN_WINDOW,
    DEFAULT_COMPRESS_RATIO,
    DEFAULT_RETAIN_COUNT,
    DEFAULT_MAX_LOOPS,
    DEFAULT_TEMPERATURE,
)
from Agent.CustomerAgent.custom.llm_client import LLMClient, LLMResponse
from Agent.CustomerAgent.custom.message_builder import MessageBuilder
from Agent.CustomerAgent.custom.tool_executor import ToolExecutor, ToolResult

logger = get_logger("CustomerAgent")


class CustomerAgent(Bot):
    """
    自定义客服 Agent

    核心循环：
    1. 加载历史消息
    2. 检查上下文压缩
    3. 构建 messages 列表
    4. 调用 LLM → 解析 tool_calls
    5. 并行执行工具 → 回传结果
    6. 循环直到无工具调用
    7. 返回最终回复

    职责已分离到子模块：
    - AgentConfig: 配置管理
    - LLMClient: LLM API 调用
    - MessageBuilder: 消息和 Prompt 构建
    - ToolExecutor: 工具执行
    - SessionManager: 会话管理（已有独立模块）
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        token_window: int = DEFAULT_TOKEN_WINDOW,
        compress_ratio: float = DEFAULT_COMPRESS_RATIO,
        retain_count: int = DEFAULT_RETAIN_COUNT,
        max_loops: int = DEFAULT_MAX_LOOPS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__()
        self._is_initialized = False

        # 配置参数
        self._config = AgentConfig(
            db_path=db_path or DEFAULT_DB_PATH,
            token_window=token_window,
            compress_ratio=compress_ratio,
            retain_count=retain_count,
            max_loops=max_loops,
            temperature=temperature,
        )

        # 子组件（延迟初始化）
        self._llm_client: Optional[LLMClient] = None
        self._active_profile = None
        self._message_builder: Optional[MessageBuilder] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._session_manager: Optional[SessionManager] = None
        self._tools: List[Dict[str, Any]] = []
        self._initialize_lock = asyncio.Lock()
        self._conversation_locks: Dict[str, asyncio.Lock] = {}
        self._fallback_session_id = f"fallback_{uuid.uuid4().hex}"

        logger.info("CustomerAgent 实例创建成功")

    async def initialize_async(self) -> bool:
        """Initialize once, even when the first messages arrive concurrently."""
        if self._is_initialized:
            return True
        async with self._initialize_lock:
            if self._is_initialized:
                return True
            return await self._initialize_async_unlocked()

    async def _initialize_async_unlocked(self) -> bool:
        """异步初始化 Agent"""
        if self._is_initialized:
            return True

        try:
            # 1. 从配置文件加载配置
            self._config = AgentConfig.load_from_config()

            # 2. 验证配置
            if not self._config.validate():
                return False

            # Snapshot the complete validated profile.  A later settings save
            # must not mutate this account's in-flight client.
            self._active_profile = self._config.profile

            # 3. 初始化 LLM 客户端
            self._llm_client = LLMClient(
                profile=self._active_profile,
                temperature=self._config.temperature,
            )
            await self._llm_client.initialize()

            # 4. 初始化会话管理器
            self._session_manager = SessionManager(
                db_path=self._config.db_path,
                token_window=self._config.token_window,
                compress_ratio=self._config.compress_ratio,
                retain_count=self._config.retain_count,
                model_name=self._config.model_name,
            )

            # 5. 初始化消息构建器
            self._message_builder = MessageBuilder(
                instructions=self._config.instructions,
            )

            # 6. 初始化工具执行器
            self._tool_executor = ToolExecutor()

            # 7. 加载工具列表
            self._tools = get_tools_for_llm()
            self._llm_client.tools = self._tools
            tool_names = [t.get("function", {}).get("name", "unknown") for t in self._tools]
            logger.info(f"已加载 {len(self._tools)} 个工具: {tool_names}")

            self._is_initialized = True
            logger.info(f"CustomerAgent 初始化成功: model={self._config.model_name}")
            return True

        except Exception as e:
            logger.error(
                f"CustomerAgent 初始化失败: error_type={type(e).__name__}"
            )
            return False

    async def async_reply(self, query: str, context: Context = None) -> Reply:
        """Reply serially per customer conversation."""
        session_id = self._session_id(context, query)
        lock = self._conversation_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            return await self._async_reply_unlocked(query, context, session_id)

    async def close(self) -> None:
        """Release per-account LLM, database, and in-memory resources."""
        async with self._initialize_lock:
            llm_client = self._llm_client
            session_manager = self._session_manager
            self._llm_client = None
            self._session_manager = None
            self._message_builder = None
            self._tool_executor = None
            self._active_profile = None
            self._tools = []
            self._conversation_locks.clear()
            self._is_initialized = False

        if llm_client is not None:
            try:
                await llm_client.close()
            except Exception as exc:
                logger.warning(
                    f"LLM client cleanup failed: error_type={type(exc).__name__}"
                )
        if session_manager is not None:
            try:
                await asyncio.to_thread(session_manager.dispose)
            except Exception as exc:
                logger.warning(
                    f"session manager cleanup failed: error_type={type(exc).__name__}"
                )

    async def _async_reply_unlocked(
        self,
        query: str,
        context: Context = None,
        session_id: Optional[str] = None,
    ) -> Reply:
        """异步回复接口"""
        # 延迟初始化
        if not self._is_initialized:
            if not await self.initialize_async():
                return Reply(ReplyType.TEXT, "AI客服初始化失败，请检查配置。")

        try:
            # 构建 session_id 和 dependencies
            if context and context.channel_type and context_scope(context).get("user_id"):
                dependencies = self._message_builder.build_dependencies(context)
            else:
                dependencies = {}
            session_id = session_id or self._session_id(context, query)

            # 加载历史并检查压缩（DB 操作放工作线程，避免阻塞事件循环）
            history = await asyncio.to_thread(self._session_manager.get_history, session_id)
            if await asyncio.to_thread(self._session_manager.should_compress, session_id):
                logger.info(f"触发上下文压缩: session_id={session_id}")
                await self._compress_with_llm(session_id)
                # 压缩后重新加载，使本轮回复使用压缩后的历史
                history = await asyncio.to_thread(self._session_manager.get_history, session_id)

            # 预取商品列表（拼多多 HTTP，放工作线程）并注入 dependencies，
            # 避免 build_messages 内同步阻塞
            # Persist the user turn before invoking the model.  This keeps the
            # durable transcript complete even when the model or a tool fails.
            await asyncio.to_thread(
                self._session_manager.add_message,
                session_id=session_id,
                role="user",
                content=query,
            )

            shop_id = dependencies.get("shop_id")
            user_id = dependencies.get("user_id")
            if shop_id and user_id:
                dependencies["product_list"] = await asyncio.to_thread(
                    self._message_builder.fetch_product_list_text, shop_id, user_id
                )
            else:
                dependencies["product_list"] = ""

            # 构建 messages
            messages = self._message_builder.build_messages(query, history, dependencies)

            # 执行 Agent 循环
            final_content = await self._run_agent_loop(
                messages, dependencies, session_id=session_id
            )

            # 保存最终回复到历史（DB 写入放工作线程，避免阻塞事件循环）
            await asyncio.to_thread(
                self._session_manager.add_message,
                session_id=session_id,
                role="assistant",
                content=final_content,
            )

            return Reply(ReplyType.TEXT, final_content or "抱歉，我暂时无法回复。")

        except Exception as e:
            logger.error(
                f"CustomerAgent 回复失败: error_type={type(e).__name__}"
            )
            return Reply(ReplyType.TEXT, "抱歉，我现在无法回复，请稍后再试。")

    async def _run_agent_loop(
        self,
        messages: List[Dict[str, Any]],
        dependencies: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> str:
        """
        Agent 循环核心

        调用 LLM → 检查 tool_calls → 并行执行工具 → 回传结果 → 循环
        """
        loop_count = 0

        while loop_count < self._config.max_loops:
            # 1. 调用 LLM
            try:
                response = await self._llm_client.chat(messages, tool_choice="auto")
            except Exception as e:
                logger.error(
                    f"LLM 调用失败: error_type={type(e).__name__}"
                )
                if loop_count == 0:
                    return "抱歉，AI 服务暂时不可用，请稍后再试。"
                # 已有中间结果，返回已生成的内容
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        return msg["content"]
                return "抱歉，AI 服务暂时不可用，请稍后再试。"

            # 2. 解析响应
            if not response.has_tool_calls:
                # 无工具调用，返回内容
                content = response.content or ""
                messages.append({"role": "assistant", "content": content})
                return content

            # 3. 保存 assistant 消息（包含 tool_calls）
            assistant_msg = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)
            if session_id and self._session_manager:
                await asyncio.to_thread(
                    self._session_manager.add_message,
                    session_id=session_id,
                    role="assistant",
                    content=json.dumps(
                        {"content": assistant_msg["content"], "tool_calls": assistant_msg["tool_calls"]},
                        ensure_ascii=False,
                    ),
                )

            # 4. 检查循环上限
            if loop_count >= self._config.max_loops - 1:
                logger.warning(f"工具调用达到上限 {self._config.max_loops}，强制结束循环")
                messages.append({
                    "role": "user",
                    "content": "[已达到最大工具调用次数，请基于已有信息给出最终回复。]",
                })
                try:
                    final_response = await self._llm_client.chat(messages)
                    return final_response.content or assistant_msg["content"]
                except Exception:
                    return assistant_msg["content"]

            # 5. 并行执行所有工具调用
            tool_results = await self._tool_executor.execute_parallel(
                response.tool_calls, dependencies
            )

            # 6. 将结果追加到消息列表
            for result in tool_results:
                messages.append(result.to_dict())
                if session_id and self._session_manager:
                    await asyncio.to_thread(
                        self._session_manager.add_message,
                        session_id=session_id,
                        role="tool",
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                    )

            loop_count += 1

        # 兜底
        return messages[-1].get("content", "")

    def _session_id(self, context: Optional[Context], query: str) -> str:
        if context is not None and context_scope(context).get("recipient_uid"):
            return make_conversation_key(context)
        return self._fallback_session_id

    def _conversation_session_id(self, context: Context) -> str:
        """会话键。客服侧消息的对话方是 to_uid，其余是 from_uid。

        两个方向必须落到同一个会话，否则 AI 看不到自己刚推过的内容。
        """
        customer_uid = ""
        if _context_value(context, "origin") == "merchant":
            customer_uid = _context_value(context, "to_uid")
        return make_conversation_key(context, customer_uid or None)

    @staticmethod
    def _context_role(context: Context) -> str:
        """会话历史里的角色：客服侧记 assistant，其余记 user。

        CONTEXT_ONLY 不只有客服侧消息：买家侧的 type=41「当前用户来自 商品详情页」
        也走这条路径。若一律记成 assistant，模型会以为那句话是自己说的，
        因此角色必须由 origin 推导，不能写死。
        """
        return "assistant" if _context_value(context, "origin") == "merchant" else "user"

    async def record_context(self, context: Context) -> bool:
        """把我方消息写入会话历史，但不生成回复。

        用于客服侧文本与商品卡（CONTEXT_ONLY）：让 AI 知道刚刚推送过什么，
        减少重复推荐。写入失败不影响主流程，调用方只需记录日志。
        """
        if not self._is_initialized:
            if not await self.initialize_async():
                return False

        content = context.content if context is not None else ""
        if not isinstance(content, str) or not content.strip():
            return False

        session_id = self._conversation_session_id(context)
        try:
            return await asyncio.to_thread(
                self._session_manager.add_message,
                session_id=session_id,
                role=self._context_role(context),
                content=content,
            )
        except Exception as exc:
            logger.warning(
                f"record_context failed: error_type={type(exc).__name__}"
            )
            return False

    async def _compress_with_llm(
        self,
        session_id: str,
    ) -> None:
        """使用 LLM 生成摘要并压缩历史。

        本方法运行在事件循环中（由 async_reply 调用），直接 await LLM 即可；
        切勿使用 asyncio.run——在已有事件循环里会抛 RuntimeError，导致
        压缩从未真正执行（历史只增不减）。
        """

        async def summary_llm(messages: List[Dict[str, Any]]) -> str:
            """异步调用 LLM 生成摘要"""
            summary_prompt = (
                "请简洁地总结以下对话的要点，保留关键信息和用户意图。\n\n"
                f"对话内容（共 {len(messages)} 条消息）：\n"
                + "\n".join(
                    f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')[:200]}"
                    for msg in messages
                    if msg.get("content")
                )
            )

            try:
                response = await self._llm_client.chat(
                    messages=[
                        {"role": "system", "content": "你是一个对话摘要助手。请简洁地总结对话要点。"},
                        {"role": "user", "content": summary_prompt},
                    ],
                    tool_choice="none",
                    use_tools=False,
                )
                return response.content or "[摘要生成失败]"
            except Exception as e:
                logger.error(
                    f"生成摘要失败: error_type={type(e).__name__}"
                )
                return "[摘要生成失败]"

        await self._session_manager.compress_history(session_id, summary_llm)
