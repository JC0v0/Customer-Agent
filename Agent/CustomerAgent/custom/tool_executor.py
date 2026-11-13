"""Bounded tool execution with serialized side effects."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from utils.logger_loguru import get_logger
from Agent.CustomerAgent.custom.tool_decorator import (
    execute_tool,
    get_tool_entry,
)


logger = get_logger("ToolExecutor")

# Only these audited tools are allowed to run concurrently.  New tools are
# serialized by default until their side-effect behavior is reviewed.
READ_ONLY_TOOL_NAMES = {
    "get_shop_products",
    "get_product_knowledge",
    "search_customer_service_knowledge",
}


class ToolResult:
    def __init__(self, tool_call_id: str, content: str):
        self.tool_call_id = tool_call_id
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


class ToolExecutor:
    # 单个工具执行超时（秒）。工具在 to_thread 线程里跑，Python 无法强杀
    # 线程，但超时后循环不再等待、直接给模型回错误结果，避免整体卡死。
    TOOL_EXEC_TIMEOUT_SECONDS = 30

    async def execute_parallel(
        self,
        tool_calls: List[Any],
        dependencies: Dict[str, Any],
    ) -> List[ToolResult]:
        if not tool_calls:
            return []

        results: List[ToolResult] = []

        async def run(call: Any) -> ToolResult:
            try:
                content = await asyncio.wait_for(
                    asyncio.to_thread(
                        execute_tool,
                        call.function.name,
                        call.function.arguments,
                        dependencies,
                    ),
                    timeout=self.TOOL_EXEC_TIMEOUT_SECONDS,
                )
                return ToolResult(call.id, content)
            except asyncio.TimeoutError:
                logger.error(
                    f"tool execution timeout: {call.function.name} "
                    f"after {self.TOOL_EXEC_TIMEOUT_SECONDS}s"
                )
                return ToolResult(
                    call.id, "[工具执行超时，请告知用户稍后重试，不要编造结果]"
                )
            except Exception as exc:
                logger.error(
                    f"tool execution failed: {call.function.name}: {type(exc).__name__}"
                )
                return ToolResult(call.id, "[工具执行失败，请稍后重试]")

        # Execute contiguous read-only calls concurrently, but keep side
        # effects in the exact order emitted by the model.  This avoids a
        # later transfer/send overtaking an earlier lookup or another send.
        readonly_batch = []

        async def flush_readonly_batch() -> None:
            if readonly_batch:
                results.extend(
                    await asyncio.gather(*(run(call) for call in readonly_batch))
                )
                readonly_batch.clear()

        for call in tool_calls:
            entry = get_tool_entry(call.function.name)
            if (
                entry is None
                or entry.side_effect
                or call.function.name not in READ_ONLY_TOOL_NAMES
            ):
                await flush_readonly_batch()
                results.append(await run(call))
            else:
                readonly_batch.append(call)
        await flush_readonly_batch()

        order = {call.id: index for index, call in enumerate(tool_calls)}
        results.sort(key=lambda result: order.get(result.tool_call_id, 0))
        return results

    def results_to_messages(self, results: List[ToolResult]) -> List[Dict[str, str]]:
        return [result.to_dict() for result in results]
