"""
异步处理工具
统一处理异步调用的复杂逻辑
"""
import asyncio
import concurrent.futures
from typing import Any, Coroutine, TypeVar, Optional
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


def run_async(coro: Coroutine[T, Any, Any]) -> T:
    """
    统一处理异步调用的工具函数

    自动处理事件循环的复杂情况：
    - 如果已有运行中的事件循环，使用线程池运行
    - 如果没有运行中的事件循环，直接运行

    Args:
        coro: 协程对象

    Returns:
        协程的返回值

    Examples:
        >>> async def fetch_data():
        ...     return "data"
        >>> result = run_async(fetch_data())
        >>> print(result)  # "data"
    """
    try:
        # 尝试获取当前运行的事件循环
        loop = asyncio.get_running_loop()

        # 如果有运行中的事件循环，使用线程池运行
        logger.debug("检测到运行中的事件循环，使用线程池执行")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    except RuntimeError:
        # 没有运行中的事件循环，直接运行
        logger.debug("未检测到运行中的事件循环，直接执行")
        return asyncio.run(coro)


async def run_async_multiple(*coros: Coroutine) -> list:
    """
    并发运行多个协程

    Args:
        *coros: 多个协程对象

    Returns:
        协程返回值列表

    Examples:
        >>> async def task1(): return "task1"
        >>> async def task2(): return "task2"
        >>> results = await run_async_multiple(task1(), task2())
        >>> print(results)  # ["task1", "task2"]
    """
    return await asyncio.gather(*coros)


def run_async_in_thread(coro: Coroutine, timeout: Optional[float] = None) -> Any:
    """
    在独立线程中运行异步协程

    适用于需要在独立线程中隔离运行的情况。

    Args:
        coro: 协程对象
        timeout: 超时时间（秒），None表示不限制

    Returns:
        协程的返回值

    Raises:
        TimeoutError: 如果超时

    Notes:
        超时会立即返回 TimeoutError，不会像旧实现那样被 ThreadPoolExecutor
        的 with 退出（shutdown(wait=True)）阻塞等待后台线程真正结束。
        后台协程仍会在其线程中继续运行直至自行结束。
    """
    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(asyncio.run, coro)
        return future.result(timeout=timeout)
    finally:
        # 关键：wait=False，避免超时后仍阻塞等待后台线程；cancel_futures=False
        # 让已在运行的协程继续收尾，不强制中断（asyncio.run 无法被安全强杀）。
        executor.shutdown(wait=False, cancel_futures=False)
