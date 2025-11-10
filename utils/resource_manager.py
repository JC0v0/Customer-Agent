"""
资源管理器模块
提供统一的资源注册、管理和清理机制
"""

import asyncio
import weakref
from typing import Any, Callable, Optional, Set
from dataclasses import dataclass, field
from utils.logger import get_logger


@dataclass
class ResourceInfo:
    """资源信息"""
    resource: Any
    cleanup_callback: Optional[Callable] = None
    description: str = ""
    created_at: float = field(default_factory=lambda: __import__('time').time())


class ResourceManager:
    """统一的资源管理器

    提供资源的注册、管理和自动清理功能，防止资源泄漏
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self._resources: list[ResourceInfo] = []
        self._cleanup_lock = asyncio.Lock()
        self._is_cleaning_up = False
        self._weak_refs: Set[weakref.ref] = set()

    def register_resource(
        self,
        resource: Any,
        cleanup_callback: Optional[Callable] = None,
        description: str = ""
    ) -> None:
        """注册需要管理的资源

        Args:
            resource: 要注册的资源对象
            cleanup_callback: 清理回调函数
            description: 资源描述
        """
        resource_info = ResourceInfo(
            resource=resource,
            cleanup_callback=cleanup_callback,
            description=description or f"Resource {type(resource).__name__}"
        )

        self._resources.append(resource_info)

        # 如果资源有弱引用支持，也注册弱引用
        try:
            weak_ref = weakref.ref(resource, self._on_resource_deleted)
            self._weak_refs.add(weak_ref)
        except TypeError:
            # 某些对象不支持弱引用
            pass

        self.logger.debug(f"注册资源: {resource_info.description}")

    def _on_resource_deleted(self, weak_ref: weakref.ref) -> None:
        """资源被垃圾回收时的回调"""
        self._weak_refs.discard(weak_ref)
        self.logger.debug("资源已被垃圾回收")

    async def cleanup_resource(self, resource_info: ResourceInfo) -> bool:
        """清理单个资源

        Args:
            resource_info: 资源信息

        Returns:
            清理是否成功
        """
        try:
            if resource_info.cleanup_callback:
                if asyncio.iscoroutinefunction(resource_info.cleanup_callback):
                    await resource_info.cleanup_callback()
                else:
                    resource_info.cleanup_callback()

                self.logger.debug(f"成功清理资源: {resource_info.description}")
                return True
            else:
                self.logger.debug(f"资源无需清理: {resource_info.description}")
                return True

        except Exception as e:
            self.logger.error(f"清理资源失败 {resource_info.description}: {e}")
            return False

    async def cleanup_all(self) -> dict:
        """清理所有注册的资源

        Returns:
            清理结果统计
        """
        async with self._cleanup_lock:
            if self._is_cleaning_up:
                self.logger.warning("资源清理已在进行中")
                return {"status": "already_cleaning"}

            self._is_cleaning_up = True

        try:
            self.logger.info(f"开始清理 {len(self._resources)} 个资源")

            success_count = 0
            failed_count = 0

            # 按注册时间倒序清理（后注册的先清理）
            for resource_info in reversed(self._resources):
                if await self.cleanup_resource(resource_info):
                    success_count += 1
                else:
                    failed_count += 1

            # 清理弱引用集合
            self._weak_refs.clear()
            self._resources.clear()

            result = {
                "status": "completed",
                "total": success_count + failed_count,
                "success": success_count,
                "failed": failed_count
            }

            self.logger.info(f"资源清理完成: {result}")
            return result

        finally:
            self._is_cleaning_up = False

    def get_resource_count(self) -> int:
        """获取当前注册的资源数量"""
        return len(self._resources)

    def get_resource_descriptions(self) -> list[str]:
        """获取所有资源的描述"""
        return [info.description for info in self._resources]

    def remove_resource(self, resource: Any) -> bool:
        """移除特定资源的注册

        Args:
            resource: 要移除的资源对象

        Returns:
            是否找到并移除了资源
        """
        for i, resource_info in enumerate(self._resources):
            if resource_info.resource is resource:
                removed_info = self._resources.pop(i)
                self.logger.debug(f"移除资源注册: {removed_info.description}")
                return True
        return False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup_all()


class WebSocketResourceManager(ResourceManager):
    """WebSocket 专用资源管理器"""

    def register_websocket(self, websocket: Any, description: str = "WebSocket连接") -> None:
        """注册 WebSocket 连接

        Args:
            websocket: WebSocket 连接对象
            description: 连接描述
        """
        def cleanup_websocket():
            try:
                if hasattr(websocket, 'close'):
                    if asyncio.iscoroutinefunction(websocket.close):
                        asyncio.create_task(websocket.close())
                    else:
                        websocket.close()
                elif hasattr(websocket, 'close_conn'):
                    websocket.close_conn()
            except Exception as e:
                self.logger.error(f"关闭 WebSocket 连接失败: {e}")

        self.register_resource(websocket, cleanup_websocket, description)


class ThreadResourceManager(ResourceManager):
    """线程资源管理器"""

    def register_thread_pool(self, executor: Any, description: str = "线程池") -> None:
        """注册线程池

        Args:
            executor: 线程池执行器
            description: 描述
        """
        def cleanup_thread_pool():
            try:
                if hasattr(executor, 'shutdown'):
                    executor.shutdown(wait=False)
                self.logger.debug(f"线程池已关闭: {description}")
            except Exception as e:
                self.logger.error(f"关闭线程池失败: {e}")

        self.register_resource(executor, cleanup_thread_pool, description)


# 全局资源管理器实例
_global_resource_manager = None


def get_global_resource_manager() -> ResourceManager:
    """获取全局资源管理器实例"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
    return _global_resource_manager


async def cleanup_all_global_resources() -> dict:
    """清理所有全局资源"""
    manager = get_global_resource_manager()
    return await manager.cleanup_all()