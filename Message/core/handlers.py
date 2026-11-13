"""
简化的消息处理器基类
提取核心接口，移除复杂的实现
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from bridge.context import Context
from utils.logger_loguru import get_logger


class MessageHandler(ABC):
    """消息处理器基类"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def can_handle(self, context: Context) -> bool:
        """
        判断是否能处理该消息

        Args:
            context: Context格式的消息

        Returns:
            bool: 是否能处理
        """
        pass

    @abstractmethod
    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """
        处理消息

        Args:
            context: Context格式的消息
            metadata: 消息元数据

        Returns:
            bool: 是否处理成功
        """
        pass

    async def on_error(self, context: Context, error: Exception) -> None:
        """
        错误处理回调（可选重写）

        Args:
            context: 消息上下文
            error: 错误对象
        """
        self.logger.error(
            f"Handler {self.__class__.__name__} error: "
            f"error_type={type(error).__name__}"
        )


class TypeBasedHandler(MessageHandler):
    """基于消息类型的处理器"""

    def __init__(self, supported_types: set):
        super().__init__()
        self.supported_types = supported_types

    def can_handle(self, context: Context) -> bool:
        """检查消息类型"""
        return context.type in self.supported_types


class ChannelBasedHandler(MessageHandler):
    """基于渠道类型的处理器"""

    def __init__(self, supported_channels: set):
        super().__init__()
        self.supported_channels = supported_channels

    def can_handle(self, context: Context) -> bool:
        """检查渠道类型"""
        # 处理 channel_type 可能为 None 的情况
        channel_type = context.channel_type
        if channel_type is None:
            return False

        # 支持字符串和枚举类型
        if hasattr(channel_type, 'value'):
            channel_str = str(channel_type.value)
        else:
            channel_str = str(channel_type)

        return channel_str in {str(ch) for ch in self.supported_channels}


class CatchAllHandler(MessageHandler):
    """兜底处理器 - 处理所有未被其他处理器处理的消息"""

    def __init__(self):
        super().__init__()

    def can_handle(self, context: Context) -> bool:
        """总是返回True，确保能处理所有消息"""
        return True

    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """兜底处理并记一行 debug（不记录完整内容以保护隐私）。

        原来这里打 6 行 INFO（含两条分隔线），每条没有被专用处理器接走的
        消息都刷一遍，日志面板里非常突兀——拼多多走专用处理器，一条都不打，
        两个渠道看起来完全不像同一个程序。收敛成一行 debug。
        """
        content_length = len(str(context.content)) if context.content else 0
        self.logger.debug(
            f"消息兜底处理: type={context.type}, channel={context.channel_type}, "
            f"content_length={content_length}"
        )
        return True  # 总是返回True，避免"没有合适的处理器"警告
