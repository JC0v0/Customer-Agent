"""
抖音WebSocket客户端
此模块提供与抖音商家后台的接入能力，用于接收和发送客服消息。
支持多店铺管理、消息队列处理和自动重连机制。

代码已拆分为多个模块（结构与拼多多一致）：
- core/douyin_config.py: 配置类
- core/douyin_connection.py: 连接管理
- core/douyin_message_handler.py: 消息处理
- core/douyin_lifecycle.py: 生命周期管理
- core/douyin_status.py: 状态查询
- core/douyin_utils.py: 全局便捷函数
"""
from utils.logger_loguru import get_logger
from bridge.context import Context, ContextType, ChannelType, make_account_key
from Channel.douyin.douyin_message import DouyinChatMessage
from Channel.channel import Channel
from Channel.douyin.utils.API.get_token import GetToken
from database import db_manager
from utils.resource_manager import WebSocketResourceManager
from core.connection_status import ConnectionStatusManager, ConnectionState, ConnectionStatus
from Message.core.queue import QueueManager
from Message.core.consumer import MessageConsumerManager
import websockets
import json
from websockets import exceptions as ws_exceptions
import asyncio
import time
from typing import Optional, Dict, List, Set, Any
from dataclasses import dataclass
from config import config

# 导入 Mixin 类
from Channel.douyin.core.douyin_config import ReconnectConfig, HeartbeatConfig
from Channel.douyin.core.douyin_connection import ConnectionMixin
from Channel.douyin.core.douyin_message_handler import MessageHandlerMixin
from Channel.douyin.core.douyin_lifecycle import LifecycleMixin
from Channel.douyin.core.douyin_status import StatusMixin
from Channel.douyin.core.douyin_utils import (
    get_douyin_connection_status,
    get_douyin_connected_count,
    get_douyin_connection_summary,
    get_douyin_heartbeat_status_all,
)


class DouyinChannel(ConnectionMixin, MessageHandlerMixin, LifecycleMixin, StatusMixin, Channel):
    """
    抖音WebSocket客户端 - 支持自动重连和心跳检查

    使用 Mixin 组合模式，代码已拆分为多个功能模块：
    - ConnectionMixin: 连接管理
    - MessageHandlerMixin: 消息处理
    - LifecycleMixin: 生命周期管理
    - StatusMixin: 状态查询

    注意：此类不再强制单例。每个 AutoReplyThread 应创建独立的 DouyinChannel 实例
    （各自的事件循环 + WebSocket 连接），但通过 DI 容器共享同一个
    ConnectionStatusManager 来维护全局连接状态。
    """

    # 飞鸽协议版本号（对应 ws_url 的 version_code 参数）
    API_VERSION = "10000"

    def __init__(self, max_concurrent_messages: int = 50, status_manager: ConnectionStatusManager = None):
        super().__init__()
        self.channel_name = "douyin"
        self.logger = get_logger("DouyinChannel")
        # ===== 防抖属性（如果 StatusMixin 未初始化） =====
        self._last_set_status_time = 0.0
        self._status_cooldown = 10.0
        # 从 DI 容器获取 ConnectionStatusManager（所有实例共享同一个）
        if status_manager is None:
            from core.di_container import container
            status_manager = container.get(ConnectionStatusManager)
        self.status_manager = status_manager

        self._stop_event: Optional[asyncio.Event] = None
        # base_url 由 ConnectionMixin 提供，此处不覆盖。
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.business_hours = config.get(
            "business_hours", {"start": "08:00", "end": "23:00"}
        )
        # Keep the legacy attribute for third-party callers.
        self.businessHours = self.business_hours

        # Each AutoReplyThread owns one DouyinChannel.  Keep queue, consumers and
        # Agent state on that channel instead of using process-global registries.
        self.queue_manager = QueueManager()
        self.consumer_manager = MessageConsumerManager(self.queue_manager)
        self._account_agent = None
        self._account_key = None

        # WebSocket优化功能
        self.reconnect_config = ReconnectConfig()
        self.heartbeat_config = HeartbeatConfig()
        self._reconnect_tasks: Dict[str, asyncio.Task] = {}
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self._health_tasks: Dict[str, asyncio.Task] = {}  # cookie 健康检查任务

        # 性能优化：并发控制和任务管理
        self.max_concurrent_messages = max_concurrent_messages
        self.message_semaphore = asyncio.Semaphore(max_concurrent_messages)
        self.processing_tasks: Set[asyncio.Task[Any]] = set()

        # 资源管理
        self.resource_manager = WebSocketResourceManager()


# 重新导出全局便捷函数，保持向后兼容
__all__ = [
    'DouyinChannel',
    'ReconnectConfig',
    'HeartbeatConfig',
    'get_douyin_connection_status',
    'get_douyin_connected_count',
    'get_douyin_connection_summary',
    'get_douyin_heartbeat_status_all',
]
