# 抖音渠道核心模块（结构与拼多多一致）
from .douyin_config import ReconnectConfig, HeartbeatConfig
from .douyin_connection import ConnectionMixin
from .douyin_message_handler import MessageHandlerMixin
from .douyin_lifecycle import LifecycleMixin
from .douyin_status import StatusMixin
from .douyin_utils import (
    get_douyin_connection_status,
    get_douyin_connected_count,
    get_douyin_connection_summary,
    get_douyin_heartbeat_status_all,
)

__all__ = [
    # 配置
    'ReconnectConfig',
    'HeartbeatConfig',
    # Mixin
    'ConnectionMixin',
    'MessageHandlerMixin',
    'LifecycleMixin',
    'StatusMixin',
    # 便捷函数
    'get_douyin_connection_status',
    'get_douyin_connected_count',
    'get_douyin_connection_summary',
    'get_douyin_heartbeat_status_all',
]
