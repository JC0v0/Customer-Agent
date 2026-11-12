# 状态查询模块（抖音渠道，结构与拼多多一致）
from typing import Dict, List, Optional, Any


class StatusMixin:
    """状态查询 Mixin"""

    def get_connection_status(self) -> List:
        """获取所有连接状态"""
        return self.status_manager.get_all_status()

    def get_connected_count(self) -> int:
        """获取当前连接数"""
        return self.status_manager.get_connected_count()

    def get_connection_info(self, shop_id: str, user_id: str) -> Optional:
        """获取指定连接的状态信息"""
        return self.status_manager.get_status(shop_id, user_id)

    def configure_reconnect(self, max_attempts: int = None, initial_delay: float = None,
                          max_delay: float = None, backoff_factor: float = None,
                          enable_auto_reconnect: bool = None) -> None:
        """配置重连参数"""
        if max_attempts is not None:
            self.reconnect_config.max_attempts = max_attempts
        if initial_delay is not None:
            self.reconnect_config.initial_delay = initial_delay
        if max_delay is not None:
            self.reconnect_config.max_delay = max_delay
        if backoff_factor is not None:
            self.reconnect_config.backoff_factor = backoff_factor
        if enable_auto_reconnect is not None:
            self.reconnect_config.enable_auto_reconnect = enable_auto_reconnect

        self.logger.info(f"重连配置已更新: max_attempts={self.reconnect_config.max_attempts}, "
                        f"initial_delay={self.reconnect_config.initial_delay}, "
                        f"enable_auto_reconnect={self.reconnect_config.enable_auto_reconnect}")

    def configure_heartbeat(self, enable_heartbeat: bool = None, heartbeat_interval: float = None,
                           heartbeat_timeout: float = None, max_heartbeat_failures: int = None) -> None:
        """配置心跳参数"""
        if enable_heartbeat is not None:
            self.heartbeat_config.enable_heartbeat = enable_heartbeat
        if heartbeat_interval is not None:
            self.heartbeat_config.heartbeat_interval = heartbeat_interval
        if heartbeat_timeout is not None:
            self.heartbeat_config.heartbeat_timeout = heartbeat_timeout
        if max_heartbeat_failures is not None:
            self.heartbeat_config.max_heartbeat_failures = max_heartbeat_failures

        self.logger.info(f"心跳配置已更新: enable_heartbeat={self.heartbeat_config.enable_heartbeat}, "
                        f"heartbeat_interval={self.heartbeat_config.heartbeat_interval}, "
                        f"max_heartbeat_failures={self.heartbeat_config.max_heartbeat_failures}")

    def get_heartbeat_status(self, shop_id: str, user_id: str) -> Dict[str, Optional[Any]]:
        """获取心跳状态信息"""
        connection_key = f"{shop_id}_{user_id}"
        has_heartbeat_task = connection_key in self._heartbeat_tasks

        status = self.status_manager.get_status(shop_id, user_id)

        return {
            "connection_key": connection_key,
            "heartbeat_enabled": self.heartbeat_config.enable_heartbeat,
            "heartbeat_running": has_heartbeat_task,
            "heartbeat_interval": self.heartbeat_config.heartbeat_interval,
            "max_failures": self.heartbeat_config.max_heartbeat_failures,
            "connection_state": status.state.value if status else None,
            "last_error": status.last_error if status else None,
            "error_count": status.error_count if status else 0
        }


__all__ = ['StatusMixin']
