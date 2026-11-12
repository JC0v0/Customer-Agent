# 配置模块（抖音渠道，结构与拼多多一致）
from dataclasses import dataclass


@dataclass
class ReconnectConfig:
    """重连配置"""
    max_attempts: int = 5          # 最大重试次数
    initial_delay: float = 2.0     # 初始延迟(秒)
    max_delay: float = 60.0        # 最大延迟(秒)
    backoff_factor: float = 2.0    # 退避因子
    enable_auto_reconnect: bool = True  # 是否启用自动重连


@dataclass
class HeartbeatConfig:
    """心跳检查配置"""
    enable_heartbeat: bool = True     # 是否启用心跳检查
    heartbeat_interval: float = 30.0   # 心跳间隔(秒)
    heartbeat_timeout: float = 10.0    # 心跳超时(秒)
    health_check_interval: float = 60.0 # 健康检查间隔(秒)
    max_heartbeat_failures: int = 3    # 最大心跳失败次数
    # Cookie 健康检查
    enable_cookie_health_check: bool = True       # 是否启用主动 cookie 健康检查
    cookie_health_check_interval: float = 300.0    # cookie 健康检查间隔(秒)，默认5分钟
    cookie_health_check_timeout: float = 15.0      # cookie 验证 HTTP 请求超时(秒)


__all__ = ['ReconnectConfig', 'HeartbeatConfig']
