# 全局便捷函数模块
from typing import Dict, List

from core.connection_status import ConnectionStatusManager


def get_pdd_connection_status() -> List:
    """获取所有拼多多连接状态 - 全局便捷函数"""
    from core.di_container import container
    sm = container.get(ConnectionStatusManager)
    return sm.get_all_status()


def get_pdd_connected_count() -> int:
    """获取当前拼多多连接数 - 全局便捷函数"""
    from core.di_container import container
    sm = container.get(ConnectionStatusManager)
    return sm.get_connected_count()


def get_pdd_connection_summary() -> Dict[str, int]:
    """获取拼多多连接状态汇总 - 全局便捷函数"""
    from core.di_container import container
    sm = container.get(ConnectionStatusManager)
    all_status = sm.get_all_status()
    summary = {
        "total": len(all_status),
        "connected": 0,
        "connecting": 0,
        "reconnecting": 0,
        "error": 0,
        "disconnected": 0
    }

    for status in all_status:
        summary[status.state.value] += 1

    return summary


def get_pdd_heartbeat_status_all() -> Dict[str, Dict]:
    """
    获取所有拼多多连接的状态信息 - 全局便捷函数
    """
    from core.di_container import container
    sm = container.get(ConnectionStatusManager)
    heartbeat_status = {}

    all_status = sm.get_all_status()
    for status in all_status:
        connection_key = f"{status.shop_id}_{status.user_id}"
        heartbeat_status[connection_key] = {
            "connection_key": connection_key,
            "connection_state": status.state.value if status else None,
            "last_error": status.last_error if status else None,
            "error_count": status.error_count if status else 0,
            "reconnect_count": status.reconnect_count if status else 0,
            "heartbeat_running": False,
        }

    return heartbeat_status


__all__ = [
    'get_pdd_connection_status',
    'get_pdd_connected_count',
    'get_pdd_connection_summary',
    'get_pdd_heartbeat_status_all',
]
