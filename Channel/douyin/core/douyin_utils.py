# 全局便捷函数模块（抖音渠道，结构与拼多多一致）
from typing import Dict, List


def get_douyin_connection_status() -> List:
    """获取所有抖音连接状态 - 全局便捷函数"""
    from core.di_container import container
    sm = container.get(ConnectionStatusManager)
    return sm.get_all_status()


def get_douyin_connected_count() -> int:
    """获取当前抖音连接数 - 全局便捷函数"""
    from core.di_container import container
    sm = container.get(ConnectionStatusManager)
    return sm.get_connected_count()


def get_douyin_connection_summary() -> Dict[str, int]:
    """获取抖音连接状态汇总 - 全局便捷函数"""
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


def get_douyin_heartbeat_status_all() -> Dict[str, Dict]:
    """
    获取所有抖音连接的状态信息 - 全局便捷函数
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
    'get_douyin_connection_status',
    'get_douyin_connected_count',
    'get_douyin_connection_summary',
    'get_douyin_heartbeat_status_all',
]
