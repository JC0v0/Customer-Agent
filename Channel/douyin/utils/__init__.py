"""
抖音API模块
统一管理所有API请求类（结构与拼多多对齐，仅保留已有实现）
"""

from .base_request import BaseRequest
from .API.get_token import GetToken
from .API.send_message import SendMessage
from .API.get_user_info import GetUserInfo
from .API.get_shop_info import GetShopInfo
from .API.order_query import OrderQuery
from .API.product_manager import ProductManager
from .API.Set_up_online import AccountMonitor

__all__ = [
    'BaseRequest',
    'GetToken',
    'SendMessage',
    'GetUserInfo',
    'GetShopInfo',
    'OrderQuery',
    'ProductManager',
    'AccountMonitor',
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'Agent-Customer Team'
__description__ = '抖音API统一请求基类和相关API实现'
