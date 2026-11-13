"""
工具模块

导入所有工具以触发 @agent_tool 装饰器注册。
"""
from Agent.CustomerAgent.tools import send_goods_link
from Agent.CustomerAgent.tools import move_conversation
from Agent.CustomerAgent.tools import get_product_list
from Agent.CustomerAgent.tools import get_product_knowledge
from Agent.CustomerAgent.tools import search_customer_service_knowledge
from Agent.CustomerAgent.tools import query_order_status

__all__ = [
    "send_goods_link",
    "move_conversation",
    "get_product_list",
    "get_product_knowledge",
    "search_customer_service_knowledge",
    "query_order_status",
]
