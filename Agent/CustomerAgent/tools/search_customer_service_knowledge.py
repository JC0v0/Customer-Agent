"""
搜索客服知识工具
====================

根据关键词和店铺ID搜索客服知识。
"""
from typing import Optional, Union
from pydantic import BaseModel, Field
from Agent.CustomerAgent.custom.tool_decorator import agent_tool

from database.knowledge_service import KnowledgeService
from core.di_container import container

# 从DI容器获取服务实例
def _get_knowledge_service() -> KnowledgeService:
    """Resolve lazily so tool imports do not depend on DI bootstrap order."""
    return container.get(KnowledgeService)


class SearchCustomerServiceKnowledgeParams(BaseModel):
    """搜索客服知识参数"""
    # 渠道必须受信任注入：抖音的 shop_id 是 douyin_<店铺名> 字符串，
    # 只按 shop_id 解析在跨渠道同名时会串店。
    channel_type: Optional[str] = Field(default=None, description="渠道类型（受信任注入）")
    query: str = Field(..., description="搜索关键词（必须提供）")
    # 不能声明成 int：抖音 shop_id 是 douyin_示例店铺 这样的字符串，
    # Pydantic 会在工具执行前抛 ValidationError，表现为抖音下客服知识
    # 永远返回「工具执行失败」。
    shop_id: Optional[Union[str, int]] = Field(default=None, description="店铺ID（必须提供）")


@agent_tool(
    name="search_customer_service_knowledge",
    description="搜索客服知识库，查找售后政策、常见问题解答、物流信息等客服知识。当用户询问售后、物流、退款等非产品特定问题时使用此工具。",
    param_model=SearchCustomerServiceKnowledgeParams,
)
def search_customer_service_knowledge(params: SearchCustomerServiceKnowledgeParams) -> str:
    """
    搜索客服知识

    Args:
        params: SearchCustomerServiceKnowledgeParams
            query: 搜索关键词
            shop_id: 店铺ID

    Returns:
        格式化的客服知识
    """
    if not params.shop_id:
        return "[错误：缺少店铺ID，无法搜索客服知识]"

    if not params.query:
        return "[错误：缺少搜索关键词，无法搜索客服知识]"

    knowledge_service = _get_knowledge_service()
    result = knowledge_service.search_knowledge(
        shop_id=params.shop_id,
        query=params.query,
        goods_id=None,
        channel_name=params.channel_type,
    )

    return knowledge_service.format_search_result(result)
