"""
获取产品知识工具
==================

根据商品ID和店铺ID获取产品详细知识。
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


class GetProductKnowledgeParams(BaseModel):
    """获取产品知识参数"""
    channel_type: Optional[str] = Field(default=None, description="渠道类型（受信任注入）")
    goods_id: Optional[int] = Field(default=None, description="商品ID（必须提供）")
    shop_id: Optional[Union[str, int]] = Field(default=None, description="店铺ID（必须提供）")


@agent_tool(
    name="get_product_knowledge",
    description="获取指定商品的详细知识信息，包括成分、使用方法、规格、价格等。当用户询问特定商品时使用此工具。",
    param_model=GetProductKnowledgeParams,
)
def get_product_knowledge(params: GetProductKnowledgeParams) -> str:
    """
    获取产品知识

    Args:
        params: GetProductKnowledgeParams
            goods_id: 商品ID
            shop_id: 店铺ID

    Returns:
        格式化的产品知识
    """
    if not params.shop_id:
        return "[错误：缺少店铺ID，无法获取产品知识]"

    if not params.goods_id:
        return "[错误：缺少商品ID，无法获取产品知识]"

    knowledge_service = _get_knowledge_service()
    result = knowledge_service.search_knowledge(
        shop_id=params.shop_id,
        query=None,
        goods_id=params.goods_id,
        channel_name=params.channel_type,
    )

    return knowledge_service.format_search_result(result)
