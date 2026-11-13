"""
获取店铺商品列表工具

用于客服主动推荐商品。

【架构统一】
抖音/拼多多均通过各自渠道的 ProductManager 获取商品列表，
不再有抖音特殊分支。抖音 ProductManager 以知识库为数据源，
拼多多 ProductManager 以真实 API 为数据源，返回格式完全对齐。
"""
from typing import Optional, Union
from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom.tool_decorator import agent_tool
from Channel.pinduoduo.utils.API.product_manager import ProductManager as PddProductManager
from Channel.douyin.utils.API.product_manager import ProductManager as DouyinProductManager
from utils.logger_loguru import get_logger

logger = get_logger("GetProductListTool")


class GetShopProductsParams(BaseModel):
    """获取商品列表参数（由 dependencies 自动注入）"""
    channel_type: Optional[str] = Field(default=None, description="渠道类型（受信任注入）")
    shop_id: Optional[Union[str, int]] = Field(default=None, description="店铺ID")
    user_id: Optional[Union[str, int]] = Field(default=None, description="用户ID（账号ID）")


def _get_product_manager(channel_type: str, shop_id, user_id):
    """根据渠道类型选择对应的 ProductManager 实现。"""
    from Agent.CustomerAgent.tools.channel_router import normalize_channel

    channel = normalize_channel(channel_type)
    if channel == "douyin":
        return DouyinProductManager(shop_id=shop_id, user_id=user_id)
    # 默认拼多多
    return PddProductManager(shop_id=shop_id, user_id=user_id)


@agent_tool(
    name="get_shop_products",
    description="获取店铺商品列表（含价格、库存），用于客服主动推荐及库存查询。",
    param_model=GetShopProductsParams,
)
def get_shop_products(params: GetShopProductsParams) -> str:
    """
    获取店铺商品列表，用于客服主动推荐。

    Returns:
        str: 格式化的商品列表信息，包含商品名称、ID、价格、销量、库存等
    """
    try:
        if not params.shop_id or not params.user_id:
            return "获取商品列表失败：缺少必要的shop_id或user_id参数"

        # 统一架构：按渠道选择 ProductManager（抖音/拼多多返回格式完全对齐）
        product_manager = _get_product_manager(
            params.channel_type, params.shop_id, params.user_id
        )

        # 调用API获取商品列表
        result = product_manager.get_product_list(
            page=1,
            size=10
        )

        if result.get("success"):
            products = result.get("products", [])
            total = result.get("total", 0)

            if not products:
                logger.info(f"获取商品列表完成，但店铺当前暂无商品 (shop_id: {params.shop_id})")
                return f"店铺当前暂无商品 (shop_id: {params.shop_id})"

            logger.info(f"获取商品列表成功: shop_id={params.shop_id}, total={total}, returned={len(products)}")
            return _format_products_output(products, total, page=1)

        else:
            error_msg = result.get("error_msg", "未知错误")
            logger.error(
                f"获取商品列表失败: shop_id={params.shop_id}, error_msg={error_msg}"
            )
            return "获取商品列表失败，请稍后重试"

    except Exception as e:
        logger.error(
            f"获取商品列表工具执行异常: {type(e).__name__}, shop_id={params.shop_id}"
        )
        return "获取商品列表失败，请稍后重试"


def _format_products_output(products, total, page):
    """格式化商品列表输出"""

    def _safe_text(value, limit=240):
        text = str(value or "")
        text = "".join(ch if ord(ch) >= 32 else " " for ch in text)
        return (
            text.replace("<", "＜")
            .replace(">", "＞")
            .replace("[", "［")
            .replace("]", "］")[:limit]
        )

    # API fields are external content; mark them as data-only for the model.
    if not products:
        return "未找到商品"

    output = f"[untrusted_product_catalog]\n商品列表 (共{int(total or 0)}个商品):\n\n"

    for product in products:
        goods_id = _safe_text(product.get("goods_id", "未知ID"), 64)
        goods_name = _safe_text(product.get("goods_name", "未命名商品"))
        price = _safe_text(product.get("price", ""), 64)
        quantity = product.get("quantity")

        output += f"商品名称: {goods_name}\n商品ID: {goods_id}\n"
        if price:
            output += f"价格: {price} 元\n"
        if quantity is not None:
            output += f"库存: {quantity} 件\n"
        output += "\n"

    return output + "[/untrusted_product_catalog]"
