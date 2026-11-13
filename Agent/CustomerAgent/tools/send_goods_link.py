"""
发送商品卡片工具

使用自定义 @agent_tool 装饰器，无需 Agno 依赖。

【架构统一】
抖音/拼多多均通过各自渠道的 ProductManager 校验商品有效性。
抖音 ProductManager 以知识库为数据源，校验通过后返回明确提示
（抖音商品卡发送功能待接入），不再直接降级为"不支持"。
"""
from typing import Optional, Union
from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom.tool_decorator import agent_tool
from Channel.pinduoduo.utils.API.product_manager import ProductManager
from Channel.douyin.utils.API.product_manager import ProductManager as DouyinProductManager
from bridge.sender import get_sender
from utils.logger_loguru import get_logger

logger = get_logger("SendGoodsLinkTool")


class SendGoodsLinkParams(BaseModel):
    """发送商品卡片参数"""
    channel_type: Optional[str] = Field(default=None, description="渠道类型（受信任注入）")
    recipient_uid: Optional[Union[str, int]] = Field(default=None, description="接收消息的用户UID")
    goods_id: Optional[int] = Field(default=None, description="商品ID。必须使用商品列表中给出的 商品ID（通常是很大的数字）。绝对不能使用列表序号(1, 2, 3...)，那是错误的！")
    shop_id: Optional[Union[str, int]] = Field(default=None, description="店铺ID")
    user_id: Optional[Union[str, int]] = Field(default=None, description="用户ID（账号ID）")


def _get_product_manager(channel_type: str, shop_id, user_id):
    """根据渠道类型选择对应的 ProductManager 实现。"""
    from Agent.CustomerAgent.tools.channel_router import normalize_channel

    channel = normalize_channel(channel_type)
    if channel == "douyin":
        return DouyinProductManager(shop_id=shop_id, user_id=user_id)
    # 默认拼多多
    return ProductManager(shop_id=shop_id, user_id=user_id)


@agent_tool(
    name="send_goods_link",
    description="向用户发送商品卡片链接，用于客服主动推荐商品。重要：goods_id 必须使用商品列表中给出的真实商品ID（大数），严禁使用列表序号（1、2、3这样的小数）！",
    param_model=SendGoodsLinkParams,
    side_effect=True,
)
def send_goods_link(params: SendGoodsLinkParams) -> str:
    """
    向用户发送商品卡片链接。

    Args:
        params: SendGoodsLinkParams，包含 recipient_uid, goods_id, shop_id, user_id

    Returns:
        str: 发送结果，成功返回 True，失败返回错误信息
    """
    try:
        if not all([params.recipient_uid, params.goods_id, params.shop_id, params.user_id]):
            logger.error("商品卡片发送失败: 缺少必要参数")
            return "发送失败：缺少必要参数"

        # 渠道隔离：按渠道选择 ProductManager 校验商品（抖音走知识库，拼多多走真实API）
        from Agent.CustomerAgent.tools.channel_router import normalize_channel

        is_douyin = normalize_channel(params.channel_type) == "douyin"

        # 防护：真实商品ID都是大数，如果goods_id很小，很可能是把列表序号误当作商品ID了
        if params.goods_id is not None and params.goods_id < 1000:
            logger.warning(f"商品ID可能错误: goods_id={params.goods_id} 太小，大概率是列表序号不是真实商品ID，请重新从商品列表中选择正确的商品ID")
            return "发送失败：商品 ID 无效，请使用商品列表中的真实商品 ID"

        # Do not trust an ID supplied by the model or caller.  Resolve it
        # through the authenticated shop API before performing the side effect.
        product_manager = _get_product_manager(
            params.channel_type, params.shop_id, params.user_id
        )
        detail = product_manager.get_product_detail(params.goods_id)
        product_info = detail.get("product_info") if isinstance(detail, dict) else None
        resolved_id = product_info.get("goods_id") if isinstance(product_info, dict) else None
        if not (
            isinstance(detail, dict)
            and detail.get("success") is True
            and resolved_id is not None
            and str(resolved_id) == str(params.goods_id)
        ):
            error_msg = detail.get("error_msg", "商品不存在") if isinstance(detail, dict) else "商品不存在"
            logger.warning(f"拒绝发送无效商品 ID: goods_id={params.goods_id}, error={error_msg}")
            return f"发送失败：商品不属于当前店铺或已失效（{error_msg}）"

        # 抖音：商品校验通过，但商品卡发送协议尚未逆向，返回明确提示
        if is_douyin:
            goods_name = product_info.get("goods_name", "该商品")
            logger.info(
                f"抖音商品校验通过，商品卡发送待接入: "
                f"goods_id={params.goods_id}, goods_name={goods_name}"
            )
            return (
                f"商品已找到：{goods_name}（ID: {params.goods_id}）。"
                f"抖音渠道商品卡片发送功能待接入，建议转人工处理，"
                f"或引导买家在店铺商品页查看该商品。"
            )

        # 拼多多：商品校验通过，执行发送
        sender = get_sender()
        result = sender.send_product_card(str(params.shop_id), str(params.user_id), str(params.recipient_uid), params.goods_id, biz_type=2)

        if result and result.get("success"):
            logger.info(f"商品卡片发送成功: goods_id={params.goods_id}")
            return "商品卡片发送成功"
        else:
            logger.error(
                f"商品卡片发送失败: goods_id={params.goods_id}, "
                f"response_keys={sorted(result.keys()) if isinstance(result, dict) else []}"
            )
            return "商品卡片发送失败，请稍后重试"

    except Exception as e:
        logger.error(f"发送商品卡片异常: {type(e).__name__}")
        return "发送商品卡片失败，请稍后重试"
