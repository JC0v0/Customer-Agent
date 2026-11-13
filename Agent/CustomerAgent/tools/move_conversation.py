"""
会话转接工具

将当前会话转接给人工客服。
"""
from typing import Optional, Union
from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom.tool_decorator import agent_tool
from bridge.sender import get_sender
from utils.logger_loguru import get_logger

logger = get_logger("TransferConversationTool")


class TransferConversationParams(BaseModel):
    """会话转接参数"""
    channel_type: Optional[str] = Field(default=None, description="渠道类型（受信任注入）")
    shop_id: Optional[Union[str, int]] = Field(default=None, description="店铺ID")
    user_id: Optional[Union[str, int]] = Field(default=None, description="用户ID（账号ID）")
    recipient_uid: Optional[str] = Field(default=None, description="接收转接的用户UID")


@agent_tool(
    name="transfer_conversation",
    description="将当前会话转接给人工客服。",
    param_model=TransferConversationParams,
    side_effect=True,
)
def transfer_conversation(params: TransferConversationParams) -> str:
    """
    将当前会话转接给人工客服。
    """
    try:
        if not all([params.shop_id, params.user_id, params.recipient_uid]):
            return "转接失败：缺少必要的会话信息"

        # 渠道取自受信任注入的 channel_type（与 send_goods_link 同一范式），
        # 不再无参调用 get_sender() —— 那样抖音会话会拿到拼多多发送器，
        # 用 douyin_ 的 shop_id 去查拼多多库必然报「未找到店铺」。
        from Agent.CustomerAgent.tools.channel_router import normalize_channel

        sender = get_sender(normalize_channel(params.channel_type))
        if not sender:
            return "转接失败：当前渠道无可用发送器"

        cs_list = sender.get_cs_list(str(params.shop_id), str(params.user_id))
        my_cs_uid = f"cs_{params.shop_id}_{params.user_id}"
        if cs_list and isinstance(cs_list, dict):
            # 过滤掉自己，不转接给自己
            available_cs_uids = [uid for uid in cs_list.keys() if uid != my_cs_uid]

            if available_cs_uids:
                # 选择第一个可用的客服
                cs_uid = available_cs_uids[0]
                # 转移会话
                transfer_result = sender.transfer_to_cs(str(params.shop_id), str(params.user_id), params.recipient_uid, cs_uid)

                if transfer_result and transfer_result.get('success'):
                    logger.info("会话转接成功")
                    return "会话转接成功"
                else:
                    logger.warning(f"会话转接失败: transfer_result={transfer_result}")
                    return "会话转接失败"
            else:
                logger.warning("会话转接失败: 当前无可用的人工客服")
                return "当前无可用的人工客服"
        logger.warning("会话转接失败：无法获取客服列表")
        return "会话转接失败：无法获取客服列表"

    except Exception as e:
        logger.error(f"转接过程中发生错误: {type(e).__name__}")
        return "转接失败，请稍后重试"
