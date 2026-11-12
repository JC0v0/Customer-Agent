"""AI 工具渠道路由辅助。

工具层原本硬编码拼多多实现（ProductManager / get_sender() 默认拼多多），
导致抖音会话触发工具时误调拼多多 API，报「无法在数据库中找到账户
shop_id=douyin_xxx」。本模块提供统一的渠道归一化，供各工具按
channel_type 路由到正确渠道实现（拼多多 / 抖音 / 淘宝）。

【受信任注入】channel_type 来自 message_builder.build_dependencies 的
Context.channel_type（受信任字段），由 tool_executor 注入工具参数模型，
禁止信任 LLM 提供的任意渠道值。
"""

from __future__ import annotations

from typing import Optional, Union

# 渠道归一化映射（大小写/别名 → 统一标识）
_CHANNEL_ALIASES = {
    "pinduoduo": "pinduoduo",
    "pdd": "pinduoduo",
    "拼多多": "pinduoduo",
    "douyin": "douyin",
    "抖音": "douyin",
    "taobao": "taobao",
    "淘宝": "taobao",
    "千牛": "taobao",
}


def normalize_channel(channel_type: Optional[Union[str, object]]) -> str:
    """将渠道标识归一化为 pinduoduo / douyin / taobao。

    兼容 ChannelType 枚举（str() 取 value）、字符串及 None。
    未知渠道保守回退 pinduoduo（与 get_sender 默认行为一致）。
    """
    if channel_type is None:
        return "pinduoduo"
    ct = str(getattr(channel_type, "value", channel_type)).strip().lower()
    return _CHANNEL_ALIASES.get(ct, "pinduoduo")


__all__ = ["normalize_channel"]
