"""
自定义 Agent 工具系统

提供 @agent_tool 装饰器、全局工具注册表、LLM 工具格式转换和工具执行能力。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, get_type_hints, get_origin, get_args
from dataclasses import dataclass
from pydantic import BaseModel, create_model, Field

from utils.logger_loguru import get_logger

logger = get_logger("agent_tools")


# ==============================================================================
# 工具注册表
# ==============================================================================

@dataclass
class ToolEntry:
    """工具条目"""
    name: str
    description: str
    param_model: type[BaseModel]  # Pydantic 模型，用于生成 schema 和实例化参数
    func: Callable
    side_effect: bool = False


# 全局工具注册表
TOOL_REGISTRY: Dict[str, ToolEntry] = {}


def _build_openai_tool(entry: ToolEntry) -> Dict[str, Any]:
    """将 ToolEntry 转换为 OpenAI tools 格式"""
    schema = entry.param_model.model_json_schema()

    # OpenAI 要求 "parameters" 而不是 "$schema" 等额外字段
    params = {
        "type": schema.get("type", "object"),
        "properties": schema.get("properties", {}),
    }
    if "required" in schema:
        params["required"] = schema["required"]

    return {
        "type": "function",
        "function": {
            "name": entry.name,
            "description": entry.description,
            "parameters": params,
        },
    }


def get_tools_for_llm() -> List[Dict[str, Any]]:
    """获取所有已注册工具的 OpenAI tools 格式列表"""
    return [_build_openai_tool(entry) for entry in TOOL_REGISTRY.values()]


def get_tool_entry(name: str) -> Optional[ToolEntry]:
    """根据名称获取工具条目"""
    return TOOL_REGISTRY.get(name)


def execute_tool(
    name: str,
    arguments: str,
    dependencies: Dict[str, str],
) -> str:
    """
    执行指定工具

    Args:
        name: 工具名称
        arguments: JSON 格式的参数字符串
        dependencies: 依赖字典（由调用方传入的上下文，如 shop_id, user_id 等）

    Returns:
        工具执行结果的字符串，失败时返回错误信息字符串
    """
    entry = TOOL_REGISTRY.get(name)
    if not entry:
        return f"[工具不存在: {name}]"

    try:
        import json

        # 解析参数
        args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments

        # 构建参数：身份/权限字段必须来自受信任的 dependencies，
        # 其他业务参数才允许使用 LLM 提供的值。
        # 使用 Pydantic 模型的字段名来构建参数字典
        params = {}
        model_fields = entry.param_model.model_fields
        authority_fields = {
            "channel_type",
            "shop_id",
            "user_id",
            "recipient_uid",
            # 买家平台昵称同样受信任注入：抖音订单接口按昵称精确反查，
            # 若允许 LLM 传入会话买家的昵称，可能查到他人的订单（越权）
            "recipient_nick",
        }

        for field_name in model_fields:
            if field_name in authority_fields:
                trusted_value = dependencies.get(field_name)
                if trusted_value is None or str(trusted_value).strip() == "":
                    return "[工具无法执行：缺少可信会话身份]"
                params[field_name] = trusted_value
            elif field_name in args_dict:
                # 优先从 LLM 提供的参数中获取
                params[field_name] = args_dict[field_name]
            elif field_name in dependencies:
                # 仅当 LLM 未提供时，从 dependencies 补充
                params[field_name] = dependencies[field_name]
            # 字段为 Optional 且 default=None，不提供则留空

        # 使用 Pydantic 模型验证参数（会自动处理 Optional 字段）
        validated = entry.param_model(**params)

        # 调用工具函数，传入 Pydantic 模型实例
        result = entry.func(validated)

        if result is None:
            return "[工具执行无返回]"
        return str(result)

    except json.JSONDecodeError as e:
        logger.error(f"工具 {name} 参数解析失败: {type(e).__name__}")
        return "[工具参数格式错误，请重试]"
    except Exception as e:
        logger.error(f"工具 {name} 执行异常: {type(e).__name__}")
        return "[工具执行失败，请稍后重试]"


# ==============================================================================
# 装饰器
# ==============================================================================

def agent_tool(
    name: str,
    description: str,
    param_model: Optional[type[BaseModel]] = None,
    side_effect: bool = False,
) -> Callable:
    """
    工具装饰器

    Args:
        name: 工具名称
        description: 工具描述（会传给 LLM）
        param_model: Pydantic BaseModel 子类，定义工具参数

    Usage:
        class SendGoodsParams(BaseModel):
            recipient_uid: str = Field(description="接收消息的用户UID")
            goods_id: int = Field(description="商品ID")

        @agent_tool(
            name="send_goods_link",
            description="向用户发送商品卡片链接",
            param_model=SendGoodsParams,
        )
        def send_goods_link(params: SendGoodsParams) -> str:
            ...
    """
    if param_model is None:
        # 如果没有提供 param_model，创建一个空模型
        param_model = create_model(f"{name.title()}Params")

    def decorator(func: Callable) -> Callable:
        # 注册到全局注册表
        TOOL_REGISTRY[name] = ToolEntry(
            name=name,
            description=description,
            param_model=param_model,
            func=func,
            side_effect=side_effect,
        )
        logger.debug(f"工具已注册: {name}")
        return func

    return decorator
