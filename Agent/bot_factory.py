"""
channel factory
"""
from config import config


def create_bot():
    """
    创建一个bot实例
    :return: bot实例
    """
    bot_type = config.get("bot_type")
    if bot_type == "coze":
        from Agent.CozeAgent.bot import CozeBot
        return CozeBot()
    elif bot_type == "openai":
        from Agent.OpenAIAgent.bot import OpenAIBot
        return OpenAIBot()
    else:
        raise RuntimeError(f"Invalid bot type: {bot_type}")