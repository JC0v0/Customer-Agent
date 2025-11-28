import json
import os
from typing import List, Dict, Optional
from openai import OpenAI
from Agent.bot import Bot
from bridge.context import Context, ContextType
from bridge.reply import Reply, ReplyType
from utils.logger import get_logger
from config import config

class OpenAIBot(Bot):
    def __init__(self):
        super().__init__()
        self.logger = get_logger("OpenAIBot")
        self.api_key = config.get("openai_api_key")
        self.base_url = config.get("openai_api_base", "https://api.openai.com/v1")
        self.model = config.get("openai_model", "gpt-3.5-turbo")
        
        if not self.api_key:
            self.logger.warning("OpenAI API Key 未配置，无法使用 OpenAIBot")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 加载 AI 设定
        self.system_prompt = self._load_ai_settings()
        
        # 简单的内存对话历史记录 {user_id: [{"role": "...", "content": "..."}]}
        self.history: Dict[str, List[Dict[str, str]]] = {}
        self.max_history_len = 20  # 保留最近20条消息

    def _load_ai_settings(self) -> str:
        """加载 AI 设定文件并构建 System Prompt"""
        try:
            settings_path = "ai_settings.json"
            if not os.path.exists(settings_path):
                self.logger.warning(f"AI 配置文件 {settings_path} 不存在，使用默认设定")
                return "你是一个有用的客服助手。"
                
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                
            persona = settings.get("persona", "")
            product_info = settings.get("product_info", "")
            reply_rules = settings.get("reply_rules", "")
            
            system_prompt = f"""
{persona}

【产品信息】
{product_info}

【回复规则】
{reply_rules}

请根据以上信息回答客户问题。
"""
            return system_prompt.strip()
            
        except Exception as e:
            self.logger.error(f"加载 AI 设定失败: {e}")
            return "你是一个有用的客服助手。"

    def reply(self, context: Context) -> Reply:
        try:
            if not self.api_key:
                return Reply(ReplyType.TEXT, "OpenAI API Key 未配置")

            # 获取用户ID作为会话标识
            user_id = context.kwargs.get("from_uid")
            if not user_id:
                user_id = "default_user"
                
            query = context.content

            # 初始化该用户的历史记录
            if user_id not in self.history:
                self.history[user_id] = []
            
            # 构建本次请求的消息列表
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.history[user_id])
            messages.append({"role": "user", "content": query})
            
            # 调用 OpenAI API
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                
                reply_content = response.choices[0].message.content
                
                # 更新历史记录
                self.history[user_id].append({"role": "user", "content": query})
                self.history[user_id].append({"role": "assistant", "content": reply_content})
                
                # 保持历史记录长度适中
                if len(self.history[user_id]) > self.max_history_len:
                    # 保留最近的 N 条，注意要成对保留 (user+assistant)
                    # 这里简单切片
                    self.history[user_id] = self.history[user_id][-self.max_history_len:]

                return Reply(ReplyType.TEXT, reply_content)
                
            except Exception as api_e:
                self.logger.error(f"OpenAI API 请求失败: {api_e}")
                return Reply(ReplyType.TEXT, "抱歉，我暂时无法回答您的问题。")

        except Exception as e:
            self.logger.error(f"OpenAIBot 处理异常: {e}")
            return Reply(ReplyType.TEXT, "系统错误")
