"""
消息处理器集合
提供各种常用的消息处理器实现
"""
from typing import Dict, Any, List, Set, Callable, Awaitable
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from Message.message_consumer import MessageHandler
from bridge.context import Context, ContextType, ChannelType
from utils.logger import get_logger
from utils.resource_manager import ThreadResourceManager
from utils.performance_monitor import monitor_async_function
from Channel.pinduoduo.utils.API.send_message import SendMessage


class AIAutoReplyHandler(MessageHandler):
    """AI自动回复处理器 - 集成CozeBot智能回复"""
    
    def __init__(self, bot=None, auto_reply_types: Set[ContextType] = None, enable_fallback: bool = True, max_workers: int = 5):
        """
        初始化AI自动回复处理器

        Args:
            bot: AI Bot实例 (如CozeBot)
            auto_reply_types: 支持自动回复的消息类型
            enable_fallback: 是否启用规则回复作为后备
            max_workers: 线程池最大工作线程数
        """
        self.bot = bot
        self.auto_reply_types = auto_reply_types or {
            ContextType.TEXT,
            ContextType.GOODS_INQUIRY,
            ContextType.GOODS_SPEC,
            ContextType.ORDER_INFO,
            ContextType.IMAGE,
            ContextType.VIDEO,
            ContextType.EMOTION
        }
        self.enable_fallback = enable_fallback
        self.max_workers = max_workers
        self.logger = get_logger()

        # 创建专用的线程池资源管理器
        self.resource_manager = ThreadResourceManager()
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ai_handler"
        )
        self.resource_manager.register_thread_pool(
            self.executor,
            f"AI处理器线程池(max_workers={max_workers})"
        )

        # 如果没有提供bot实例，尝试创建默认的CozeBot
        if not self.bot:
            try:
                from Agent.bot_factory import create_bot
                self.bot = create_bot()
                self.logger.debug("已创建默认AI Bot实例")
            except Exception as e:
                self.logger.warning(f"创建AI Bot失败: {e}，将使用规则回复")
                self.bot = None

    def __del__(self):
        """析构函数，确保线程池被正确关闭"""
        try:
            if hasattr(self, 'resource_manager'):
                asyncio.create_task(self.resource_manager.cleanup_all())
        except Exception as e:
            self.logger.error(f"清理AI处理器资源失败: {e}")
    
    def can_handle(self, context: Context) -> bool:
        """检查是否可以处理该消息"""
        # 支持拼多多渠道的多种消息类型
        return (context.type in self.auto_reply_types and 
                context.channel_type == ChannelType.PINDUODUO)
    
    def _preprocess_message(self, context: Context) -> str:
        """
        消息预处理 - 将不同类型的消息转换为AI可理解的格式
        
        Args:
            context: 消息上下文
            
        Returns:
            处理后的消息内容（JSON字符串格式）
        """
        import json
        
        try:
            # 处理商品咨询类型
            if context.type == ContextType.GOODS_INQUIRY or context.type == ContextType.GOODS_SPEC:
                try:
                    goods_info = context.content
                    message = f'商品：{goods_info.get("goods_name")},商品价格：{goods_info.get("goods_price")},商品规格：{goods_info.get("goods_spec")}'
                    return json.dumps([{"type": "text", "text": message}], ensure_ascii=False)
                except Exception as e:
                    self.logger.error(f"处理商品咨询消息失败: {str(e)}")
                    return json.dumps([{"type": "text", "text": "收到商品咨询"}], ensure_ascii=False)
           
            # 处理订单信息类型
            elif context.type == ContextType.ORDER_INFO:
                try:
                    order_info = context.content
                    order_id = order_info.get("order_id")
                    goods_name = order_info.get("goods_name")
                    message = f"订单：{order_id}，商品：{goods_name}"
                    return json.dumps([{"type": "text", "text": message}], ensure_ascii=False)
                except Exception as e:
                    self.logger.error(f"处理订单信息消息失败: {str(e)}")
                    return json.dumps([{"type": "text", "text": "收到订单查询"}], ensure_ascii=False)

            # 文本消息处理
            elif context.type == ContextType.TEXT:
                # 基础文本处理
                return json.dumps([{"type": "text", "text": context.content}], ensure_ascii=False)
                
            # 表情消息处理
            elif context.type == ContextType.EMOTION:
                return json.dumps([{"type": "text", "text": f"表情: {context.content}"}], ensure_ascii=False)
            
            # 图片消息处理
            elif context.type == ContextType.IMAGE:
                return json.dumps([{"type": "text", "text": f"图片: {context.content}"}], ensure_ascii=False)
                
            # 视频消息处理
            elif context.type == ContextType.VIDEO:
                return json.dumps([{"type": "text", "text": f"视频: {context.content}"}], ensure_ascii=False)
                
            # 默认处理
            else:
                self.logger.warning(f"未知消息类型: {context.type}")
                return json.dumps([{"type": "text", "text": str(context.content)}], ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"消息预处理失败: {e}")
            return json.dumps([{"type": "text", "text": "消息处理失败"}], ensure_ascii=False)

    @monitor_async_function("ai_message_handler", {"handler": "AIAutoReplyHandler"})
    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """处理消息并发送AI回复 - 集成性能监控"""
        try:
            shop_id = context.kwargs.get('shop_id')
            user_id = context.kwargs.get('user_id')
            from_uid = context.kwargs.get('from_uid')
            username = context.kwargs.get("username")
            nickname = context.kwargs.get("nickname")
            if not all([shop_id, user_id, from_uid]):
                self.logger.error("缺少必要的用户或店铺信息")
                return False

            try:
                self.logger.info(f"'{username}'收到用户'{nickname}'消息: 消息类型：{context.type},消息内容：{context.content}")
                reply = await self._get_ai_reply(context)
                await self._send_reply(reply, shop_id, user_id, from_uid)
                self.logger.info(f"'{username}'回复用户'{nickname}'消息: 消息类型：{reply.type},消息内容：{reply.content}")
            except Exception as e:
                self.logger.error(f"AI回复生成失败: {e}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"AI自动回复处理失败: {e}")
            return False

    async def _get_ai_reply(self, context: Context):
        """获取AI Bot回复 - 优化版本使用专用线程池"""
        if not self.bot:
            self.logger.warning("AI Bot实例不可用，无法获取回复")
            from bridge.reply import Reply, ReplyType
            return Reply(ReplyType.TEXT, "AI服务暂时不可用，请稍后再试")

        try:
            # 预处理消息内容
            processed_content = self._preprocess_message(context)

            # 创建新的context对象，将预处理后的内容传递给bot
            processed_context = Context(
                type=ContextType.TEXT,  # 统一转换为TEXT类型
                content=processed_content,
                channel_type=context.channel_type,
                kwargs=context.kwargs
            )

            # 使用专用线程池运行同步的bot.reply方法
            loop = asyncio.get_running_loop()

            # 添加超时控制，防止长时间阻塞
            reply = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor,  # 使用专用的线程池
                    self.bot.reply,
                    processed_context
                ),
                timeout=30.0  # 30秒超时
            )

            return reply

        except asyncio.TimeoutError:
            self.logger.error("AI回复超时")
            from bridge.reply import Reply, ReplyType
            return Reply(ReplyType.TEXT, "AI回复超时，请稍后再试")

        except Exception as e:
            self.logger.error(f"获取AI回复失败: {e}", exc_info=True)
            from bridge.reply import Reply, ReplyType
            return Reply(ReplyType.TEXT, "AI服务暂时不可用，请稍后再试")
        
    async def _send_reply(self, reply, shop_id: str, user_id: str, from_uid: str) -> bool:
        """发送回复消息"""
        try:
            sender = SendMessage(shop_id, user_id)
            
            # 处理不同类型的回复
            if hasattr(reply, '__iter__') and not isinstance(reply, str):
                # 处理多个回复的情况
                for single_reply in reply:
                    success = await self._send_single_reply(single_reply, sender, from_uid)
                    if not success:
                        return False
                return True
            else:
                # 处理单个回复
                return await self._send_single_reply(reply, sender, from_uid)
                
        except Exception as e:
            self.logger.error(f"发送回复失败: {e}")
            return False
    
    async def _send_single_reply(self, reply, sender, from_uid: str) -> bool:
        """发送单个回复"""
        try:
            from bridge.reply import ReplyType
            
            if hasattr(reply, 'type') and hasattr(reply, 'content'):
                # 处理Reply对象，只处理TEXT类型
                if reply.type == ReplyType.TEXT:
                    result = sender.send_text(from_uid, reply.content)
                else:
                    # 非TEXT类型转为文本发送
                    result = sender.send_text(from_uid, str(reply.content))
                    
            else:
                # 处理字符串类型的回复
                result = sender.send_text(from_uid, str(reply))
            
            if result:
                return True
            else:
                self.logger.error("AI回复发送失败")
                return False
                
        except Exception as e:
            self.logger.error(f"发送单个回复失败: {e}")
            return False
    

class KeywordTriggerHandler(MessageHandler):
    """关键词触发处理器"""
    
    def __init__(self, keyword_rules: Dict[str, Callable[[Context, Dict[str, Any]], Awaitable[bool]]]):
        """
        初始化关键词触发处理器
        
        Args:
            keyword_rules: 关键词规则字典 {关键词: 处理函数}
        """
        self.keyword_rules = keyword_rules
        self.logger = get_logger()
    
    def can_handle(self, context: Context) -> bool:
        """检查消息是否包含关键词"""
        if context.type != ContextType.TEXT:
            return False
        
        # 确保content是字符串
        if not isinstance(context.content, str):
            return False
            
        message = context.content.lower()
        return any(keyword in message for keyword in self.keyword_rules.keys())
    
    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """根据关键词触发相应处理"""
        try:
            # 确保content是字符串
            if not isinstance(context.content, str):
                return False
                
            message = context.content.lower()
            
            for keyword, handler_func in self.keyword_rules.items():
                if keyword in message:
                    self.logger.info(f"触发关键词: {keyword}")
                    return await handler_func(context, metadata)
                    
        except Exception as e:
            self.logger.error(f"关键词触发处理失败: {e}")
            
        return False


class CustomerServiceTransferHandler(MessageHandler):
    """客服转接处理器"""
    
    def __init__(self, transfer_keywords: List[str] = None):
        """
        初始化客服转接处理器
        
        Args:
            transfer_keywords: 触发转接的关键词列表
        """
        self.transfer_keywords = transfer_keywords or [
            '人工客服', '转人工', '人工', '客服', '投诉', '举报', 
            '不满意', '解决不了', '要求赔偿'
        ]
        self.logger = get_logger()
    
    def can_handle(self, context: Context) -> bool:
        """检查是否需要转接人工客服"""
        if context.type != ContextType.TEXT:
            return False
        
        # 确保content是字符串
        if not isinstance(context.content, str):
            return False
            
        message = context.content.lower()
        return any(keyword in message for keyword in self.transfer_keywords)
    
    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """转接到人工客服"""
        try:
            shop_id = context.kwargs.get('shop_id')
            user_id = context.kwargs.get('user_id')
            from_uid = context.kwargs.get('from_uid')
            
            if not all([shop_id, user_id, from_uid]):
                return False
            
            # 获取可用的客服列表
            sender = SendMessage(shop_id, user_id)
            cs_list = sender.getAssignCsList()
            my_cs_uid = f"cs_{shop_id}_{user_id}"
            
            if cs_list and isinstance(cs_list, dict):
                # 过滤掉自己，不转接给自己
                available_cs_uids = [uid for uid in cs_list.keys() if uid != my_cs_uid]

                if available_cs_uids:
                    # 选择第一个可用的客服
                    cs_uid = available_cs_uids[0]
                    target_cs = cs_list[cs_uid]
                    cs_name = target_cs.get('username', '客服')
                    
                    # 转移会话
                    transfer_result = sender.move_conversation(from_uid, cs_uid)
                    
                    if transfer_result and transfer_result.get('success'):

                        self.logger.info(f"会话已成功转接给 {cs_name} ({cs_uid})")
                        return True
                    else:
                        self.logger.error("会话转接失败")
                else:
                    self.logger.warning("没有其他可用的客服进行转接")
                    sender.send_text(from_uid, "抱歉，当前没有其他客服在线，请您稍后再试。")
            
            return False
            
        except Exception as e:
            self.logger.error(f"客服转接处理失败: {e}")
            return False


class BusinessHoursHandler(MessageHandler):
    """营业时间处理器"""
    
    def __init__(self, business_hours: Dict[str, str] = None):
        """
        初始化营业时间处理器
        
        Args:
            business_hours: 营业时间配置 {'start': '08:00', 'end': '23:00'}
        """
        self.business_hours = business_hours or {'start': '08:00', 'end': '23:00'}
        self.logger = get_logger()
    
    def can_handle(self, context: Context) -> bool:
        """检查是否在非营业时间"""
        return not self._is_business_hours()
    
    async def handle(self, context: Context, metadata: Dict[str, Any]) -> bool:
        """处理非营业时间的消息"""
        try:
            shop_id = context.kwargs.get('shop_id')
            user_id = context.kwargs.get('user_id')
            from_uid = context.kwargs.get('from_uid')
            
            if not all([shop_id, user_id, from_uid]):
                return False
            
            current_time = datetime.now().strftime('%H:%M:%S')
            start_time = self.business_hours['start']
            end_time = self.business_hours['end']
            
            reply = (f"您好！当前时间是 {current_time}，我们的营业时间是 {start_time}-{end_time}。"
                    f"现在是非营业时间，您可以先留言，我们会在营业时间内尽快回复您。")
            
            sender = SendMessage(shop_id, user_id)
            sender.send_text(from_uid, reply)
            self.logger.info(f"非营业时间自动回复:回复 {reply} 给 {from_uid}")
            return True
            
        except Exception as e:
            self.logger.error(f"营业时间处理失败: {e}")
            
        return False
    
    def _is_business_hours(self) -> bool:
        """检查当前是否在营业时间内"""
        now = datetime.now().time()
        start_time = datetime.strptime(self.business_hours['start'], '%H:%M').time()
        end_time = datetime.strptime(self.business_hours['end'], '%H:%M').time()
        
        return start_time <= now <= end_time


# 便捷函数：创建预配置的处理器
def create_ai_handler(bot=None, enable_fallback: bool = True, max_workers: int = 5) -> AIAutoReplyHandler:
    """
    创建AI自动回复处理器

    Args:
        bot: AI Bot实例，如果为None会自动创建CozeBot
        enable_fallback: 是否启用规则回复作为后备
        max_workers: 线程池最大工作线程数
    """
    return AIAutoReplyHandler(bot=bot, enable_fallback=enable_fallback, max_workers=max_workers)


def create_coze_ai_handler(max_workers: int = 5) -> AIAutoReplyHandler:
    """创建基于CozeBot的AI回复处理器"""
    try:
        from Agent.bot_factory import create_bot
        bot = create_bot()
        return AIAutoReplyHandler(bot=bot, enable_fallback=True, max_workers=max_workers)
    except Exception as e:
        return AIAutoReplyHandler(bot=None, enable_fallback=True, max_workers=max_workers)





def handler_chain(use_ai: bool = True, businessHours: Dict[str, str] = None) -> List[MessageHandler]:
    """
    创建完整的处理器链
    
    Args:
        use_ai: 是否使用AI回复处理器
    """
    handlers = [
        BusinessHoursHandler(business_hours=businessHours),                     # 营业时间检查
        CustomerServiceTransferHandler()           # 客服转接（紧急情况）
    ]
    
    # 添加AI处理器（处理所有其他消息类型）
    if use_ai:
        handlers.append(create_ai_handler())
    else:
        handlers.append(AIAutoReplyHandler(bot=None, enable_fallback=True))
    
    return handlers

