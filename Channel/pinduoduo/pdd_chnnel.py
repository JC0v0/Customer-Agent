"""
拼多多WebSocket客户端

此模块提供与拼多多商家后台的WebSocket通信功能，用于接收和发送客服消息。
支持多店铺管理、消息队列处理和自动重连机制。
"""
from utils.logger import get_logger
from bridge.context import Context, ContextType, ChannelType
from Channel.pinduoduo.pdd_message import PDDChatMessage
from Channel.channel import Channel
from Channel.pinduoduo.utils.API.get_token import GetToken
from database import db_manager
from utils.resource_manager import WebSocketResourceManager
import websockets
import json
import asyncio
from typing import Optional, Set
# 导入消息处理系统
from Message import put_message
from config import config

class PDDChannel(Channel):
    def __init__(self, max_concurrent_messages: int = 50):
        super().__init__()
        self.channel_name = "pinduoduo"
        self.logger = get_logger("PDDChannel")
        self.base_url = "wss://m-ws.pinduoduo.com/"
        self.ws = None
        self._stop_event = None  # 停止事件
        self.businessHours = config.get("businessHours")

        # 性能优化：并发控制和任务管理
        self.max_concurrent_messages = max_concurrent_messages
        self.message_semaphore = asyncio.Semaphore(max_concurrent_messages)
        self.processing_tasks: Set[asyncio.Task] = set()

        # 资源管理
        self.resource_manager = WebSocketResourceManager()

    async def start_account(self, shop_id: str, user_id: str, on_success: callable, on_failure: callable) -> None:
        """
        启动指定店铺下账号
        :param shop_id: 店铺ID
        :param user_id: 用户ID
        :param on_success: 连接成功回调
        :param on_failure: 连接失败回调
        """
        account_info = db_manager.get_account(self.channel_name, shop_id, user_id)
        if not account_info:
            error_msg = f"账号 {user_id} 在数据库中不存在"
            self.logger.error(error_msg)
            on_failure(error_msg)
            return
        
        username = account_info.get("username", user_id)
        # 启动账号
        await self.init(shop_id, user_id, username, on_success, on_failure)

    async def stop_account(self, shop_id: str, user_id: str) -> None:
        """
        停止指定店铺下账号
        :param shop_id: 店铺ID
        :param user_id: 用户ID
        """
        try:
            # 检查账号是否存在
            account_info = db_manager.get_account(self.channel_name, shop_id, user_id)
            if not account_info:
                self.logger.warning(f"账号 {user_id} 不存在，无法停止")
                return
            
            username = account_info.get("username", user_id)
            self.logger.info(f"正在停止店铺 {shop_id} 账号 {username}")
            
            # 关闭WebSocket连接
            if self.ws and not self.ws.closed:
                await self.ws.close()
                self.logger.info(f"已关闭店铺 {shop_id} 账号 {username} 的WebSocket连接")
            else:
                self.logger.warning(f"店铺 {shop_id} 账号 {username} 的WebSocket连接已经关闭或不存在")
            
            # 停止消息消费者
            queue_name = f"pdd_{shop_id}"
            await self._cleanup_resources(queue_name)
            
            self.logger.info(f"成功停止店铺 {shop_id} 账号 {username}")
            
        except Exception as e:
            self.logger.error(f"停止店铺 {shop_id} 账号 {user_id} 时发生错误: {str(e)}")


    async def init(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable):
        """
        初始化WebSocket连接和消息处理系统
        """
        try:
            # 创建停止事件
            self._stop_event = asyncio.Event()
            
            # 获取访问令牌
            token = GetToken(shop_id, user_id)
            access_token = token.get_token()
            
            # 设置队列名称
            queue_name = f"pdd_{shop_id}"
            
            # 初始化消息消费者和处理器（只创建一次）
            await self._setup_message_consumer(queue_name)
            
            # 构建WebSocket连接URL
            params = {
                "access_token": access_token,
                "role": "mall_cs",
                "client": "web",
                "version": "202506091557"
            }
            query = "&".join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{self.base_url}?{query}"
            
            self.logger.debug(f"正在连接到拼多多WebSocket: {shop_id}-{username}")
            
            # 建立WebSocket连接
            async with websockets.connect(
                full_url,
                ping_interval=20,
                ping_timeout=20
            ) as websocket:
                self.ws = websocket
                # 注册WebSocket连接到资源管理器
                self.resource_manager.register_websocket(
                    websocket,
                    f"PDD WebSocket ({shop_id}-{username})"
                )

                self.logger.debug(f"WebSocket连接已建立: {shop_id}-{username}")

                # 连接成功，调用成功回调
                on_success()

                # 创建消息接收任务
                message_task = asyncio.create_task(
                    self._message_loop(websocket, shop_id, user_id, username, queue_name)
                )

                # 等待停止事件或消息任务完成
                stop_task = asyncio.create_task(self._stop_event.wait())

                try:
                    done, pending = await asyncio.wait(
                        [message_task, stop_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # 取消未完成的任务
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    if stop_task in done:
                        self.logger.debug(f"收到停止信号: {shop_id}-{username}")
                    else:
                        self.logger.debug(f"消息循环自然结束: {shop_id}-{username}")

                except asyncio.CancelledError:
                    self.logger.debug(f"WebSocket任务被取消: {shop_id}-{username}")
                    message_task.cancel()
                    try:
                        await message_task
                    except asyncio.CancelledError:
                        pass
                    
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.warning(f"WebSocket连接已关闭: {shop_id}-{username}, 错误: {str(e)}")
            on_failure(f"WebSocket连接已关闭: {e}")
        except Exception as e:
            self.logger.error(f"WebSocket连接错误: {shop_id}-{username}, 错误: {str(e)}")
            on_failure(f"WebSocket连接错误: {e}")
        finally:
            # 清理资源
            await self._cleanup_resources(f"pdd_{shop_id}")
    
    async def _message_loop(self, websocket, shop_id: str, user_id: str, username: str, queue_name: str):
        """消息接收循环 - 优化版本支持并发处理"""
        try:
            async for message in websocket:
                if self._stop_event and self._stop_event.is_set():
                    self.logger.info(f"停止事件已设置，退出消息循环: {shop_id}-{username}")
                    break

                # 创建并发处理任务
                task = asyncio.create_task(
                    self._process_websocket_message_concurrent(
                        message, shop_id, user_id, username, queue_name
                    )
                )

                # 添加到任务跟踪集合
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning(f"WebSocket连接在消息循环中关闭: {shop_id}-{username}")
        except Exception as e:
            self.logger.error(f"消息循环错误: {shop_id}-{username}, 错误: {str(e)}")

    async def _process_websocket_message_concurrent(
        self, message: str, shop_id: str, user_id: str, username: str, queue_name: str
    ):
        """并发处理WebSocket消息"""
        async with self.message_semaphore:
            try:
                await self._process_websocket_message(message, shop_id, user_id, username, queue_name)
            except Exception as e:
                self.logger.error(f"并发处理消息失败: {e}")

    async def cleanup_processing_tasks(self):
        """清理所有处理任务"""
        if not self.processing_tasks:
            return

        self.logger.info(f"清理 {len(self.processing_tasks)} 个处理任务")
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(f"清理任务失败: {e}")

        self.processing_tasks.clear()
    
    def request_stop(self):
        """请求停止WebSocket连接"""
        if self._stop_event:
            self._stop_event.set()
    
    async def _setup_message_consumer(self, queue_name: str):
        """
        设置消息消费者和处理器链
        """
        try:
            from Message.message_consumer import message_consumer_manager
            from Message.message_handler import handler_chain
            
            # 检查消费者是否已存在
            existing_consumer = message_consumer_manager.get_consumer(queue_name)
            if existing_consumer:
                self.logger.info(f"消费者 {queue_name} 已存在，跳过创建")
                return
            
            # 创建新的消费者
            consumer = message_consumer_manager.create_consumer(queue_name, max_concurrent=10)
            
            # 添加处理器链
            handlers = handler_chain(use_ai=True, businessHours=self.businessHours)
            for handler in handlers:
                consumer.add_handler(handler)
            
            # 启动消费者
            await message_consumer_manager.start_consumer(queue_name)
            self.logger.debug(f"消息消费者已启动: {queue_name}")
            
        except Exception as e:
            self.logger.error(f"设置消息消费者失败: {e}")
            raise
    
    async def _process_websocket_message(self, message: str, shop_id: str, user_id: str, username: str, queue_name: str):
        """
        处理单条WebSocket消息
        """
        try:
            # 解析消息
            message_data = json.loads(message)
            self.logger.debug(f"收到消息: {json.dumps(message_data, indent=2, ensure_ascii=False)}")
            
            # 转换为PDD消息对象
            pdd_message = PDDChatMessage(message_data)
            
            # 转换为Context格式
            context = self._convert_to_context(pdd_message, shop_id, user_id, username)
            if context:
                # 根据消息类型决定处理方式
                if self._should_process_immediately(context):
                    # 立即处理的消息类型
                    await self._handle_immediate_message(context, shop_id, user_id)
                    self.logger.debug(f"立即处理消息: {context.type}, ID: {pdd_message.msg_id}")
                elif self._should_queue_message(context):
                    # 需要放入队列的消息类型
                    msg_id = await put_message(queue_name, context)
                    self.logger.debug(f"消息已入队: {queue_name}, ID: {msg_id}, 类型: {context.type}")
                else:
                    # 忽略的消息类型
                    self.logger.debug(f"忽略消息: {context.type}, ID: {pdd_message.msg_id}")
            else:
                self.logger.warning("消息转换失败，跳过处理")
                
        except json.JSONDecodeError:
            self.logger.error(f"JSON解析失败: {message}")
        except Exception as e:
            self.logger.error(f"处理WebSocket消息失败: {e}")
    
    def _should_process_immediately(self, context: Context) -> bool:
        """
        判断消息是否需要立即处理（不放入队列）
        
        立即处理的消息类型：
        - 系统状态消息（心跳、连接状态等）
        - 认证消息（登录验证等）
        - 撤回消息（需要及时响应）
        - 系统提示消息
        - 商城客服消息（其他客服发的消息）
        - 转接消息
        """
        immediate_types = {
            ContextType.SYSTEM_STATUS,    # 系统状态
            ContextType.AUTH,             # 认证消息
            ContextType.WITHDRAW,         # 撤回消息
            ContextType.SYSTEM_HINT,      # 系统提示
            ContextType.MALL_CS,          # 商城客服消息
            ContextType.TRANSFER          # 转接消息
        }
        
        return context.type in immediate_types
    
    def _should_queue_message(self, context: Context) -> bool:
        """
        判断消息是否需要放入队列处理
        
        放入队列的消息类型：
        - 用户文本消息（需要AI分析和回复）
        - 图片消息（需要识别和处理）
        - 视频消息（需要分析处理）
        - 表情消息（需要智能回复）
        - 商品咨询（需要详细业务处理）
        - 订单信息（需要查询和处理）
        - 商品卡片（需要业务逻辑处理）
        """
        queue_types = {
            ContextType.TEXT,             # 文本消息
            ContextType.IMAGE,            # 图片消息
            ContextType.VIDEO,            # 视频消息
            ContextType.EMOTION,          # 表情消息
            ContextType.GOODS_INQUIRY,    # 商品咨询
            ContextType.ORDER_INFO,       # 订单信息
            ContextType.GOODS_CARD,       # 商品卡片
            ContextType.GOODS_SPEC,       # 商品规格
        }
        
        return context.type in queue_types
    
    async def _handle_immediate_message(self, context: Context, shop_id: str, user_id: str):
        """
        立即处理消息
        """
        username = context.kwargs.get("username")
        recipient_uid = context.kwargs.get('from_uid')
        try:
            from Channel.pinduoduo.utils.API.send_message import SendMessage
            send_message = SendMessage(shop_id, user_id)
            if context.type == ContextType.AUTH:
                # 认证消息处理
                auth_info = context.content
                if isinstance(auth_info, dict):
                    result = auth_info.get('result')
                    if result == 'ok':
                        self.logger.info(f"{username}认证成功")
                    else:
                        self.logger.warning(f"{username}认证失败")
                        
            elif context.type == ContextType.WITHDRAW:
                # 撤回消息处理
                self.logger.info(f"收到撤回消息: {context.content}")
                send_message.send_text(recipient_uid,"[玫瑰]")

            elif context.type == ContextType.SYSTEM_STATUS:
                # 系统状态消息
                self.logger.debug(f"系统状态消息: {context.content}")
                
            elif context.type == ContextType.SYSTEM_HINT:
                # 系统提示消息
                self.logger.info(f"系统提示: {context.content}")
                
            elif context.type == ContextType.MALL_CS:
                # 其他客服消息，通常不需要回复
                self.logger.debug(f"收到客服消息: {context.content}")
                
            elif context.type == ContextType.SYSTEM_BIZ:
                # 系统业务消息
                self.logger.info(f"系统业务消息: {context.content}")
                
            elif context.type == ContextType.MALL_SYSTEM_MSG:
                # 商城系统消息
                self.logger.info(f"商城系统消息: {context.content}")
                
            elif context.type == ContextType.TRANSFER:
                # 转接消息
                self.logger.info(f"转接消息: {context.content}")
                send_message.send_text(recipient_uid,"[玫瑰]")
                
        except Exception as e:
            self.logger.error(f"立即处理消息失败: {e}")
    
    async def _cleanup_resources(self, queue_name: str):
        """
        清理资源 - 优化版本支持完整资源管理
        """
        try:
            # 清理处理任务
            await self.cleanup_processing_tasks()

            # 清理WebSocket资源
            await self.resource_manager.cleanup_all()

            # 停止消息消费者
            from Message.message_consumer import message_consumer_manager
            await message_consumer_manager.stop_consumer(queue_name)
            self.logger.debug(f"已停止消息消费者: {queue_name}")

            # 清理WebSocket连接引用
            self.ws = None

        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")  
    
    def _convert_to_context(self, pdd_message: PDDChatMessage, shop_id: str, user_id: str, username: str) -> Optional[Context]:
        """
        将拼多多消息转换为Context格式
        
        Args:
            pdd_message: 拼多多消息对象
            shop_id: 店铺ID
            user_id: 用户ID
            username: 用户名
            
        Returns:
            Context对象或None
        """
        try:
            # 直接从pdd_message中获取Context类型
            context_type = pdd_message.user_msg_type
            
            # 构建额外参数
            kwargs = {
                'msg_id': pdd_message.msg_id,
                'from_user': pdd_message.from_user,
                'from_uid': pdd_message.from_uid,
                'to_user': pdd_message.to_user,
                'to_uid': pdd_message.to_uid,
                'nickname': pdd_message.nickname,
                'timestamp': pdd_message.timestamp,
                'user_msg_type': pdd_message.user_msg_type,
                'shop_id': shop_id,
                'user_id': user_id,
                'username': username,
                'raw_data': pdd_message.raw_data
            }
            
            # 创建Context对象
            context = Context(
                type=context_type,
                content=pdd_message.content,
                kwargs=kwargs,
                channel_type=ChannelType.PINDUODUO
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"转换消息格式时发生错误: {e}")
            return None