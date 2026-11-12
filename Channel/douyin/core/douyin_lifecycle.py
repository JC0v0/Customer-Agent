# 生命周期管理模块（抖音渠道，结构与拼多多一致）
import asyncio
import os
import time
import websockets
from urllib.parse import urlencode
from websockets import exceptions as ws_exceptions
from typing import Optional, Any
from utils.logger_loguru import get_logger
from utils.runtime_path import get_temp_path
from Channel.douyin.utils.API.get_token import GetToken
from Channel.douyin.message_rules import (
    Origin,
    classify_merchant_sender,
    clear_self_sent,
    is_agent_message,
    is_self_client_message,
    is_self_sent,
    resolve_http_origin,
)
from config import config
from bridge.context import (
    ChannelType,
    make_account_key,
    make_handoff_key,
    make_queue_name,
)
from core.session_state import SessionState

# 会话凭据 / 发送帧模板路径（统一走运行时临时目录，兼容打包环境）
PIGEON_SESSION_PATH = str(get_temp_path("dy_session_live.json"))
PIGEON_TEMPLATE_PATH = str(get_temp_path("dy_send_capture.json"))
# 接收用的请求模板（v3 get_user_message 的 protobuf body 抓包）。
# 与发送模板是两份不同的抓包，不能共用：发送帧是 WS 帧，接收是 HTTP body。
PIGEON_USRMSG_TEMPLATE_PATH = str(get_temp_path("dy_usermsg_template.json"))
# HTTP 拉取轮询间隔（秒）。
# 飞鸽 WS 只推会话状态，消息正文要靠 HTTP 拉取。间隔直接决定买家消息的
# 可见延迟；空轮询已经不打日志、失败已限频，因此用短间隔保证及时接收。
# 实测复用连接后单次请求 0.2 秒返回：0.4 秒间隔 + 一次请求 ≈ 0.6 秒一轮，
# 买家发完消息到派发的平均延迟在半秒以内；WS 一有推送还会立刻唤醒。
PIGEON_PULL_INTERVAL = 0.4

# 同账号只允许一条消息循环：新实例启动时取消旧循环，避免双投递复发。
_ACTIVE_PIGEON_LOOPS: dict = {}

# 登录过期后的自动刷新冷却（秒）。perform_relogin 内部另有 60 秒 ReloginGuard，
# 这里是更外层的闸：浏览器刷新一次要几秒到几十秒，不能每轮都拉。
DOUYIN_RELOGIN_COOLDOWN_SECONDS = 300


class LifecycleMixin:
    """生命周期管理 Mixin"""

    async def start_account(self, shop_id: str, user_id: str, on_success: callable, on_failure: callable):
        """启动指定店铺下账号"""
        requested_account_key = make_account_key(
            ChannelType.DOUYIN, shop_id, user_id
        )
        bound_account_key = getattr(self, "_account_key", None)
        if bound_account_key and bound_account_key != requested_account_key:
            message = "一个连接实例只能绑定一个账号"
            self.logger.error(message)
            on_failure(message)
            return
        account_info = db_manager.get_account(self.channel_name, shop_id, user_id)
        if not account_info:
            error_msg = f"账号 {user_id} 在数据库中不存在"
            self.logger.error(error_msg)
            on_failure(error_msg)
            return

        self._account_key = requested_account_key
        username = account_info.get("username", user_id)
        connection_key = f"{shop_id}_{user_id}"

        self.status_manager.update_status(shop_id, user_id, username, ConnectionState.CONNECTING)

        if connection_key in self._reconnect_tasks:
            self._reconnect_tasks[connection_key].cancel()
            del self._reconnect_tasks[connection_key]

        if self.reconnect_config.enable_auto_reconnect:
            connect_task = asyncio.create_task(
                self._connect_with_retry(shop_id, user_id, username, on_success, on_failure)
            )
        else:
            connect_task = asyncio.create_task(
                self._connect_single_attempt(shop_id, user_id, username, on_success, on_failure)
            )

        self._reconnect_tasks[connection_key] = connect_task

    async def stop_account(self, shop_id: str, user_id: str):
        """停止指定店铺下账号"""
        try:
            account_info = db_manager.get_account(self.channel_name, shop_id, user_id)
            if not account_info:
                self.logger.warning(f"账号 {user_id} 不存在，无法停止")
                return

            username = account_info.get("username", user_id)
            connection_key = f"{shop_id}_{user_id}"

            self.logger.info(f"正在停止店铺 {shop_id} 账号 {username}")

            if self._stop_event:
                self._stop_event.set()

            if connection_key in self._reconnect_tasks:
                task = self._reconnect_tasks[connection_key]
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except asyncio.CancelledError:
                        self.logger.debug(f"重连任务已被取消: {connection_key}")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"重连任务取消超时: {connection_key}")
                    except Exception as task_error:
                        self.logger.error(
                            f"等待重连任务完成时出错: error_type={type(task_error).__name__}"
                        )
                self._reconnect_tasks.pop(connection_key, None)
                self.logger.debug(f"已清理重连任务: {connection_key}")

            if connection_key in self._heartbeat_tasks:
                task = self._heartbeat_tasks[connection_key]
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except asyncio.CancelledError:
                        self.logger.debug(f"心跳任务已被取消: {connection_key}")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"心跳任务取消超时: {connection_key}")
                    except Exception as task_error:
                        self.logger.error(
                            f"等待心跳任务完成时出错: error_type={type(task_error).__name__}"
                        )
                del self._heartbeat_tasks[connection_key]
                self.logger.debug(f"已清理心跳任务: {connection_key}")

            if connection_key in self._health_tasks:
                task = self._health_tasks[connection_key]
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except asyncio.CancelledError:
                        self.logger.debug(f"Cookie健康检查任务已被取消: {connection_key}")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Cookie健康检查任务取消超时: {connection_key}")
                    except Exception as task_error:
                        self.logger.error(
                            f"等待Cookie健康检查任务完成时出错: error_type={type(task_error).__name__}"
                        )
                del self._health_tasks[connection_key]
                self.logger.debug(f"已清理Cookie健康检查任务: {connection_key}")

            self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)

            if self.ws:
                await self._safe_close_websocket(self.ws)
                self.logger.info(f"已关闭店铺 {shop_id} 账号 {username} 的WebSocket连接")
            else:
                self.logger.warning(f"店铺 {shop_id} 账号 {username} 的WebSocket连接已经关闭或不存在")

            await self.cleanup_processing_tasks()

            queue_name = make_queue_name(ChannelType.DOUYIN, shop_id, user_id)
            await self._cleanup_resources(queue_name)

            self.logger.info(f"成功停止店铺 {shop_id} 账号 {username}")

        except Exception as e:
            self.logger.error(
                f"停止店铺 {shop_id} 账号 {user_id} 时发生错误: "
                f"error_type={type(e).__name__}"
            )
            self.logger.exception(
                f"停止店铺 {shop_id} 账号 {user_id} 详细堆栈",
                exc_info=True,
            )

    async def init(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable):
        """初始化消息处理系统。

        传输层优先级：
          1. pigeon 纯协议（方案B，与拼多多同构）：当 use_pigeon=True 或
             存在会话凭据快照 dy_session_live.json 时使用。WS protobuf 收发，
             会话凭据(token+pigeon_sign)浏览器抓取一次复用。
          2. 浏览器桥（方案A，兜底）：use_bridge=True 时使用，DOM 收发。
        """
        self._stop_event = asyncio.Event()

        use_pigeon = getattr(self, "use_pigeon", None)
        if use_pigeon is None:
            use_pigeon = os.path.exists(PIGEON_SESSION_PATH)
        if use_pigeon:
            # pigeon 优先；浏览器桥仅在调用方显式开启时才作为兼容兜底。
            connected = await self._pigeon_connect(shop_id, user_id, username, on_success, on_failure)
            if connected:
                await self._pigeon_wait(shop_id, user_id, username, on_success, on_failure)
                return
            if getattr(self, "use_bridge", False):
                self.logger.warning(f"消息通道回退浏览器桥: {shop_id}-{username}")
                await self._init_bridge(shop_id, user_id, username, on_success, on_failure)
                return
            raise ConnectionError("飞鸽 pigeon 会话不可用，请重新登录以刷新会话凭据")

        # 方案A：浏览器桥模式（默认兜底）。绕过 pigeon 签名/WS protobuf，直接驱动飞鸽网页。
        if getattr(self, "use_bridge", False):
            await self._init_bridge(shop_id, user_id, username, on_success, on_failure)
            return

        # ------- 以下为纯协议模式（WS）占位实现，当前不可用 -------
        try:

            # TODO(传输缝 #1): GetToken.get_token() 当前raise NotImplementedError。
            # 抖音真实取 token（抖店开放平台 / 商家后台端点与签名）需由你补全。
            # 补全后此处会返回字符串 token。
            def _get_access_token():
                token = GetToken(shop_id, user_id)
                return token.get_token()

            access_token = await asyncio.to_thread(_get_access_token)

            queue_name = make_queue_name(ChannelType.DOUYIN, shop_id, user_id)
            await self._setup_message_consumer(queue_name, shop_id, user_id)

            # TODO(传输缝 #2): 下面的 query 沿用拼多多参数结构，与飞鸽实证协议
            # （token / aid / device_id / access_key / pigeon_sign）不一致，此分支
            # 暂不可用。实际连接由 PigeonTransport.ws_url() 构造。
            params = {
                "access_token": access_token,
                "role": "mall_cs",
                "client": "web",
                "version": self.API_VERSION
            }
            query = urlencode(params)
            full_url = f"{self.base_url}?{query}"

            self.logger.debug(f"正在连接到抖音WebSocket: {shop_id}-{username}")

            async with websockets.connect(
                full_url,
                ping_interval=60,
                ping_timeout=30,
                max_size=10**7,
                compression=None,
                close_timeout=10
            ) as websocket:
                self.ws = websocket
                self.resource_manager.register_websocket(
                    websocket,
                    f"Douyin WebSocket ({shop_id}-{username})"
                )
                self.logger.debug(f"WebSocket连接已建立: {shop_id}-{username}")

                if self.ws and not self._is_ws_closed(self.ws):
                    self.logger.debug(f"WebSocket连接正常: {shop_id}-{username}")
                else:
                    self.logger.error(f"WebSocket连接异常: {shop_id}-{username}")

                self.status_manager.update_status(shop_id, user_id, username, ConnectionState.CONNECTED)
                self.logger.debug(f"暂时跳过在线状态设置: {shop_id}-{username}")

                on_success()

                heartbeat_task = None
                if self.heartbeat_config.enable_heartbeat:
                    connection_key = f"{shop_id}_{user_id}"
                    heartbeat_task = asyncio.create_task(
                        self._heartbeat_loop(websocket, shop_id, user_id, username)
                    )
                    self._heartbeat_tasks[connection_key] = heartbeat_task
                    self.logger.debug(f"心跳检查已启动: {shop_id}-{username}")

                health_task = None
                if self.heartbeat_config.enable_cookie_health_check:
                    connection_key = f"{shop_id}_{user_id}"
                    health_task = asyncio.create_task(
                        self._cookie_health_loop(shop_id, user_id, username)
                    )
                    self._health_tasks[connection_key] = health_task
                    self.logger.debug(f"Cookie 健康检查已启动: {shop_id}-{username}")

                message_task = asyncio.create_task(
                    self._message_loop(websocket, shop_id, user_id, username, queue_name)
                )

                stop_task = asyncio.create_task(self._stop_event.wait())

                try:
                    tasks = [message_task, stop_task]
                    if heartbeat_task:
                        tasks.append(heartbeat_task)
                    if health_task:
                        tasks.append(health_task)

                    done, pending = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    should_cleanup = False
                    if stop_task in done:
                        self.logger.debug(f"收到停止信号: {shop_id}-{username}")
                        should_cleanup = True
                    else:
                        self.logger.warning(f"消息循环异常结束: {shop_id}-{username}")
                        should_cleanup = True

                    for task in pending:
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=3.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                            pass
                        except Exception as e:
                            self.logger.debug(
                                f"等待任务取消时出错: error_type={type(e).__name__}"
                            )

                    if should_cleanup:
                        await self._cleanup_resources(queue_name)

                except asyncio.CancelledError:
                    self.logger.debug(f"WebSocket任务被取消: {shop_id}-{username}")
                    message_task.cancel()
                    if heartbeat_task:
                        heartbeat_task.cancel()
                    if health_task:
                        health_task.cancel()
                    try:
                        await asyncio.wait_for(message_task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                        pass
                    if heartbeat_task:
                        try:
                            await asyncio.wait_for(heartbeat_task, timeout=3.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                            pass
                    if health_task:
                        try:
                            await asyncio.wait_for(health_task, timeout=3.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                            pass
                    await self._cleanup_resources(
                        make_queue_name(ChannelType.DOUYIN, shop_id, user_id)
                    )

        except ws_exceptions.ConnectionClosed as e:
            self.status_manager.update_status(
                shop_id, user_id, username, ConnectionState.ERROR, type(e).__name__
            )
            self.logger.warning(
                f"WebSocket连接已关闭: {shop_id}-{username}, 错误类型: {type(e).__name__}"
            )
            on_failure("WebSocket连接已关闭，请稍后重试")
        except Exception as e:
            self.status_manager.update_status(
                shop_id, user_id, username, ConnectionState.ERROR, type(e).__name__
            )
            self.logger.error(
                f"WebSocket连接错误: {shop_id}-{username}, 错误类型: {type(e).__name__}"
            )
            on_failure("WebSocket连接错误，请检查网络后重试")
            await self._cleanup_resources(
                make_queue_name(ChannelType.DOUYIN, shop_id, user_id)
            )

    async def _init_bridge(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable):
        """飞鸽浏览器桥初始化（方案A 传输层）。

        启动常驻浏览器桥（复用 profile 登录态），随后进入桥消息轮询循环，
        把桥检测到的买家消息送入与 WS 相同的消息处理管线。
        """
        from Channel.douyin.feige_bridge import FeigeBridge

        try:
            queue_name = make_queue_name(ChannelType.DOUYIN, shop_id, user_id)
            await self._setup_message_consumer(queue_name, shop_id, user_id)

            bridge = FeigeBridge.instance()
            try:
                await asyncio.to_thread(bridge.start)
            except Exception as e:
                self.logger.error(
                    f"消息通道启动异常: {shop_id}-{username}, "
                    f"error_type={type(e).__name__}"
                )

            if not bridge.is_ready:
                self.status_manager.update_status(
                    shop_id, user_id, username, ConnectionState.ERROR, "bridge_not_ready"
                )
                self.logger.error(
                    f"浏览器未就绪（可能未登录）: {shop_id}-{username}，"
                    f"请在弹出的浏览器中登录抖店账号后重试"
                )
                on_failure("飞鸽浏览器桥未就绪：请在弹出的浏览器中登录抖店账号后重试")
                return

            self.status_manager.update_status(
                shop_id, user_id, username, ConnectionState.CONNECTED
            )
            self.logger.info(f"连接已建立: {shop_id}-{username}")
            on_success()

            message_task = asyncio.create_task(
                self._bridge_message_loop(shop_id, user_id, username, queue_name)
            )
            stop_task = asyncio.create_task(self._stop_event.wait())

            try:
                done, pending = await asyncio.wait(
                    [message_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if message_task in done and stop_task not in done:
                    # Propagate a transport failure to ConnectionMixin so it
                    # follows the same bounded retry/status path as PDD.
                    message_task.result()
                    raise ConnectionError("飞鸽消息循环意外结束")
                for task in pending:
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                        pass
                await self._cleanup_resources(
                    make_queue_name(ChannelType.DOUYIN, shop_id, user_id)
                )
            except asyncio.CancelledError:
                message_task.cancel()
                try:
                    await asyncio.wait_for(message_task, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                    pass
                await self._cleanup_resources(
                    make_queue_name(ChannelType.DOUYIN, shop_id, user_id)
                )
        except Exception as e:
            self.status_manager.update_status(
                shop_id, user_id, username, ConnectionState.ERROR, type(e).__name__
            )
            self.logger.error(
                f"连接错误: {shop_id}-{username}, error_type={type(e).__name__}"
            )
            on_failure("飞鸽浏览器桥连接错误，请检查日志")

    async def _bridge_message_loop(self, shop_id: str, user_id: str, username: str, queue_name: str):
        """飞鸽桥消息轮询循环：轮询桥事件并送入消息处理管线。"""
        from Channel.douyin.feige_bridge import FeigeBridge

        bridge = FeigeBridge.instance()
        self.logger.info(f"消息循环开始: {shop_id}-{username}")
        try:
            while not (self._stop_event and self._stop_event.is_set()):
                ev = await asyncio.to_thread(bridge.get_event, 2.0)
                if not ev:
                    continue
                while len(self.processing_tasks) >= self.max_concurrent_messages:
                    done, _ = await asyncio.wait(
                        list(self.processing_tasks),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    self.processing_tasks.difference_update(done)
                task = asyncio.create_task(
                    self._process_bridge_event(
                        ev, shop_id, user_id, username, queue_name
                    )
                )
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)
        except asyncio.CancelledError:
            self.logger.debug(f"消息循环被取消: {shop_id}-{username}")
        except Exception as e:
            self.logger.error(
                f"消息循环异常: {shop_id}-{username}, "
                f"error_type={type(e).__name__}"
            )

    def request_stop(self):
        """请求停止WebSocket连接"""
        if self._stop_event:
            self._stop_event.set()

    async def _pigeon_connect(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable) -> bool:
        """pigeon 纯协议连接（方案B，与拼多多同构）。成功返回 True 并调用 on_success。

        读取会话凭据快照(dy_session_live.json) + 发送帧模板(dy_send_capture.json)，
        建立持久 WS 连接。失败返回 False（不调用 on_failure，由调用方决定回退桥）。
        """
        from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport, PigeonSession

        try:
            queue_name = make_queue_name(ChannelType.DOUYIN, shop_id, user_id)
            await self._setup_message_consumer(queue_name, shop_id, user_id)

            if not os.path.exists(PIGEON_SESSION_PATH):
                self.status_manager.update_status(
                    shop_id, user_id, username, ConnectionState.ERROR, "no_pigeon_session"
                )
                self.logger.error(
                    f"会话凭据缺失: {shop_id}-{username}，请重新登录以刷新会话凭据"
                )
                return False

            transport = PigeonTransport(
                PigeonSession.from_json(PIGEON_SESSION_PATH), PIGEON_TEMPLATE_PATH
            )
            try:
                ok = await asyncio.to_thread(transport.connect, 15)
            except Exception as e:
                self.logger.error(
                    f"连接异常: {shop_id}-{username}, "
                    f"error_type={type(e).__name__}"
                )
                ok = False
            if not ok or not transport.is_connected():
                self.status_manager.update_status(
                    shop_id, user_id, username, ConnectionState.ERROR, "pigeon_conn_failed"
                )
                self.logger.error(
                    f"会话凭据已过期: {shop_id}-{username}，请重新登录以刷新会话凭据"
                )
                transport.close()
                return False

            PigeonTransport.set_instance(transport)
            self.pigeon_transport = transport
            self.status_manager.update_status(
                shop_id, user_id, username, ConnectionState.CONNECTED
            )
            self.logger.info(f"连接已建立: {shop_id}-{username}")
            on_success()
            return True
        except Exception as e:
            self.status_manager.update_status(
                shop_id, user_id, username, ConnectionState.ERROR, type(e).__name__
            )
            self.logger.error(
                f"连接错误: {shop_id}-{username}, error_type={type(e).__name__}"
            )
            return False

    async def _pigeon_wait(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable):
        """pigeon 消息循环等待（连接成功后调用，阻塞直到停止）。"""
        from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport

        transport = PigeonTransport.instance()
        if transport is None:
            self.logger.error(f"传输实例缺失，无法进入消息循环: {shop_id}-{username}")
            return
        queue_name = make_queue_name(ChannelType.DOUYIN, shop_id, user_id)
        try:
            message_task = asyncio.create_task(
                self._pigeon_message_loop(shop_id, user_id, username, queue_name)
            )
            stop_task = asyncio.create_task(self._stop_event.wait())
            try:
                done, pending = await asyncio.wait(
                    [message_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if message_task in done and stop_task not in done:
                    # Let ConnectionMixin retry a failed protocol connection
                    # and publish the same final error state as PDD.
                    message_task.result()
                    raise ConnectionError("飞鸽消息循环意外结束")
                for task in pending:
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                        pass
            except asyncio.CancelledError:
                message_task.cancel()
                try:
                    await asyncio.wait_for(message_task, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                    pass
        finally:
            transport.close()
            self.pigeon_transport = None
            await self._cleanup_resources(
                make_queue_name(ChannelType.DOUYIN, shop_id, user_id)
            )

    async def _pigeon_message_loop(self, shop_id: str, user_id: str, username: str, queue_name: str):
        """pigeon 消息接收循环：与拼多多同构，一条来源派发。

        接收分工：
          1) WS poll → 只用来唤醒拉取（pull_now.set()），本身不派发。
             派发只走一条路径，同一条消息就不可能被两个来源各派发一次。
          2) 独立 HTTP task → 每 PIGEON_PULL_INTERVAL 秒请求一次 v3
             get_user_message 时间线接口，WS 一有状态变化就立刻唤醒。
             一次请求同时取回：
               - 买家消息 → 送入消息管线，并回填 security_sender_id/shop_id/talk_id
               - 商家侧消息 → 判定人工介入，静默转人工（见 _mark_handoff_by_merchant）

        不用「当前会话列表」接口：会话被系统关闭后就不在那个列表里（实测
        total=0），而买家仍会继续往该会话发消息，用它做主路径会彻底收不到。

        【架构说明】HTTP 拉取为独立 asyncio task，与 WS poll 并行，避免 poll
        阻塞时饿死拉取（此前串行循环曾出现 98 秒零拉取的问题）。
        日志风格与拼多多一致：只有「消息循环开始」、30 秒心跳、失败告警。
        """
        from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport, PigeonSession
        from tools.reverse_probe.douyin.pigeon_http import PigeonHttp

        # 同账号只允许一条消息循环。
        # UI 连点两次「开始自动回复」会 new 两个 DouyinChannel，两条 pull loop
        # 并行拉同一批消息，同一条消息就被插两次队、AI 回两遍。拼多多靠
        # 「一个 WS 连接」天然幂等，抖音是 HTTP 拉取，必须显式互斥。
        loop_key = "%s_%s" % (shop_id, user_id)
        previous = _ACTIVE_PIGEON_LOOPS.get(loop_key)
        current = asyncio.current_task()
        if previous is not None and previous is not current and not previous.done():
            self.logger.warning(
                f"检测到同账号的旧消息循环，先停止: {shop_id}-{username}"
            )
            previous.cancel()
            try:
                await asyncio.wait_for(previous, timeout=5.0)
            except (
                asyncio.CancelledError,
                asyncio.TimeoutError,
                asyncio.InvalidStateError,
                # 旧循环若跑在另一个事件循环里，await 会抛 RuntimeError；
                # 此时取消已发出，继续往下走即可，不能让启动失败。
                RuntimeError,
            ):
                pass
        _ACTIVE_PIGEON_LOOPS[loop_key] = current

        transport = PigeonTransport.instance()
        http = None
        seen_msg = set()
        # WS 推「会话状态变化」时唤醒拉取循环，立刻补拉正文。
        pull_now = asyncio.Event()
        # 接收只有一条路径：v3 get_user_message 时间线接口。
        #
        # 不能用「当前会话列表」做唯一主路径：会话被系统关闭后就不在那个
        # 列表里（真机实测 total=0），但买家仍会继续往这个会话发消息，
        # 表现就是「发了消息程序毫无反应」。v3 按时间线拉，不看会话状态。
        try:
            http = PigeonHttp(
                PigeonSession.from_json(PIGEON_SESSION_PATH),
                PIGEON_USRMSG_TEMPLATE_PATH,
            )
        except Exception as e:
            # 拉取器起不来就等于「一条消息都收不到」，绝不能只打一行 warning
            # 然后带着 http=None 继续跑：那样 WS 还活着、状态还是「已连接」，
            # 表面健康实则完全不收消息。这里直接抛给连接层重试并暴露错误。
            self.logger.error(
                f"消息拉取初始化失败，无法接收消息: {shop_id}-{username}, "
                f"error_type={type(e).__name__}, err={e}, "
                f"template={PIGEON_USRMSG_TEMPLATE_PATH}"
            )
            raise ConnectionError(
                f"飞鸽消息拉取初始化失败: {type(e).__name__}: {e}"
            )

        # 启动拉取 task（与 WS poll 并行）
        http_task = None
        if http is not None:
            http_task = asyncio.create_task(
                self._pigeon_http_pull_loop(
                    http, transport, seen_msg, shop_id, user_id, username,
                    queue_name, pull_now
                )
            )

        try:
            while not (self._stop_event and self._stop_event.is_set()):
                # 与拼多多一致：消息循环不打迭代心跳，存活状态统一由
                # _heartbeat_loop（30s 的「心跳成功」DEBUG）反映。
                if transport is None or not transport.is_connected():
                    raise ConnectionError("飞鸽 websocket 已断开")
                # 1) WS 推送事件：只用来唤醒拉取，不直接派发。
                #    派发只走会话列表这一条路径，同一条消息就不可能被两个
                #    来源各派发一次（这是「一条消息回两遍」的根源）。
                evs = await asyncio.to_thread(transport.poll, 1.0)
                for ev in evs:
                    if ev.get("type") != "is_ack":
                        pull_now.set()
                # 检查拉取 task 是否异常退出
                if http_task and http_task.done():
                    exc = http_task.exception()
                    if exc:
                        self.logger.error(
                            f"消息拉取任务异常退出: {shop_id}-{username}, "
                            f"error_type={type(exc).__name__}"
                        )
                    else:
                        self.logger.warning(
                            f"消息拉取任务已退出: {shop_id}-{username}"
                        )
                    http_task = None  # 避免重复日志
        except asyncio.CancelledError:
            self.logger.debug(f"消息循环被取消: {shop_id}-{username}")
        except Exception as e:
            self.logger.error(
                f"消息循环异常: {shop_id}-{username}, "
                f"error_type={type(e).__name__}"
            )
            raise
        finally:
            if http_task and not http_task.done():
                http_task.cancel()
                try:
                    await asyncio.wait_for(http_task, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                    pass
            # 交还注册位，别让下一个实例误取消一个已经结束的 task。
            if _ACTIVE_PIGEON_LOOPS.get(loop_key) is current:
                _ACTIVE_PIGEON_LOOPS.pop(loop_key, None)

    async def _pigeon_http_pull_loop(self, http, transport, seen_msg, shop_id,
                                     user_id, username, queue_name, pull_now):
        """v3 时间线增量拉取：一条路径同时拿买家消息与商家侧消息。

        为什么不用「当前会话列表」：会话被系统关闭后就不在那个列表里
        （真机实测 total=0），而买家仍会继续往这个会话发消息，表现就是
        「发了消息程序毫无反应」。get_user_message 按时间线拉，不看会话状态。

        启动基线：首轮成功时把已有消息全部标记为已见、一律不派发。不能用
        「本机时间」判断新旧——平台消息时间戳实测比本机慢 66~246 秒，拿本机
        时间去比会把刚发的真实消息也判成历史消息丢掉（这正是「点开始自动回复
        之后发的消息收不到」的直接原因）。
        """
        since_us = 0
        baseline_ready = False
        last_error_logged_at = 0.0
        last_relogin_at = 0.0
        consecutive_errors = 0

        while not (self._stop_event and self._stop_event.is_set()):
            msgs = None
            try:
                msgs = await asyncio.to_thread(http.pull_messages, since_us)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                now_monotonic = time.monotonic()
                if now_monotonic - last_error_logged_at >= 60.0:
                    last_error_logged_at = now_monotonic
                    self.logger.warning(
                        f"消息拉取失败: {shop_id}-{username}, "
                        f"error_type={type(exc).__name__}, err={exc}"
                    )
                msgs = None

            # 连续失败说明凭据可能失效：自动刷新一次，成功则不用人工登录。
            if consecutive_errors >= 6:
                now_monotonic = time.monotonic()
                if now_monotonic - last_relogin_at >= DOUYIN_RELOGIN_COOLDOWN_SECONDS:
                    last_relogin_at = now_monotonic
                    consecutive_errors = 0
                    asyncio.create_task(
                        self._auto_refresh_douyin_credentials(
                            shop_id, user_id, username, transport, http
                        )
                    )

            # 拿不到响应时不要建立基线：空基线会把之后拉到的全部历史
            # 当成新消息，一次性回复几天前的会话。
            if msgs is not None and not baseline_ready:
                newest_us = 0
                for m in msgs:
                    ts = int(m.get("ts") or 0)
                    if ts > newest_us:
                        newest_us = ts
                    # 买家侧和商家侧共用同一套去重键：商家侧也要靠它把
                    # 「点开始自动回复之前的历史人工回复」挡掉，否则一启动
                    # 就会把上一轮 AI 的回复当成人工介入，会话直接静默转人工。
                    seen_msg.add(self._pigeon_dedupe_key({
                        "msg_id": m.get("msg_id"),
                        "conv_id": str(m.get("conv_id") or ""),
                        "ts": ts,
                        "text": m.get("text"),
                    }))
                    self._fill_conv_state(transport, m)
                # 一条历史都没有时用「本机时间 - 300 秒」做基线：
                # 平台时间戳最多比本机慢 246 秒，留 300 秒能保证之后
                # 的真实新消息一定 > 基线。
                if newest_us:
                    since_us = newest_us - 5 * 1000 * 1000
                else:
                    since_us = int(time.time() * 1e6) - 300 * 1000 * 1000
                baseline_ready = True
                self.logger.info(
                    f"接收基线建立: {shop_id}-{username}, 已有 {len(msgs)} 条"
                    f"（一律不回复），只回此后新消息"
                )
                msgs = []

            if msgs:
                newest_us = since_us
                # 先处理商家侧：同一条时间线里人工回复可能紧跟买家消息，
                # 必须先标记转人工，AI 才不会抢在人工前面回。
                for m in msgs:
                    ts = int(m.get("ts") or 0)
                    if ts > newest_us:
                        newest_us = ts
                    if resolve_http_origin(m) is not Origin.MERCHANT:
                        continue
                    # 只认商家坐席（CurrentServer / role=2）。店铺机器人
                    # 与本程序 AI 回复都是 role=4，不能算人工介入。
                    if not is_agent_message(m):
                        continue
                    if is_self_client_message(m.get("client_message_id")):
                        continue
                    conv = str(m.get("conv_id") or "")
                    text = str(m.get("text") or "").strip()
                    if not conv or not text:
                        continue
                    # 商家侧同样要按「会话 + 时间戳 + 正文」去重。游标每轮回退
                    # 5 秒，同一条人工回复会被反复读到；不去重就会每 0.4 秒
                    # 重新标记一次转人工，日志刷屏且掩盖真正的新消息。
                    dedupe_key = self._pigeon_dedupe_key({
                        "msg_id": m.get("msg_id"),
                        "conv_id": conv,
                        "ts": ts,
                        "text": text,
                    })
                    if dedupe_key and dedupe_key in seen_msg:
                        continue
                    if dedupe_key:
                        seen_msg.add(dedupe_key)
                    self._mark_handoff_by_merchant({
                        "conv_id": conv,
                        "text": text,
                        "ts": ts,
                        "sender_kind": "human",
                        "src": str(m.get("sender_biz_role") or ""),
                    }, shop_id, username)

                for m in msgs:
                    ts = int(m.get("ts") or 0)
                    if resolve_http_origin(m) is not Origin.CUSTOMER:
                        continue
                    conv = str(m.get("conv_id") or "")
                    text = str(m.get("text") or "").strip()
                    if not conv or not text:
                        continue
                    # 回填发送所需上下文：买家 ID / 店铺 ID / 当前子会话 ID。
                    # v3 的消息条目自带这三个字段，不需要再额外请求。
                    self._fill_conv_state(transport, m)
                    dedupe_key = self._pigeon_dedupe_key({
                        "msg_id": m.get("msg_id"),
                        "conv_id": conv,
                        "ts": ts,
                        "text": text,
                    })
                    if dedupe_key and dedupe_key in seen_msg:
                        continue
                    if dedupe_key:
                        seen_msg.add(dedupe_key)
                    if len(seen_msg) > 50000:
                        seen_msg.clear()
                    ev = {
                        "type": "text",
                        "conv_id": conv,
                        "msg_id": str(m.get("msg_id") or ""),
                        "text": text,
                        "ts": ts,
                        "security_receiver_id": m.get("security_sender_id") or "",
                        "shop_id": m.get("shop_id") or "",
                        "sender_role": "3",
                        "sender_biz_role": m.get("sender_biz_role") or "",
                        "client_message_id": m.get("client_message_id") or "",
                    }
                    self.logger.debug(
                        f"received message: type=text, shop_id={shop_id}, "
                        f"conv={conv}, text={text[:40]!r}"
                    )
                    await self._dispatch_pigeon_event(
                        ev, shop_id, user_id, username, queue_name
                    )
                # 只推进到「已处理的最大时间戳」，回退 5 秒防止同秒多条被跳过，
                # 重复部分靠 seen_msg 挡住。
                if newest_us > since_us:
                    since_us = newest_us - 5 * 1000 * 1000

            try:
                if pull_now is None:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=PIGEON_PULL_INTERVAL
                    )
                else:
                    await asyncio.wait_for(
                        pull_now.wait(), timeout=PIGEON_PULL_INTERVAL
                    )
                    pull_now.clear()
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            if self._stop_event and self._stop_event.is_set():
                break

    @staticmethod
    def _fill_conv_state(transport, message) -> None:
        """把消息自带的发送上下文回填到 transport.conv_state。

        发送时 SendMessage 靠 conv_state 里的 security_receiver_id / shop_id /
        talk_id 组帧；基线阶段就回填好，第一条新消息到达时也能直接回复，
        不必等第二轮拉取把上下文补齐。
        """
        conv = str(message.get("conv_id") or "")
        if not conv or transport is None:
            return
        state = transport.conv_state.setdefault(conv, {})
        if message.get("security_sender_id"):
            state["security_receiver_id"] = message["security_sender_id"]
        if message.get("shop_id"):
            state["shop_id"] = message["shop_id"]
        if message.get("talk_id"):
            state["talk_id"] = str(message["talk_id"])

    def _mark_handoff_by_merchant(self, item, shop_id, username) -> None:
        """商家侧出现新消息 → 静默转人工（与拼多多 MALL_CS 分支一致）。

        判定依据是「登录的客服商家账号自己发出了消息」：
          - 来源为商家侧（sender_role=2 / CurrentServer），由 message_rules 判定
          - 不是本程序刚发出的那条：AI 的回复同样以商家侧身份回流，
            不排除的话每发一次就自我转人工一次
        命中后只做 mark_handoff，不回复、不发企业微信通知，到期自动恢复。

        handoff key 的 uid 维度用 conv_id：消息派发时 from_uid 就是 conv_id，
        ai_handler 检查 is_handoff 用的也是它，两边必须一致，否则标记等于没打。
        """
        conv_id = str(item.get("conv_id") or "")
        text = str(item.get("text") or "").strip()
        if not conv_id or not text:
            return
        # 本程序自己发的回声：消费掉指纹后直接返回。
        if is_self_sent(shop_id, conv_id, text):
            clear_self_sent(shop_id, conv_id, text)
            return
        sender_kind = str(item.get("sender_kind") or "")
        if sender_kind == "robot":
            self.logger.debug(
                f"商家侧机器人消息，不转人工: conv={conv_id}, "
                f"text={text[:30]!r}"
            )
            return
        session_key = make_handoff_key(ChannelType.DOUYIN, shop_id, conv_id)
        try:
            if SessionState().is_handoff(session_key):
                return
            SessionState().mark_handoff(session_key)
            self.logger.info(
                f"检测到商家侧消息，会话静默转人工（不再回复且不通知）: "
                f"session_key={session_key}, shop={shop_id}, "
                f"src={item.get('src') or '?'}, text={text[:30]!r}"
            )
        except Exception as exc:
            self.logger.error(
                f"人工介入静默标记失败（不影响主流程）: "
                f"error_type={type(exc).__name__}, exc={exc}"
            )

    async def _auto_refresh_douyin_credentials(self, shop_id, user_id, username,
                                               transport=None, http=None) -> bool:
        """pigeon 凭据过期后自动刷新，对齐拼多多的 perform_relogin。

        抖音有两套凭据：网页 cookie（活得久）和 IM 网关的 token/pigeon_sign
        （约 1~2 小时）。商品查询只用 cookie 所以一直正常，收发的 155/10005
        却是因为后者过期。perform_relogin 阶段 1 会开无头浏览器复用已登录的
        profile 重抓 token/sign，cookie 还在时全自动完成，不弹窗。
        """
        from Channel.douyin.cookie_utils import perform_relogin

        self.logger.info(f"抖音凭据已过期，开始自动刷新: {shop_id}-{username}")
        password = ""
        try:
            from database import db_manager
            account = db_manager.get_account(self.channel_name, shop_id, user_id)
            if account:
                password = account.get("password") or ""
        except Exception as exc:
            self.logger.warning(
                f"读取账号信息失败，按无密码刷新: {shop_id}-{username}, "
                f"error_type={type(exc).__name__}"
            )

        try:
            ok = await asyncio.to_thread(
                perform_relogin,
                self.channel_name, shop_id, user_id, username, password, False,
            )
        except Exception as exc:
            self.logger.error(
                f"自动刷新抖音凭据异常: {shop_id}-{username}, "
                f"error_type={type(exc).__name__}"
            )
            return False

        if ok:
            self._reload_pigeon_session(transport, http)
            self.logger.info(f"抖音凭据自动刷新成功，已切换到新凭据: {shop_id}-{username}")
            return True
        self.logger.error(
            f"抖音凭据自动刷新失败，需要手动重新登录: {shop_id}-{username}"
        )
        return False

    @staticmethod
    def _reload_pigeon_session(transport=None, http=None):
        """把刚写入磁盘的凭据灌回正在使用的传输层与 HTTP 客户端。

        这两处都持有 PigeonSession 对象，不更新的话刷新完还是拿旧 sign 发。
        PigeonConvList 每轮重新读文件，不需要处理。
        """
        try:
            from tools.reverse_probe.douyin.pigeon_transport import (
                PigeonSession,
                PigeonTransport,
            )
            fresh = PigeonSession.from_json(PIGEON_SESSION_PATH)
        except Exception:
            return None
        target = transport
        if target is None:
            target = PigeonTransport.instance()
        if target is not None:
            target.session = fresh
        if http is not None:
            http.session = fresh
        return fresh

    async def _dispatch_pigeon_event(self, ev, shop_id, user_id, username, queue_name):
        """带并发控制的 pigeon 事件派发（复用消息处理管线）。"""
        while len(self.processing_tasks) >= self.max_concurrent_messages:
            done, _ = await asyncio.wait(
                list(self.processing_tasks),
                return_when=asyncio.FIRST_COMPLETED,
            )
            self.processing_tasks.difference_update(done)
        task = asyncio.create_task(
            self._process_pigeon_event(
                ev, shop_id, user_id, username, queue_name
            )
        )
        self.processing_tasks.add(task)
        task.add_done_callback(self.processing_tasks.discard)

    @staticmethod
    def _pigeon_dedupe_key(event):
        """同一条消息在 WS 与会话列表两条链路上的统一去重键。

        不能优先用消息 id：两个接口对同一条消息给出的 id 字段并不一致
        （一个带 message_logid，一个只有 serverMessageId），按 id 去重会
        让同一条消息派发两次。conv_id + 服务端时间戳 + 文本是跨接口稳定的。
        """
        if not event:
            return ""
        conv_id = str(event.get("conv_id") or "")
        timestamp = str(event.get("ts") or "")
        text = str(event.get("text") or "")
        if conv_id and timestamp and text:
            return f"{conv_id}:{timestamp}:{text}"
        # 时间戳缺失时才退回 id，至少能挡住同一接口内的重复投递。
        msg_id = str(event.get("msg_id") or "")
        return "mid:" + msg_id if msg_id else ""

    async def stop_all_connections(self):
        """停止所有连接并清理所有任务"""
        try:
            self.logger.info("正在停止所有连接...")

            if self._stop_event:
                self._stop_event.set()

            for connection_key, task in list(self._reconnect_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        self.logger.debug(f"任务已取消或超时: {connection_key}")
                    except Exception as e:
                        self.logger.error(
                            f"停止任务时出错: {connection_key}, "
                            f"error_type={type(e).__name__}"
                        )
                self._reconnect_tasks.pop(connection_key, None)

            for connection_key, task in list(self._heartbeat_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        self.logger.debug(f"心跳任务已取消或超时: {connection_key}")
                    except Exception as e:
                        self.logger.error(
                            f"停止心跳任务时出错: {connection_key}, "
                            f"error_type={type(e).__name__}"
                        )
                del self._heartbeat_tasks[connection_key]

            for connection_key, task in list(self._health_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        self.logger.debug(f"Cookie健康检查任务已取消或超时: {connection_key}")
                    except Exception as e:
                        self.logger.error(
                            f"停止Cookie健康检查任务时出错: {connection_key}, "
                            f"error_type={type(e).__name__}"
                        )
                del self._health_tasks[connection_key]

            if self.ws:
                await self._safe_close_websocket(self.ws)
                self.ws = None

            self.logger.info("所有连接已停止")

        except Exception as e:
            self.logger.error(
                f"停止所有连接时发生错误: error_type={type(e).__name__}"
            )

    async def _heartbeat_loop(self, websocket, shop_id: str, user_id: str, username: str):
        """心跳检查循环"""
        connection_key = f"{shop_id}_{user_id}"
        consecutive_failures = 0

        try:
            while not (self._stop_event and self._stop_event.is_set()):
                try:
                    start_time = time.time()
                    await websocket.ping()
                    response_time = time.time() - start_time

                    consecutive_failures = 0
                    self.logger.debug(f"心跳成功: {shop_id}-{username}, 响应时间: {response_time:.3f}s")

                    status = self.status_manager.get_status(shop_id, user_id)
                    if status and status.state == ConnectionState.CONNECTED:
                        pass

                    await asyncio.sleep(self.heartbeat_config.heartbeat_interval)

                except asyncio.TimeoutError:
                    consecutive_failures += 1
                    self.logger.warning(f"心跳超时: {shop_id}-{username}, 连续失败: {consecutive_failures}")
                    await asyncio.sleep(self.heartbeat_config.heartbeat_timeout)

                except Exception as e:
                    consecutive_failures += 1
                    self.logger.warning(
                        f"心跳失败: {shop_id}-{username}, "
                        f"error_type={type(e).__name__}, "
                        f"连续失败: {consecutive_failures}"
                    )

                    if consecutive_failures >= self.heartbeat_config.max_heartbeat_failures:
                        self.logger.error(f"心跳检查失败次数过多，标记连接为错误状态: {shop_id}-{username}")
                        self.status_manager.update_status(
                            shop_id, user_id, username,
                            ConnectionState.ERROR,
                            f"心跳检查失败: 连续{consecutive_failures}次失败"
                        )
                        break

                    await asyncio.sleep(self.heartbeat_config.heartbeat_timeout)

        except asyncio.CancelledError:
            self.logger.debug(f"心跳循环被取消: {shop_id}-{username}")
        except Exception as e:
            self.logger.error(
                f"心跳循环异常: {shop_id}-{username}, "
                f"error_type={type(e).__name__}"
            )
        finally:
            if connection_key in self._heartbeat_tasks:
                del self._heartbeat_tasks[connection_key]
            self.logger.debug(f"心跳循环已结束: {shop_id}-{username}")

    async def _cookie_health_loop(self, shop_id: str, user_id: str, username: str):
        """Cookie 健康检查循环，定期验证 cookie 有效性并主动刷新

        验证走飞鸽 backstage/currentuser 端点；网络异常或响应未知时返回 None，
        按安全策略暂不重登。确认失效后由 perform_relogin 拉起浏览器完成重登。
        """
        from Channel.douyin.cookie_utils import check_cookies_valid, perform_relogin
        from Channel.douyin.cookie_cache import cookie_cache

        connection_key = f"{shop_id}_{user_id}"

        try:
            while not (self._stop_event and self._stop_event.is_set()):
                await asyncio.sleep(self.heartbeat_config.cookie_health_check_interval)

                if self._stop_event and self._stop_event.is_set():
                    break

                # 从共享缓存或 DB 加载当前 cookie（db_manager 已按 self.channel_name 隔离）
                cookies = cookie_cache.get(self.channel_name, shop_id, user_id)
                if not cookies:
                    account_info = db_manager.get_account(self.channel_name, shop_id, user_id)
                    if account_info:
                        cookies_data = account_info.get('cookies')
                        if isinstance(cookies_data, str):
                            import json
                            try:
                                cookies = json.loads(cookies_data)
                            except json.JSONDecodeError:
                                cookies = None
                        elif isinstance(cookies_data, dict):
                            cookies = cookies_data

                if not cookies:
                    self.logger.debug(f"无 cookie 可检查: {shop_id}-{username}")
                    continue

                # 轻量级 HTTP 验证（网络异常时返回 None，视作状态未知）
                is_valid = await asyncio.to_thread(
                    check_cookies_valid,
                    self.channel_name, shop_id, user_id, cookies,
                    self.heartbeat_config.cookie_health_check_timeout,
                )

                if is_valid is False:
                    self.logger.warning(
                        f"Cookie 健康检查失败: {shop_id}-{username}，触发主动重登"
                    )
                    account_info = db_manager.get_account(self.channel_name, shop_id, user_id)
                    if account_info:
                        # 抖音登录可能需要人工完成短信验证码或滑块验证，重登必须用
                        # 可见浏览器，不能像拼多多那样走 headless。
                        success = await asyncio.to_thread(
                            perform_relogin,
                            self.channel_name, shop_id, user_id,
                            account_info.get('username', username),
                            account_info.get('password', ''),
                            False,
                        )
                        if success:
                            self.logger.info(f"主动重登成功: {shop_id}-{username}")
                        else:
                            self.logger.error(f"主动重登失败: {shop_id}-{username}")
                elif is_valid is True:
                    self.logger.debug(f"Cookie 健康检查通过: {shop_id}-{username}")
                else:
                    self.logger.debug(f"Cookie 健康检查状态未知，暂不重登: {shop_id}-{username}")

        except asyncio.CancelledError:
            self.logger.debug(f"Cookie 健康检查循环被取消: {shop_id}-{username}")
        except Exception as e:
            self.logger.error(
                f"Cookie 健康检查循环异常: {shop_id}-{username}, "
                f"error_type={type(e).__name__}"
            )
        finally:
            if connection_key in self._health_tasks:
                del self._health_tasks[connection_key]
            self.logger.debug(f"Cookie 健康检查循环已结束: {shop_id}-{username}")

    async def _message_loop(self, websocket, shop_id: str, user_id: str, username: str, queue_name: str):
        """消息接收循环"""
        try:
            self.logger.info(f"消息循环开始: {shop_id}-{username}")

            async for message in websocket:
                if self._stop_event and self._stop_event.is_set():
                    self.logger.info(f"停止事件已设置，退出消息循环: {shop_id}-{username}")
                    break
                # Bound pending processing tasks before reading more work.
                while len(self.processing_tasks) >= self.max_concurrent_messages:
                    done, _ = await asyncio.wait(
                        list(self.processing_tasks),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    self.processing_tasks.difference_update(done)

                task = asyncio.create_task(
                    self._process_websocket_message_concurrent(
                        message, shop_id, user_id, username, queue_name
                    )
                )

                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)

        except ws_exceptions.ConnectionClosed as cc:
            self.logger.warning(f"WebSocket连接正常关闭: {shop_id}-{username}, 代码: {cc.code}")
        except ws_exceptions.ConnectionClosedError as cce:
            self.logger.error(
                f"WebSocket连接异常关闭: {shop_id}-{username}, "
                f"error_type={type(cce).__name__}"
            )
        except Exception as e:
            self.logger.error(
                f"消息循环错误: {shop_id}-{username}, "
                f"error_type={type(e).__name__}"
            )

    async def _process_websocket_message_concurrent(self, message: str, shop_id: str, user_id: str, username: str, queue_name: str):
        """并发处理WebSocket消息"""
        async with self.message_semaphore:
            try:
                await self._process_websocket_message(message, shop_id, user_id, username, queue_name)
            except Exception as e:
                self.logger.error(
                    f"并发处理消息失败: error_type={type(e).__name__}"
                )

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
                    self.logger.error(
                        f"清理任务失败: error_type={type(e).__name__}"
                    )

        self.processing_tasks.clear()

    async def _cleanup_reconnect_tasks(self):
        """清理所有重连任务"""
        try:
            for connection_key, task in list(self._reconnect_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    except asyncio.InvalidStateError:
                        self.logger.debug(f"重连任务在不同的的事件循环中: {connection_key}")
                    except Exception as e:
                        self.logger.error(
                            f"清理重连任务失败: {connection_key}, "
                            f"error_type={type(e).__name__}"
                        )
            self._reconnect_tasks.clear()
        except Exception as e:
            self.logger.error(
                f"清理重连任务列表失败: error_type={type(e).__name__}"
            )

    async def _cleanup_heartbeat_tasks(self):
        """清理所有心跳任务"""
        try:
            for connection_key, task in list(self._heartbeat_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    except asyncio.InvalidStateError:
                        self.logger.debug(f"心跳任务在不同的的事件循环中: {connection_key}")
                    except Exception as e:
                        self.logger.error(
                            f"清理心跳任务失败: {connection_key}, "
                            f"error_type={type(e).__name__}"
                        )
            self._heartbeat_tasks.clear()
        except Exception as e:
            self.logger.error(
                f"清理心跳任务列表失败: error_type={type(e).__name__}"
            )

    async def _cleanup_health_tasks(self):
        """清理所有 Cookie 健康检查任务"""
        try:
            for connection_key, task in list(self._health_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    except asyncio.InvalidStateError:
                        self.logger.debug(f"Cookie健康检查任务在不同的的事件循环中: {connection_key}")
                    except Exception as e:
                        self.logger.error(
                            f"清理Cookie健康检查任务失败: {connection_key}, "
                            f"error_type={type(e).__name__}"
                        )
            self._health_tasks.clear()
        except Exception as e:
            self.logger.error(
                f"清理Cookie健康检查任务列表失败: error_type={type(e).__name__}"
            )

    async def _cleanup_resources(self, queue_name: str):
        """清理资源"""
        try:
            await self.cleanup_processing_tasks()
            await self._cleanup_reconnect_tasks()
            await self._cleanup_heartbeat_tasks()
            await self._cleanup_health_tasks()
            await self.resource_manager.cleanup_all()

            try:
                await self.consumer_manager.stop_consumer(queue_name, remove=True)
                self.queue_manager.remove_queue(queue_name)
                self.logger.debug(f"已停止消息消费者: {queue_name}")
            except asyncio.InvalidStateError:
                self.logger.debug(f"消息消费者已在其他事件循环中停止: {queue_name}")
            except Exception as e:
                self.logger.warning(
                    f"停止消息消费者失败: {queue_name}, "
                    f"error_type={type(e).__name__}"
                )

            # CustomerAgent is transient per account; release its HTTP pool
            # and SQLite engine after the consumer has drained.
            account_agent = getattr(self, "_account_agent", None)
            if account_agent is not None:
                try:
                    await account_agent.close()
                except Exception as e:
                    self.logger.warning(
                        f"Agent resource cleanup failed: "
                        f"error_type={type(e).__name__}"
                    )
                finally:
                    self._account_agent = None

            self.ws = None
            self._account_key = None

        except Exception as e:
            self.logger.error(
                f"清理资源失败: error_type={type(e).__name__}"
            )


# 延迟导入避免循环依赖
from database import db_manager
from core.connection_status import ConnectionState

__all__ = ['LifecycleMixin']
