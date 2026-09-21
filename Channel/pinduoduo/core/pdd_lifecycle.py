# 生命周期管理模块
import asyncio
import time
import websockets
from urllib.parse import urlencode
from websockets import exceptions as ws_exceptions
from typing import Optional, Any
from utils.logger_loguru import get_logger
from Channel.pinduoduo.utils.API.get_token import GetToken
from config import config
from bridge.context import ChannelType, make_account_key, make_queue_name


class LifecycleMixin:
    """生命周期管理 Mixin"""

    async def start_account(self, shop_id: str, user_id: str, on_success: callable, on_failure: callable):
        """启动指定店铺下账号"""
        requested_account_key = make_account_key(
            ChannelType.PINDUODUO, shop_id, user_id
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
                # 必须用 pop：上面的 await 会让出事件循环，被取消的连接任务
                # 会在自己的 CancelledError 分支里调用 _cleanup_resources，
                # 而它会清空本字典。此时 del 会抛 KeyError，导致后面的状态
                # 更新与资源释放全部被跳过。
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
                self._heartbeat_tasks.pop(connection_key, None)
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
                self._health_tasks.pop(connection_key, None)
                self.logger.debug(f"已清理Cookie健康检查任务: {connection_key}")

            self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)

            if self.ws:
                await self._safe_close_websocket(self.ws)
                self.logger.info(f"已关闭店铺 {shop_id} 账号 {username} 的WebSocket连接")
            else:
                self.logger.warning(f"店铺 {shop_id} 账号 {username} 的WebSocket连接已经关闭或不存在")

            await self.cleanup_processing_tasks()

            queue_name = make_queue_name(ChannelType.PINDUODUO, shop_id, user_id)
            await self._cleanup_resources(queue_name)

            self.logger.info(f"成功停止店铺 {shop_id} 账号 {username}")

        except Exception as e:
            self.logger.error(
                f"停止店铺 {shop_id} 账号 {user_id} 时发生错误: "
                f"error_type={type(e).__name__}"
            )

    async def init(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable):
        """初始化WebSocket连接和消息处理系统"""
        try:
            self._stop_event = asyncio.Event()

            # GetToken 同步 HTTP（可能触发重登，耗时较长），放工作线程避免阻塞事件循环
            def _get_access_token():
                token = GetToken(shop_id, user_id)
                return token.get_token()

            access_token = await asyncio.to_thread(_get_access_token)

            queue_name = make_queue_name(ChannelType.PINDUODUO, shop_id, user_id)
            await self._setup_message_consumer(queue_name, shop_id, user_id)

            params = {
                "access_token": access_token,
                "role": "mall_cs",
                "client": "web",
                "version": self.API_VERSION
            }
            query = urlencode(params)
            full_url = f"{self.base_url}?{query}"

            self.logger.debug(f"正在连接到拼多多WebSocket: {shop_id}-{username}")

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
                    f"PDD WebSocket ({shop_id}-{username})"
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
                        make_queue_name(ChannelType.PINDUODUO, shop_id, user_id)
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
                make_queue_name(ChannelType.PINDUODUO, shop_id, user_id)
            )

    def request_stop(self):
        """请求停止WebSocket连接"""
        if self._stop_event:
            self._stop_event.set()

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
                self._heartbeat_tasks.pop(connection_key, None)

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
                self._health_tasks.pop(connection_key, None)

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
        """Cookie 健康检查循环，定期验证 cookie 有效性并主动刷新"""
        from Channel.pinduoduo.cookie_utils import check_cookies_valid, perform_relogin
        from Channel.pinduoduo.cookie_cache import cookie_cache

        connection_key = f"{shop_id}_{user_id}"

        try:
            while not (self._stop_event and self._stop_event.is_set()):
                await asyncio.sleep(self.heartbeat_config.cookie_health_check_interval)

                if self._stop_event and self._stop_event.is_set():
                    break

                # 从共享缓存或 DB 加载当前 cookie
                cookies = cookie_cache.get("pinduoduo", shop_id, user_id)
                if not cookies:
                    account_info = db_manager.get_account("pinduoduo", shop_id, user_id)
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

                # 轻量级 HTTP 验证
                is_valid = await asyncio.to_thread(
                    check_cookies_valid,
                    "pinduoduo", shop_id, user_id, cookies,
                    self.heartbeat_config.cookie_health_check_timeout,
                )

                if is_valid is False:
                    self.logger.warning(
                        f"Cookie 健康检查失败: {shop_id}-{username}，触发主动重登"
                    )
                    account_info = db_manager.get_account("pinduoduo", shop_id, user_id)
                    if account_info:
                        success = await asyncio.to_thread(
                            perform_relogin,
                            "pinduoduo", shop_id, user_id,
                            account_info.get('username', username),
                            account_info.get('password', ''),
                            True,
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
            # 本方法由连接任务自身在收尾时调用（见 init 的取消/异常分支），
            # 该任务也登记在本字典里。若把它一并取消，等于让清理者在自己的
            # await 点上收到 CancelledError，清理流程会中途异常退出。
            current = asyncio.current_task()
            for connection_key, task in list(self._reconnect_tasks.items()):
                if task is current or task.done():
                    continue
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
            current = asyncio.current_task()
            for connection_key, task in list(self._heartbeat_tasks.items()):
                if task is current or task.done():
                    continue
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
            current = asyncio.current_task()
            for connection_key, task in list(self._health_tasks.items()):
                if task is current or task.done():
                    continue
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
