# 后台线程模块
import asyncio
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPainterPath
from PyQt6.QtCore import Qt
from utils.logger_loguru import get_logger
from utils.safe_image_fetch import fetch_image


class LogoLoaderThread(QThread):
    """异步加载Logo的线程"""
    logo_loaded = pyqtSignal(QPixmap)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            image_data = fetch_image(self.url)

            pixmap = QPixmap()
            pixmap.loadFromData(image_data)

            if pixmap.isNull():
                raise ValueError("Loaded data is not a valid image.")

            # 创建圆形pixmap
            size = 60
            circular_pixmap = QPixmap(size, size)
            circular_pixmap.fill(Qt.GlobalColor.transparent)

            painter = QPainter(circular_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            path = QPainterPath()
            path.addEllipse(0, 0, size, size)

            painter.setClipPath(path)

            # 缩放并绘制原始图片
            scaled_pixmap = pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(0, 0, scaled_pixmap)
            painter.end()

            self.logo_loaded.emit(circular_pixmap)
        except Exception as e:
            get_logger().error(
                f"Failed to load logo: error_type={type(e).__name__}"
            )
            self.logo_loaded.emit(QPixmap())


class AutoReplyThread(QThread):
    """自动回复线程 - 每个账号独立的WebSocket连接线程"""

    connection_success = pyqtSignal()  # 连接成功信号
    connection_failed = pyqtSignal(str)  # 连接失败信号

    def __init__(self, account_data: dict):
        super().__init__()
        self.account_data = account_data
        self.channel = None
        self.loop = None
        self._shutdown_task = None
        self._stop_requested = False
        self.logger = get_logger("AutoReplyThread")

    def run(self):
        """按账号所属渠道启动后端引擎"""
        channel_name = str(
            self.account_data.get("channel_name") or "pinduoduo"
        ).strip().lower()
        if channel_name == "douyin":
            from Channel.douyin.douyin_channel import DouyinChannel
            channel_cls = DouyinChannel
        else:
            from Channel.pinduoduo.pdd_channel import PDDChannel
            channel_cls = PDDChannel

        try:
            # 为当前线程创建并设置新的事件循环
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # 每个账号一个独立渠道实例
            self.channel = channel_cls()

            # 定义成功和失败的回调函数
            def on_success():
                self.connection_success.emit()

            def on_failure(error_msg):
                self.connection_failed.emit(error_msg)

            # 启动引擎，并传递回调
            self._start_task = self.loop.create_task(
                self.channel.start_account(
                    shop_id=self.account_data['shop_id'],
                    user_id=self.account_data['user_id'],
                    on_success=on_success,
                    on_failure=on_failure
                )
            )

            # 保持事件循环运行，直到显式停止
            self.loop.run_forever()

        except Exception as e:
            self.logger.error(f"自动回复线程启动失败: {type(e).__name__}")
            self.connection_failed.emit("自动回复线程启动失败，请稍后重试")
        finally:
            loop = self.loop
            if loop is not None:
                try:
                    if loop.is_running():
                        loop.stop()

                    # 先在事件循环内取消并等待剩余任务，再关闭 loop，避免
                    # cleanup 协程在 loop 关闭后继续访问 asyncio 资源。
                    if not loop.is_closed():
                        pending = [
                            task for task in asyncio.all_tasks(loop)
                            if not task.done()
                        ]
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(
                                asyncio.gather(*pending, return_exceptions=True)
                            )
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.close()
                except Exception as cleanup_error:
                    self.logger.error(
                        "自动回复线程清理失败: "
                        f"error_type={type(cleanup_error).__name__}"
                    )
                finally:
                    try:
                        if asyncio.get_event_loop() is loop:
                            asyncio.set_event_loop(None)
                    except RuntimeError:
                        pass

    async def _shutdown_async(self):
        """在工作线程自己的事件循环中完成连接和任务清理。"""
        try:
            if self.channel:
                shop_id = self.account_data.get("shop_id")
                user_id = self.account_data.get("user_id")
                if shop_id is not None and user_id is not None:
                    await asyncio.wait_for(
                        self.channel.stop_account(shop_id, user_id),
                        timeout=4.0,
                    )
                else:
                    await asyncio.wait_for(
                        self.channel.stop_all_connections(),
                        timeout=4.0,
                    )
        except asyncio.TimeoutError:
            self.logger.warning("自动回复连接清理超时，将取消剩余任务")
        except Exception as e:
            self.logger.warning(
                "自动回复连接清理失败，将取消剩余任务: "
                f"error_type={type(e).__name__}"
            )
        finally:
            current_task = asyncio.current_task()
            pending = [
                task for task in asyncio.all_tasks()
                if task is not current_task and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            asyncio.get_running_loop().stop()

    def stop(self):
        """停止后端引擎"""
        try:
            loop = self.loop
            if loop is None or loop.is_closed() or not loop.is_running():
                return
            if self._stop_requested:
                return

            self._stop_requested = True

            # stop() 通常从 Qt 主线程调用；所有 asyncio 对象必须回到其
            # 所属事件循环线程中操作，不能直接跨线程 cancel/set。
            def schedule_shutdown():
                if loop.is_closed():
                    return
                if self._shutdown_task is None or self._shutdown_task.done():
                    self._shutdown_task = loop.create_task(self._shutdown_async())

            loop.call_soon_threadsafe(schedule_shutdown)

        except Exception as e:
            self.logger.error(
                f"停止自动回复线程失败: error_type={type(e).__name__}"
            )

    def is_running(self) -> bool:
        """检查线程是否在运行"""
        # 实际的运行状态由 PDDChannel 内部管理，这里仅表示线程是否已启动
        return self.isRunning()


class SetStatusThread(QThread):
    """设置账号状态的线程"""

    status_set_success = pyqtSignal(dict, int)  # 设置成功信号
    status_set_failed = pyqtSignal(dict, str)   # 设置失败信号

    def __init__(self, account_data: dict, target_status: int):
        super().__init__()
        self.account_data = account_data
        self.target_status = target_status
        self.logger = get_logger()

    def run(self):
        """在后台线程中执行状态更新"""
        from Channel.pinduoduo.utils.API.Set_up_online import AccountMonitor
        from service.account_service import account_service

        try:
            # 1. 调用API设置平台状态
            cookies = self.account_data.get("cookies")
            if not cookies:
                raise ValueError("账号缺少cookies，无法设置状态")

            # 获取账户信息用于自动重新登录
            shop_id = self.account_data.get("shop_id")
            user_id = self.account_data.get("user_id")
            channel_name = self.account_data.get("channel_name", "pinduoduo")

            account_monitor = AccountMonitor(cookies, shop_id=shop_id, user_id=user_id, channel_name=channel_name)

            api_success = account_monitor.set_csstatus(self.target_status)

            if not api_success:
                # API调用失败
                self.status_set_failed.emit(self.account_data, "平台状态设置失败")
                return

            # 2. 更新数据库状态
            db_success = account_service.update_account_status(
                channel_name=self.account_data["channel_name"],
                shop_id=self.account_data["shop_id"],
                user_id=self.account_data["user_id"],
                status=self.target_status
            )

            if db_success:
                # 发射成功信号
                self.status_set_success.emit(self.account_data, self.target_status)
            else:
                # 发射失败信号
                self.status_set_failed.emit(self.account_data, "数据库状态更新失败")

        except KeyError:
            # 如果缺少 'user_id' 等关键信息
            self.status_set_failed.emit(self.account_data, "账号数据不完整，无法设置状态")
        except Exception as e:
            # 其他异常
            self.status_set_failed.emit(self.account_data, str(e))


__all__ = ['LogoLoaderThread', 'AutoReplyThread', 'SetStatusThread']
