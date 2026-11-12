# 连接管理模块（抖音渠道，结构与拼多多一致）
import asyncio
import websockets
from websockets import exceptions as ws_exceptions
from typing import Optional, Any
from utils.logger_loguru import get_logger


class ConnectionMixin:
    """连接管理 Mixin

    base_url 为抓包实证的飞鸽 WebSocket 端点。实际收发由
    PigeonTransport.ws_url() 构造完整连接串（含 token、aid、device_id、
    access_key、pigeon_sign）；init 中沿用拼多多参数结构的旧分支暂不可用，
    详见 douyin_lifecycle.init 的 TODO(传输缝)。
    """

    # 飞鸽 WebSocket 端点（tools/reverse_probe/douyin/pigeon_transport.py 抓包实证）
    base_url: str = "wss://ws.fxg.jinritemai.com/ws/v2"

    async def _connect_with_retry(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable):
        """带重连机制的WebSocket连接"""
        logger = get_logger("DouyinChannel")

        for attempt in range(self.reconnect_config.max_attempts):
            if self._stop_event and self._stop_event.is_set():
                logger.info(f"收到停止信号，取消重连: {shop_id}-{username}")
                self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)
                return

            try:
                if attempt > 0:
                    self.status_manager.update_status(shop_id, user_id, username, ConnectionState.RECONNECTING)
                    logger.info(f"尝试重连 ({attempt + 1}/{self.reconnect_config.max_attempts}): {shop_id}-{username}")

                await self._connect_single_attempt(shop_id, user_id, username, on_success, on_failure)
                return

            except Exception as e:
                if self._stop_event and self._stop_event.is_set():
                    logger.info(f"连接被停止信号中断: {shop_id}-{username}")
                    self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)
                    return

                if attempt == self.reconnect_config.max_attempts - 1:
                    self.status_manager.update_status(
                        shop_id, user_id, username, ConnectionState.ERROR, type(e).__name__
                    )
                    logger.error(
                        f"连接失败，已达到最大重试次数: {shop_id}-{username}, 错误类型: {type(e).__name__}"
                    )
                    on_failure("连接失败，已达到最大重试次数，请检查网络和账号状态")
                    return

                delay = min(
                    self.reconnect_config.initial_delay * (self.reconnect_config.backoff_factor ** attempt),
                    self.reconnect_config.max_delay
                )

                logger.warning(
                    f"连接失败，{delay:.1f}秒后重试: {shop_id}-{username}, 错误类型: {type(e).__name__}"
                )

                try:
                    for _ in range(int(delay * 10)):
                        if self._stop_event and self._stop_event.is_set():
                            logger.info(f"重连延迟被停止信号中断: {shop_id}-{username}")
                            self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)
                            return
                        await asyncio.sleep(0.1)
                except (asyncio.CancelledError, RuntimeError):
                    logger.info(f"重连延迟被中断或事件循环关闭: {shop_id}-{username}")
                    self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)
                    return

    async def _connect_single_attempt(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable):
        """单次WebSocket连接尝试"""
        await self.init(shop_id, user_id, username, on_success, on_failure)

    def _is_ws_closed(self, ws: Any) -> bool:
        """检查WebSocket是否已关闭"""
        try:
            closed = getattr(ws, "closed", None)
            if isinstance(closed, bool):
                return closed
            return False
        except Exception:
            return False

    async def _safe_close_websocket(self, ws: Any):
        """安全关闭WebSocket"""
        try:
            close_fn = getattr(ws, "close", None)
            if close_fn:
                result = close_fn()
                if asyncio.iscoroutine(result):
                    await result
        except Exception as e:
            self.logger.debug(
                f"关闭WebSocket失败: error_type={type(e).__name__}"
            )


# 延迟导入避免循环依赖
from core.connection_status import ConnectionState
__all__ = ['ConnectionMixin']
