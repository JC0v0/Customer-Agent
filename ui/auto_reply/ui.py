# 自动回复主界面模块
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFrame, QWidget, QMessageBox
from PyQt6.QtGui import QFont
from qfluentwidgets import (SubtitleLabel, CaptionLabel, PushButton, PrimaryPushButton,
                            ScrollArea, FluentIcon as FIF)
from utils.logger_loguru import get_logger
from service.account_service import account_service
from .card import AutoReplyCard
from .manager import auto_reply_manager
from .threads import SetStatusThread


class AutoReplyUI(QFrame):
    """自动回复主界面"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.logger = get_logger()
        self.accounts_data = []
        self._loaded_once = False
        # 已提示过的重登失败（shop_id_user_id + last_error），避免每 10 秒重复弹窗
        self._notified_relogin_errors = set()
        self.setupUI()
        QTimer.singleShot(300, self._maybeLoadOnShow)

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.updateStats)
        self.stats_timer.start(5000)

        self.sync_timer = QTimer()
        self.sync_timer.timeout.connect(self._sync_auto_reply_status)
        self.sync_timer.start(10000)

    def closeEvent(self, event):
        """窗口关闭时清理定时器"""
        try:
            if hasattr(self, 'stats_timer'):
                self.stats_timer.stop()
            if hasattr(self, 'sync_timer'):
                self.sync_timer.stop()
            event.accept()
        except Exception as e:
            self.logger.error(f"清理定时器失败: error_type={type(e).__name__}")
            event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        self._maybeLoadOnShow()

    def _maybeLoadOnShow(self):
        if not self._loaded_once and self.isVisible():
            self._loaded_once = True
            self.loadAccountsFromDB()

    def setupUI(self):
        """设置主界面UI"""
        from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(25)

        header_widget = self.createHeaderWidget()
        content_widget = self.createContentWidget()

        self.refresh_btn.clicked.connect(self.reloadAccounts)
        self.start_all_btn.clicked.connect(self.onStartAllAutoReply)
        self.stop_all_btn.clicked.connect(self.stopAllAutoReply)

        main_layout.addWidget(header_widget)
        main_layout.addWidget(content_widget, 1)
        self.setObjectName("自动回复")

    def createHeaderWidget(self):
        """创建头部区域"""
        from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(20)

        title_label = SubtitleLabel("自动回复管理")
        self.stats_label = CaptionLabel("共 0 个账号")
        self.running_stats_label = CaptionLabel("运行中: 0 个")
        self.running_stats_label.setStyleSheet("color: #28a745; font-weight: bold;")

        title_area = QWidget()
        title_layout = QVBoxLayout(title_area)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)
        title_layout.addWidget(title_label)
        title_layout.addWidget(self.stats_label)
        title_layout.addWidget(self.running_stats_label)

        self.refresh_btn = PushButton("刷新")
        self.refresh_btn.setIcon(FIF.UPDATE)
        self.refresh_btn.setFixedSize(80, 40)

        self.start_all_btn = PrimaryPushButton("开始所有")
        self.start_all_btn.setIcon(FIF.PLAY_SOLID)
        self.start_all_btn.setFixedSize(120, 40)

        self.stop_all_btn = PushButton("停止所有")
        self.stop_all_btn.setIcon(FIF.CANCEL)
        self.stop_all_btn.setFixedSize(120, 40)

        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)
        buttons_layout.addWidget(self.refresh_btn)
        buttons_layout.addWidget(self.start_all_btn)
        buttons_layout.addWidget(self.stop_all_btn)

        header_layout.addWidget(title_area)
        header_layout.addStretch()
        header_layout.addWidget(buttons_widget)

        return header_widget

    def createContentWidget(self):
        """创建内容区域"""
        from PyQt6.QtWidgets import QVBoxLayout
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area = ScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            ScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        self.accounts_container = QWidget()
        self.accounts_layout = QVBoxLayout(self.accounts_container)
        self.accounts_layout.setSpacing(15)
        self.accounts_layout.setContentsMargins(20, 20, 20, 20)
        self.accounts_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.accounts_container.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
            }
        """)

        self.scroll_area.setWidget(self.accounts_container)
        content_layout.addWidget(self.scroll_area)

        return content_widget

    def loadAccountsFromDB(self):
        """从数据库加载账号数据"""
        try:
            self.accounts_data.clear()

            # 使用批量查询一次性获取所有账号数据，减少N+1查询
            all_accounts = account_service.get_all_accounts_with_details()
            self.accounts_data.extend(all_accounts)

            self.refreshAccountList()

        except Exception as e:
            self.logger.error(f"加载账号数据失败: error_type={type(e).__name__}")

    def refreshAccountList(self):
        """刷新账号列表"""
        self.clearAccountList()

        for account_data in self.accounts_data:
            account_card = AutoReplyCard(account_data)

            account_card.online_clicked.connect(self.onAccountOnline)
            account_card.offline_clicked.connect(self.onAccountOffline)
            account_card.auto_reply_clicked.connect(self.onAutoReplyToggle)

            is_running = auto_reply_manager.is_running(account_data)
            self.logger.debug(f"账号 {account_data['username']} 状态检查: running={is_running}")
            account_card.setAutoReplyStatus(is_running)

            self.accounts_layout.addWidget(account_card)

        self.accounts_layout.addStretch()
        self.updateStats()
        QTimer.singleShot(2000, self._sync_auto_reply_status)

    def clearAccountList(self):
        """清空账号列表"""
        while self.accounts_layout.count():
            child = self.accounts_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def updateStats(self):
        """更新统计信息"""
        count = len(self.accounts_data)
        running_count = auto_reply_manager.get_running_count()
        self.stats_label.setText(f"共 {count} 个账号")
        self.running_stats_label.setText(f"运行中: {running_count} 个")

    def _sync_auto_reply_status(self):
        """同步自动回复状态"""
        try:
            updated_count = 0

            for i in range(self.accounts_layout.count() - 1):
                widget = self.accounts_layout.itemAt(i).widget()
                if isinstance(widget, AutoReplyCard):
                    is_running = auto_reply_manager.is_running(widget.account_data)
                    current_status = widget.auto_reply_status

                    if is_running != current_status:
                        if current_status and not is_running:
                            account_key = "_".join(
                                str(widget.account_data.get(key, ""))
                                for key in ("channel_name", "shop_id", "user_id")
                            )
                            if account_key in auto_reply_manager.running_accounts:
                                thread = auto_reply_manager.running_accounts[account_key]
                                if hasattr(thread, 'isRunning') and thread.isRunning():
                                    continue

                        self.logger.info(f"同步状态: {widget.account_data['username']} 从 {current_status} 更新为 {is_running}")
                        widget.setAutoReplyStatus(is_running)
                        updated_count += 1

            if updated_count > 0:
                self.logger.info(f"状态同步完成，更新了 {updated_count} 个账号的状态")
                self.updateStats()

        except Exception as e:
            self.logger.error(f"同步自动回复状态失败: error_type={type(e).__name__}")

        self._check_relogin_failures()

    def _check_relogin_failures(self):
        """检查运行中账号的 Cookie 重登失败状态，提示用户手动重登。

        由 10 秒定时器调用；同一账号的同一错误只提示一次（去重）。
        """
        try:
            from core.di_container import container
            from core.connection_status import ConnectionStatusManager, ConnectionState

            status_manager = container.get(ConnectionStatusManager)
            pending_messages = []

            for status in status_manager.get_all_status():
                if status.state != ConnectionState.ERROR:
                    continue
                if not status.last_error or "重新登录失败" not in status.last_error:
                    continue

                account_key = "_".join([
                    "pinduoduo",
                    str(status.shop_id),
                    str(status.user_id),
                ])
                if account_key not in auto_reply_manager.running_accounts:
                    continue

                notify_key = f"{status.shop_id}_{status.user_id}:{status.last_error}"
                if notify_key in self._notified_relogin_errors:
                    continue
                self._notified_relogin_errors.add(notify_key)
                pending_messages.append(f"「{status.username}」：{status.last_error}")

            if pending_messages:
                detail = "\n".join(pending_messages)
                self.logger.error(f"检测到 Cookie 重登失败:\n{detail}")
                QMessageBox.warning(
                    self,
                    "Cookie 登录失效",
                    "以下账号的 Cookie 已过期，且自动重新登录失败：\n\n"
                    f"{detail}\n\n"
                    "请到「账号管理」页手动重新登录这些账号。",
                )

        except Exception as e:
            self.logger.error(f"检查重登失败状态失败: error_type={type(e).__name__}")

    def reloadAccounts(self):
        """重新加载账号"""
        self.loadAccountsFromDB()

    def onStartAllAutoReply(self):
        """开始所有符合条件的账号的自动回复"""
        try:
            eligible_accounts = [
                acc_data for acc_data in self.accounts_data
                if acc_data.get("status") == 1 and not auto_reply_manager.is_running(acc_data)
            ]

            if not eligible_accounts:
                QMessageBox.information(self, "提示", "没有符合条件的账号可以启动自动回复。\n\n(需要账号状态为'在线'且当前未在回复中)")
                return

            reply = QMessageBox.question(
                self,
                "确认开始",
                f"找到 {len(eligible_accounts)} 个可启动的账号。确定要全部开始自动回复吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return

            started_count = 0
            for account_data in eligible_accounts:
                success = auto_reply_manager.start_auto_reply(account_data)
                if success:
                    started_count += 1
                    self._connect_auto_reply_signals(account_data)

            self._update_all_cards_auto_reply_status()
            self.updateStats()

            QMessageBox.information(self, "操作完成", f"已成功为 {started_count} / {len(eligible_accounts)} 个账号启动自动回复。")

        except Exception as e:
            self.logger.error(f"开始所有自动回复失败: error_type={type(e).__name__}")
            QMessageBox.critical(self, "错误", "开始所有自动回复失败，请稍后重试")

    def stopAllAutoReply(self):
        """停止所有自动回复"""
        try:
            running_count = auto_reply_manager.get_running_count()

            if running_count == 0:
                QMessageBox.information(self, "提示", "当前没有正在运行的自动回复")
                return

            reply = QMessageBox.question(
                self,
                "确认停止",
                f"确定要停止所有 {running_count} 个正在运行的自动回复吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                auto_reply_manager.stop_all()
                self._update_all_cards_auto_reply_status()
                self.updateStats()
                QMessageBox.information(self, "成功", "已停止所有自动回复")

        except Exception as e:
            self.logger.error(f"停止所有自动回复失败: error_type={type(e).__name__}")
            QMessageBox.critical(self, "错误", "停止所有自动回复失败，请稍后重试")

    def _update_all_cards_auto_reply_status(self):
        """更新所有卡片的自动回复状态"""
        try:
            for i in range(self.accounts_layout.count() - 1):
                widget = self.accounts_layout.itemAt(i).widget()
                if isinstance(widget, AutoReplyCard):
                    is_running = auto_reply_manager.is_running(widget.account_data)
                    widget.setAutoReplyStatus(is_running)

        except Exception as e:
            self.logger.error(f"更新卡片状态失败: error_type={type(e).__name__}")

    def onAccountOnline(self, account_data: dict):
        """账号上线回调"""
        try:
            account_card = self.findAccountCard(account_data)
            if account_card:
                account_card.setButtonLoading("online", True)

            self.status_thread = SetStatusThread(account_data, 1)
            self.status_thread.status_set_success.connect(self.onStatusSetSuccess)
            self.status_thread.status_set_failed.connect(self.onStatusSetFailed)
            self.status_thread.start()

        except Exception as e:
            self.logger.error(f"启动上线操作失败: error_type={type(e).__name__}")
            QMessageBox.critical(self, "错误", "启动上线操作失败，请稍后重试")

    def onAccountOffline(self, account_data: dict):
        """账号离线回调"""
        try:
            account_card = self.findAccountCard(account_data)
            if account_card:
                account_card.setButtonLoading("offline", True)

            self.status_thread = SetStatusThread(account_data, 3)
            self.status_thread.status_set_success.connect(self.onStatusSetSuccess)
            self.status_thread.status_set_failed.connect(self.onStatusSetFailed)
            self.status_thread.start()

        except Exception as e:
            self.logger.error(f"启动离线操作失败: error_type={type(e).__name__}")
            QMessageBox.critical(self, "错误", "启动离线操作失败，请稍后重试")

    def findAccountCard(self, account_data: dict):
        """查找对应的账号卡片"""
        for i in range(self.accounts_layout.count() - 1):
            widget = self.accounts_layout.itemAt(i).widget()
            if isinstance(widget, AutoReplyCard) and widget.account_data == account_data:
                return widget
        return None

    def onStatusSetSuccess(self, account_data: dict, new_status: int):
        """状态设置成功回调"""
        try:
            account_card = self.findAccountCard(account_data)
            if account_card:
                account_card.setButtonLoading("online", False)
                account_card.setButtonLoading("offline", False)

            self.updateCardStatus(account_data, new_status)

            status_text = "在线" if new_status == 1 else "离线"
            self.logger.info(f"账号 '{account_data['username']}' 已成功设置为{status_text}状态")

        except Exception as e:
            self.logger.error(f"处理状态设置成功回调失败: error_type={type(e).__name__}")

    def onStatusSetFailed(self, account_data: dict, error_message: str):
        """状态设置失败回调"""
        try:
            account_card = self.findAccountCard(account_data)
            if account_card:
                account_card.setButtonLoading("online", False)
                account_card.setButtonLoading("offline", False)

            self.logger.error(f"设置账号 '{account_data['username']}' 状态失败：{error_message}")
            QMessageBox.warning(self, "失败", f"设置账号 '{account_data['username']}' 状态失败：{error_message}")

        except Exception as e:
            self.logger.error(f"处理状态设置失败回调失败: error_type={type(e).__name__}")

    def onAutoReplyToggle(self, account_data: dict):
        """自动回复开关回调"""
        try:
            account_card = self.findAccountCard(account_data)
            if not account_card:
                self.logger.error("找不到对应的账号卡片")
                return

            current_status = auto_reply_manager.is_running(account_data)

            if current_status:
                self._stop_auto_reply(account_data, account_card)
            else:
                self._start_auto_reply(account_data, account_card)

        except Exception as e:
            self.logger.error(f"自动回复开关操作失败: error_type={type(e).__name__}")
            QMessageBox.critical(self, "错误", "自动回复操作失败，请稍后重试")

    def _start_auto_reply(self, account_data: dict, account_card):
        """启动自动回复"""
        try:
            if account_data.get("status") != 1:
                QMessageBox.warning(self, "提示", "账号必须先上线才能开始自动回复！")
                return

            account_card.auto_reply_btn.setText("启动中...")
            account_card.auto_reply_btn.setEnabled(False)

            success = auto_reply_manager.start_auto_reply(account_data)

            if success:
                account_card.setAutoReplyStatus(True)
                self.logger.info(f"账号 '{account_data['username']}' 自动回复启动成功")
                self._connect_auto_reply_signals(account_data)
            else:
                account_card.auto_reply_btn.setText("开始回复")
                account_card.auto_reply_btn.setEnabled(True)
                QMessageBox.warning(self, "失败", f"启动账号 '{account_data['username']}' 自动回复失败！")

        except Exception as e:
            self.logger.error(f"启动自动回复失败: error_type={type(e).__name__}")
            account_card.auto_reply_btn.setText("开始回复")
            account_card.auto_reply_btn.setEnabled(True)
            QMessageBox.critical(self, "错误", "启动自动回复失败，请稍后重试")

    def _stop_auto_reply(self, account_data: dict, account_card):
        """停止自动回复"""
        try:
            account_card.auto_reply_btn.setText("停止中...")
            account_card.auto_reply_btn.setEnabled(False)

            success = auto_reply_manager.stop_auto_reply(account_data)
            account_card.setAutoReplyStatus(auto_reply_manager.is_running(account_data))

            if success:
                self.logger.info(f"账号 '{account_data['username']}' 自动回复停止成功")
            else:
                self.logger.warning(f"账号 '{account_data['username']}' 自动回复停止可能未完全成功")

            self.updateStats()

        except Exception as e:
            self.logger.error(f"停止自动回复失败: error_type={type(e).__name__}")
            account_card.setAutoReplyStatus(False)
            QMessageBox.critical(self, "错误", "停止自动回复失败，请稍后重试")
            self.updateStats()

    def _connect_auto_reply_signals(self, account_data: dict):
        """连接自动回复相关信号"""
        try:
            account_key = "_".join(
                str(account_data.get(key, ""))
                for key in ("channel_name", "shop_id", "user_id")
            )

            if account_key in auto_reply_manager.running_accounts:
                thread = auto_reply_manager.running_accounts[account_key]

                thread.connection_success.connect(
                    lambda: self._on_auto_reply_success(account_data)
                )
                thread.connection_failed.connect(
                    lambda error: self._on_auto_reply_failed(account_data, error)
                )

        except Exception as e:
            self.logger.error(f"连接自动回复信号失败: error_type={type(e).__name__}")

    def _on_auto_reply_success(self, account_data: dict):
        """自动回复连接成功回调"""
        try:
            account_card = self.findAccountCard(account_data)
            if account_card:
                account_card.auto_reply_btn.setText("停止回复")
                account_card.auto_reply_btn.setEnabled(True)

            self.logger.info(f"账号 '{account_data['username']}' 自动回复连接成功")
            self.updateStats()

        except Exception as e:
            self.logger.error(f"处理自动回复成功回调失败: error_type={type(e).__name__}")

    def _on_auto_reply_failed(self, account_data: dict, error: str):
        """自动回复连接失败回调"""
        try:
            account_card = self.findAccountCard(account_data)
            if account_card:
                account_card.setAutoReplyStatus(False)
                account_card.auto_reply_btn.setText("开始回复")
                account_card.auto_reply_btn.setEnabled(True)

            self.logger.error(f"账号 '{account_data['username']}' 自动回复连接失败: {error}")
            QMessageBox.warning(self, "连接失败", f"账号 '{account_data['username']}' 自动回复连接失败：{error}")
            self.updateStats()

        except Exception as e:
            self.logger.error(f"处理自动回复失败回调失败: error_type={type(e).__name__}")

    def updateCardStatus(self, account_data: dict, new_status: int):
        """更新卡片状态"""
        for i in range(self.accounts_layout.count() - 1):
            widget = self.accounts_layout.itemAt(i).widget()
            if isinstance(widget, AutoReplyCard) and widget.account_data == account_data:
                widget.updateStatus(new_status)
                break


__all__ = ['AutoReplyUI']
