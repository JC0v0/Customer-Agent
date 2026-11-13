# 账号管理界面

import asyncio
from typing import Any, Dict, Optional
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSignal as Signal, QTimer
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QVBoxLayout, QWidget, QLabel,
                            QMessageBox, QDialog, QFormLayout)
from PyQt6.QtGui import QFont, QPixmap
from qfluentwidgets import (CardWidget, SubtitleLabel, CaptionLabel, BodyLabel,
                           PrimaryPushButton, PushButton, StrongBodyLabel,
                           InfoBadge, ScrollArea, FluentIcon as FIF)
from service.account_service import account_service
from utils.logger_loguru import get_logger

logger = get_logger("UserUI")

class LogoLoaderThread(QThread):
    """异步加载Logo的线程"""
    logo_loaded = pyqtSignal(QPixmap)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        # load_logo 任何失败都会返回非空占位图，并在 DEBUG 记录真实原因。
        # 旧实现解码失败时抛 ValueError，日志只留下 error_type=ValueError、
        # 丢掉具体消息，界面上还会显示「加载失败」。
        from ui.logo_helper import load_logo

        self.logo_loaded.emit(load_logo(self.url))

class LoginThread(QThread):
    """登录验证线程"""
    login_finished = Signal(object)  # 登录完成信号，传递结果(字典或bool)
    
    def __init__(self, account_data):
        super().__init__()
        self.account_data = account_data
        
    def run(self):
        """在后台线程中执行登录"""
        try:
            # 创建事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 执行登录
            result = loop.run_until_complete(
                account_service.login(
                    name=self.account_data["username"],
                    password=self.account_data["password"],
                    channel_name=self.account_data.get("channel_name", "pinduoduo"),
                )
            )
            
            loop.close()
            self.login_finished.emit(result)
            
        except Exception as e:
            logger.error(f"登录线程异常: error_type={type(e).__name__}")
            self.login_finished.emit(False)


class AccountCard(CardWidget):
    """账号卡片组件"""
    
    # 定义信号
    edit_clicked = pyqtSignal(dict)  # 编辑按钮点击信号，传递账号信息
    delete_clicked = pyqtSignal(dict)  # 删除按钮点击信号，传递账号信息
    verify_clicked = pyqtSignal(dict)  # 验证按钮点击信号，传递账号信息
    
    def __init__(self, account_data: dict, parent=None):
        super().__init__(parent)
        self.account_data = account_data
        self.shop_id = account_data.get("shop_id", "")
        self.shop_name = account_data.get("shop_name", "")
        self.shop_logo = account_data.get("shop_logo")
        self.account_name = account_data.get("username", "")
        self.platform = account_data.get("channel_name", "")
        self.status = self.getStatusText(account_data.get("status", 0))
        self.setupUI()
        self.connectSignals()
        self.loadLogo()
    
    def getStatusText(self, status_code: int) -> str:
        """将状态码转换为文本"""
        status_map = {
            0: "休息",
            1: "在线", 
            3: "离线",
            None: "未验证"
        }
        return status_map.get(status_code, "未知")
        
    def setupUI(self):
        """设置卡片UI"""
        # 设置卡片样式
        self.setFixedHeight(120)
        # 主布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(20)
        
        # Logo
        logo_widget = self.createLogoWidget()
        
        # 左侧信息区域
        info_widget = self.createInfoWidget()
        
        # 右侧操作区域
        self.action_widget = self.createActionWidget()
        
        # 添加到主布局
        layout.addWidget(logo_widget)
        layout.addWidget(info_widget, 1)
        layout.addWidget(self.action_widget, 0)
    
    def createLogoWidget(self):
        """创建Logo显示区域"""
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(65, 65)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet("border-radius: 30px; border: 1px solid #e0e0e0; background-color: #f5f5f5;")
        self.logo_label.setText("加载中...")
        return self.logo_label
    
    def loadLogo(self):
        """异步加载Logo"""
        if self.shop_logo:
            def _start():
                self.logo_loader_thread = LogoLoaderThread(self.shop_logo)
                self.logo_loader_thread.logo_loaded.connect(self.setLogo)
                self.logo_loader_thread.start()
            QTimer.singleShot(200, _start)
        else:
            self.logo_label.setText("无Logo")

    def setLogo(self, pixmap: QPixmap):
        """设置Logo"""
        if not pixmap.isNull():
            self.logo_label.setPixmap(pixmap)
        else:
            self.logo_label.setText("加载失败")
        
    def createInfoWidget(self):
        """创建信息显示区域"""
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setSpacing(8)
        info_layout.setContentsMargins(0, 0, 0, 0)

        # 第一行：店铺名称 + 平台标签 + 平台logo
        first_row = QWidget()
        first_row_layout = QHBoxLayout(first_row)
        first_row_layout.setContentsMargins(0, 0, 0, 0)
        first_row_layout.setSpacing(10)
        
        # 店铺名称
        shop_name_label = StrongBodyLabel(self.shop_name)
        shop_name_label.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        shop_name_label.setStyleSheet("color: #2c3e50;")
        
        # 平台标签
        platform_badge = InfoBadge.info(self.platform, self)
        platform_badge.setFont(QFont("Microsoft YaHei", 9))

        first_row_layout.addWidget(shop_name_label)
        first_row_layout.addWidget(platform_badge)
        first_row_layout.addStretch()
        

        # 第二行：店铺ID
        second_row = self.createInfoRow("店铺ID:", self.shop_id)
        
        # 第三行：账号名称
        third_row = self.createInfoRow("账号:", self.account_name)
        
        # 添加到布局
        info_layout.addWidget(first_row)
        info_layout.addWidget(second_row)
        info_layout.addWidget(third_row)
        info_layout.addStretch()
        
        return info_widget
    
    def createInfoRow(self, label_text: str, value_text: str):
        """创建信息行"""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        
        # 标签
        label = CaptionLabel(label_text)
        label.setStyleSheet("color: #7f8c8d; font-weight: 500;")
        label.setFixedWidth(60)
        
        # 值
        value = BodyLabel(value_text)
        value.setStyleSheet("color: #34495e;")
        
        row_layout.addWidget(label)
        row_layout.addWidget(value)
        row_layout.addStretch()
        
        return row_widget
    
    def createActionWidget(self):
        """创建操作区域"""
        action_widget = QWidget()
        action_layout = QVBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(10)
        action_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # 状态标签
        status_badge = self.createStatusBadge()
        
        # 操作按钮容器
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)
        
        # 验证按钮
        self.verify_btn = PushButton("验证")
        self.verify_btn.setIcon(FIF.SYNC)
        self.verify_btn.setFixedSize(100, 32)
        
        # 编辑按钮
        self.edit_btn = PushButton("编辑")
        self.edit_btn.setIcon(FIF.EDIT)
        self.edit_btn.setFixedSize(100, 32)
       
        # 删除按钮
        self.delete_btn = PushButton("删除")
        self.delete_btn.setIcon(FIF.DELETE)
        self.delete_btn.setFixedSize(100, 32)

        buttons_layout.addWidget(self.verify_btn)
        buttons_layout.addWidget(self.edit_btn)
        buttons_layout.addWidget(self.delete_btn)
        
        # 添加到操作布局
        action_layout.addWidget(status_badge, 0, Qt.AlignmentFlag.AlignRight)
        action_layout.addWidget(buttons_widget)
        action_layout.addStretch()
        
        return action_widget
    
    def createStatusBadge(self):
        """创建状态标签"""
        if self.status == "在线":
            status_badge = InfoBadge.success("● 在线", self)
        elif self.status == "离线":
            status_badge = InfoBadge.error("● 离线", self)
        elif self.status == "未验证":
            status_badge = InfoBadge.warning("● 未验证", self)
        elif self.status == "休息":
            status_badge = InfoBadge.info("● 休息", self)
        else:
            status_badge = InfoBadge.info(f"● {self.status}", self)
        
        status_badge.setFont(QFont("Microsoft YaHei", 9, QFont.Weight.Bold))
        return status_badge
    
    def connectSignals(self):
        """连接信号"""
        self.verify_btn.clicked.connect(lambda: self.verify_clicked.emit(self.account_data))
        self.edit_btn.clicked.connect(lambda: self.edit_clicked.emit(self.account_data))
        self.delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self.account_data))
    
    def setVerifyStatus(self, is_verifying: bool):
        """设置验证状态"""
        if is_verifying:
            self.verify_btn.setText("验证中...")
            self.verify_btn.setEnabled(False)
        else:
            self.verify_btn.setText("验证")
            self.verify_btn.setEnabled(True)
    
    def updateStatus(self, new_status: int):
        """更新账号状态"""
        self.account_data["status"] = new_status
        self.status = self.getStatusText(new_status)
        
        # 重新创建状态标签
        old_badge = self.action_widget.layout().itemAt(0).widget()
        if old_badge:
            old_badge.deleteLater()
            
        new_badge = self.createStatusBadge()
        self.action_widget.layout().insertWidget(0, new_badge, 0, Qt.AlignmentFlag.AlignRight)


class UserManagerWidget(QFrame):
    """用户管理主界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.accounts_data = []  # 存储账号数据
        self._loaded_once = False
        self.setupUI()
        QTimer.singleShot(300, self._maybeLoadOnShow)
        
    def showEvent(self, event):
        super().showEvent(event)
        self._maybeLoadOnShow()
    
    def _maybeLoadOnShow(self):
        if not self._loaded_once and self.isVisible():
            self._loaded_once = True
            self.loadAccountsFromDB()
        
    def setupUI(self):
        """设置主界面UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(25)
        
        # 创建头部区域
        header_widget = self.createHeaderWidget()
        
        # 创建内容区域
        content_widget = self.createContentWidget()
        
        # 连接按钮信号
        self.add_btn.clicked.connect(self.onAddAccount)
        self.refresh_btn.clicked.connect(self.reloadAccounts)
        
        # 添加到主布局
        main_layout.addWidget(header_widget)
        main_layout.addWidget(content_widget, 1)
        
        # 设置对象名
        self.setObjectName("账号管理")
    
    def createHeaderWidget(self):
        """创建头部区域"""
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(20)
        
        # 标题
        title_label = SubtitleLabel("账号管理")
        # 统计信息
        self.stats_label = CaptionLabel("共 0 个账号")
        
        # 左侧标题区域
        title_area = QWidget()
        title_layout = QVBoxLayout(title_area)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)
        title_layout.addWidget(title_label)
        title_layout.addWidget(self.stats_label)
        
        # 添加账号按钮
        self.add_btn = PrimaryPushButton("添加账号")
        self.add_btn.setIcon(FIF.ADD)
        self.add_btn.setFixedSize(120, 40)
        
        # 刷新按钮
        self.refresh_btn = PushButton("刷新")
        self.refresh_btn.setIcon(FIF.UPDATE)
        self.refresh_btn.setFixedSize(80, 40)
        
        # 按钮容器
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)
        buttons_layout.addWidget(self.refresh_btn)
        buttons_layout.addWidget(self.add_btn)
        
        # 添加到头部布局
        header_layout.addWidget(title_area)
        header_layout.addStretch()
        header_layout.addWidget(buttons_widget)
        
        return header_widget
    
    def createContentWidget(self):
        """创建内容区域"""
        content_widget = QWidget()# 内容区域
        content_layout = QVBoxLayout(content_widget)# 内容区域布局
        content_layout.setContentsMargins(0, 0, 0, 0)# 设置内容区域边距
        content_layout.setSpacing(0)# 设置内容区域间距
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)# 设置内容区域顶部对齐

        # 滚动区域
        self.scroll_area = ScrollArea()# 滚动区域
        self.scroll_area.setWidgetResizable(True)# 设置滚动区域可调整大小
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)# 设置水平滚动条不可见
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)# 设置垂直滚动条根据需要显示
        # 去除边框
        self.scroll_area.setStyleSheet("""
            ScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # 账号列表容器
        self.accounts_container = QWidget()
        self.accounts_layout = QVBoxLayout(self.accounts_container)
        self.accounts_layout.setSpacing(15)  # 设置账号卡片之间的间距
        self.accounts_layout.setContentsMargins(20, 20, 20, 20)  # 设置容器内边距
        self.accounts_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # 顶部对齐
        
        # 设置容器样式，去除边框
        self.accounts_container.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
            }
        """)
        
        # 设置滚动区域内容
        self.scroll_area.setWidget(self.accounts_container)
        
        # 添加到内容布局
        content_layout.addWidget(self.scroll_area)
        
        return content_widget
    
    def loadAccountsFromDB(self):
        """从数据库加载账号数据"""
        try:
            self.accounts_data = []
            
            # 获取所有渠道
            channels = account_service.get_all_channels()
            
            for channel in channels:
                channel_name = channel["channel_name"]
                
                # 获取该渠道下的所有店铺
                shops = account_service.get_shops_by_channel(channel_name)
                
                for shop in shops:
                    shop_id = shop["shop_id"]
                    
                    # 获取该店铺下的所有账号
                    accounts = account_service.get_accounts_by_shop(channel_name, shop_id)
                    
                    for account in accounts:
                        account_data = {
                            "channel_name": channel_name,
                            "shop_id": shop_id,
                            "shop_name": shop["shop_name"],
                            "shop_logo": shop.get("shop_logo"),
                            "user_id": account["user_id"],
                            "username": account["username"],
                            "password": account["password"],
                            "status": account["status"],
                            "cookies": account.get("cookies")
                        }
                        self.accounts_data.append(account_data)
            
            self.refreshAccountList()
            
        except Exception as e:
            logger.error(f"加载账号数据失败: error_type={type(e).__name__}")

    

    def refreshAccountList(self):
        """刷新账号列表"""
        # 清空现有卡片
        self.clearAccountList()
        
        # 添加账号卡片
        for account_data in self.accounts_data:
            account_card = AccountCard(account_data)
            
            # 连接卡片信号
            account_card.verify_clicked.connect(self.onVerifyAccount)
            account_card.edit_clicked.connect(self.onEditAccount)
            account_card.delete_clicked.connect(self.onDeleteAccount)
            
            self.accounts_layout.addWidget(account_card)
        
        # 添加弹性空间
        self.accounts_layout.addStretch()
        
        # 更新统计信息
        self.updateStats()
    
    def clearAccountList(self):
        """清空账号列表"""
        while self.accounts_layout.count():
            child = self.accounts_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def updateStats(self):
        """更新统计信息"""
        count = len(self.accounts_data)
        self.stats_label.setText(f"共 {count} 个账号")

    def reloadAccounts(self):
        """重新加载账号"""
        self.loadAccountsFromDB()
    
    def onVerifyAccount(self, account_data: dict):
        """验证账号"""
        # 找到对应的账号卡片
        account_card = None
        for i in range(self.accounts_layout.count() - 1):  # -1 因为最后一个是stretch
            widget = self.accounts_layout.itemAt(i).widget()
            if isinstance(widget, AccountCard) and widget.account_data == account_data:
                account_card = widget
                break
        
        if not account_card:
            return
            
        # 设置验证状态
        account_card.setVerifyStatus(True)
        
        # 创建并启动登录线程
        self.login_thread = LoginThread(account_data)
        self.login_thread.login_finished.connect(
            lambda result: self.onLoginFinished(account_card, account_data, result)
        )
        self.login_thread.start()
    
    def onLoginFinished(self, account_card: AccountCard, account_data: dict, result: Optional[Dict[str, Any]]):
        """登录完成回调"""
        # 恢复验证按钮状态
        account_card.setVerifyStatus(False)
        
        try:
            if result:  # 登录成功，result是一个包含信息的字典
                # 首先更新cookies（重要！）
                cookies_updated = account_service.update_account_cookies(
                    channel_name=account_data["channel_name"],
                    shop_id=account_data["shop_id"],
                    user_id=account_data["user_id"],
                    cookies=result.get("cookies")
                )
                
                # 然后更新账号状态为在线
                status_updated = account_service.update_account_status(
                    account_data["channel_name"],
                    account_data["shop_id"],
                    account_data["user_id"],
                    1  # 在线状态
                )
                
                if cookies_updated and status_updated:
                    # 更新卡片状态显示
                    account_card.updateStatus(1)
                    QMessageBox.information(self, "验证成功", f"账号 '{account_data['username']}")
               
            else:  # 登录失败，result为False
                # 登录失败，更新数据库状态为离线
                account_service.update_account_status(
                    account_data["channel_name"],
                    account_data["shop_id"],
                    account_data["user_id"],
                    3  # 离线状态
                )
                
                # 更新卡片状态显示
                account_card.updateStatus(3)
                
                QMessageBox.warning(self, "验证失败", f"账号 '{account_data['username']}' 验证失败，请检查账号密码！")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新账号状态时发生错误：{str(e)}")
            
        # 重新加载数据以确保同步
        self.reloadAccounts()

    def onAddAccount(self):
        """通过登录添加账号"""
        dialog = AddAccountDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            account_info = dialog.getAccountInfo()
            
            # 禁用添加按钮，防止重复点击
            self.add_btn.setEnabled(False)
            self.add_btn.setText("添加中...")

            # 创建并启动登录线程
            self.add_account_thread = LoginThread(account_info)
            self.add_account_thread.login_finished.connect(self.onAddAccountLoginFinished)
            self.add_account_thread.start()

    def onAddAccountLoginFinished(self, result: Optional[Dict[str, Any]]):
        """处理添加账号时的登录结果"""
        # 恢复添加按钮状态
        self.add_btn.setEnabled(True)
        self.add_btn.setText("添加账号")

        if not result:
            QMessageBox.warning(self, "添加失败", "登录验证失败，请检查账号和密码后重试。")
            return

        try:
            channel_name = result["channel_name"]
            shop_id = result["shop_id"]
            shop_name = result["shop_name"]
            user_id = result["user_id"]
            username = result["username"]
            # 1. 检查账号是否已存在
            existing_account = account_service.get_account(channel_name, shop_id, user_id)
            if existing_account:
                QMessageBox.information(self, "提示", f"账号 '{username}' 已存在于店铺 '{shop_name}' 中，无需重复添加。")
                return

            # 2. 检查店铺是否存在，不存在则添加
            existing_shop = account_service.get_shop(channel_name, shop_id)
            if not existing_shop:
                account_service.add_shop(
                    channel_name=channel_name,
                    shop_id=shop_id,
                    shop_name=shop_name,
                    shop_logo=result.get("shop_logo"),
                    description=f"由登录自动添加"
                )
                logger.info(f"新店铺 '{shop_name}' 已自动添加。")

            # 3. 添加新账号
            # 注意：这里的add_account可能需要更多参数，我们传递所有已知信息
            success = account_service.add_account(
                channel_name=channel_name,
                shop_id=shop_id,
                username=username,
                password=result["password"],
                user_id=result.get("user_id"),
                cookies=result.get("cookies")
            )

            if success:
                QMessageBox.information(self, "成功", f"账号 '{username}' 已成功添加到店铺 '{shop_name}'！")
                self.reloadAccounts()
            else:
                QMessageBox.warning(self, "失败", "账号添加失败，数据写入时发生错误。")

        except Exception as e:
            QMessageBox.critical(self, "严重错误", f"添加账号过程中发生严重错误：{str(e)}")
            logger.error(
                f"添加账号过程中发生错误: error_type={type(e).__name__}"
            )
    
    def onEditAccount(self, account_data: dict):
        """编辑账号回调"""
        dialog = EditAccountDialog(account_data, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_data = dialog.getAccountData()
            try:
                # 检查密码或状态是否变化
                username_changed = new_data["username"] != account_data["username"]
                password_changed = new_data["password"] != account_data["password"]
                status_changed = new_data["status"] != account_data["status"]

                if not (username_changed or password_changed or status_changed):
                    QMessageBox.information(self, "提示", "账号信息未发生变化。")
                    return

                update_info_success = True
                update_status_success = True

                # 如果用户名或密码发生变化，则统一更新
                if username_changed or password_changed:
                    update_info_success = account_service.update_account_info(
                        channel_name=new_data["channel_name"],
                        shop_id=new_data["shop_id"],
                        user_id=new_data["user_id"],
                        username=new_data["username"],
                        password=new_data["password"],
                        cookies=account_data.get("cookies")
                    )
                
                # 如果状态变化，仅更新状态
                if status_changed:
                    update_status_success = account_service.update_account_status(
                        channel_name=new_data["channel_name"],
                        shop_id=new_data["shop_id"],
                        user_id=new_data["user_id"],
                        status=new_data["status"]
                    )
                
                if update_info_success and update_status_success:
                    QMessageBox.information(self, "成功", "账号信息更新成功！")
                    self.reloadAccounts()
                else:
                    error_parts = []
                    if not update_info_success:
                        error_parts.append("基本信息(用户名/密码)")
                    if not update_status_success:
                        error_parts.append("状态")
                    error_message = f"更新以下信息失败: {', '.join(error_parts)}。\n数据库未找到对应条目或发生错误，请检查日志。"
                    QMessageBox.warning(self, "更新失败", error_message)
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"更新账号信息时发生错误：{str(e)}")
    

    
    def onDeleteAccount(self, account_data: dict):
        """删除账号回调"""
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除账号 '{account_data['username']}' 吗？\n"
            f"店铺：{account_data['shop_name']}\n"
            "此操作不可撤销！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                success = account_service.delete_account(
                    account_data["channel_name"],
                    account_data["shop_id"],
                    account_data["user_id"]
                )
                
                if success:
                    QMessageBox.information(self, "成功", "账号删除成功！")
                    self.reloadAccounts()
                else:
                    QMessageBox.warning(self, "失败", "账号删除失败！")
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除账号时发生错误：{str(e)}")


class EditAccountDialog(QDialog):
    """编辑账号对话框"""
    
    def __init__(self, account_data: dict, parent=None):
        super().__init__(parent)
        self.account_data = account_data
        self.setWindowTitle("编辑账号")
        self.setModal(True)
        self.resize(400, 350)
        self.setupUI()
    
    def setupUI(self):
        """设置对话框UI"""
        from qfluentwidgets import LineEdit, ComboBox
        
        layout = QFormLayout(self)
        
        # 显示不可编辑的信息
        
        # 渠道信息（只读）
        self.channel_label = QLabel(self.account_data["channel_name"])
        self.channel_label.setStyleSheet("color: #666; font-weight: bold;")
        layout.addRow("渠道:", self.channel_label)
        
        # 店铺信息（只读）
        shop_info = f"{self.account_data['shop_name']} ({self.account_data['shop_id']})"
        self.shop_label = QLabel(shop_info)
        self.shop_label.setStyleSheet("color: #666; font-weight: bold;")
        layout.addRow("店铺:", self.shop_label)
        
        # 用户名（可编辑）
        self.username_edit = LineEdit()
        self.username_edit.setText(self.account_data["username"])
        self.username_edit.setPlaceholderText("请输入用户名")
        layout.addRow("用户名:", self.username_edit)
        
        # 密码（可编辑）
        self.password_edit = LineEdit()
        self.password_edit.setText(self.account_data["password"])
        self.password_edit.setEchoMode(LineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("输入新密码")
        
        # 显示密码按钮
        self.show_password_btn = PushButton("显示")
        self.show_password_btn.setFixedSize(60, 32)
        self.show_password_btn.clicked.connect(self.togglePasswordVisibility)
        
        # 密码行容器
        password_container = QWidget()
        password_layout = QHBoxLayout(password_container)
        password_layout.setContentsMargins(0, 0, 0, 0)
        password_layout.addWidget(self.password_edit)
        password_layout.addWidget(self.show_password_btn)
        
        layout.addRow("密码:", password_container)
        
        # 状态选择
        self.status_combo = ComboBox()
        status_options = [
            ("休息", 0),
            ("在线", 1), 
            ("离线", 3),
            ("未验证", None)
        ]
        
        current_status_index = 0
        for i, (text, value) in enumerate(status_options):
            self.status_combo.addItem(text)
            self.status_combo.setItemData(i, value)
            if value == self.account_data["status"]:
                current_status_index = i
        
        self.status_combo.setCurrentIndex(current_status_index)
        layout.addRow("状态:", self.status_combo)
        
        # 按钮
        from qfluentwidgets import PrimaryPushButton
        buttons_layout = QHBoxLayout()
        self.ok_btn = PrimaryPushButton("确定")
        self.cancel_btn = PushButton("取消")
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.cancel_btn)
        buttons_layout.addWidget(self.ok_btn)
        
        layout.addRow(buttons_layout)
        
        # 连接信号
        self.ok_btn.clicked.connect(self.validateAndAccept)
        self.cancel_btn.clicked.connect(self.reject)
    
    def togglePasswordVisibility(self):
        """切换密码显示/隐藏"""
        from qfluentwidgets import LineEdit
        if self.password_edit.echoMode() == LineEdit.EchoMode.Password:
            self.password_edit.setEchoMode(LineEdit.EchoMode.Normal)
            self.show_password_btn.setText("隐藏")
        else:
            self.password_edit.setEchoMode(LineEdit.EchoMode.Password)
            self.show_password_btn.setText("显示")
    
    def validateAndAccept(self):
        """验证输入并接受"""
        if not self.username_edit.text().strip():
            QMessageBox.warning(self, "输入错误", "用户名不能为空！")
            return
            
        if not self.password_edit.text().strip():
            QMessageBox.warning(self, "输入错误", "密码不能为空！")
            return
            
        self.accept()
    
    def getAccountData(self) -> dict:
        """获取编辑后的账号数据"""
        return {
            "channel_name": self.account_data["channel_name"],
            "shop_id": self.account_data["shop_id"],
            "shop_name": self.account_data["shop_name"],
            "user_id": self.account_data["user_id"],  # 传递user_id
            "username": self.username_edit.text().strip(),
            "password": self.password_edit.text().strip(),
            "status": self.status_combo.currentData()
        }


class AddAccountDialog(QDialog):
    """添加账号对话框（通过登录）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("通过登录添加账号")
        self.setModal(True)
        self.resize(400, 240)
        self.setupUI()
    
    def setupUI(self):
        """设置对话框UI"""
        from qfluentwidgets import ComboBox, LineEdit, PrimaryPushButton, PushButton

        layout = QFormLayout(self)
        layout.setSpacing(15)

        self.channel_combo = ComboBox(self)
        self.channel_combo.addItem("拼多多")
        self.channel_combo.setItemData(0, "pinduoduo")
        self.channel_combo.addItem("抖音")
        self.channel_combo.setItemData(1, "douyin")
        self.channel_combo.setFixedSize(300, 32)
        
        # 账号信息
        self.username_edit = LineEdit(self)
        self.username_edit.setPlaceholderText("请输入账号用户名")
        self.username_edit.setFixedSize(300, 32)
        self.password_edit = LineEdit(self)
        self.password_edit.setEchoMode(LineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("请输入密码")
        self.password_edit.setFixedSize(300, 32)
        layout.addRow("渠道:", self.channel_combo)
        layout.addRow("用户名:", self.username_edit)
        layout.addRow("密码:", self.password_edit)
        
        # 按钮
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 15, 0, 0)
        buttons_layout.addStretch()
        
        self.ok_btn = PrimaryPushButton("确定")
        self.cancel_btn = PushButton("取消")
        
        buttons_layout.addWidget(self.cancel_btn)
        buttons_layout.addWidget(self.ok_btn)
        
        layout.addRow(buttons_widget)
        
        # 连接信号
        self.ok_btn.clicked.connect(self.validateAndAccept)
        self.cancel_btn.clicked.connect(self.reject)
    
    def validateAndAccept(self):
        """验证输入并接受"""
        if not self.username_edit.text().strip():
            QMessageBox.warning(self, "输入错误", "用户名不能为空！")
            return
            
        if not self.password_edit.text().strip():
            QMessageBox.warning(self, "输入错误", "密码不能为空！")
            return
            
        self.accept()
        
    def getAccountInfo(self) -> dict:
        """获取账号信息"""
        return {
            "channel_name": self.channel_combo.currentData() or "pinduoduo",
            "username": self.username_edit.text().strip(),
            "password": self.password_edit.text().strip()
        }
