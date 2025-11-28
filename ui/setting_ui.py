from PyQt6.QtCore import Qt, QTime
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QMessageBox
)

from qfluentwidgets import (
    CardWidget, SubtitleLabel, CaptionLabel, StrongBodyLabel,
    PrimaryPushButton, PushButton, LineEdit, PasswordLineEdit,
    TimePicker, ScrollArea, InfoBar, InfoBarPosition,
    FluentIcon as FIF, ComboBox
)

from utils.logger import get_logger
from config import config


# -----------------------------------------------------------------------------
# Coze 配置卡片
# -----------------------------------------------------------------------------
class CozeConfigCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._buildUI()

    def _buildUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title = StrongBodyLabel("Coze AI 配置")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(12)

        self.api_base = LineEdit()
        self.api_base.setPlaceholderText("https://api.coze.cn")

        self.token = PasswordLineEdit()
        self.token.setPlaceholderText("输入您的 Coze API Token")

        self.bot_id = LineEdit()
        self.bot_id.setPlaceholderText("输入您的 Bot ID")

        form.addRow("API Base URL:", self.api_base)
        form.addRow("API Token:", self.token)
        form.addRow("Bot ID:", self.bot_id)

        layout.addLayout(form)

        description = CaptionLabel(
            "请在 Coze 平台获取 API Token 与 Bot ID。\n"
            "API Token 用于身份认证；Bot ID 用于选择机器人。"
        )
        description.setStyleSheet("color: #666;")
        layout.addWidget(description)

    def getConfig(self):
        return {
            "coze_api_base": self.api_base.text().strip() or "https://api.coze.cn",
            "coze_token": self.token.text().strip(),
            "coze_bot_id": self.bot_id.text().strip()
        }

    def setConfig(self, cfg):
        self.api_base.setText(cfg.get("coze_api_base", "https://api.coze.cn"))
        self.token.setText(cfg.get("coze_token", ""))
        self.bot_id.setText(cfg.get("coze_bot_id", ""))


# -----------------------------------------------------------------------------
# OpenAI 配置卡片
# -----------------------------------------------------------------------------
class OpenAIConfigCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._buildUI()

    def _buildUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title = StrongBodyLabel("OpenAI 配置")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(12)

        self.api_base = LineEdit()
        self.api_base.setPlaceholderText("https://api.openai.com/v1")

        self.api_key = PasswordLineEdit()
        self.api_key.setPlaceholderText("输入您的 OpenAI API Key")

        self.model = LineEdit()
        self.model.setPlaceholderText("gpt-3.5-turbo")

        form.addRow("API Base URL:", self.api_base)
        form.addRow("API Key:", self.api_key)
        form.addRow("Model:", self.model)

        layout.addLayout(form)

        description = CaptionLabel(
            "可以配置 OpenAI、DeepSeek、Moonshot 等兼容接口。\n"
            "Model 字段填写您的模型名称，如 gpt-4o-mini。"
        )
        description.setStyleSheet("color: #666;")
        layout.addWidget(description)

    def getConfig(self):
        return {
            "openai_api_base": self.api_base.text().strip() or "https://api.openai.com/v1",
            "openai_api_key": self.api_key.text().strip(),
            "openai_model": self.model.text().strip() or "gpt-3.5-turbo"
        }

    def setConfig(self, cfg):
        self.api_base.setText(cfg.get("openai_api_base", "https://api.openai.com/v1"))
        self.api_key.setText(cfg.get("openai_api_key", ""))
        self.model.setText(cfg.get("openai_model", "gpt-3.5-turbo"))


# -----------------------------------------------------------------------------
# 业务时间配置卡片
# -----------------------------------------------------------------------------
class BusinessHoursCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._buildUI()

    def _buildUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title = StrongBodyLabel("业务时间设置")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(12)

        self.start_time = TimePicker()
        self.start_time.setTime(QTime(8, 0))

        self.end_time = TimePicker()
        self.end_time.setTime(QTime(23, 0))

        form.addRow("开始时间:", self.start_time)
        form.addRow("结束时间:", self.end_time)

        layout.addLayout(form)

        description = CaptionLabel(
            "设置 AI 客服的自动回复工作时段。\n"
            "非工作时间系统将不自动响应消息。"
        )
        description.setStyleSheet("color: #666;")
        layout.addWidget(description)

    def getConfig(self):
        return {
            "businessHours": {
                "start": self.start_time.getTime().toString("HH:mm"),
                "end": self.end_time.getTime().toString("HH:mm")
            }
        }

    def setConfig(self, cfg):
        hours = cfg.get("businessHours", {})
        start = QTime.fromString(hours.get("start", "08:00"), "HH:mm")
        end = QTime.fromString(hours.get("end", "23:00"), "HH:mm")

        if start.isValid():
            self.start_time.setTime(start)
        if end.isValid():
            self.end_time.setTime(end)


# -----------------------------------------------------------------------------
# 主界面 SettingUI
# -----------------------------------------------------------------------------
class SettingUI(QFrame):
    """设置界面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger("SettingUI")

        self._buildUI()
        self.loadConfig()

        self.setObjectName("设置")

    # -------------------------------------------------------------------------
    def _buildUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)

        # Header
        header = self._buildHeader()

        # Scroll Content
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(20)

        # 三个卡片
        # Bot Type Selector
        type_card = CardWidget()
        type_layout = QHBoxLayout(type_card)
        type_layout.setContentsMargins(20, 16, 20, 16)
        
        type_label = StrongBodyLabel("AI 模型选择")
        type_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        
        self.bot_type_combo = ComboBox()
        self.bot_type_combo.addItems(["Coze", "OpenAI"])
        self.bot_type_combo.setFixedWidth(150)
        self.bot_type_combo.currentTextChanged.connect(self.onBotTypeChanged)
        
        type_layout.addWidget(type_label)
        type_layout.addStretch()
        type_layout.addWidget(self.bot_type_combo)

        self.coze_card = CozeConfigCard()
        self.openai_card = OpenAIConfigCard()
        self.hours_card = BusinessHoursCard()

        content_layout.addWidget(type_card)
        content_layout.addWidget(self.coze_card)
        content_layout.addWidget(self.openai_card)
        content_layout.addWidget(self.hours_card)
        content_layout.addStretch()

        scroll.setWidget(content)

        layout.addWidget(header)
        layout.addWidget(scroll, 1)

    def onBotTypeChanged(self, text):
        if text == "Coze":
            self.coze_card.show()
            self.openai_card.hide()
        else:
            self.coze_card.hide()
            self.openai_card.show()

    # -------------------------------------------------------------------------
    def _buildHeader(self):
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        title = SubtitleLabel("系统设置")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        desc = CaptionLabel("配置 AI 客服的基本参数与工作时间")
        desc.setStyleSheet("color: #666;")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_layout.addWidget(title)
        left_layout.addWidget(desc)

        # Buttons
        self.reset_btn = PushButton("重置")
        self.reset_btn.setIcon(FIF.UPDATE)
        self.reset_btn.setFixedSize(90, 40)

        self.save_btn = PrimaryPushButton("保存")
        self.save_btn.setIcon(FIF.SAVE)
        self.save_btn.setFixedSize(90, 40)

        btn_box = QWidget()
        btn_layout = QHBoxLayout(btn_box)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(10)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.save_btn)

        # 信号
        self.save_btn.clicked.connect(self.onSaveConfig)
        self.reset_btn.clicked.connect(self.onResetConfig)

        layout.addWidget(left)
        layout.addStretch()
        layout.addWidget(btn_box)

        return header

    # -------------------------------------------------------------------------
    def loadConfig(self):
        """载入配置到 UI"""
        try:
            self.coze_card.setConfig(config.data)
            self.openai_card.setConfig(config.data)
            self.hours_card.setConfig(config.data)
            
            bot_type = config.get("bot_type", "coze")
            self.bot_type_combo.setCurrentText("Coze" if bot_type == "coze" else "OpenAI")
            self.onBotTypeChanged(self.bot_type_combo.currentText())

            self.logger.info("配置加载成功")
        except Exception as e:
            self.logger.error(f"加载失败: {e}")

    # -------------------------------------------------------------------------
    def onSaveConfig(self):
        """保存配置"""
        try:
            bot_type = "coze" if self.bot_type_combo.currentText() == "Coze" else "openai"
            new_cfg = {"bot_type": bot_type}
            new_cfg.update(self.coze_card.getConfig())
            new_cfg.update(self.openai_card.getConfig())
            new_cfg.update(self.hours_card.getConfig())

            config.update(new_cfg)
            config.save()

            InfoBar.success(
                title="保存成功",
                content="配置已成功保存！",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=1800,
                parent=self
            )
            self.logger.info("配置已保存")

        except Exception as e:
            self.logger.error(f"保存失败: {e}")
            QMessageBox.critical(self, "保存失败", str(e))

    # -------------------------------------------------------------------------
    def onResetConfig(self):
        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要重置所有配置吗？\n这将重新加载配置文件中的内容。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                config.reload()
                self.loadConfig()

                InfoBar.success(
                    title="重置成功",
                    content="配置已重置为配置文件内容！",
                    orient=Qt.Orientation.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=1800,
                    parent=self
                )
                self.logger.info("配置已重置")

            except Exception as e:
                self.logger.error(f"重置失败: {e}")
                QMessageBox.critical(self, "重置失败", str(e))
