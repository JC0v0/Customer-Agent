"""
知识库管理UI模块
==============

提供产品知识和客服知识管理界面，包含：
- 顶部店铺选择器
- 两个标签页：产品知识 / 客服知识
- 自动同步产品知识（拼多多API + LLM提取）
- 客服知识人工添加/编辑/删除
"""
from __future__ import annotations
import asyncio
import os
from typing import TYPE_CHECKING, Optional, List, Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QStackedWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QPushButton, QMessageBox, QDialog, QDialogButtonBox, QInputDialog,
    QLineEdit, QTextEdit, QCheckBox, QProgressBar, QFrame, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from qfluentwidgets import (
    PrimaryPushButton, PushButton,
    InfoBar, InfoBarPosition, TableWidget, SegmentedWidget,
    ComboBox,
)

from core.di_container import container
from database.knowledge_service import KnowledgeService
from database.product_sync import ProductSyncService, SyncProgress
from database.models import ProductKnowledge, CustomerServiceKnowledge, Shop
from utils.logger_loguru import get_logger

if TYPE_CHECKING:
    from database.knowledge_service import KnowledgeService

logger = get_logger("KnowledgeUI")


class SyncWorker(QThread):
    """同步工作线程"""
    progress_updated = pyqtSignal(int, int, int, int, int, str, str)  # current, total, success, failed, degraded, name, phase
    sync_finished = pyqtSignal(int, int, int, bool)  # success, failed, degraded, cancelled

    def __init__(
        self,
        shop_db_id: int,
        pdd_shop_id: str,
        user_id: str,
        is_full_sync: bool,
        product_sync: ProductSyncService,
        parent=None,
    ):
        super().__init__(parent)
        self.shop_db_id = shop_db_id
        self.pdd_shop_id = pdd_shop_id
        self.user_id = user_id
        self.is_full_sync = is_full_sync
        self.product_sync = product_sync
        self._cancelled = False

    def run(self):
        """运行同步"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def progress_callback(progress: SyncProgress):
            self.progress_updated.emit(
                progress.current,
                progress.total,
                progress.success,
                progress.failed,
                progress.degraded,
                progress.current_goods_name,
                progress.phase,
            )

        result = loop.run_until_complete(
            self.product_sync.sync_shop(
                shop_id=int(self.pdd_shop_id),
                shop_db_id=self.shop_db_id,
                user_id=self.user_id,
                is_full_sync=self.is_full_sync,
                progress_callback=progress_callback,
            )
        )

        loop.close()
        self.sync_finished.emit(result.success, result.failed, result.degraded, result.cancelled)


class ProductDetailDialog(QDialog):
    """产品知识详情对话框，支持编辑"""

    def __init__(self, product: ProductKnowledge, parent=None):
        super().__init__(parent)
        self.product = product
        self.setWindowTitle("产品知识详情")
        self.resize(700, 500)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 商品名称
        self.name_label = QLabel("商品名称:")
        self.name_edit = QLineEdit(self.product.goods_name)
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_edit)

        # 提取内容
        self.content_label = QLabel("提取知识:")
        self.content_edit = QTextEdit()
        self.content_edit.setPlainText(self.product.extracted_content or "")
        self.content_edit.setPlaceholderText("LLM提取的产品知识会显示在这里，你可以手动编辑")
        layout.addWidget(self.content_label)
        layout.addWidget(self.content_edit)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        """获取编辑后的数据"""
        return {
            "goods_name": self.name_edit.text().strip(),
            "extracted_content": self.content_edit.toPlainText().strip(),
        }


class CsAddEditDialog(QDialog):
    """客服知识添加/编辑对话框"""

    def __init__(
        self,
        shop_id: int,
        existing: Optional[CustomerServiceKnowledge] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.shop_id = shop_id
        self.existing = existing
        self.default_tags = ["物流", "售后", "支付", "商品规格", "优惠券", "会员", "发货时间", "退换货"]
        self.setWindowTitle("添加客服知识" if not existing else "编辑客服知识")
        self.resize(650, 500)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        self.title_label = QLabel("标题:")
        self.title_edit = QLineEdit()
        if self.existing:
            self.title_edit.setText(self.existing.title)
        layout.addWidget(self.title_label)
        layout.addWidget(self.title_edit)

        # 内容
        self.content_label = QLabel("内容:")
        self.content_edit = QTextEdit()
        self.content_edit.setPlaceholderText("输入客服知识内容，例如：退换货政策说明...")
        if self.existing:
            self.content_edit.setText(self.existing.content)
        layout.addWidget(self.content_label)
        layout.addWidget(self.content_edit)

        # 标签 - 预设复选框
        self.tags_label = QLabel("选择标签 (可多选):")
        layout.addWidget(self.tags_label)

        self.tag_checkboxes: List[QCheckBox] = []
        existing_tags = []
        if self.existing and self.existing.tags:
            existing_tags = [t.strip() for t in self.existing.tags.split(',') if t.strip()]

        tags_frame = QFrame()
        tags_layout = QHBoxLayout(tags_frame)
        tags_layout.setSpacing(8)

        for tag in self.default_tags:
            cb = QCheckBox(tag)
            if tag in existing_tags:
                cb.setChecked(True)
            tags_layout.addWidget(cb)
            self.tag_checkboxes.append(cb)

        layout.addWidget(tags_frame)

        # 自定义标签
        self.custom_label = QLabel("自定义标签 (逗号分隔):")
        self.custom_edit = QLineEdit()
        if self.existing and self.existing.tags:
            # 已有标签中不在预设列表的合并到自定义
            existing_custom = [
                t for t in existing_tags
                if t not in self.default_tags
            ]
            if existing_custom:
                self.custom_edit.setText(','.join(existing_custom))
        layout.addWidget(self.custom_label)
        layout.addWidget(self.custom_edit)

        # 启用
        self.enabled_cb = QCheckBox("启用")
        self.enabled_cb.setChecked(True)
        if self.existing:
            self.enabled_cb.setChecked(self.existing.enabled)
        layout.addWidget(self.enabled_cb)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        """获取数据"""
        title = self.title_edit.text().strip()
        content = self.content_edit.toPlainText().strip()
        enabled = self.enabled_cb.isChecked()

        # 收集选中的预设标签
        selected_tags = [
            cb.text() for cb in self.tag_checkboxes
                if cb.isChecked()
        ]

        # 添加自定义标签
        custom = self.custom_edit.text().strip()
        if custom:
            selected_tags.extend([t.strip() for t in custom.split(',') if t.strip()])

        # 去重
        selected_tags = list(dict.fromkeys(selected_tags))
        tags_str = ','.join(selected_tags) if selected_tags else None

        return {
            "title": title,
            "content": content,
            "tags": tags_str,
            "enabled": enabled,
        }


class KnowledgeUI(QWidget):
    """知识库管理主界面"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName('KnowledgeUI')
        self.resize(900, 700)

        # 从DI容器获取服务
        self.knowledge_service: KnowledgeService = container.get(KnowledgeService)
        self.product_sync = ProductSyncService(self.knowledge_service)

        # 当前选中的店铺
        self.current_shop_id: Optional[int] = None
        # 店铺缓存 {shop_id: shop_name}
        self._shop_cache: Dict[int, str] = {}

        # 懒加载标志：只在首次切换到对应标签页时加载数据
        self._product_loaded = False
        self._cs_loaded = False
        # 标签缓存，避免重复重建下拉框
        self._last_cs_tags: tuple = ()

        self._init_ui()
        self._load_shops()

    def _init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 顶部店铺选择栏
        shop_bar = QHBoxLayout()
        shop_bar.setSpacing(8)

        shop_label = QLabel("当前店铺:")
        shop_label.setFixedWidth(60)
        self.shop_combo = ComboBox()
        self.shop_combo.currentIndexChanged.connect(self._on_shop_changed)

        shop_bar.addWidget(shop_label)
        shop_bar.addWidget(self.shop_combo)
        shop_bar.addStretch()

        main_layout.addLayout(shop_bar)

        # SegmentedWidget 标签切换
        self.pivot = SegmentedWidget(self)
        self.pivot.setFixedWidth(300)
        self.stacked_widget = QStackedWidget(self)

        # 初始化两个页面
        self._init_product_tab()
        self._init_cs_tab()

        # 添加页面到 stacked_widget
        self.stacked_widget.addWidget(self.product_tab)
        self.stacked_widget.addWidget(self.cs_tab)

        # 添加 SegmentedWidget 按钮（带懒加载）
        self.pivot.addItem(
            routeKey="product",
            text="产品知识",
            onClick=lambda: self._switch_to_product_tab()
        )
        self.pivot.addItem(
            routeKey="customer_service",
            text="客服知识",
            onClick=lambda: self._switch_to_cs_tab()
        )
        self.pivot.setCurrentItem("product")

        # 居中放置 SegmentedWidget
        pivot_layout = QHBoxLayout()
        pivot_layout.addStretch()
        pivot_layout.addWidget(self.pivot)
        pivot_layout.addStretch()
        main_layout.addLayout(pivot_layout)

        main_layout.addWidget(self.stacked_widget)

        self.setLayout(main_layout)
        logger.info("KnowledgeUI 初始化完成")

    def _load_shops(self):
        """加载店铺列表到下拉框"""
        self.shop_combo.clear()
        self._shop_cache.clear()
        shops = self.knowledge_service.get_all_shops()
        if not shops:
            self.shop_combo.addItem("请先在账号管理添加店铺")
            self.shop_combo.setItemData(0, None)
            return

        for i, shop in enumerate(shops):
            self.shop_combo.addItem(shop.shop_name)
            self.shop_combo.setItemData(i, shop.id)
            self._shop_cache[shop.id] = shop.shop_name

        # 默认选中第一个
        if len(shops) > 0:
            self.shop_combo.setCurrentIndex(0)
            self.current_shop_id = shops[0].id
            # 懒加载：只刷新当前可见的标签页
            if self.stacked_widget.currentWidget() == self.product_tab:
                self._refresh_product_table()
                self._product_loaded = True
            else:
                self._refresh_cs_table()
                self._cs_loaded = True

    def _switch_to_product_tab(self):
        """切换到产品知识标签页（懒加载）"""
        self.stacked_widget.setCurrentWidget(self.product_tab)
        if not self._product_loaded and self.current_shop_id is not None:
            self._refresh_product_table()
            self._product_loaded = True

    def _switch_to_cs_tab(self):
        """切换到客服知识标签页（懒加载）"""
        self.stacked_widget.setCurrentWidget(self.cs_tab)
        if not self._cs_loaded and self.current_shop_id is not None:
            self._refresh_cs_table()
            self._cs_loaded = True

    def _on_shop_changed(self, index: int):
        """店铺切换（懒加载，只刷新当前可见标签页）"""
        shop_id = self.shop_combo.itemData(index)
        if shop_id is not None:
            self.current_shop_id = shop_id
            self._product_loaded = False
            self._cs_loaded = False
            self._last_cs_tags = ()
            # 只刷新当前可见的标签页
            if self.stacked_widget.currentWidget() == self.product_tab:
                self._refresh_product_table()
                self._product_loaded = True
            else:
                self._refresh_cs_table()
                self._cs_loaded = True

    def _init_product_tab(self):
        """初始化产品知识标签页"""
        self.product_tab = QWidget()
        layout = QVBoxLayout(self.product_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        self.sync_btn = PrimaryPushButton("同步产品知识")
        self.sync_btn.clicked.connect(self._on_sync_clicked)
        self.clear_btn = PushButton("清空全部")
        self.clear_btn.clicked.connect(self._on_clear_clicked)

        toolbar.addWidget(self.sync_btn)
        toolbar.addWidget(self.clear_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.cancel_sync_btn = PushButton("取消")
        self.cancel_sync_btn.clicked.connect(self._on_cancel_sync)
        self.cancel_sync_btn.setVisible(False)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.cancel_sync_btn)
        layout.addLayout(progress_layout)

        # 产品表格
        self.product_table = TableWidget()
        self.product_table.setColumnCount(5)
        self.product_table.setHorizontalHeaderLabels(["商品ID", "商品名称", "价格", "同步时间", "操作"])
        self.product_table.setAlternatingRowColors(True)  # 交替行颜色
        self.product_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)  # 选择整行
        self.product_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)  # 单选
        self.product_table.verticalHeader().setVisible(False)  # 隐藏行号
        self.product_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.product_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.product_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.product_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.product_table.setColumnWidth(4, 180)  # 操作列固定宽度
        self.product_table.verticalHeader().setDefaultSectionSize(50)  # 设置默认行高
        layout.addWidget(self.product_table)

    def _init_cs_tab(self):
        """初始化客服知识标签页"""
        self.cs_tab = QWidget()
        layout = QVBoxLayout(self.cs_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        self.add_cs_btn = PrimaryPushButton("添加客服知识")
        self.add_cs_btn.clicked.connect(self._on_add_cs_clicked)

        # 标签筛选
        self.tag_label = QLabel("标签筛选:")
        self.tag_combo = ComboBox()
        self.tag_combo.addItem("全部", None)
        self.tag_combo.currentIndexChanged.connect(self._on_tag_filter_changed)

        self.batch_import_btn = PushButton("批量导入")
        self.batch_import_btn.clicked.connect(self._on_batch_import_clicked)

        toolbar.addWidget(self.add_cs_btn)
        toolbar.addWidget(self.batch_import_btn)
        toolbar.addStretch()
        toolbar.addWidget(self.tag_label)
        toolbar.addWidget(self.tag_combo)
        layout.addLayout(toolbar)

        # 客服知识表格
        self.cs_table = TableWidget()
        self.cs_table.setColumnCount(6)
        self.cs_table.setHorizontalHeaderLabels(["标题", "内容", "标签", "状态", "更新时间", "操作"])
        self.cs_table.setAlternatingRowColors(True)  # 交替行颜色
        self.cs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)  # 选择整行
        self.cs_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)  # 单选
        self.cs_table.verticalHeader().setVisible(False)  # 隐藏行号
        self.cs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.cs_table.setColumnWidth(0, 160)  # 标题列固定宽度
        self.cs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.cs_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.cs_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.cs_table.setColumnWidth(5, 180)  # 操作列固定宽度
        self.cs_table.verticalHeader().setDefaultSectionSize(50)  # 设置默认行高
        layout.addWidget(self.cs_table)

    def _refresh_product_table(self):
        """刷新产品知识表格"""
        if self.current_shop_id is None:
            return

        products = self.knowledge_service.list_products_by_shop(self.current_shop_id)
        self.product_table.setRowCount(len(products))

        for row, product in enumerate(products):
            # 商品ID
            item = QTableWidgetItem(str(product.goods_id))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.product_table.setItem(row, 0, item)

            # 商品名称
            item = QTableWidgetItem(product.goods_name)
            self.product_table.setItem(row, 1, item)

            # 价格
            price_str = product.price or ""
            item = QTableWidgetItem(price_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.product_table.setItem(row, 2, item)

            # 同步时间
            dt_str = product.last_extracted_at.strftime("%Y-%m-%d %H:%M") if product.last_extracted_at else ""
            item = QTableWidgetItem(dt_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.product_table.setItem(row, 3, item)

            # 操作按钮 - 详情/编辑 删除
            # 使用容器放按钮
            cell_widget = QWidget()
            btn_layout = QHBoxLayout(cell_widget)
            btn_layout.setContentsMargins(4, 4, 4, 4)
            btn_layout.setSpacing(4)

            detail_btn = PushButton("详情")
            detail_btn.clicked.connect(lambda _, r=row: self._view_product(r))
            delete_btn = PushButton("删除")
            delete_btn.clicked.connect(lambda _, r=row: self._on_delete_product(r))

            btn_layout.addWidget(detail_btn)
            btn_layout.addWidget(delete_btn)
            cell_widget.setLayout(btn_layout)
            self.product_table.setCellWidget(row, 4, cell_widget)

    def _refresh_cs_table(self):
        """刷新客服知识表格"""
        if self.current_shop_id is None:
            return

        # 更新标签下拉框（只有标签变化时才重建，避免卡顿）
        all_tags = tuple(sorted(self.knowledge_service.get_all_tags(self.current_shop_id)))
        current_selection = self.tag_combo.currentData()
        if all_tags != self._last_cs_tags:
            self._last_cs_tags = all_tags

            self.tag_combo.blockSignals(True)
            self.tag_combo.clear()
            self.tag_combo.addItem("全部")
            self.tag_combo.setItemData(0, None)
            for i, tag in enumerate(all_tags, 1):
                self.tag_combo.addItem(tag)
                self.tag_combo.setItemData(i, tag)
            # 恢复选中
            if current_selection is None:
                self.tag_combo.setCurrentIndex(0)
            else:
                # 查找索引
                for i in range(self.tag_combo.count()):
                    if self.tag_combo.itemData(i) == current_selection:
                        self.tag_combo.setCurrentIndex(i)
                        break
            self.tag_combo.blockSignals(False)

        # 获取数据
        if current_selection is None:
            cs_list = self.knowledge_service.list_customer_service_with_disabled(self.current_shop_id)
        else:
            cs_list = self.knowledge_service.filter_customer_service_by_tag(self.current_shop_id, current_selection)

        self.cs_table.setRowCount(len(cs_list))

        for row, cs in enumerate(cs_list):
            # 标题
            item = QTableWidgetItem(cs.title)
            item.setData(Qt.ItemDataRole.UserRole, cs.id)
            self.cs_table.setItem(row, 0, item)

            # 内容（截断避免过长）
            content_preview = cs.content
            if len(content_preview) > 60:
                content_preview = content_preview[:60] + "..."
            item = QTableWidgetItem(content_preview)
            item.setToolTip(cs.content)
            self.cs_table.setItem(row, 1, item)

            # 标签
            item = QTableWidgetItem(cs.tags or "")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.cs_table.setItem(row, 2, item)

            # 状态
            status_text = "启用" if cs.enabled else "禁用"
            item = QTableWidgetItem(status_text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.cs_table.setItem(row, 3, item)

            # 更新时间
            dt_str = cs.updated_at.strftime("%Y-%m-%d %H:%M") if cs.updated_at else ""
            item = QTableWidgetItem(dt_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.cs_table.setItem(row, 4, item)

            # 操作按钮
            cell_widget = QWidget()
            btn_layout = QHBoxLayout(cell_widget)
            btn_layout.setContentsMargins(4, 4, 4, 4)
            btn_layout.setSpacing(4)

            edit_btn = PushButton("编辑")
            edit_btn.clicked.connect(lambda _, r=row: self._on_edit_cs(r))
            delete_btn = PushButton("删除")
            delete_btn.clicked.connect(lambda _, r=row: self._on_delete_cs(r))

            btn_layout.addWidget(edit_btn)
            btn_layout.addWidget(delete_btn)
            cell_widget.setLayout(btn_layout)
            self.cs_table.setCellWidget(row, 5, cell_widget)

            # 禁用行灰色显示
            if not cs.enabled:
                for col in range(self.cs_table.columnCount()):
                    if self.cs_table.item(row, col):
                        self.cs_table.item(row, col).setForeground(Qt.GlobalColor.gray)

    def _view_product(self, row: int):
        """查看/编辑产品详情"""
        # 获取商品ID
        product_id = self.product_table.item(row, 0).text()
        goods_id = int(product_id)
        product = self.knowledge_service.get_product_by_goods_id(self.current_shop_id, goods_id)
        if not product:
            QMessageBox.warning(self, "错误", "产品不存在")
            return

        dialog = ProductDetailDialog(product, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            # 更新
            if product.extracted_content != data["extracted_content"] or product.goods_name != data["goods_name"]:
                with self.knowledge_service.get_session() as session:
                    prod = session.get(ProductKnowledge, product.id)
                    if prod:
                        prod.goods_name = data["goods_name"]
                        prod.extracted_content = data["extracted_content"]
                        session.commit()
                        self._show_message("success", "更新成功")
                        self._refresh_product_table()

    def _on_delete_product(self, row: int):
        """删除产品"""
        # 获取商品ID（第0列）
        product_id = self.product_table.item(row, 0).text()
        goods_id = int(product_id)
        product = self.knowledge_service.get_product_by_goods_id(self.current_shop_id, goods_id)
        if not product:
            QMessageBox.warning(self, "错误", "产品不存在")
            return

        confirm = QMessageBox.question(
            self, "确认删除",
            f"确定要删除产品 «{product.goods_name}» 吗？\n\n删除后无法恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            success = self.knowledge_service.delete_product(product.id)
            if success:
                self._show_message("success", "删除成功")
                self._refresh_product_table()
            else:
                self._show_message("error", "删除失败")

    def _on_clear_clicked(self):
        """清空全部产品知识"""
        if self.current_shop_id is None:
            return

        confirm = QMessageBox.question(
            self, "确认清空",
            f"确定要清空当前店铺的所有产品知识吗？\n\n清空后无法恢复，请谨慎操作。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            total_deleted = self.knowledge_service.clear_products_by_shop(self.current_shop_id)
            self._show_message("success", f"已清空，共删除 {total_deleted} 条记录")
            self._refresh_product_table()

    def _on_sync_clicked(self):
        """点击同步按钮，弹出选择同步模式"""
        if self.current_shop_id is None:
            self._show_message("warning", "请先选择店铺")
            return

        # 获取当前选中的店铺
        shop_to_sync = self._get_shop_by_id(self.current_shop_id)
        if not shop_to_sync:
            self._show_message("error", "无法获取店铺信息")
            return

        # 弹出选择同步模式对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择同步模式")
        dialog.resize(350, 180)

        layout = QVBoxLayout(dialog)

        label = QLabel(f"即将同步店铺 «{shop_to_sync.shop_name}» 的产品知识，请选择同步模式:")
        layout.addWidget(label)

        incremental_btn = PushButton("增量同步（仅同步本地不存在的商品，推荐）")
        full_btn = PrimaryPushButton("全量同步（同步所有商品，覆盖已提取知识）")

        layout.addWidget(incremental_btn)
        layout.addWidget(full_btn)

        def start_sync(is_full):
            dialog.close()
            self._start_sync(shop_to_sync, is_full)

        incremental_btn.clicked.connect(lambda: start_sync(False))
        full_btn.clicked.connect(lambda: start_sync(True))

        dialog.setLayout(layout)
        dialog.exec()

    def _get_shop_by_id(self, shop_id: int) -> Optional[Shop]:
        """根据ID获取店铺对象"""
        # 根据ID查询店铺对象，同时预加载关联的accounts，避免懒加载问题
        with self.knowledge_service.get_session() as session:
            from sqlalchemy import select
            from sqlalchemy.orm import joinedload
            stmt = select(Shop).where(Shop.id == shop_id).options(joinedload(Shop.accounts))
            return session.scalar(stmt)

    def _start_sync(self, shop: Shop, is_full_sync: bool):
        """开始同步"""
        # 获取pdd shop_id和user_id
        pdd_shop_id = shop.shop_id
        # A shop may have multiple isolated accounts. Never silently select
        # the first one because that can sync with the wrong credentials.
        if not shop.accounts:
            self._show_message("error", "店铺没有账号信息")
            return

        if len(shop.accounts) == 1:
            user_id = shop.accounts[0].user_id
        else:
            labels = [f"{account.username} ({account.user_id})" for account in shop.accounts]
            selected, ok = QInputDialog.getItem(
                self,
                "选择同步账号",
                "当前店铺有多个账号，请选择本次同步使用的账号：",
                labels,
                0,
                False,
            )
            if not ok or selected not in labels:
                return
            user_id = shop.accounts[labels.index(selected)].user_id

        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.cancel_sync_btn.setVisible(True)
        self.sync_btn.setEnabled(False)

        # 创建工作线程
        self._sync_worker = SyncWorker(
            shop_db_id=shop.id,
            pdd_shop_id=pdd_shop_id,
            user_id=user_id,
            is_full_sync=is_full_sync,
            product_sync=self.product_sync,
            parent=self,
        )

        # 连接信号
        def on_progress(
            current: int, total: int, success: int, failed: int, degraded: int,
            current_name: str, phase: str,
        ):
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            # 根据阶段显示不同的提示
            if phase == "fetching":
                self.progress_label.setText(f"[1/3] 抓取商品列表: {current_name} ({current}/{total})")
            elif phase == "saving_basic":
                self.progress_label.setText(f"[2/3] 保存商品信息: {current_name} ({current}/{total}, 成功 {success})")
                # 第二阶段开始后刷新一次表格，让用户能看到商品
                if current == 1 or current % 10 == 0:
                    self._refresh_product_table()
            elif phase == "extracting":
                self.progress_label.setText(
                    f"[3/3] 提取商品知识: {current_name} ({current}/{total}, "
                    f"成功 {success}, 降级 {degraded}, 失败 {failed})"
                )
                # 提取阶段也定期刷新，显示更新的知识
                if current % 5 == 0:
                    self._refresh_product_table()
            else:
                self.progress_label.setText(f"正在同步: {current_name} ({current}/{total}, 成功 {success})")

        def on_finished(success: int, failed: int, degraded: int, cancelled: bool):
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            self.cancel_sync_btn.setVisible(False)
            self.sync_btn.setEnabled(True)

            # 最后刷新一次表格
            self._refresh_product_table()

            if cancelled:
                self._show_message("info", "同步已取消")
            else:
                msg = f"同步完成: 成功 {success}, 降级 {degraded}, 失败 {failed}"
                if degraded:
                    msg += "。降级商品仅补基本信息，已有知识未被覆盖"
                self._show_message("warning" if failed or degraded else "success", msg)

        self._sync_worker.progress_updated.connect(on_progress)
        self._sync_worker.sync_finished.connect(on_finished)
        self._sync_worker.start()

    def _on_cancel_sync(self):
        """取消同步"""
        if hasattr(self, '_sync_worker') and self._sync_worker.isRunning():
            self.product_sync.cancel()
            self.cancel_sync_btn.setEnabled(False)

    def _on_add_cs_clicked(self):
        """添加客服知识"""
        if self.current_shop_id is None:
            self._show_message("warning", "请先选择店铺")
            return

        dialog = CsAddEditDialog(self.current_shop_id, None, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            self.knowledge_service.add_customer_service(
                shop_id=self.current_shop_id,
                title=data["title"],
                content=data["content"],
                tags=data["tags"],
                enabled=data["enabled"],
            )
            self._show_message("success", "添加成功")
            self._refresh_cs_table()

    def _on_batch_import_clicked(self):
        """批量导入客服知识"""
        if self.current_shop_id is None:
            self._show_message("warning", "请先选择店铺")
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "选择客服话术文件",
            "",
            "Excel 文件 (*.xls *.xlsx)",
        )
        if not filepath:
            return

        try:
            rows, parse_skipped = self._parse_excel(filepath)
        except Exception as e:
            self._show_message("error", f"文件读取失败: {e}")
            return

        success, import_skipped = self.knowledge_service.batch_import_customer_service(
            self.current_shop_id, rows
        )
        total_skipped = parse_skipped + import_skipped
        self._show_message("success", f"导入完成：成功 {success} 条，跳过 {total_skipped} 条")
        self._refresh_cs_table()

    def _parse_excel(self, filepath: str) -> tuple[list, int]:
        """解析 Excel 文件，返回 (有效行列表, 跳过行数)

        列顺序：0=一级分类, 1=二级分类, 2=话术标题, 3=话术内容
        """
        import pandas as pd

        df = pd.read_excel(filepath, header=0, dtype=str)
        df = df.fillna("")

        rows = []
        skipped = 0
        for _, row in df.iterrows():
            values = row.tolist()
            # 补齐不足4列的情况
            while len(values) < 4:
                values.append("")

            cat1 = str(values[0]).strip()
            cat2 = str(values[1]).strip()
            title = str(values[2]).strip()
            content = str(values[3]).strip()

            if not cat1 or not content:
                skipped += 1
                continue

            tags = f"{cat1},{cat2}" if cat2 else cat1
            rows.append({"title": title, "content": content, "tags": tags})

        return rows, skipped

    def _on_edit_cs(self, row: int):
        """编辑客服知识"""
        cs_id = self._get_cs_id_from_row(row)
        cs = self.knowledge_service.get_customer_service_by_id(cs_id)
        if not cs:
            self._show_message("error", "知识不存在")
            return

        dialog = CsAddEditDialog(cs.shop_id, cs, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            updated = self.knowledge_service.update_customer_service(
                cs_id,
                title=data["title"],
                content=data["content"],
                tags=data["tags"],
                enabled=data["enabled"],
            )
            if updated:
                self._show_message("success", "更新成功")
                self._refresh_cs_table()

    def _on_delete_cs(self, row: int):
        """删除客服知识"""
        cs_id = self._get_cs_id_from_row(row)
        cs = self.knowledge_service.get_customer_service_by_id(cs_id)
        if not cs:
            self._show_message("error", "知识不存在")
            return

        confirm = QMessageBox.question(
            self, "确认删除",
            f"确定要删除客服知识 «{cs.title}» 吗？\n\n删除后无法恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            success = self.knowledge_service.delete_customer_service(cs_id)
            if success:
                self._show_message("success", "删除成功")
                self._refresh_cs_table()
            else:
                self._show_message("error", "删除失败")

    def _get_cs_id_from_row(self, row: int) -> int:
        """从表格行读取稳定的客服知识ID。"""
        title_item = self.cs_table.item(row, 0)
        if title_item is None:
            return 0
        stored_id = title_item.data(Qt.ItemDataRole.UserRole)
        if stored_id is not None:
            return int(stored_id)

        # 兼容旧表格实例：标题查询只作为最后兜底，并取最新记录。
        title = title_item.text()
        # 直接查询当前店铺下的客服知识
        with self.knowledge_service.get_session() as session:
            from sqlalchemy import select
            stmt = select(CustomerServiceKnowledge).where(
                CustomerServiceKnowledge.shop_id == self.current_shop_id,
                CustomerServiceKnowledge.title == title,
            ).order_by(CustomerServiceKnowledge.id.desc())
            cs = session.scalar(stmt)
            if cs:
                return cs.id
        return 0

    def _on_tag_filter_changed(self, index: int):
        """标签筛选变化"""
        self._refresh_cs_table()

    def _show_message(self, level: str, content: str):
        """显示消息条"""
        method = getattr(InfoBar, level)
        method(
            title="",
            content=content,
            orient=InfoBarPosition.TOP,
            parent=self,
        )

    def showEvent(self, event):
        """显示时刷新"""
        super().showEvent(event)
        # 刷新店铺列表，可能有新增
        self._load_shops()
