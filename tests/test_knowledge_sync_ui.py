"""知识同步进度/UI 回归；只使用 offscreen 控件与模拟服务。"""

import asyncio
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PyQt6.QtWidgets import QApplication, QLabel, QProgressBar, QPushButton

from database.product_sync import SyncProgress
from service import knowledge_io
from ui import Knowledge_ui as knowledge_ui


class KnowledgeSyncUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def _worker(self, service=None):
        return knowledge_ui.SyncWorker(
            shop_db_id=1, pdd_shop_id="100", user_id="fake-account", is_full_sync=True,
            product_sync=service or mock.Mock(),
        )

    def test_worker_emits_success_failure_and_downgrade_counts(self):
        result = SyncProgress(5, 5, 1, 1, "Item", phase="extracting", degraded=3)

        async def sync_shop(**kwargs):
            kwargs["progress_callback"](result)
            return result

        worker = self._worker(SimpleNamespace(sync_shop=sync_shop))
        progress, finished = [], []
        worker.progress_updated.connect(lambda *args: progress.append(args))
        worker.sync_finished.connect(lambda *args: finished.append(args))
        try:
            worker.run()
        finally:
            asyncio.set_event_loop(None)
            worker.deleteLater()
        self.assertEqual(progress, [(5, 5, 1, 1, 3, "Item", "extracting")])
        self.assertEqual(finished, [(1, 1, 3, False)])

    def _view(self):
        view = SimpleNamespace(
            progress_bar=QProgressBar(), progress_label=QLabel(), cancel_sync_btn=QPushButton(),
            sync_btn=QPushButton(), product_sync=mock.Mock(),
            _refresh_product_table=mock.Mock(), _show_message=mock.Mock(),
        )
        for widget in (view.progress_bar, view.progress_label, view.cancel_sync_btn, view.sync_btn):
            self.addCleanup(widget.deleteLater)
        worker = self._worker()
        self.addCleanup(worker.deleteLater)
        shop = SimpleNamespace(id=1, shop_id="100", accounts=[SimpleNamespace(user_id="fake-account")])
        with mock.patch.object(knowledge_ui, "SyncWorker", return_value=worker), mock.patch.object(worker, "start"):
            knowledge_ui.KnowledgeUI._start_sync(view, shop, True)
        return view, worker

    def test_degraded_progress_and_finish_are_visible_and_not_green_success(self):
        view, worker = self._view()
        worker.progress_updated.emit(5, 5, 1, 1, 3, "Item", "extracting")
        for value in ("成功 1", "降级 3", "失败 1"):
            self.assertIn(value, view.progress_label.text())
        worker.sync_finished.emit(1, 1, 3, False)
        level, message = view._show_message.call_args.args
        self.assertEqual(level, "warning")
        self.assertIn("降级 3", message)
        self.assertIn("已有知识未被覆盖", message)
        self.assertTrue(view.sync_btn.isEnabled())
        self.assertTrue(view.progress_bar.isHidden())

    def test_failure_without_degradation_also_warns(self):
        view, worker = self._view()
        worker.sync_finished.emit(0, 2, 0, False)
        self.assertEqual(view._show_message.call_args.args[0], "warning")
        self.assertIn("失败 2", view._show_message.call_args.args[1])

    def test_all_success_keeps_success_notification(self):
        view, worker = self._view()
        worker.sync_finished.emit(2, 0, 0, False)
        self.assertEqual(view._show_message.call_args.args[0], "success")
        self.assertIn("成功 2", view._show_message.call_args.args[1])


class CustomerServiceIoUITests(unittest.TestCase):
    """导出 / 模板 / 导入结果的回调：按钮接上了才算做完。"""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.messages = []
        self.view = SimpleNamespace(
            current_shop_id=1,
            knowledge_service=mock.Mock(),
            _show_message=lambda level, text: self.messages.append((level, text)),
            _refresh_cs_table=mock.Mock(),
        )

    def _path(self, name):
        return str(Path(self.tmp.name) / name)

    def test_export_writes_selected_shop_knowledge(self):
        self.view.knowledge_service.list_customer_service_with_disabled.return_value = [
            SimpleNamespace(title="发货", content="当天发出", tags="物流", enabled=True),
        ]
        target = self._path("export.xlsx")
        with mock.patch.object(knowledge_ui.QFileDialog, "getSaveFileName", return_value=(target, "")):
            knowledge_ui.KnowledgeUI._on_export_cs_clicked(self.view)
        self.assertTrue(Path(target).exists())
        self.assertEqual(self.messages[-1][0], "success")
        self.view.knowledge_service.list_customer_service_with_disabled.assert_called_once_with(1)
        self.assertEqual(knowledge_io.parse_workbook(target).rows[0]["title"], "发货")

    def test_export_requires_a_shop_and_some_data(self):
        self.view.current_shop_id = None
        with mock.patch.object(knowledge_ui.QFileDialog, "getSaveFileName") as dialog:
            knowledge_ui.KnowledgeUI._on_export_cs_clicked(self.view)
        dialog.assert_not_called()
        self.assertEqual(self.messages[-1][0], "warning")

        self.view.current_shop_id = 1
        self.view.knowledge_service.list_customer_service_with_disabled.return_value = []
        with mock.patch.object(knowledge_ui.QFileDialog, "getSaveFileName") as dialog:
            knowledge_ui.KnowledgeUI._on_export_cs_clicked(self.view)
        dialog.assert_not_called()
        self.assertIn("没有客服知识", self.messages[-1][1])

    def test_cancelling_the_save_dialog_writes_nothing(self):
        self.view.knowledge_service.list_customer_service_with_disabled.return_value = [
            SimpleNamespace(title="t", content="c", tags=None, enabled=True),
        ]
        with mock.patch.object(knowledge_ui.QFileDialog, "getSaveFileName", return_value=("", "")):
            knowledge_ui.KnowledgeUI._on_export_cs_clicked(self.view)
        self.assertEqual(self.messages, [])

    def test_template_button_produces_an_importable_file(self):
        target = self._path("template.xlsx")
        with mock.patch.object(knowledge_ui.QFileDialog, "getSaveFileName", return_value=(target, "")):
            knowledge_ui.KnowledgeUI._on_download_template_clicked(self.view)
        self.assertTrue(Path(target).exists())
        self.assertEqual(len(knowledge_io.parse_workbook(target).rows), 2)

    def _report(self, parsed, success=1, duplicated=0):
        with mock.patch.object(knowledge_ui, "QMessageBox", wraps=knowledge_ui.QMessageBox) as box:
            instance = mock.Mock()
            box.return_value = instance
            knowledge_ui.KnowledgeUI._report_import_result(self.view, parsed, success, duplicated)
        return instance

    def test_import_report_lists_skipped_rows_with_reasons(self):
        parsed = knowledge_io.parse_rows(["标题", "内容"], [["", "只有内容"], ["ok", "ok"]])
        instance = self._report(parsed)
        summary = instance.setText.call_args.args[0]
        self.assertIn("成功导入 1 条", summary)
        self.assertIn("格式问题跳过 1 条", summary)
        self.assertIn("第 2 行", instance.setDetailedText.call_args.args[0])
        instance.exec.assert_called_once()

    def test_unrecognized_header_is_surfaced_to_the_user(self):
        parsed = knowledge_io.parse_rows(["A", "B", "C", "D"], [["分类", "", "标题", "内容"]])
        detail = self._report(parsed).setDetailedText.call_args.args[0]
        self.assertIn("未识别到已知表头", detail)
        self.assertIn("下载模板", detail)

    def test_clean_import_reports_without_details(self):
        parsed = knowledge_io.parse_rows(["标题", "内容"], [["ok", "ok"]])
        instance = self._report(parsed, success=1, duplicated=2)
        self.assertIn("重复跳过 2 条", instance.setText.call_args.args[0])
        instance.setDetailedText.assert_not_called()


if __name__ == "__main__":
    unittest.main()
