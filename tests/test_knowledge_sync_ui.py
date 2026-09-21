"""知识同步进度/UI 回归；只使用 offscreen 控件与模拟服务。"""

import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest import mock

from PyQt6.QtWidgets import QApplication, QLabel, QProgressBar, QPushButton

from database.product_sync import SyncProgress
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


if __name__ == "__main__":
    unittest.main()
