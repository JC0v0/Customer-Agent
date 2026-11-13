"""关键词转人工处理器（Message.handlers.keyword_handler）单元测试。

`KeywordDetectionHandler` 是「关键词转人工」核心能力，此前没有专属测试。
覆盖大小写不敏感匹配、非文本消息忽略、营业时间窗口（含跨天）、数据库
失败时回退默认关键词等路径。
"""

import datetime as _dt
import unittest
from unittest import mock

from bridge.context import ChannelType, Context, ContextType
from Message.handlers import keyword_handler
from Message.handlers.keyword_handler import KeywordDetectionHandler


def _context(content, msg_type: ContextType = ContextType.TEXT) -> Context:
    return Context.create_pinduoduo_context(
        content=content,
        msg_id="msg-1",
        from_uid="customer-1",
        shop_id="shop-1",
        user_id="account-1",
        user_msg_type=msg_type,
        channel_type=ChannelType.PINDUODUO,
    )


def _patch_noon():
    """把 handler 模块内的 datetime.now() 固定为营业时间内的 12:00。"""
    return mock.patch.object(
        keyword_handler,
        "datetime",
        mock.Mock(now=mock.Mock(return_value=_dt.datetime(2026, 1, 1, 12, 0, 0))),
    )


class KeywordDetectionHandlerTests(unittest.TestCase):
    def _handler(self, keywords, business_hours=None) -> KeywordDetectionHandler:
        with mock.patch.object(keyword_handler, "db_manager") as mock_db:
            mock_db.get_all_keywords.return_value = [{"keyword": k} for k in keywords]
            return KeywordDetectionHandler(business_hours=business_hours)

    def test_keyword_match_is_case_insensitive(self):
        handler = self._handler(["人工", "Refund"])
        with _patch_noon():
            self.assertTrue(handler.can_handle(_context("我要转人工")))
            self.assertTrue(handler.can_handle(_context("I want a refund")))
            self.assertFalse(handler.can_handle(_context("你好，想咨询物流")))

    def test_non_text_message_is_ignored(self):
        handler = self._handler(["人工"])
        with _patch_noon():
            self.assertFalse(
                handler.can_handle(_context("人工", msg_type=ContextType.IMAGE))
            )

    def test_empty_content_is_ignored(self):
        handler = self._handler(["人工"])
        with _patch_noon():
            self.assertFalse(handler.can_handle(_context(None)))

    def test_outside_business_hours_is_ignored(self):
        handler = self._handler(["人工"], business_hours={"start": "08:00", "end": "23:00"})
        with mock.patch.object(
            keyword_handler,
            "datetime",
            mock.Mock(now=mock.Mock(return_value=_dt.datetime(2026, 1, 1, 2, 0, 0))),
        ):
            self.assertFalse(handler.can_handle(_context("转人工")))

    def test_overnight_business_hours_window(self):
        handler = self._handler(["人工"], business_hours={"start": "22:00", "end": "06:00"})
        with mock.patch.object(
            keyword_handler,
            "datetime",
            mock.Mock(now=mock.Mock(return_value=_dt.datetime(2026, 1, 1, 23, 0, 0))),
        ):
            self.assertTrue(handler.can_handle(_context("转人工")))
        with mock.patch.object(
            keyword_handler,
            "datetime",
            mock.Mock(now=mock.Mock(return_value=_dt.datetime(2026, 1, 1, 12, 0, 0))),
        ):
            self.assertFalse(handler.can_handle(_context("转人工")))

    def test_db_failure_falls_back_to_default_keywords(self):
        with mock.patch.object(keyword_handler, "db_manager") as mock_db:
            mock_db.get_all_keywords.side_effect = RuntimeError("db down")
            handler = KeywordDetectionHandler()
        self.assertIn("人工", handler.get_keywords())
        with _patch_noon():
            self.assertTrue(handler.can_handle(_context("我要转人工")))

    def test_reload_keywords_picks_up_new_entries(self):
        with mock.patch.object(keyword_handler, "db_manager") as mock_db:
            mock_db.get_all_keywords.return_value = [{"keyword": "人工"}]
            handler = KeywordDetectionHandler()
            self.assertNotIn("退款", handler.get_keywords())

            mock_db.get_all_keywords.return_value = [
                {"keyword": "人工"},
                {"keyword": "退款"},
            ]
            handler.reload_keywords()
            self.assertIn("退款", handler.get_keywords())


if __name__ == "__main__":
    unittest.main()
