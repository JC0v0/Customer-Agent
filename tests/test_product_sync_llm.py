"""商品知识提取回归：JSON 兼容、真实成功/降级区分、旧内容保护。"""

import copy
from datetime import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.knowledge_service import KnowledgeService
from database.models import Base, Channel, Shop, ProductKnowledge
from database.product_sync import ProductSyncService, SyncProgress
from utils.llm_provider import LLMProfile, LLMProvider
from utils.llm_transport import NormalizedResponse


def profile():
    return LLMProfile(provider=LLMProvider.DEEPSEEK, model_name="deepseek-chat", api_key="fake-key")


class ProductSyncTransportTests(unittest.IsolatedAsyncioTestCase):
    def _service(self):
        return ProductSyncService(object(), request_delay=0)

    async def _extract(self, content):
        with mock.patch("database.product_sync.async_completion", new=mock.AsyncMock(
            return_value=NormalizedResponse(content, [], {}, object())
        )):
            return await self._service()._extract_product_knowledge(
                {"goods_name": "Item", "price": "9.9"}, {"specifications": []}, profile=profile(),
            )

    async def test_product_sync_uses_shared_snapshot_and_preserves_image_json_operation(self):
        response = NormalizedResponse('{"brand":"Brand"}', [], {}, object())
        with mock.patch("database.product_sync.async_completion", new=mock.AsyncMock(return_value=response)) as completion:
            result, succeeded = await self._service()._extract_product_knowledge(
                {"goods_name": "Item", "price": "9.9", "sold_quantity": 3, "thumb_url": "https://img.test/item.jpg"},
                {"specifications": [{"name": "size", "value": "M"}]}, profile=profile(),
            )
        self.assertTrue(succeeded)
        self.assertIn("Brand", result)
        self.assertEqual(completion.await_args.args[0].provider, LLMProvider.DEEPSEEK)
        self.assertFalse(completion.await_args.kwargs["use_tools"])
        self.assertEqual(completion.await_args.kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(completion.await_args.args[1][1]["content"][1]["type"], "image_url")

    async def test_malformed_json_degrades_instead_of_storing_raw_error(self):
        result, succeeded = await self._extract("not-json secret-provider-payload")
        self.assertFalse(succeeded)
        self.assertIn("Item", result)
        self.assertNotIn("secret-provider-payload", result)

    async def test_provider_failure_falls_back_to_basic_info_without_raw_error(self):
        with mock.patch("database.product_sync.async_completion", new=mock.AsyncMock(
            side_effect=RuntimeError("secret-key provider payload")
        )), mock.patch("database.product_sync.logger") as logger:
            result, succeeded = await self._service()._extract_product_knowledge(
                {"goods_name": "Item", "price": "9.9"}, {"specifications": []}, profile=profile(),
            )
        self.assertFalse(succeeded)
        self.assertIn("Item", result)
        self.assertNotIn("secret-key", result + str(logger.mock_calls))

    async def test_missing_profile_keeps_basic_info_fallback(self):
        service = self._service()
        with mock.patch.object(service, "_snapshot_llm_profile", return_value=None), mock.patch(
            "database.product_sync.async_completion", new=mock.AsyncMock()
        ) as completion:
            result, succeeded = await service._extract_product_knowledge(
                {"goods_name": "Item", "price": "9.9"}, {"specifications": []},
            )
        self.assertFalse(succeeded)
        self.assertIn("Item", result)
        completion.assert_not_awaited()

    async def test_complete_markdown_fences_are_accepted(self):
        for text in ('```json\n{"brand":"Brand"}\n```',
                     '```JSON\r\n{"brand":"Brand"}\r\n```',
                     '```\n{"brand":"Brand"}\n```'):
            with self.subTest(text=text):
                result, succeeded = await self._extract(text)
                self.assertTrue(succeeded)
                self.assertIn("Brand", result)
                self.assertNotIn("```", result)

    async def test_empty_and_non_object_json_do_not_count_as_success(self):
        for text in ("", "{}", "[]", "null", '"text"', '{"error":"unavailable"}',
                     '{"brand":"  ", "key_points":[], "faq":[]}', '{"brand":null}'):
            with self.subTest(text=text):
                result, succeeded = await self._extract(text)
                self.assertFalse(succeeded)
                self.assertNotIn("##", result)

    async def test_wrong_schema_types_are_degraded(self):
        for value in ({"brand": 42}, {"description": {}}, {"key_points": "text"},
                      {"key_points": [42]}, {"faq": "text"}, {"faq": [42]},
                      {"faq": [{"question": "q", "answer": []}]}):
            with self.subTest(value=value):
                _, succeeded = await self._extract(json.dumps(value))
                self.assertFalse(succeeded)

    async def test_valid_optional_fields_are_formatted(self):
        result, succeeded = await self._extract(json.dumps({
            "brand": None, "description": "  description  ", "key_points": ["", " point "],
            "usage": " usage ", "faq": [{"question": " q ", "answer": " a "}],
        }))
        self.assertTrue(succeeded)
        for token in ("## 产品描述", "description", "1. point", "## 使用说明", "**Q:** q", "**A:** a"):
            self.assertIn(token, result)

    async def test_truncated_or_prose_wrapped_output_is_not_guessed(self):
        for text in ('{"brand":"Brand"', 'Result: {"brand":"Brand"}',
                     '```json\n{"brand":"Brand"}\n``` trailing text'):
            with self.subTest(text=text):
                _, succeeded = await self._extract(text)
                self.assertFalse(succeeded)


class TemporaryKnowledgeMixin:
    """所有数据库测试只使用临时 SQLite，不触碰用户的知识库。"""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = create_engine(f"sqlite:///{Path(self.tmp.name) / 'knowledge.db'}")
        self.addCleanup(self.engine.dispose)
        self.sessions = sessionmaker(bind=self.engine)
        with mock.patch("database.knowledge_service.db_manager", SimpleNamespace(
            Session=self.sessions, engine=self.engine,
        )):
            self.knowledge = KnowledgeService()
        with self.sessions() as session:
            session.add(Channel(id=1, channel_name="pinduoduo"))
            session.flush()
            session.add_all([
                Shop(id=1, channel_id=1, shop_id="test-shop", shop_name="Test shop"),
                Shop(id=2, channel_id=1, shop_id="other-shop", shop_name="Other shop"),
            ])
            session.commit()
        self.old_time = datetime(2026, 8, 12, 12, 0, 0)

    def _add(self, goods_id=1, content=None, shop_id=1):
        with self.sessions() as session:
            session.add(ProductKnowledge(
                shop_id=shop_id, goods_id=goods_id, goods_name=f"Item {goods_id}",
                extracted_content=content,
                last_extracted_at=self.old_time if content and content.strip() else None,
            ))
            session.commit()

    def _get(self, goods_id=1, shop_id=1):
        return self.knowledge.get_product_by_goods_id(shop_id, goods_id)


class ProductKnowledgePersistenceTests(TemporaryKnowledgeMixin, unittest.TestCase):
    def test_downgrade_keeps_existing_text_and_success_timestamp(self):
        self._add(content="Existing manually edited knowledge")
        saved = self.knowledge.update_product_extracted_content(
            1, 1, specifications='["new spec"]', extracted_content="basic fallback", extraction_succeeded=False,
        )
        self.assertTrue(saved)
        item = self._get()
        self.assertEqual(item.extracted_content, "Existing manually edited knowledge")
        self.assertEqual(item.last_extracted_at, self.old_time)
        self.assertEqual(item.specifications, '["new spec"]')

    def test_downgrade_fills_only_empty_content_without_success_timestamp(self):
        for goods_id, empty in enumerate((None, "", " \t\r\n"), start=1):
            self._add(goods_id=goods_id, content=empty)
            self.assertTrue(self.knowledge.update_product_extracted_content(
                1, goods_id, extracted_content="basic fallback", extraction_succeeded=False,
            ))
            self.assertEqual(self._get(goods_id).extracted_content, "basic fallback")
            self.assertIsNone(self._get(goods_id).last_extracted_at)

    def test_success_updates_content_and_timestamp(self):
        self._add(content="Old knowledge")
        self.assertTrue(self.knowledge.update_product_extracted_content(
            1, 1, extracted_content="New structured knowledge", extraction_succeeded=True,
        ))
        self.assertEqual(self._get().extracted_content, "New structured knowledge")
        self.assertGreater(self._get().last_extracted_at, self.old_time)

    def test_late_downgrade_cannot_overwrite_recent_success(self):
        self._add()
        self.knowledge.update_product_extracted_content(1, 1, extracted_content="Successful knowledge")
        saved_time = self._get().last_extracted_at
        self.knowledge.update_product_extracted_content(1, 1, extracted_content="late fallback", extraction_succeeded=False)
        self.assertEqual(self._get().extracted_content, "Successful knowledge")
        self.assertEqual(self._get().last_extracted_at, saved_time)

    def test_basic_sync_never_changes_success_timestamp_or_existing_content(self):
        self._add(content="Keep this knowledge")
        self.knowledge.add_or_update_product(shop_id=1, goods_id=1, goods_name="Updated name", extracted_content=None)
        self.assertEqual(self._get().last_extracted_at, self.old_time)
        self.assertEqual(self._get().extracted_content, "Keep this knowledge")
        new = self.knowledge.add_or_update_product(shop_id=1, goods_id=2, goods_name="New item")
        self.assertIsNone(new.last_extracted_at)
        self.assertIsNone(new.extracted_content)

    def test_missing_row_is_failure_and_other_shops_are_not_touched(self):
        self._add(content="Other shop content", shop_id=2)
        self.assertFalse(self.knowledge.update_product_extracted_content(
            1, 1, extracted_content="basic fallback", extraction_succeeded=False,
        ))
        self.assertEqual(self._get(shop_id=2).extracted_content, "Other shop content")
        self.assertEqual(self._get(shop_id=2).last_extracted_at, self.old_time)


class ProductSyncOutcomeTests(TemporaryKnowledgeMixin, unittest.IsolatedAsyncioTestCase):
    def _manager(self, products):
        manager = mock.Mock()
        manager.get_product_list.side_effect = lambda page, size: {
            "success": True, "total": len(products), "products": products if page == 1 else [],
        }
        manager.get_product_detail.return_value = {"success": True, "product_info": {"specifications": ["size: M"]}}
        return manager

    async def test_mixed_results_count_success_degraded_failed_and_protect_old_content(self):
        self._add(goods_id=1, content="Keep old LLM knowledge")
        self._add(goods_id=4, content="Old fourth item")
        products = [{"goods_id": i, "goods_name": f"Item {i}", "price": "9.9"} for i in range(1, 6)]
        manager = self._manager(products)
        manager.get_product_detail.side_effect = lambda gid: (
            {"success": False} if gid == 5 else
            {"success": True, "product_info": {"specifications": ["size: M"]}}
        )

        async def completion(profile, messages, **kwargs):
            text = messages[1]["content"][0]["text"]
            if "Item 1" in text or "Item 2" in text:
                raise RuntimeError("fake-provider-error")
            return NormalizedResponse('not-json' if "Item 3" in text else '{"brand":"New brand"}', [], {}, object())

        service = ProductSyncService(self.knowledge, request_delay=0)
        snapshots = []
        with mock.patch("database.product_sync.ProductManager", return_value=manager), mock.patch.object(
            service, "_snapshot_llm_profile", return_value=profile()
        ), mock.patch("database.product_sync.async_completion", side_effect=completion):
            result = await service.sync_shop(100, 1, "fake-account", is_full_sync=True,
                                             progress_callback=lambda p: snapshots.append(copy.copy(p)))

        self.assertEqual((result.success, result.degraded, result.failed, result.current), (1, 3, 1, 5))
        self.assertEqual(result.current, result.success + result.degraded + result.failed)
        self.assertEqual(self._get(1).extracted_content, "Keep old LLM knowledge")
        self.assertEqual(self._get(1).last_extracted_at, self.old_time)
        for gid in (2, 3):
            self.assertIn(f"Item {gid}", self._get(gid).extracted_content)
            self.assertNotIn("not-json", self._get(gid).extracted_content)
            self.assertIsNone(self._get(gid).last_extracted_at)
        self.assertIn("New brand", self._get(4).extracted_content)
        self.assertGreater(self._get(4).last_extracted_at, self.old_time)
        self.assertIsNone(self._get(5).last_extracted_at)
        self.assertEqual(snapshots[-1].degraded, 3)

    async def test_failed_database_update_is_not_reported_as_success_or_degraded(self):
        manager = self._manager([{"goods_id": 1, "goods_name": "Item"}])
        for succeeded in (True, False):
            with self.subTest(succeeded=succeeded):
                service = ProductSyncService(self.knowledge, request_delay=0)
                with mock.patch("database.product_sync.ProductManager", return_value=manager), mock.patch.object(
                    service, "_snapshot_llm_profile", return_value=profile()
                ), mock.patch.object(service, "_extract_product_knowledge", new=mock.AsyncMock(
                    return_value=("Knowledge", succeeded)
                )), mock.patch.object(self.knowledge, "update_product_extracted_content", return_value=False):
                    result = await service.sync_shop(100, 1, "fake-account", is_full_sync=True)
                self.assertEqual((result.success, result.degraded, result.failed, result.current), (0, 0, 1, 1))

    def test_new_progress_field_keeps_existing_positional_contract(self):
        progress = SyncProgress(1, 1, 0, 0, "Item", True, "extracting")
        self.assertTrue(progress.cancelled)
        self.assertEqual(progress.phase, "extracting")
        self.assertEqual(progress.degraded, 0)


if __name__ == "__main__":
    unittest.main()
