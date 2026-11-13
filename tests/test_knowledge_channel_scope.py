"""知识库店铺解析必须支持按渠道限定，避免同名 shop_id 串店。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.knowledge_service import KnowledgeService
from database.models import Base, Channel, ProductKnowledge, Shop


class KnowledgeChannelScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        engine = create_engine(f"sqlite:///{Path(self.tmp.name) / 'kb.db'}")
        self.addCleanup(engine.dispose)
        sessions = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)

        with sessions() as session:
            pdd = Channel(channel_name="pinduoduo")
            dy = Channel(channel_name="douyin")
            session.add_all([pdd, dy])
            session.flush()
            pdd_shop = Shop(channel_id=pdd.id, shop_id="same", shop_name="示例店铺")
            dy_shop = Shop(channel_id=dy.id, shop_id="same", shop_name="示例店铺")
            session.add_all([pdd_shop, dy_shop])
            session.flush()
            session.add_all([
                ProductKnowledge(
                    shop_id=pdd_shop.id, goods_id=1, goods_name="拼多多商品"
                ),
                ProductKnowledge(
                    shop_id=dy_shop.id, goods_id=1, goods_name="抖音商品"
                ),
            ])
            session.commit()

        with mock.patch(
            "database.knowledge_service.db_manager",
            SimpleNamespace(Session=sessions, engine=engine),
        ):
            self.knowledge = KnowledgeService()

    def test_resolve_shop_id_honours_channel(self):
        self.assertEqual(
            self.knowledge._resolve_shop_id("same", channel_name="pinduoduo"),
            1,
        )
        self.assertEqual(
            self.knowledge._resolve_shop_id("same", channel_name="douyin"),
            2,
        )

    def test_search_knowledge_honours_channel(self):
        result = self.knowledge.search_knowledge(
            shop_id="same",
            goods_id=1,
            channel_name="douyin",
        )
        self.assertEqual(
            result["product_knowledge"][0].goods_name, "抖音商品"
        )

    def test_get_all_shops_preloads_channel_for_ui(self):
        """知识库下拉框要显示渠道，两个同名店铺必须能区分。

        真实故障（2026-11-13，打包版）：`shops` 表里拼多多与抖音各有一条
        「示例店铺」，下拉框只显示 `shop_name`，两条都叫「示例店铺」，
        给抖音店铺录知识时会选到拼多多那条，商品同步也走错渠道 provider。

        `get_all_shops` 若不预加载 `channel`，session 关闭后访问
        `shop.channel` 会抛 DetachedInstanceError，`_load_shops` 直接崩。
        """
        shops = self.knowledge.get_all_shops()
        self.assertEqual(len(shops), 2)

        labels = []
        for shop in shops:
            labels.append(
                "[{}] {} ({})".format(
                    shop.channel.channel_name, shop.shop_name, shop.shop_id
                )
            )

        self.assertEqual(len(set(labels)), 2, "同名店铺的显示标签必须能区分")
        self.assertTrue(any(label.startswith("[douyin]") for label in labels))
        self.assertTrue(any(label.startswith("[pinduoduo]") for label in labels))


class CustomerServiceKnowledgeToolTests(unittest.TestCase):
    """客服知识工具必须能处理抖音的字符串 shop_id。

    真实故障（2026-11-13）：`SearchCustomerServiceKnowledgeParams.shop_id`
    声明为 `int`，而抖音 shop_id 是 `douyin_示例店铺`。Pydantic 在工具函数
    执行前就抛 ValidationError，`tool_decorator` 统一吞成
    「[工具执行失败，请稍后重试]」——抖音会话里客服知识永远查不出来，
    日志上也看不到具体原因。
    """

    def test_accepts_douyin_string_shop_id(self):
        from Agent.CustomerAgent.tools.search_customer_service_knowledge import (
            SearchCustomerServiceKnowledgeParams,
        )

        params = SearchCustomerServiceKnowledgeParams(
            channel_type="douyin",
            query="退货",
            shop_id="douyin_示例店铺",
        )
        self.assertEqual(params.shop_id, "douyin_示例店铺")

    def test_passes_channel_name_to_knowledge_service(self):
        from unittest import mock

        from Agent.CustomerAgent.tools import search_customer_service_knowledge as tool

        captured = {}

        class FakeService:
            def search_knowledge(self, **kwargs):
                captured.update(kwargs)
                return {"product_knowledge": [], "customer_service_knowledge": []}

            def format_search_result(self, result):
                return "ok"

        params = tool.SearchCustomerServiceKnowledgeParams(
            channel_type="douyin",
            query="退货",
            shop_id="douyin_示例店铺",
        )
        with mock.patch.object(tool, "_get_knowledge_service", return_value=FakeService()):
            tool.search_customer_service_knowledge(params)

        self.assertEqual(captured.get("channel_name"), "douyin")
        self.assertEqual(captured.get("shop_id"), "douyin_示例店铺")


if __name__ == "__main__":
    unittest.main()
