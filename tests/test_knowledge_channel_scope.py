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
            pdd_shop = Shop(channel_id=pdd.id, shop_id="same", shop_name="拼多多店")
            dy_shop = Shop(channel_id=dy.id, shop_id="same", shop_name="抖音店")
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


if __name__ == "__main__":
    unittest.main()
