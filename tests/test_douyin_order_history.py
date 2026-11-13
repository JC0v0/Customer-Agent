"""抖音订单号兜底必须读取真实存在的 Agent 会话历史。"""
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from Agent.CustomerAgent.custom.session_manager import AgentMessage, Base
from Channel.douyin.utils.API.order_query import OrderQuery


class DouyinOrderHistoryTests(unittest.TestCase):
    def test_extracts_order_ids_from_current_conversation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = create_engine(f"sqlite:///{Path(tmp) / 'agent.db'}")
            self.addCleanup(engine.dispose)
            Base.metadata.create_all(engine)
            sessions = sessionmaker(bind=engine)

            account_key = "douyin|douyin_shop|user-1"
            raw = f"{account_key}|customer|conv-1"
            session_id = f"conversation_{sha256(raw.encode('utf-8')).hexdigest()}"
            other_session_id = "conversation_other"

            with sessions() as session:
                session.add_all([
                    AgentMessage(
                        session_id=session_id,
                        role="user",
                        content="订单号 6912345678901234567 到哪了",
                    ),
                    AgentMessage(
                        session_id=session_id,
                        role="assistant",
                        content="订单号 6912345678901234568",
                    ),
                    AgentMessage(
                        session_id=other_session_id,
                        role="user",
                        content="订单号 6912345678901234569",
                    ),
                ])
                session.commit()

            with mock.patch(
                "Channel.douyin.utils.base_request.BaseRequest._init_account_info",
                lambda self: None,
            ), mock.patch(
                "database.db_manager",
                SimpleNamespace(Session=sessions),
            ):
                oq = OrderQuery(shop_id="douyin_shop", user_id="user-1")
                order_ids = oq._extract_order_ids(buyer_uid="conv-1", days=7)

            self.assertEqual(order_ids, ["6912345678901234567"])

            # Windows 下 SQLite 连接未释放会让临时目录删除报 WinError 32，
            # addCleanup 要等 with 块退出后才执行，这里先显式释放。
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
