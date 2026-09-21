"""拼多多消息解析（P1 修正）的回归测试。

覆盖三处已修缺陷，并锁住边界行为，避免回归：
  1. 表情 content 取值路径缺 message 前缀 -> 恒为 None
  2. timestamp 双重缺陷（字段名错 + 从未赋值）-> 恒为 None
  3. sub_type=2 商品卡片被误判为 text

另锁住「撤回本轮不处理」这一决策，避免后续被无意改动。
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Channel.pinduoduo.pdd_message import (  # noqa: E402
    PDDChatMessage,
    PDDMsgType,
    PDDSubType,
)
from bridge.context import ContextType  # noqa: E402


def push(message: dict) -> dict:
    """包成服务端下发的 push 信封。"""
    return {"response": "push", "message": message}


def from_customer(message: dict) -> dict:
    """标记为买家发来的消息。"""
    m = dict(message)
    m["from"] = {"role": "user", "uid": "6840347447"}
    m["to"] = {"role": "mall_cs", "uid": "591119888"}
    return push(m)


class EmotionContentTest(unittest.TestCase):
    """买家表情的 content 必须能取到 info.description。"""

    def test_emotion_description_is_extracted(self):
        msg = from_customer({
            "type": 5,
            "content": "[在吗]",
            "info": {"description": "在的", "id": "duoduoji01", "index": 55},
        })
        parsed = PDDChatMessage(msg)
        self.assertEqual(parsed.user_msg_type, ContextType.EMOTION)
        self.assertEqual(parsed.content, "在的")

    def test_emotion_without_info_does_not_raise(self):
        parsed = PDDChatMessage(from_customer({"type": 5, "content": "[x]"}))
        self.assertEqual(parsed.user_msg_type, ContextType.EMOTION)
        self.assertIsNone(parsed.content)

    def test_emotion_missing_message_prefix_would_fail(self):
        """回归护栏：路径必须带 message 前缀。

        若有人把路径改回 ("info", "description")，该断言会失败。
        """
        source = (REPO_ROOT / "Channel" / "pinduoduo" / "pdd_message.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('("message", "info", "description")', source)
        self.assertNotIn('ContextType.EMOTION, ("info", "description")', source)


class TimestampTest(unittest.TestCase):
    """timestamp 必须从 message.ts 取值并真正赋给实例。"""

    def test_ts_is_read(self):
        parsed = PDDChatMessage(from_customer({"type": 0, "content": "hi", "ts": "1789897477"}))
        self.assertEqual(parsed.timestamp, "1789897477")

    def test_legacy_time_field_still_works(self):
        parsed = PDDChatMessage(from_customer({"type": 0, "content": "hi", "time": "1700000000"}))
        self.assertEqual(parsed.timestamp, "1700000000")

    def test_ts_takes_precedence_over_time(self):
        parsed = PDDChatMessage(
            from_customer({"type": 0, "content": "hi", "ts": "111", "time": "222"})
        )
        self.assertEqual(parsed.timestamp, "111")

    def test_missing_timestamp_is_none(self):
        parsed = PDDChatMessage(from_customer({"type": 0, "content": "hi"}))
        self.assertIsNone(parsed.timestamp)


class GoodsCardSubTypeTest(unittest.TestCase):
    """sub_type=2 是我方商品卡片，不应落入纯文本分支。"""

    CARD = {
        "type": 0,
        "sub_type": 2,
        "template_name": "goods_info_card",
        "content": "https://mobile.yangkeduo.com/goods.html?goods_id=627273454388",
        "info": {
            "goodsID": 627273454388,
            "goodsName": "葵花晕车贴",
            "goodsPrice": "10.9",
            "goodsThumbUrl": "https://img.pddpic.com/x.jpeg",
            "linkUrl": "goods.html?goods_id=627273454388",
        },
    }

    def test_sub_type_2_maps_to_goods_card(self):
        parsed = PDDChatMessage(from_customer(self.CARD))
        self.assertEqual(parsed.user_msg_type, ContextType.GOODS_CARD)
        self.assertNotEqual(parsed.user_msg_type, ContextType.TEXT)

    def test_goods_card_fields_are_extracted(self):
        parsed = PDDChatMessage(from_customer(self.CARD))
        self.assertIsInstance(parsed.content, dict)
        self.assertEqual(parsed.content["goods_id"], 627273454388)
        self.assertEqual(parsed.content["goods_name"], "葵花晕车贴")

    def test_sub_type_0_still_goods_inquiry(self):
        """对照：sub_type=0 仍是买家商品咨询。"""
        card = dict(self.CARD)
        card["sub_type"] = 0
        parsed = PDDChatMessage(from_customer(card))
        self.assertEqual(parsed.user_msg_type, ContextType.GOODS_INQUIRY)

    def test_sub_type_1_still_order_info(self):
        parsed = PDDChatMessage(from_customer({
            "type": 0,
            "sub_type": 1,
            "info": {"orderSequenceNo": "260915-088636209523086"},
        }))
        self.assertEqual(parsed.user_msg_type, ContextType.ORDER_INFO)

    def test_unknown_sub_type_falls_back_to_text(self):
        parsed = PDDChatMessage(from_customer({"type": 0, "sub_type": 99, "content": "hi"}))
        self.assertEqual(parsed.user_msg_type, ContextType.TEXT)


class WithdrawDecisionTest(unittest.TestCase):
    """撤回本轮明确不处理，此测试锁定该决策。"""

    def test_withdraw_path_is_unchanged(self):
        source = (REPO_ROOT / "Channel" / "pinduoduo" / "pdd_message.py").read_text(
            encoding="utf-8"
        )
        # 仍是缺少 message 前缀的旧路径
        self.assertIn('("info", "withdraw_hint")', source)

    def test_unknown_type_is_not_dropped(self):
        """未知 type 目前落到 SYSTEM_STATUS 占位，不抛异常。"""
        parsed = PDDChatMessage(from_customer({"type": 999, "content": "x"}))
        self.assertEqual(parsed.user_msg_type, ContextType.SYSTEM_STATUS)
        self.assertIn("999", str(parsed.content))


class EnumCoverageTest(unittest.TestCase):
    """枚举应覆盖实测观察到的全部类型。"""

    OBSERVED = [0, 1, 5, 8, 14, 20, 24, 30, 31, 41, 56, 64, 97, 1002]

    def test_all_observed_types_are_defined(self):
        defined = {int(t) for t in PDDMsgType}
        missing = sorted(set(self.OBSERVED) - defined)
        self.assertEqual(missing, [], f"未覆盖的实测类型: {missing}")

    def test_sub_types_are_defined(self):
        self.assertEqual(int(PDDSubType.GOODS_INQUIRY), 0)
        self.assertEqual(int(PDDSubType.ORDER_INFO), 1)
        self.assertEqual(int(PDDSubType.CS_GOODS_CARD), 2)


if __name__ == "__main__":
    unittest.main()
