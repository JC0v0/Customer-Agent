"""P2 / P3 / P4 回归测试：动作表、路由分派、预处理器。

锁住的核心不变量：

1. 每条消息都必须落到一个动作，**不允许静默丢弃**
2. 客服侧消息不进入 AI 回复，但可注入上下文
3. AI 白名单与动作表一致（此前 GOODS_CARD 有落差）
4. 图片保留 URL、空内容有兜底、表情缺描述有兜底
5. type=64 按结构探测，不产出全 null 字段
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridge.context import Context, ContextType, make_conversation_key  # noqa: E402
from Channel.pinduoduo.message_rules import (  # noqa: E402
    CUSTOMER_REPLY_CONTEXT_TYPES,
    Action,
    Origin,
    PDDMsgType,
    PDDSubType,
    classify,
    effective_action,
    resolve_origin,
)
from Channel.pinduoduo.pdd_message import PDDChatMessage  # noqa: E402
from Channel.pinduoduo.core.pdd_message_handler import MessageHandlerMixin  # noqa: E402
from Message.handlers.ai_handler import AIReplyHandler  # noqa: E402
from Message.handlers.preprocessor import MessagePreprocessor  # noqa: E402


def envelope(role, msg_type, sub_type=None, content="x", template=None, info=None):
    message = {"type": msg_type, "content": content}
    if role is not None:
        message["from"] = {"role": role, "uid": "u1"}
    if sub_type is not None:
        message["sub_type"] = sub_type
    if template:
        message["template_name"] = template
    if info:
        message["info"] = info
    return {"response": "push", "message": message}


class OriginTest(unittest.TestCase):
    def test_roles_map_to_origins(self):
        cases = [
            ("user", Origin.CUSTOMER),
            ("mall_cs", Origin.MERCHANT),
            ("system", Origin.SYSTEM),
            ("mall", Origin.SYSTEM),
        ]
        for role, expected in cases:
            self.assertIs(resolve_origin(role, "push"), expected, role)

    def test_missing_from_is_unknown_not_error(self):
        self.assertIs(resolve_origin(None, "push"), Origin.UNKNOWN)

    def test_system_push_response_without_from_is_system(self):
        self.assertIs(resolve_origin(None, "system_push"), Origin.SYSTEM)


class ClassifyTest(unittest.TestCase):
    def test_customer_reply_types(self):
        for t in (PDDMsgType.TEXT, PDDMsgType.IMAGE, PDDMsgType.EMOTION,
                  PDDMsgType.VIDEO, PDDMsgType.GOODS_SPEC):
            self.assertIs(classify(Origin.CUSTOMER, t), Action.REPLY, t)

    def test_customer_text_sub_types(self):
        self.assertIs(classify(Origin.CUSTOMER, PDDMsgType.TEXT, PDDSubType.GOODS_INQUIRY), Action.REPLY)
        self.assertIs(classify(Origin.CUSTOMER, PDDMsgType.TEXT, PDDSubType.ORDER_INFO), Action.REPLY)
        # 未知 sub_type 仍按文本处理，不能丢
        self.assertIs(classify(Origin.CUSTOMER, PDDMsgType.TEXT, 99), Action.REPLY)

    def test_withdraw_is_ignored_by_decision(self):
        self.assertIs(classify(Origin.CUSTOMER, PDDMsgType.WITHDRAW), Action.IGNORE)

    def test_customer_transfer_is_handoff(self):
        self.assertIs(classify(Origin.CUSTOMER, PDDMsgType.TRANSFER), Action.HANDOFF)

    def test_merchant_never_replies(self):
        """客服侧任何类型都不能产生 REPLY，否则会与人工抢答。"""
        for t in list(PDDMsgType):
            self.assertIsNot(classify(Origin.MERCHANT, t), Action.REPLY, t)

    def test_merchant_text_is_context_only(self):
        self.assertIs(classify(Origin.MERCHANT, PDDMsgType.TEXT), Action.CONTEXT_ONLY)

    def test_unknown_type_is_unknown_action(self):
        self.assertIs(classify(Origin.CUSTOMER, 4242), Action.UNKNOWN)

    def test_effective_action_degrades_unknown_to_context(self):
        self.assertIs(effective_action(Action.UNKNOWN), Action.CONTEXT_ONLY)
        self.assertIs(effective_action(Action.REPLY), Action.REPLY)

    def test_every_action_is_reachable_or_explicit(self):
        """动作表不应产生无法路由的动作。"""
        valid = set(Action)
        for origin in Origin:
            for t in list(PDDMsgType) + [4242]:
                self.assertIn(classify(origin, t), valid)


class ParserMetadataTest(unittest.TestCase):
    def test_origin_and_action_are_exposed(self):
        msg = PDDChatMessage(envelope("user", 0, content="hi"))
        self.assertIs(msg.origin, Origin.CUSTOMER)
        self.assertIs(msg.action, Action.REPLY)

    def test_merchant_card_keeps_goods_card(self):
        msg = PDDChatMessage(envelope("mall_cs", 0, sub_type=2, content="card"))
        self.assertIs(msg.user_msg_type, ContextType.GOODS_CARD)
        self.assertIs(msg.action, Action.CONTEXT_ONLY)

    def test_merchant_text_collapses_to_mall_cs(self):
        msg = PDDChatMessage(envelope("mall_cs", 0, content="hello"))
        self.assertIs(msg.user_msg_type, ContextType.MALL_CS)

    def test_merchant_transfer_is_parsed_not_short_circuited(self):
        """此前 mall_cs 短路使 handle_transfer 永不可达。"""
        msg = PDDChatMessage(envelope("mall_cs", PDDMsgType.TRANSFER, content="转移"))
        self.assertIs(msg.user_msg_type, ContextType.TRANSFER)
        self.assertIs(msg.action, Action.OBSERVE)

    def test_non_push_responses_are_observe(self):
        for resp in ("auth", "mall_system_msg", "system_push"):
            msg = PDDChatMessage({"response": resp, "message": {"type": 30, "content": "c"}})
            self.assertIs(msg.action, Action.OBSERVE, resp)

    def test_template_name_is_exposed(self):
        msg = PDDChatMessage(envelope("user", 0, template="user_goods_card",
                                      sub_type=0, content="c"))
        self.assertEqual(msg.template_name, "user_goods_card")


class GoodsSpecStructureTest(unittest.TestCase):
    """type=64 按结构探测，不产出全 null。"""

    NESTED = {
        "type": 64,
        "template_name": "product_manual",
        "content": "商品说明书",
        "info": {"key": "mall-product-manual-card", "data": {
            "sub_title": "点击查看使用/安装说明",
            "title": "商品说明书",
            "goods_info": {"goods_id": 622704236240, "goods_name": "葵花宝宝内热贴"},
        }},
    }

    def test_nested_snake_case_is_extracted(self):
        msg = PDDChatMessage({"response": "push", "message": dict(
            self.NESTED, **{"from": {"role": "user", "uid": "u"}})})
        self.assertIs(msg.user_msg_type, ContextType.GOODS_SPEC)
        self.assertEqual(msg.content["goods_id"], 622704236240)
        self.assertEqual(msg.content["goods_name"], "葵花宝宝内热贴")
        self.assertEqual(msg.content["title"], "商品说明书")

    def test_flat_camel_case_still_works(self):
        flat = {"type": 64, "content": "c", "info": {"data": {
            "goodsID": 111, "goodsName": "旧结构", "goodsPrice": "9.9", "spec": "规格"}}}
        msg = PDDChatMessage({"response": "push",
                              "message": dict(flat, **{"from": {"role": "user", "uid": "u"}})})
        self.assertEqual(msg.content["goods_id"], 111)
        self.assertEqual(msg.content["goods_spec"], "规格")

    def test_unknown_template_falls_back_to_raw_info(self):
        unknown = {"type": 64, "content": "c", "template_name": "brand_new_card",
                   "info": {"data": {"whatever": 1}}}
        msg = PDDChatMessage({"response": "push",
                              "message": dict(unknown, **{"from": {"role": "user", "uid": "u"}})})
        # 不产出全 null：至少保留原始 info
        self.assertIn("raw_info", msg.content)
        self.assertIsNotNone(msg.content["raw_info"])


class PreprocessorTest(unittest.TestCase):
    def setUp(self):
        self.pre = MessagePreprocessor()

    def test_image_keeps_url(self):
        url = "https://chat-img.pddugc.com/x.jpeg"
        out = self.pre.process(url, ContextType.IMAGE)
        self.assertIn(url, out)

    def test_video_stays_placeholder(self):
        self.assertEqual(self.pre.process("https://v/x.mp4", ContextType.VIDEO), "[视频消息]")

    def test_emotion_uses_description(self):
        self.assertEqual(self.pre.process("在的", ContextType.EMOTION), "在的")

    def test_emotion_without_description_has_fallback(self):
        self.assertEqual(self.pre.process(None, ContextType.EMOTION), "[表情]")
        self.assertEqual(self.pre.process("None", ContextType.EMOTION), "[表情]")

    def test_card_title_and_detail_are_extracted(self):
        """卡片的 title / text 之前提不出来，会导致整串 JSON 进模型。"""
        payload = json.dumps({
            "title": "待用户寄出",
            "text": "您的售后申请已通过",
            "goods_name": "鼻舒油",
            "order_id": "260915-088",
        }, ensure_ascii=False)
        out = self.pre.process(payload, ContextType.ORDER_INFO)
        self.assertNotIn("{", out)
        for token in ("待用户寄出", "您的售后申请已通过", "鼻舒油", "260915-088"):
            self.assertIn(token, out)

    def test_card_detail_not_duplicated_when_same_as_title(self):
        payload = json.dumps({"title": "说明书", "sub_title": "说明书"},
                             ensure_ascii=False)
        self.assertEqual(self.pre.process(payload, ContextType.GOODS_SPEC), "说明书")

    def test_empty_content_has_fallback(self):
        for empty in (None, ""):
            self.assertEqual(self.pre.process(empty, ContextType.TEXT),
                             "[收到一条无法解析的消息]")


class AiWhitelistAlignmentTest(unittest.TestCase):
    def test_default_reply_types_match_action_table(self):
        self.assertEqual(AIReplyHandler._default_reply_types(), set(CUSTOMER_REPLY_CONTEXT_TYPES))

    def test_every_customer_reply_type_passes_whitelist(self):
        types = AIReplyHandler._default_reply_types()
        for t, sub in [(0, None), (1, None), (5, None), (14, None),
                       (0, PDDSubType.GOODS_INQUIRY), (0, PDDSubType.ORDER_INFO)]:
            msg = PDDChatMessage(envelope("user", t, sub))
            self.assertIn(msg.user_msg_type, types, f"type={t} sub={sub}")

    def test_merchant_types_are_not_whitelisted(self):
        types = AIReplyHandler._default_reply_types()
        for t, sub in [(0, None), (0, 2), (5, None), (24, None)]:
            msg = PDDChatMessage(envelope("mall_cs", t, sub))
            self.assertNotIn(msg.user_msg_type, types, f"type={t} sub={sub}")


class _StubLogger:
    def __init__(self):
        self.records = []

    def _log(self, level, msg):
        self.records.append((level, str(msg)))

    def debug(self, msg, *a, **k):
        self._log("debug", msg)

    def info(self, msg, *a, **k):
        self._log("info", msg)

    def warning(self, msg, *a, **k):
        self._log("warning", msg)

    def error(self, msg, *a, **k):
        self._log("error", msg)

    def levels(self):
        return [lv for lv, _ in self.records]


class _StubQueue:
    def __init__(self):
        self.puts = []

    async def put(self, context):
        self.puts.append(context)
        return len(self.puts)


class _StubQueueManager:
    def __init__(self):
        self.queue = _StubQueue()

    def get_or_create_queue(self, name):
        return self.queue


class _StubAgent:
    def __init__(self):
        self.recorded = []

    async def record_context(self, context):
        self.recorded.append(context)
        return True


class _Harness(MessageHandlerMixin):
    """只提供分派所需的最小依赖，避免依赖 DB 与网络。"""

    def __init__(self):
        self.logger = _StubLogger()
        self.queue_manager = _StubQueueManager()
        self._account_agent = _StubAgent()
        self.immediate_calls = []

    async def _handle_immediate_message(self, context, shop_id, user_id):
        self.immediate_calls.append(context)


def make_context(msg: PDDChatMessage):
    # 与 _convert_to_context 保持一致：dict 内容会被序列化成 JSON 字符串，
    # 测试替身若用 "{}" 占位，就测不出「内容被解析层丢掉」这类问题。
    content = msg.content
    if isinstance(content, dict):
        content = json.dumps(content, ensure_ascii=False)
    return Context.create_pinduoduo_context(
        content="" if content is None else str(content),
        msg_id=str(msg.msg_id or ""),
        from_user=msg.from_user, from_uid=msg.from_uid,
        to_user=msg.to_user, to_uid=msg.to_uid,
        user_msg_type=msg.user_msg_type,
        shop_id="591119888", user_id="149439461",
        origin=msg.origin.value, action=msg.action.value,
    )


class DispatchTest(unittest.TestCase):
    """路由分派：每条消息都有归宿。"""

    def setUp(self):
        self.h = _Harness()

    def _dispatch(self, raw):
        msg = PDDChatMessage(raw)
        ctx = make_context(msg)
        asyncio.run(self.h._dispatch_by_action(ctx, "591119888", "149439461", "q", msg))
        return msg

    def test_reply_goes_to_queue(self):
        self._dispatch(envelope("user", 0, content="hi"))
        self.assertEqual(len(self.h.queue_manager.queue.puts), 1)

    def test_handoff_uses_immediate_path(self):
        self._dispatch(envelope("user", PDDMsgType.TRANSFER, content="转接"))
        self.assertEqual(len(self.h.immediate_calls), 1)
        self.assertEqual(len(self.h.queue_manager.queue.puts), 0)

    def test_context_only_is_recorded_not_queued(self):
        self._dispatch(envelope("mall_cs", 0, content="我方回复"))
        self.assertEqual(len(self.h._account_agent.recorded), 1)
        self.assertEqual(len(self.h.queue_manager.queue.puts), 0)

    def test_merchant_order_card_is_recorded_not_just_logged(self):
        """订单卡带售后状态与订单号，买家常紧接着追问，必须进上下文。"""
        info = {
            "title": "待用户寄出",
            "text": "您的售后申请已通过，点击售后详情可查看退货地址",
            "goods_info": {
                "goods_id": 651549062035,
                "goods_name": "葵花苍耳子鼻舒油",
                "order_sequence_no": "260915-088636209523086",
            },
        }
        msg = self._dispatch(envelope("mall_cs", PDDMsgType.ORDER_CARD,
                                      content="[待用户寄出]",
                                      template="order_card_auto_reply", info=info))
        # 不再被压成 MALL_CS，否则订单号与售后状态会整体丢失
        self.assertIs(msg.user_msg_type, ContextType.ORDER_INFO)
        self.assertIs(msg.action, Action.CONTEXT_ONLY)
        self.assertEqual(len(self.h.queue_manager.queue.puts), 0)
        self.assertEqual(len(self.h._account_agent.recorded), 1)
        recorded = self.h._account_agent.recorded[0].content
        for token in ("待用户寄出", "退货地址", "260915-088636209523086"):
            self.assertIn(token, recorded)

    def test_structured_context_is_summarized_not_raw_json(self):
        """结构化内容写进会话历史前必须摘要，不能塞整串 JSON。"""
        info = {"data": {"goods_info": {"goods_name": "晕车贴", "spec": "10 贴"}}}
        self._dispatch(envelope("mall_cs", PDDMsgType.GOODS_SPEC,
                                content="商品说明书",
                                template="product_manual", info=info))
        recorded = self.h._account_agent.recorded[0].content
        self.assertNotIn("{", recorded)
        self.assertIn("晕车贴", recorded)

    def test_plain_text_context_is_untouched(self):
        """纯文本不应被格式化逻辑改写。"""
        self._dispatch(envelope("mall_cs", 0, content="亲，8贴就是四对哦～"))
        self.assertEqual(self.h._account_agent.recorded[0].content,
                         "亲，8贴就是四对哦～")

    def test_ignored_message_is_logged(self):
        self._dispatch(envelope("user", PDDMsgType.WITHDRAW, content="撤回"))
        self.assertEqual(len(self.h.queue_manager.queue.puts), 0)
        self.assertEqual(len(self.h._account_agent.recorded), 0)
        self.assertIn("debug", self.h.logger.levels())

    def test_transfer_notice_is_logged_at_info(self):
        """默认日志级别是 INFO，转接记在 debug 等于没记。"""
        self._dispatch(envelope("mall_cs", PDDMsgType.TRANSFER,
                                content="小景 将该会话转移给 思雨，并留言：催发货"))
        self.assertIn("info", self.h.logger.levels())
        joined = " ".join(m for _, m in self.h.logger.records)
        self.assertIn("转移给", joined)

    def test_system_push_is_logged_at_warning(self):
        """异地登录提示需要运维可见。"""
        self._dispatch(envelope("system", PDDMsgType.SYSTEM_PUSH,
                                content="账户在别处登录 请刷新重登。"))
        self.assertIn("warning", self.h.logger.levels())

    def test_auth_is_logged_at_info(self):
        """auth 鉴权结果必须可见；内容已归一化为 JSON 字符串。"""
        msg = PDDChatMessage({"response": "auth", "auth": {"result": "ok"},
                              "uid": "cs_1_2"})
        ctx = Context.create_pinduoduo_context(
            content=json.dumps({"uid": "cs_1_2", "result": "ok", "status": 1}),
            user_msg_type=msg.user_msg_type,
            shop_id="1", user_id="2",
            origin=msg.origin.value, action=msg.action.value,
        )
        asyncio.run(self.h._dispatch_by_action(ctx, "1", "2", "q", msg))
        self.assertIn("info", self.h.logger.levels())
        joined = " ".join(m for _, m in self.h.logger.records)
        self.assertIn("ok", joined)

    def test_auth_result_extraction_handles_both_shapes(self):
        self.assertEqual(self.h._extract_auth_result({"result": "ok"}), "ok")
        self.assertEqual(self.h._extract_auth_result('{"result": "ok"}'), "ok")
        self.assertEqual(self.h._extract_auth_result("raw"), "raw")

    def test_unknown_type_is_not_silently_dropped(self):
        """核心不变量：未识别也不能什么都不做。"""
        msg = self._dispatch(envelope("user", 4242, content="未知"))
        self.assertIs(msg.action, Action.UNKNOWN)
        # 降级为 CONTEXT_ONLY：有记录或有日志，不能完全静默
        acted = len(self.h._account_agent.recorded) + len(self.h.logger.records)
        self.assertGreater(acted, 0)

    def test_unknown_type_emits_warning_with_details(self):
        """降级必须伴随告警，否则动作表缺项会悄悄消失、无从补表。"""
        self._dispatch(envelope("user", 4242, sub_type=7,
                                template="brand_new_card", content="未知"))
        warnings = [m for lv, m in self.h.logger.records if lv == "warning"]
        self.assertEqual(len(warnings), 1, self.h.logger.records)
        text = warnings[0]
        for token in ("4242", "brand_new_card", "sub_type=7", "customer"):
            self.assertIn(token, text)

    def test_known_action_does_not_warn(self):
        """正常消息不能被告警噪声污染。"""
        self._dispatch(envelope("user", 0, content="hi"))
        self.assertNotIn("warning", self.h.logger.levels())

    def test_no_message_is_dropped_for_any_observed_type(self):
        observed = [0, 1, 5, 8, 14, 20, 24, 30, 31, 41, 56, 64, 97, 1002]
        for role in ("user", "mall_cs", "system"):
            for t in observed:
                self.setUp()
                self._dispatch(envelope(role, t, content="c"))
                acted = (len(self.h.queue_manager.queue.puts)
                         + len(self.h._account_agent.recorded)
                         + len(self.h.immediate_calls)
                         + len(self.h.logger.records))
                self.assertGreater(acted, 0, f"被静默丢弃: role={role} type={t}")


class ContextRoleTest(unittest.TestCase):
    """CONTEXT_ONLY 写入会话历史时的角色必须由来源推导。"""

    @staticmethod
    def _ctx(origin):
        return Context.create_pinduoduo_context(
            content="c", from_uid="6840347447", to_uid="591119888",
            user_msg_type=ContextType.TEXT,
            shop_id="591119888", user_id="149439461", origin=origin,
        )

    def test_merchant_context_is_assistant(self):
        from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
        self.assertEqual(CustomerAgent._context_role(self._ctx("merchant")), "assistant")

    def test_customer_context_is_user(self):
        """type=41「当前用户来自 商品详情页」是买家侧，记成 assistant 会让模型认领。"""
        from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
        for origin in ("customer", "system", "unknown", None):
            self.assertEqual(
                CustomerAgent._context_role(self._ctx(origin)), "user", origin
            )

    def test_user_source_is_customer_context_only(self):
        msg = PDDChatMessage(envelope("user", PDDMsgType.USER_SOURCE,
                                      template="user_source",
                                      content="[当前用户来自 商品详情页]"))
        self.assertIs(msg.origin, Origin.CUSTOMER)
        self.assertIs(msg.action, Action.CONTEXT_ONLY)


class ConversationKeyTest(unittest.TestCase):
    """客服侧消息必须与买家侧落到同一个会话键。"""

    def _ctx(self, origin, from_uid, to_uid):
        return Context.create_pinduoduo_context(
            content="c", from_uid=from_uid, to_uid=to_uid,
            user_msg_type=ContextType.TEXT,
            shop_id="591119888", user_id="149439461",
            origin=origin, channel_type=None,
        )

    def test_merchant_uses_to_uid(self):
        from bridge.context import _context_value
        ctx = self._ctx("merchant", "591119888", "6840347447")
        customer = _context_value(ctx, "to_uid")
        self.assertEqual(make_conversation_key(ctx, customer),
                         make_conversation_key(self._ctx("customer", "6840347447", "591119888")))

    def test_customer_uses_from_uid(self):
        ctx = self._ctx("customer", "6840347447", "591119888")
        self.assertNotEqual(make_conversation_key(ctx),
                            make_conversation_key(ctx, "other"))


if __name__ == "__main__":
    unittest.main()
