"""「下载即用」骨架资产与登录录帧测试。

背景：pigeon 通道原本要求用户机器上预先放好手工抓包文件，否则连不上。
改造后：
1. 包内脱敏骨架（无凭据）随程序分发，兜底 10018 发送帧和接收 body；
2. 登录流程自动录制 WS 帧序列（握手/注册帧，按账号隔离落盘）；
3. 传输层运行时优先用真实录帧，缺发送帧时回退骨架重建。

这里锁住：骨架确实不含凭据、骨架回退能重建出可用帧、录帧按账号隔离。
"""
import base64
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Channel.douyin.session_paths import (
    existing_pigeon_capture_path,
    pigeon_capture_path,
)
from tools.reverse_probe.douyin.pigeon_assets import (
    enter_skeleton_path,
    load_enter_skeleton_frame,
    load_send_skeleton_frame,
    load_start_skeleton_frame,
    load_usermsg_skeleton,
    read_capture_frames,
    send_skeleton_path,
    start_skeleton_path,
    usermsg_skeleton_path,
)
from tools.reverse_probe.douyin.pigeon_proto import (
    get_bytes_field,
    parse_fields,
)
from tools.reverse_probe.douyin.pigeon_transport import (
    PigeonSession,
    PigeonTransport,
    PigeonTemplateError,
    _kvmap,
    _kvmap_field,
)


class SkeletonAssetTests(unittest.TestCase):
    def test_skeleton_files_exist_and_load(self):
        self.assertTrue(send_skeleton_path().exists(), "发送帧骨架文件缺失")
        self.assertTrue(usermsg_skeleton_path().exists(), "接收 body 骨架文件缺失")
        self.assertIsInstance(load_send_skeleton_frame(), bytes)
        self.assertIsNotNone(load_usermsg_skeleton().get("body_b64"))

    def test_send_skeleton_carries_no_credentials(self):
        """骨架里的会话绑定字段必须是空值——这是它能够入库/分发的前提。"""
        frame = load_send_skeleton_frame()
        f8 = get_bytes_field(frame, 8)
        self.assertIsNotNone(f8, "骨架不是预期的 10018 帧结构")
        # f4 = token，必须清空
        self.assertFalse(get_bytes_field(f8, 4), "骨架里残留 token")
        # f8 层 KV 里的 device / sign 必须清空
        sess_kv = _kvmap(f8)
        for key in ("session_did", "pigeon_sign"):
            self.assertFalse(sess_kv.get(key), f"骨架里残留 {key}")
        # 消息体 KV 里的店铺/买家标识必须清空
        inner = get_bytes_field(f8, 8)
        payload = get_bytes_field(inner, 100)
        self.assertIsNotNone(payload, "骨架缺少 f8.f8.f100 消息体")
        kv = _kvmap(payload)
        for key in ("shop_id", "security_receiver_id"):
            self.assertFalse(kv.get(key), f"骨架里残留 {key}")

    def test_fallback_transport_raises_without_capture(self):
        """没有任何录帧文件时，握手/注册帧无从取得，必须抛可操作异常。"""
        session = PigeonSession("tok", "sign")
        with self.assertRaises(PigeonTemplateError) as ctx:
            PigeonTransport(session, "no_such_capture.json")
        self.assertIn("重新登录", str(ctx.exception))

    def test_fallback_transport_rebuilds_message_frame(self):
        """录帧缺 10018 时用骨架兜底，重建帧注入当前会话凭据。"""
        with tempfile.TemporaryDirectory() as tmp:
            # 最小握手/注册帧：顶层 f1=cmd
            def frame_b64(cmd):
                value = bytes([0x08]) + _varint(cmd)
                return base64.b64encode(value).decode("ascii")

            capture = Path(tmp) / "cap.json"
            capture.write_text(json.dumps({
                "ws_sent": [
                    {"b64": frame_b64(10008)},
                    {"b64": frame_b64(10009)},
                    {"b64": frame_b64(10011)},
                ],
            }), encoding="utf-8")

            session = PigeonSession(
                "TOKEN_NEW_abcdef", "SIGN_NEW_xyz", device_id="DEV_NEW_123"
            )
            transport = PigeonTransport(session, str(capture))
            self.assertTrue(
                transport.used_send_skeleton,
                "录帧里没有 10018，应标记为骨架兜底",
            )
            frame = transport._rebuild_message_frame(
                transport._text_cmd, "7123456789", "你好测试", "BUYER_987",
                "SHOP_555", "T" * 60, 42,
            )
            self.assertIsNotNone(frame)
            self.assertIn(b"TOKEN_NEW_abcdef", frame, "token 未注入")
            self.assertIn(b"SIGN_NEW_xyz", frame, "pigeon_sign 未注入")
            self.assertIn(b"DEV_NEW_123", frame, "device_id 未注入")
            self.assertIn("你好测试".encode("utf-8"), frame, "文本未写入")
            # 模板里的旧凭据不得残留
            self.assertNotIn(b"100000001", frame)


class StartConversationSkeletonTests(unittest.TestCase):
    """10016（开会话）骨架：缺了它，连接和收消息都正常，但一条也发不出去。

    2026-11-13 真机：登录录帧只有握手/注册帧，10018 有骨架兜底、
    10016 没有，于是每次回复都报 ``missing_start_template``。
    """

    def test_start_skeleton_file_exists_and_has_no_credentials(self):
        self.assertTrue(start_skeleton_path().exists(), "开会话帧骨架文件缺失")
        frame = load_start_skeleton_frame()
        self.assertIsInstance(frame, bytes)
        f8 = get_bytes_field(frame, 8)
        self.assertIsNotNone(f8, "骨架不是预期的 10016 帧结构")
        self.assertFalse(get_bytes_field(f8, 4), "骨架里残留 token")
        sess_kv = _kvmap(f8)
        for key in ("session_did", "pigeon_sign"):
            self.assertFalse(sess_kv.get(key), f"骨架里残留 {key}")
        inner = get_bytes_field(f8, 8)
        payload = get_bytes_field(inner, 100)
        self.assertIsNotNone(payload, "骨架缺少 f8.f8.f100 消息体")
        kv = _kvmap(payload)
        for key in ("shop_id", "security_receiver_id",
                    "s:client_message_id", "p:check_Send"):
            self.assertFalse(kv.get(key), f"骨架里残留 {key}")

    def test_rebuild_start_frame_from_skeleton(self):
        """只有握手帧的录帧 + 包内骨架，必须能重建出开会话帧。"""
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "cap.json"
            capture.write_text(json.dumps({
                "ws_sent": [{"b64": _frame_b64(10010)}, {"b64": _frame_b64(10011)}],
            }), encoding="utf-8")

            session = PigeonSession(
                "TOKEN_NEW_abcdef", "SIGN_NEW_xyz", device_id="DEV_NEW_123"
            )
            transport = PigeonTransport(session, str(capture))
            self.assertTrue(transport.used_start_skeleton, "无 10016 时应标记骨架兜底")

            frame = transport._rebuild_start_conversation_frame(
                "7123456789", "BUYER_987", "SHOP_555", "T" * 60, 42
            )
            self.assertIsNotNone(frame, "开会话帧重建失败（会报 missing_start_template）")
            self.assertIn(b"TOKEN_NEW_abcdef", frame, "token 未注入")
            self.assertIn(b"SIGN_NEW_xyz", frame, "pigeon_sign 未注入")

            f8 = get_bytes_field(frame, 8)
            inner = get_bytes_field(f8, 8)
            payload = get_bytes_field(inner, 100)
            kv = _kvmap(payload)
            self.assertEqual(kv.get("shop_id"), "SHOP_555", "shop_id 未写入")
            self.assertEqual(
                kv.get("security_receiver_id"), "BUYER_987",
                "security_receiver_id 未写入",
            )
            # conv_id(f3) / short_id(f14) / 会话票据(f7) 必须换成当前会话的
            fields = {}
            for num, wt, value, _ in parse_fields(payload):
                fields[num] = value
            self.assertEqual(fields.get(3), 7123456789, "conv_id 未写入")
            self.assertEqual(fields.get(14), 42, "conversation_short_id 未写入")
            self.assertEqual(fields.get(7), b"T" * 60, "会话票据未写入")

    def test_missing_both_skeletons_raises(self):
        """骨架也缺时必须抛可操作异常，而不是静默发不出去。"""
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "cap.json"
            capture.write_text(json.dumps({
                "ws_sent": [{"b64": _frame_b64(10010)}],
            }), encoding="utf-8")
            with mock.patch(
                "tools.reverse_probe.douyin.pigeon_transport.load_start_skeleton_frame",
                return_value=None,
            ):
                with self.assertRaises(PigeonTemplateError) as ctx:
                    PigeonTransport(PigeonSession("tok", "sign"), str(capture))
                self.assertIn("开会话帧(10016)", str(ctx.exception))


class EnterSkeletonTests(unittest.TestCase):
    """进会话帧（10014）骨架：协议发送取票据的前置帧。

    这张帧的回包带「目标会话的票据 + 会话短 ID」，正文帧缺这两个值会被
    服务端 ACK 后静默丢弃。骨架进版本库前必须清掉 token / device_id /
    pigeon_sign，运行时由 ``_refresh_session_frame`` 按当前登录会话补回。
    """

    def test_enter_skeleton_exists_and_carries_no_credentials(self):
        self.assertTrue(enter_skeleton_path().exists(), "进会话帧骨架文件缺失")
        frame = load_enter_skeleton_frame()
        self.assertIsInstance(frame, bytes)

        # 顶层 f5 的 pigeon_sign 必须清空
        self.assertFalse(_kvmap(frame).get("pigeon_sign"), "骨架顶层残留 pigeon_sign")

        f8 = get_bytes_field(frame, 8)
        self.assertIsNotNone(f8, "骨架不是预期的进会话帧结构")
        self.assertFalse(get_bytes_field(f8, 4), "骨架里残留 token")
        self.assertFalse(get_bytes_field(f8, 9), "骨架里残留 device_id")
        for key in ("pigeon_sign", "session_did", "session_aid"):
            self.assertFalse(
                _kvmap_field(f8, 15).get(key), f"骨架里残留 {key}"
            )

    def test_enter_skeleton_uses_placeholder_conversation(self):
        """会话号/买家键换成占位值，不能带真实买家信息进版本库。"""
        frame = load_enter_skeleton_frame()
        f8 = get_bytes_field(frame, 8)
        inner = get_bytes_field(f8, 8)
        payload = get_bytes_field(inner, 610)
        wrapper = get_bytes_field(payload, 1)
        self.assertIsNotNone(wrapper, "骨架缺少 f8.f8.f610.f1 载荷")

        conv = None
        buyer = b""
        for f, wt, v, _ in parse_fields(wrapper):
            if f == 2 and wt == 0:
                conv = int(v)
            elif f == 20 and wt == 2:
                buyer = v
        self.assertEqual(conv, 7000000000000000001, "会话号占位值被改动")
        self.assertEqual(buyer, b"A" * 88, "买家键占位值被改动")

    def test_rebuild_injects_session_and_target_conversation(self):
        """骨架重建后必须补回当前会话凭据，并写入目标会话号与买家键。"""
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "cap.json"
            capture.write_text(json.dumps({
                "ws_sent": [{"b64": _frame_b64(10008)}],
            }), encoding="utf-8")

            session = PigeonSession(
                "TOKEN_ENTER_abc", "SIGN_ENTER_xyz", device_id="DEV_ENTER_123"
            )
            transport = PigeonTransport(session, str(capture))
            transport._enter_template = load_enter_skeleton_frame()

            frame = transport._rebuild_enter_frame(
                "7123456789", "BUYER_987", "SHOP_555"
            )
            self.assertIsNotNone(frame, "骨架重建失败")
            self.assertIn(b"TOKEN_ENTER_abc", frame, "token 未注入")
            self.assertIn(b"SIGN_ENTER_xyz", frame, "pigeon_sign 未注入")
            self.assertIn(b"DEV_ENTER_123", frame, "device_id 未注入")

            f8 = get_bytes_field(frame, 8)
            inner = get_bytes_field(f8, 8)
            wrapper = get_bytes_field(get_bytes_field(inner, 610), 1)
            conv = None
            buyer = b""
            for f, wt, v, _ in parse_fields(wrapper):
                if f == 2 and wt == 0:
                    conv = int(v)
                elif f == 20 and wt == 2:
                    buyer = v
            self.assertEqual(conv, 7123456789, "目标会话号未写入")
            self.assertIn(b"BUYER_987", buyer, "买家键未写入")
            self.assertNotIn(b"7000000000000000001", frame, "占位会话号残留")


def _frame_b64(cmd):
    value = bytes([0x08]) + _varint(cmd)
    return base64.b64encode(value).decode("ascii")


class HandshakeRenumberingTests(unittest.TestCase):
    """握手帧命令号随客户端版本变，不能写死。

    2026 年三次实测：9 月旧协议 10008→10009；headless 新连接 10010；
    11-13 真机 GUI 登录又变成 10005→10006。写死白名单的结果就是每次
    抖音改版，用户都被「缺少抓包文件」挡在门外。
    """

    def test_is_handshake_cmd_covers_all_known_generations(self):
        from tools.reverse_probe.douyin.pigeon_assets import is_handshake_cmd

        for cmd in (10005, 10006, 10008, 10009, 10010):
            self.assertTrue(is_handshake_cmd(cmd), f"{cmd} 应算握手帧")
        for cmd in (10011, 10013, 10016, 10018, 10024):
            self.assertFalse(is_handshake_cmd(cmd), f"{cmd} 不是握手帧")

    def test_transport_sends_new_generation_handshake_in_capture_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "cap.json"
            capture.write_text(json.dumps({
                "ws_sent": [
                    {"b64": _frame_b64(10005)},
                    {"b64": _frame_b64(10006)},
                    {"b64": _frame_b64(10011)},
                ],
            }), encoding="utf-8")

            transport = PigeonTransport(PigeonSession("tok", "sign"), str(capture))
            from tools.reverse_probe.douyin.pigeon_transport import _cmd_of

            self.assertEqual(
                [_cmd_of(f) for f in transport._handshake], [10005, 10006],
                "新代握手指令必须按录制顺序发出",
            )
            self.assertEqual(_cmd_of(transport._register), 10011)

    def test_positional_fallback_when_no_cmd_looks_like_handshake(self):
        """整条编号规则都变了（握手帧号也 > 10011）时，按顺序取前两帧当握手。

        位置比编号稳定：浏览器新建连接的头两帧必然是认证握手。
        """
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "cap.json"
            capture.write_text(json.dumps({
                "ws_sent": [
                    {"b64": _frame_b64(10031)},
                    {"b64": _frame_b64(10032)},
                    {"b64": _frame_b64(10040)},
                ],
            }), encoding="utf-8")

            transport = PigeonTransport(PigeonSession("tok", "sign"), str(capture))
            from tools.reverse_probe.douyin.pigeon_transport import _cmd_of

            self.assertEqual(
                [_cmd_of(f) for f in transport._handshake], [10031, 10032],
                "编号全变时也要能按录制顺序取到握手帧",
            )

    def test_shell_10018_is_not_used_as_send_template(self):
        """页面加载时录到的 10018 可能是没有业务载荷的空壳帧。

        拿它当发送模板会发出空包（服务端静默丢弃），所以必须回退骨架。
        """
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "cap.json"
            capture.write_text(json.dumps({
                "ws_sent": [
                    {"b64": _frame_b64(10005)},
                    {"b64": _frame_b64(10011)},
                    {"b64": _frame_b64(10018)},
                ],
            }), encoding="utf-8")

            transport = PigeonTransport(PigeonSession("tok", "sign"), str(capture))
            self.assertTrue(
                transport.used_send_skeleton,
                "空壳 10018 不应被当成发送模板",
            )
            frame = transport._rebuild_message_frame(
                transport._text_cmd, "7123456789", "你好", "BUYER_1",
                "SHOP_1", "T" * 60, 42,
            )
            self.assertIsNotNone(frame, "回退骨架后仍无法重建发送帧")


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)






class DimensionMaskingTests(unittest.TestCase):
    """骨架里的 hierarchical_dimension 必须已做等长掩码。

    早期脱敏脚本把它当成「不含账号身份的客户端画像」保留，实际值是真实
    抓包带下来的设备/买家维度串，随包分发会把上一个账号的指纹带给所有
    新用户。掩码只换数字、不改长度，protobuf 布局保持不变。
    """

    def _skeleton_blobs(self):
        blobs = []
        paths = [
            send_skeleton_path(),
            start_skeleton_path(),
            enter_skeleton_path(),
            usermsg_skeleton_path(),
        ]
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            key = "body_b64" if "body_b64" in data else "b64"
            blobs.append((path.name, base64.b64decode(data[key])))
        return blobs

    def test_assets_carry_no_unmasked_dimension_ids(self):
        pattern = re.compile(rb"\d{4}(?:[_-]\d{4}){7,}")
        for name, blob in self._skeleton_blobs():
            for match in pattern.finditer(blob):
                for byte in match.group(0):
                    self.assertIn(
                        byte, b"0_-",
                        "%s 里残留未掩码的真实维度值" % name,
                    )

    def test_mask_keeps_length_and_is_idempotent(self):
        from tools.reverse_probe.douyin.strip_capture import _mask_dimension_ids

        sample = (
            b'{"dynamic_dimension":"6972_9230_4541_1131_8272_6599_9420_8656_'
            b'0925_4050_3823_3994_2997"}'
        )
        once = _mask_dimension_ids(sample)
        self.assertEqual(len(once), len(sample), "掩码改变了字节数")
        self.assertNotIn(b"6972", once, "真实维度值未被掩码")
        self.assertIn(b"dynamic_dimension", once, "字段名不应被改掉")
        self.assertEqual(_mask_dimension_ids(once), once, "掩码不幂等")

    def test_scrub_skeleton_file_masks_existing_asset(self):
        from tools.reverse_probe.douyin.strip_capture import scrub_skeleton_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.json"
            raw = (
                b'{"dynamic_dimension":"6972_9230_4541_1131_8272_6599_9420_8656_'
                b'0925_4050_3823_3994_2997"}'
            )
            path.write_text(json.dumps({
                "desc": "test",
                "b64": base64.b64encode(raw).decode("ascii"),
            }), encoding="utf-8")

            count = scrub_skeleton_file(str(path))
            self.assertEqual(count, 1, "应命中 1 处维度串")

            data = json.loads(path.read_text(encoding="utf-8"))
            scrubbed = base64.b64decode(data["b64"])
            self.assertEqual(len(scrubbed), len(raw), "二次脱敏改变了字节数")
            self.assertNotIn(b"6972", scrubbed)
            self.assertEqual(scrub_skeleton_file(str(path)), 0, "重复脱敏不应再改动")


if __name__ == "__main__":
    unittest.main()
