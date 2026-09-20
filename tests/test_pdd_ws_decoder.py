"""拼多多 WS 抓包解码器的回归测试。

这套协议来自真实抓包反推，解析逻辑踩过两个坑，用测试固定住：
  1. WebSocket opcode 2(binary) 是数据帧，不能当成协议控制帧
  2. titan.notifyDataLite 的负载「大块 gzip、小块明文」两种变体都要支持

同时验证「通知 ↔ ACK」配对依赖 seq_id 与数字序号双重一致。
"""

import base64
import gzip
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DECODER_PATH = REPO_ROOT / "scripts" / "decode_pdd_ws.py"

# 按路径加载脚本模块，避免依赖 scripts/ 是否是包
_spec = importlib.util.spec_from_file_location("decode_pdd_ws", DECODER_PATH)
decoder = importlib.util.module_from_spec(_spec)
sys.modules["decode_pdd_ws"] = decoder
_spec.loader.exec_module(decoder)


# ---------------------------------------------------------------------------
# 构造测试用帧的辅助函数（protobuf 最小编码器）
# ---------------------------------------------------------------------------
def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return bytes(out)


def _field_varint(field: int, value: int) -> bytes:
    return _varint((field << 3) | 0) + _varint(value)


def _field_bytes(field: int, value: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(value)) + value


def _frame(body: bytes, corr_id: int = 101, version: int = 10, flags: int = 1) -> bytes:
    """组装完整二进制帧：16 字节大端头 + body。"""
    return struct.pack(">HHIII", version, 102, corr_id, flags, len(body)) + body


def _notify_body(seq_id: str, msg_seq: int, notification: dict, compress: bool) -> bytes:
    inner = (
        _field_bytes(1, b"149439461")
        + _field_varint(3, 5)
        + _field_varint(4, msg_seq)
        + _field_bytes(5, seq_id.encode("ascii"))
        + _field_bytes(6, json.dumps(notification, ensure_ascii=False).encode("utf-8"))
    )
    payload = gzip.compress(inner) if compress else inner
    return (
        _field_bytes(1, b"titan.notifyDataLite")
        + _field_varint(2, 0)
        + _field_bytes(10, payload)
        + _field_varint(13, 0)
        + _field_varint(14, 0)
    )


def _ack_body(seq_id: str, msg_seq: int) -> bytes:
    inner = _field_bytes(1, seq_id.encode("ascii")) + _field_varint(2, msg_seq)
    return (
        _field_varint(1, 2)
        + _field_bytes(2, b"titan.notifyDataLite.ack")
        + _field_varint(3, 1)
        + _field_varint(4, 0)
        + _field_bytes(9, inner)
        + _field_varint(11, 103)
    )


def _row(event: str, frame: bytes, ts: str, request_id: str = "1") -> dict:
    return {
        "ts": ts,
        "event": event,
        "requestId": request_id,
        "url": "wss://titan-ws.pinduoduo.com/?access_token=<REDACTED>",
        "opcode": 2,
        "opcodeName": "binary",
        "len": len(frame),
        "payload": base64.b64encode(frame).decode("ascii"),
    }


def _notification(message: dict) -> dict:
    return {
        "push_type": 2,
        "push_data": {"seq_type": 0, "seq_id": 0, "data": [{"chat_type_id": 1, "message": message}]},
        "target_id": "",
        "custom_data": "",
    }


class ProtobufParsingTest(unittest.TestCase):
    def test_varint_roundtrip(self):
        for value in (0, 1, 127, 128, 300, 20007, 2 ** 32 - 1):
            encoded = _varint(value)
            self.assertEqual(decoder.read_varint(encoded, 0), (value, len(encoded)))

    def test_varint_truncated_raises(self):
        with self.assertRaises(decoder.DecodeError):
            decoder.read_varint(b"\x80\x80", 0)

    def test_length_overflow_raises(self):
        # 声明 100 字节但实际只有 2 字节
        with self.assertRaises(decoder.DecodeError):
            decoder.parse_message(_varint((1 << 3) | 2) + _varint(100) + b"ab")

    def test_unsupported_wire_type_raises(self):
        with self.assertRaises(decoder.DecodeError):
            decoder.parse_message(_varint((1 << 3) | 3))


class FrameDecodingTest(unittest.TestCase):
    def _decode_single(self, event: str, frame: bytes) -> dict:
        return decoder.decode_binary_frame(event, frame)

    def test_gzip_variant(self):
        seq = "1789893033491#fb13a7bd"
        body = _notify_body(seq, 20007, _notification({"type": 0, "content": "在吗"}), compress=True)
        decoded = self._decode_single("frameReceived", _frame(body))
        self.assertEqual(decoded["method"], "titan.notifyDataLite")
        self.assertEqual(decoded["seq_id"], seq)
        self.assertEqual(decoded["msg_seq"], 20007)
        self.assertTrue(decoded["compressed"])
        self.assertNotIn("error", decoded)

    def test_plain_variant(self):
        seq = "2_20023_5163e4de-e7e0-433e-a369-268bbd4f4716"
        body = _notify_body(seq, 20023, _notification({"type": 5, "content": "[在吗]"}), compress=False)
        decoded = self._decode_single("frameReceived", _frame(body))
        self.assertEqual(decoded["seq_id"], seq)
        self.assertEqual(decoded["msg_seq"], 20023)
        self.assertFalse(decoded["compressed"])
        self.assertNotIn("error", decoded)

    def test_heartbeat_frame_has_no_method(self):
        decoded = self._decode_single("frameSent", _frame(b"", version=0))
        self.assertEqual(decoded["method"], "<heartbeat>")

    def test_header_length_mismatch_raises(self):
        frame = struct.pack(">HHIII", 10, 102, 1, 1, 999) + b"abc"
        with self.assertRaises(decoder.DecodeError):
            self._decode_single("frameReceived", frame)

    def test_binary_frame_is_not_treated_as_control_frame(self):
        """opcode 2 是数据帧；把它当控制帧会得出错误结论。"""
        self.assertNotIn("binary", {"close", "ping", "pong"})


class CapturePairingTest(unittest.TestCase):
    """端到端：通知与 ACK 必须按 seq_id + 数字序号配对成功。"""

    def _write_capture(self, rows) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "capture.jsonl"
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        return tmp

    def test_notify_and_ack_pair_without_error(self):
        seq = "1789893073354#149439461#97c2dcb4"
        rows = [
            {"ts": "2026-09-20T16:31:06.900", "event": "created", "requestId": "1",
             "url": "wss://titan-ws.pinduoduo.com/"},
            _row("frameReceived",
                 _frame(_notify_body(seq, 20008, _notification({
                     "type": 0, "sub_type": 0, "template_name": "user_goods_card",
                     "from": {"role": "user", "uid": "6840347447"},
                     "to": {"role": "mall_cs", "uid": "591119888"},
                     "content": "https://mobile.yangkeduo.com/goods.html?goods_id=627273454388",
                     "info": {"goodsID": 627273454388, "goodsName": "葵花晕车贴", "goodsPrice": "10.9"},
                 }), compress=True)),
                 "2026-09-20T16:31:06.930"),
            _row("frameSent", _frame(_ack_body(seq, 20008), corr_id=103),
                 "2026-09-20T16:31:06.931"),
        ]
        path = self._write_capture(rows)

        result, messages, errors = decoder.decode_capture(path)

        self.assertEqual(errors, [], f"解码不应有错误: {errors}")
        self.assertEqual(result["unacked_notifications"], [])
        self.assertEqual(result["notifications"], 1)
        self.assertEqual(len(messages), 1)

        message = messages[0]
        self.assertEqual(message["type_name"], "GOODS_INQUIRY")
        self.assertEqual(message["from_role"], "user")
        self.assertEqual(message["detail"]["goods_id"], 627273454388)

        notify = next(e for e in result["timeline"] if e.get("method") == "titan.notifyDataLite")
        ack = next(e for e in result["timeline"] if e.get("method") == "titan.notifyDataLite.ack")
        self.assertEqual(ack["acks_frame_n"], notify["n"])
        self.assertEqual(notify["acked_by_frame_n"], ack["n"])
        self.assertEqual(ack["ack_seq_id"], notify["seq_id"])
        self.assertEqual(ack["ack_msg_seq"], notify["msg_seq"])

    def test_unmatched_ack_is_reported(self):
        """找不到对应通知的 ACK 必须被报出来，不能静默忽略。"""
        rows = [
            _row("frameSent", _frame(_ack_body("missing-seq", 1), corr_id=103),
                 "2026-09-20T16:31:06.931"),
        ]
        path = self._write_capture(rows)
        result, _messages, _errors = decoder.decode_capture(path)
        ack = next(e for e in result["timeline"] if e.get("method") == "titan.notifyDataLite.ack")
        self.assertNotIn("acks_frame_n", ack)


if __name__ == "__main__":
    unittest.main()
