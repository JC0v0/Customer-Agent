# -*- coding: utf-8 -*-
"""v3 get_user_message 响应解码回归。

关键回归点：f12048.f1 下面可能有**多个** f2 消息块。
真机实测（2026-11-11 00:38）同一个响应里有两个块：
  block0：20 条，时间停在 00:02（旧快照）
  block1：20 条，一路到 00:36（含最新消息，也就是刚发出的回复）
只解析第一个块时，客户端拿到的永远是那份旧快照，游标早已越过它，
增量过滤后恒为 0 条 —— 表现为「前面能收，后来一条都收不到」。
"""
from __future__ import annotations

import time
import unittest

from tools.reverse_probe.douyin.pigeon_http import PigeonHttp
from tools.reverse_probe.douyin.pigeon_proto import enc_bytes_field, enc_varint_field

BASE_TS = int(time.time() * 1e6)


def _kv(pairs):
    sub = b""
    for key, value in pairs.items():
        sub += enc_bytes_field(1, key.encode("utf-8"))
        sub += enc_bytes_field(2, str(value).encode("utf-8"))
    return enc_bytes_field(9, sub)


def _item(conv_id, ts, text, biz_role, role="", client_message_id=""):
    item = b""
    item += enc_varint_field(5, conv_id)
    item += enc_varint_field(4, ts)
    item += enc_varint_field(6, 1000)
    item += enc_bytes_field(8, text.encode("utf-8"))
    item += _kv({
        "s:sender_biz_role": biz_role,
        "im_sender_role": role,
        "s:client_message_id": client_message_id,
    })
    return item


def _block(items):
    out = b""
    for item in items:
        out += enc_bytes_field(2, item)
    return out


def _response(blocks):
    inner = enc_varint_field(1, BASE_TS)
    for block in blocks:
        inner += enc_bytes_field(2, block)
    f12048 = enc_bytes_field(1, inner)
    f6 = enc_bytes_field(12048, f12048)
    return enc_bytes_field(6, f6)


class TestMultiBlockDecode(unittest.TestCase):
    def test_newest_message_in_second_block_is_decoded(self):
        old_block = _block([_item(111, BASE_TS - 600 * 10 ** 6, "旧消息", "Buyer")])
        new_block = _block([_item(111, BASE_TS, "刚发的新消息", "CurrentServer", "")])

        messages = PigeonHttp._decode_response(_response([old_block, new_block]))

        texts = [m["text"] for m in messages]
        self.assertIn("旧消息", texts)
        self.assertIn("刚发的新消息", texts)
        self.assertEqual(len(messages), 2)

    def test_single_block_still_works(self):
        block = _block([_item(111, BASE_TS, "只有一块", "Buyer")])

        messages = PigeonHttp._decode_response(_response([block]))

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["text"], "只有一块")
        self.assertEqual(messages[0]["sender_biz_role"], "Buyer")

    def test_same_message_in_both_blocks_is_deduped(self):
        item = _item(111, BASE_TS, "重复的一条", "Buyer")
        block = _block([item])

        messages = PigeonHttp._decode_response(_response([block, block]))

        self.assertEqual(len(messages), 1)

    def test_missing_blocks_returns_empty(self):
        self.assertEqual(PigeonHttp._decode_response(b""), [])


if __name__ == "__main__":
    unittest.main()
