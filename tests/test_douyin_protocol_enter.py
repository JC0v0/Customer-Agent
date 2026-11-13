# -*- coding: utf-8 -*-
"""pigeon 进会话取票相关的单元测试（不联网，只验帧构造与选帧逻辑）。

2026-11-14 真机结论：
  - 进会话帧结构是 f8.f8.f610，f610.f1 里 f2 = 会话号、f20 = 买家复合键
  - 它的回包带「目标会话的票据 + 会话短 ID」，正文帧 f7/f14 就靠这两样
  - 客户端升级会换命令号，所以选帧必须按结构认，不能按 10014 认
"""
from tools.reverse_probe.douyin.pigeon_proto import (
    enc_bytes_field,
    enc_varint_field,
    get_bytes_field,
    parse_fields,
)
from types import SimpleNamespace

from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport


def _enter_frame(conv_id: int, buyer: str, cmd: int = 10014) -> bytes:
    wrapper = (
        enc_bytes_field(1, b"")
        + enc_varint_field(2, conv_id)
        + enc_varint_field(3, 10)
        + enc_bytes_field(20, buyer.encode("utf-8"))
    )
    payload = enc_bytes_field(1, wrapper)
    inner = enc_bytes_field(610, payload)
    f8 = (
        enc_varint_field(1, 100)
        + enc_varint_field(2, cmd)
        + enc_bytes_field(3, b"1.1.4-beta.7")
        + enc_bytes_field(8, inner)
    )
    return (
        enc_varint_field(1, cmd)
        + enc_varint_field(2, 1794590000000)
        + enc_varint_field(3, 10001)
        + enc_varint_field(4, 1)
        + enc_bytes_field(8, f8)
    )


def _transport():
    session = SimpleNamespace(token="", device_id="", aid=0, pigeon_sign="")
    transport = PigeonTransport.__new__(PigeonTransport)
    transport.session = session
    transport._enter_template = None
    transport._enter_cmd = 10014
    transport._ticket_cache = {}
    transport._frames_by_cmd = {}
    return transport


def test_pick_enter_frame_ignores_heartbeat_with_same_cmd():
    """同号心跳帧（只带 f16001）不能被当成进会话帧。"""
    from tools.reverse_probe.douyin.pigeon_transport import _pick_enter_frame

    heartbeat = (
        enc_varint_field(1, 10014)
        + enc_varint_field(2, 1)
        + enc_bytes_field(8, enc_bytes_field(8, enc_bytes_field(16001, b"12345")))
    )
    real = _enter_frame(7706532626512807168, "A" * 88)
    cmd, frame = _pick_enter_frame({10014: heartbeat, 10012: real})
    assert cmd == 10012
    assert frame == real


def test_pick_enter_frame_requires_buyer_id():
    """只有会话号没有买家键的列表类请求不算进会话帧。"""
    from tools.reverse_probe.douyin.pigeon_transport import _pick_enter_frame

    wrapper = enc_varint_field(2, 7706532626512807168) + enc_varint_field(3, 10)
    inner = enc_bytes_field(610, enc_bytes_field(1, wrapper))
    f8 = enc_varint_field(2, 10013) + enc_bytes_field(8, inner)
    frame = enc_varint_field(1, 10013) + enc_bytes_field(8, f8)
    assert _pick_enter_frame({10013: frame}) == (None, None)


def test_rebuild_enter_frame_replaces_conv_and_buyer_key():
    transport = _transport()
    transport._enter_template = _enter_frame(7000000000000000001, "A" * 88)

    frame = transport._rebuild_enter_frame("7706532626512807168", "B" * 88, "100000001")
    assert frame is not None

    inner = get_bytes_field(get_bytes_field(frame, 8), 8)
    wrapper = get_bytes_field(get_bytes_field(inner, 610), 1)
    values = {}
    for f, wt, v, _ in parse_fields(wrapper):
        values[f] = v
    assert values[2] == 7706532626512807168
    assert values[20].decode() == "B" * 88 + ":100000001::2:1:pigeon"


def test_rebuild_enter_frame_keeps_command_number():
    transport = _transport()
    transport._enter_template = _enter_frame(7000000000000000001, "A" * 88, cmd=10012)

    frame = transport._rebuild_enter_frame("7706532626512807168", "B" * 88, "100000001")
    assert parse_fields(frame)[0][2] == 10012


def test_rebuild_enter_frame_without_template_returns_none():
    transport = _transport()
    assert transport._rebuild_enter_frame("7706532626512807168", "B" * 88, "1") is None
