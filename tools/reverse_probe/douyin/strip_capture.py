"""把抓包产物脱敏成「结构骨架」，供随包分发。

背景
----
pigeon 通道的发送帧必须复用真实抓包的结构（protobuf 字段布局 + 命令号），
手写重建风险很高。但原始抓包里带着 token / pigeon_sign / device_id / 店铺号 /
买家 id，不能进版本库，也不能随程序分发。

好在发送前 ``PigeonTransport._rebuild_message_frame`` 会把所有业务字段重建一遍，
模板只剩「字段骨架」的作用。因此可以把凭据替换成空串，保留字段结构，
得到一个既能入库、又能被正常重建的骨架文件。

保留 vs 清空
------------
清空：token、pigeon_sign、device_id、shop_id、买家 id、会话号、会话票据、
      消息 id、时间戳。这些运行时都会被覆盖，留着只会泄露。
保留：协议版本号、构建号、app_name、浏览器环境、屏幕尺寸、时区。这些是
      「客户端画像」，不含账号身份，且代码不会覆盖它们。
掩码：hierarchical_dimension 里的长数字串（dynamic_dimension）。它是真实抓包
      带下来的设备/买家维度值，代码不会覆盖，随包分发等于把上一个账号的
      指纹带给所有新用户。字段结构必须留着（服务端风控会读），所以只做
      **等长数字掩码**，不删字段、不改长度，protobuf 布局完全不变。

用法
----
    python -m tools.reverse_probe.douyin.strip_capture \
        --capture temp/dy_send_capture.json \
        --out Channel/douyin/assets/pigeon_send_frame.json
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .pigeon_proto import get_bytes_field, parse_fields, rebuild

# 命令号 -> 帧序号（抓包里的 ws_sent 顺序），用于挑选要保留的帧
# 正文帧是 10019（10018 是会话语义帧，不带正文，发它只会拿到 ACK 而不投递）
_SEND_FRAME_CMD = 10019
# 10016 = 「开会话」帧。发送序列里它排在正文帧之前，缺它就没法发消息
# （2026-11-13 真机：10018 有骨架兜底、10016 没有，于是报 missing_start_template）。
_START_FRAME_CMD = 10016
# 10014 = 「进会话」帧，回包带该会话的票据与会话短 ID，是协议发送的前置条件
_ENTER_FRAME_CMD = 10014

# 顶层字段里需要清掉的 varint（时间戳 / 会话号）
_VARINT_CLEAR = {
    "f2": 0,           # 帧级时间戳，发送前会被写成当前时间
}

# f8.f8.f100 里的 varint（消息体内部）
_MSG_VARINT_CLEAR = {
    "f3": 0,           # conv_id
    "f14": 0,          # conversation_short_id
}

# f8.f8.f100 里直接清空的字符串字段
_MSG_STR_CLEAR = (
    "f4",              # 文本正文
    "f7",              # 会话票据（从 10015 ACK 取）
    "f8",              # client_message_id
    "f50",             # conversation_key
)

# f8 这一层的 KV（f15）里要清空的 key
_F8_KV_CLEAR = (
    "pigeon_sign",
    "session_did",
)

# f8.f8.f100 的 KV（f5）里要清空的 key
_MSG_KV_CLEAR = (
    "shop_id",
    "security_receiver_id",
    "p:check_Send",
    "s:client_message_id",
    "track_info",
    "sender_id",
)

# 顶层 KV（f5）里要清空的 key
_TOP_KV_CLEAR = (
    "pigeon_sign",
)

# f8 这一层直接清空的字符串字段
_F8_STR_CLEAR = (
    "f4",              # token
    "f9",              # device_id
)

# 进会话帧（10014）的 f8.f15 里要清空的 KV。这几个值运行时会由
# ``PigeonTransport._refresh_session_frame`` 按当前登录会话重新填回去，
# 骨架里留着只会把上一个账号的设备指纹带进版本库。
_ENTER_F8_KV_CLEAR = (
    "pigeon_sign",
    "session_did",
    "session_aid",
)

# 进会话帧只保留「结构」，会话号与买家键换成占位值，
# 运行时由 ``_rebuild_enter_frame`` 按目标会话重写。
_ENTER_PLACEHOLDER_CONV = 7000000000000000001
_ENTER_PLACEHOLDER_BUYER = "A" * 88

# hierarchical_dimension 里的 dynamic_dimension 长这样：
#   6972_9230_4541_1131_8272_6599_...
# 4 位一组、下划线分隔、20 组以上。这是真实抓包的设备/买家维度值，
# 运行时代码不会覆盖它，必须随骨架一起脱敏。
# 用等长掩码而不是删除：字段留着，只把数字换成 0，字节数不变，
# protobuf 的长度前缀全部保持原样，服务端仍能按原结构解析。
_DIMENSION_ID_RE = re.compile(rb"\d{4}(?:[_-]\d{4}){7,}")


def _mask_dimension_ids(blob: bytes) -> bytes:
    """把长数字维度串按等长规则换成全 0，保持字节数完全一致。"""

    def replace(match: "re.Match[bytes]") -> bytes:
        token = match.group(0)
        out = bytearray()
        for byte in token:
            if 48 <= byte <= 57:
                out.append(48)  # ASCII '0'
            else:
                out.append(byte)
        return bytes(out)

    return _DIMENSION_ID_RE.sub(replace, blob)


def scrub_skeleton_file(path: str) -> int:
    """对已经生成的骨架 JSON 做二次脱敏，返回被掩码的数字串个数。

    用于修复历史骨架：早期版本把 ``hierarchical_dimension`` 当成
    「不含账号身份的客户端画像」保留了下来，里面其实是真实抓包值。
    直接对 b64 解码后的原始字节做等长替换，不重新解析字段，
    因此不会改动 protobuf 结构，也不会影响运行时重建。
    """
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as f:
        data = json.load(f)

    key = "body_b64" if "body_b64" in data else "b64"
    raw = base64.b64decode(data[key])
    masked = _mask_dimension_ids(raw)
    # 全 0 占位串本身仍然匹配正则，只按 findall 计数会导致重复运行时
    # 每次都「命中」并重写文件。用掩码前后是否变化判断是否真的需要写。
    if masked == raw:
        return 0

    count = 0
    for token in _DIMENSION_ID_RE.findall(raw):
        for byte in token:
            if 49 <= byte <= 57:  # 非 0 数字，说明这处还是真实值
                count += 1
                break

    data[key] = base64.b64encode(masked).decode("ascii")
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return count


def _clear_kv(blob: bytes, kv_field: int, keys: Tuple[str, ...]) -> bytes:
    """把指定 kv 子消息里命中 key 的 value 置空，key 本身保留。"""
    fields = parse_fields(blob)
    out: List[Tuple[int, int, Any, Any]] = []
    for f, wt, v, _ in fields:
        if f != kv_field or wt != 2:
            out.append((f, wt, v, None))
            continue
        kvs = parse_fields(v)
        rebuilt: List[Tuple[int, int, Any, Any]] = []
        for sf, swt, sv, _ in kvs:
            if sf == 2 and swt == 2:
                rebuilt.append((sf, swt, b"", None))
            else:
                rebuilt.append((sf, swt, sv, None))
        # 判断这条 kv 的 key 是否命中，命中才清空
        key_name = ""
        for sf, swt, sv, _ in kvs:
            if sf == 1 and swt == 2:
                key_name = sv.decode("utf-8", "replace")
                break
        if key_name in keys:
            out.append((f, wt, rebuild(rebuilt), None))
        else:
            out.append((f, wt, v, None))
    return rebuild(out)


def _clear_fields(blob: bytes, str_fields: Tuple[str, ...], varint_fields: Dict[str, int]) -> bytes:
    fields = parse_fields(blob)
    out: List[Tuple[int, int, Any, Any]] = []
    for f, wt, v, _ in fields:
        name = "f%d" % f
        if wt == 2 and name in str_fields:
            out.append((f, wt, b"", None))
            continue
        if wt == 0 and name in varint_fields:
            out.append((f, wt, varint_fields[name], None))
            continue
        out.append((f, wt, v, None))
    return rebuild(out)


def strip_send_frame(frame: bytes) -> bytes:
    """清空一帧 10018 里的凭据，保留结构与客户端画像。"""
    # 1) 顶层 KV：pigeon_sign
    frame = _clear_kv(frame, 5, _TOP_KV_CLEAR)
    # 2) 顶层 varint：时间戳
    frame = _clear_fields(frame, (), _VARINT_CLEAR)

    # 3) f8 层：token / device_id / KV
    f8 = None
    for f, wt, v, _ in parse_fields(frame):
        if f == 8 and wt == 2:
            f8 = v
            break
    if f8 is None:
        raise ValueError("帧里没有 f8，不是合法的 pigeon 发送帧")
    f8 = _clear_fields(f8, _F8_STR_CLEAR, {})
    f8 = _clear_kv(f8, 15, _F8_KV_CLEAR)

    # 4) f8.f8 层：消息主体
    inner = None
    for f, wt, v, _ in parse_fields(f8):
        if f == 8 and wt == 2:
            inner = v
            break
    if inner is None:
        raise ValueError("帧里没有 f8.f8，不是合法的 pigeon 发送帧")

    payload = None
    for f, wt, v, _ in parse_fields(inner):
        if f == 100 and wt == 2:
            payload = v
            break
    if payload is None:
        raise ValueError("帧里没有 f8.f8.f100 消息体")

    payload = _clear_fields(payload, _MSG_STR_CLEAR, _MSG_VARINT_CLEAR)
    payload = _clear_kv(payload, 5, _MSG_KV_CLEAR)
    inner = _replace_sub(inner, 100, payload)
    f8 = _replace_sub(f8, 8, inner)
    frame = _replace_sub(frame, 8, f8)
    return _mask_dimension_ids(frame)


def _replace_sub(blob: bytes, field_no: int, new_value: bytes) -> bytes:
    """替换某个子消息字段（wt=2），保持字段位置。"""
    fields = parse_fields(blob)
    out: List[Tuple[int, int, Any, Any]] = []
    done = False
    for f, wt, v, _ in fields:
        if f == field_no and wt == 2 and not done:
            out.append((f, wt, new_value, None))
            done = True
        else:
            out.append((f, wt, v, None))
    if not done:
        out.append((field_no, 2, new_value, None))
    return rebuild(out)


def _replace_varint(blob: bytes, field_no: int, new_value: int) -> bytes:
    """替换某个 varint 字段（wt=0），保持字段位置。"""
    fields = parse_fields(blob)
    out: List[Tuple[int, int, Any, Any]] = []
    done = False
    for f, wt, v, _ in fields:
        if f == field_no and wt == 0 and not done:
            out.append((f, wt, int(new_value), None))
            done = True
        else:
            out.append((f, wt, v, None))
    if not done:
        out.append((field_no, 0, int(new_value), None))
    return rebuild(out)


def looks_like_enter_frame(frame: bytes) -> bool:
    """按结构判断是否进会话帧（f8.f8.f610.f1，且 f1.f2 是会话号）。

    不能按命令号判断：登录录帧里的 10014 往往只是一帧 5 字节心跳，
    真正的进会话帧挂在 10012/10013 上，命令号随客户端版本漂移。
    """
    try:
        f8 = get_bytes_field(frame, 8)
        if not f8:
            return False
        inner = get_bytes_field(f8, 8)
        if not inner:
            return False
        payload = get_bytes_field(inner, 610)
        if not payload:
            return False
        wrapper = get_bytes_field(payload, 1)
        if not wrapper:
            return False
        for f, wt, v, _ in parse_fields(wrapper):
            if f == 2 and wt == 0 and int(v) > 10 ** 15:
                return True
    except Exception:
        return False
    return False


def strip_enter_frame(frame: bytes) -> bytes:
    """清空一帧进会话（10014）里的凭据，只保留字段结构。

    清掉的四个值运行时会按当前登录会话重新写入
    （``PigeonTransport._refresh_session_frame``）：token、device_id、
    session_did、pigeon_sign。会话号与买家键换成占位值，
    运行时由 ``_rebuild_enter_frame`` 按目标会话重写。
    """
    frame = _clear_kv(frame, 5, _TOP_KV_CLEAR)
    frame = _clear_fields(frame, (), _VARINT_CLEAR)

    f8 = get_bytes_field(frame, 8)
    if not f8:
        raise ValueError("帧里没有 f8，不是合法的 pigeon 进会话帧")
    f8 = _clear_fields(f8, _F8_STR_CLEAR, {})
    f8 = _clear_kv(f8, 15, _ENTER_F8_KV_CLEAR)

    inner = get_bytes_field(f8, 8)
    if not inner:
        raise ValueError("帧里没有 f8.f8")
    payload = get_bytes_field(inner, 610)
    if not payload:
        raise ValueError("帧里没有 f8.f8.f610，不是进会话帧")
    wrapper = get_bytes_field(payload, 1)
    if not wrapper:
        raise ValueError("帧里没有 f8.f8.f610.f1 载荷")

    wrapper = _replace_varint(wrapper, 2, _ENTER_PLACEHOLDER_CONV)
    wrapper = _replace_sub(wrapper, 20, _ENTER_PLACEHOLDER_BUYER.encode("ascii"))
    payload = _replace_sub(payload, 1, wrapper)
    inner = _replace_sub(inner, 610, payload)
    f8 = _replace_sub(f8, 8, inner)
    return _mask_dimension_ids(_replace_sub(frame, 8, f8))



def strip_usermsg_body(body: bytes) -> bytes:
    """清空 HTTP 拉消息请求体里的凭据，保留时间戳字段结构。"""
    body = _clear_fields(body, ("f4",), {})
    body = _clear_kv(body, 15, _F8_KV_CLEAR)
    body = _clear_fields(body, ("f9",), {})
    return _mask_dimension_ids(body)


def load_sent_frames(capture_path: str) -> List[bytes]:
    """读出抓包里所有「客户端发出」的帧。

    兼容两种落盘格式：程序自己存的是 ``{"ws_sent": [...], "ws_recv": [...]}``，
    临时探针存的是 ``[{"dir": "SENT"/"RECV", ...}]`` 列表。
    """
    with open(capture_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = data.get("ws_sent", [])
    else:
        items = data or []

    frames: List[bytes] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        b64 = item.get("b64")
        if not b64:
            continue
        direction = str(item.get("dir", "SENT")).upper()
        if direction.startswith("R"):
            continue
        frames.append(base64.b64decode(b64))
    return frames


def build_send_skeleton(capture_path: str) -> Dict[str, Any]:
    """从抓包文件里挑出 10019 正文帧，脱敏后返回骨架字典。"""
    target = None
    for frame in load_sent_frames(capture_path):
        if _cmd_of_frame(frame) == _SEND_FRAME_CMD:
            target = frame
            break
    if target is None:
        raise ValueError("抓包里没有 cmd=%d 的发送帧" % _SEND_FRAME_CMD)

    stripped = strip_send_frame(target)
    return {
        "desc": (
            "pigeon 文本发送帧（10019）结构骨架。所有凭据已清空，"
            "运行时由 PigeonTransport._rebuild_message_frame 重建。"
            "来源：strip_capture.py 从真实抓包生成。"
        ),
        "cmd": _SEND_FRAME_CMD,
        "b64": base64.b64encode(stripped).decode("ascii"),
    }


def build_start_skeleton(capture_path: str) -> Dict[str, Any]:
    """从抓包文件里挑出 10016（开会话）帧，脱敏后返回骨架字典。

    10016 与 10018 的 protobuf 布局完全一致（f8.f8.f100 消息体 + 同一组 kv），
    差别只是不带文本正文。所以直接复用 ``strip_send_frame`` 清凭据。
    运行时 ``_rebuild_message_frame(10016, ...)`` 会把 conv_id、会话票据、
    短会话 ID、shop_id、security_receiver_id、sender_role 全部重写。
    """
    target = None
    for frame in load_sent_frames(capture_path):
        if _cmd_of_frame(frame) == _START_FRAME_CMD:
            target = frame
            break
    if target is None:
        raise ValueError("抓包里没有 cmd=%d 的会话帧" % _START_FRAME_CMD)

    stripped = strip_send_frame(target)
    return {
        "desc": (
            "pigeon 开会话帧（10016）结构骨架。所有凭据已清空，"
            "运行时由 PigeonTransport._rebuild_message_frame 重建。"
            "来源：strip_capture.py 从真实抓包生成。"
        ),
        "cmd": _START_FRAME_CMD,
        "b64": base64.b64encode(stripped).decode("ascii"),
    }


def build_enter_skeleton(capture_path: str) -> Dict[str, Any]:
    """从抓包文件里挑出「进会话」帧，脱敏后返回骨架字典。

    进会话帧的回包带目标会话的票据与会话短 ID。挑帧必须按结构
    （``looks_like_enter_frame``）而不是按命令号，原因见该函数注释。
    """
    target = None
    for frame in load_sent_frames(capture_path):
        if looks_like_enter_frame(frame):
            target = frame
            break
    if target is None:
        raise ValueError("抓包里没有进会话帧（f8.f8.f610.f1）")

    stripped = strip_enter_frame(target)
    return {
        "desc": (
            "pigeon 进会话帧（10014）结构骨架。token / device_id / pigeon_sign "
            "已清空，会话号与买家键为占位值；运行时由 "
            "PigeonTransport._rebuild_enter_frame 重建。"
            "来源：strip_capture.py 从真实抓包生成。"
        ),
        "cmd": _ENTER_FRAME_CMD,
        "b64": base64.b64encode(stripped).decode("ascii"),
    }


def build_usermsg_skeleton(capture_path: str) -> Dict[str, Any]:
    """从接收 body 模板里清空凭据，返回骨架字典。"""
    with open(capture_path, encoding="utf-8") as f:
        data = json.load(f)
    body = base64.b64decode(data["body_b64"])
    stripped = strip_usermsg_body(body)
    return {
        "desc": (
            "pigeon v3 get_user_message 请求体骨架。凭据已清空，"
            "运行时由 PigeonHttp._build_body 填入当前账号的 token/device/sign。"
        ),
        "url_base": data.get("url_base"),
        "body_b64": base64.b64encode(stripped).decode("ascii"),
    }


def _cmd_of_frame(frame: bytes) -> Optional[int]:
    """读取帧顶层的 f1（命令号）。"""
    for f, wt, v, _ in parse_fields(frame):
        if f == 1 and wt == 0:
            return int(v)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="抓包脱敏：生成可入库的协议骨架")
    parser.add_argument("--capture", help="原始 dy_send_capture.json")
    parser.add_argument("--usermsg", help="原始 dy_usermsg_template.json")
    parser.add_argument("--out", help="发送帧骨架输出路径")
    parser.add_argument("--out-usermsg", help="请求体骨架输出路径")
    parser.add_argument("--out-start", help="开会话帧骨架输出路径")
    parser.add_argument("--out-enter", help="进会话帧骨架输出路径")
    parser.add_argument(
        "--scrub-assets",
        help="对已生成骨架所在目录做二次脱敏（等长掩码长数字维度串），不需要 --capture",
    )
    args = parser.parse_args(argv)

    if args.scrub_assets:
        folder = Path(args.scrub_assets)
        total = 0
        for path in sorted(folder.glob("*.json")):
            count = scrub_skeleton_file(str(path))
            if count > 0:
                print("已掩码 %s: %d 处" % (path.name, count))
                total += count
        print("骨架二次脱敏完成: 共 %d 处" % total)
        return 0

    if not args.capture:
        parser.error("需要 --capture，或使用 --scrub-assets")
    if not any((args.out, args.out_start, args.out_enter)):
        parser.error("至少要指定一个输出路径（--out / --out-start / --out-enter）")

    if args.out:
        send = build_send_skeleton(args.capture)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(send, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("发送帧骨架已写入: %s (%d 字节)" % (
            args.out, len(base64.b64decode(send["b64"]))))

    if args.out_start:
        start = build_start_skeleton(args.capture)
        with open(args.out_start, "w", encoding="utf-8") as f:
            json.dump(start, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("开会话帧骨架已写入: %s (%d 字节)" % (
            args.out_start, len(base64.b64decode(start["b64"]))))

    if args.out_enter:
        enter = build_enter_skeleton(args.capture)
        with open(args.out_enter, "w", encoding="utf-8") as f:
            json.dump(enter, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("进会话帧骨架已写入: %s (%d 字节)" % (
            args.out_enter, len(base64.b64decode(enter["b64"]))))

    if args.usermsg and args.out_usermsg:
        body = build_usermsg_skeleton(args.usermsg)
        with open(args.out_usermsg, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("请求体骨架已写入: %s (%d 字节)" % (
            args.out_usermsg, len(base64.b64decode(body["body_b64"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
