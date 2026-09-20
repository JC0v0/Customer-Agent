"""
拼多多 WebSocket 抓包离线解码器
================================

把 capture_pdd_ws.py 抓到的 JSONL 转成可读格式，便于人工查看。

协议结构（由真实抓包反推；未验证的推断均已标注）
------------------------------------------------
WebSocket 二进制帧
  ├─ 16 字节大端头，按 >HHIII 解析：
  │    [0] 版本/类型（应用帧=10，心跳帧=0）
  │    [1] 常量 102（应用帧）或 0（心跳帧），含义未确认
  │    [2] 关联 ID，服务端响应会回填同一个值
  │    [3] 标志位
  │    [4] 后续 payload 字节数
  └─ protobuf 体
       ├─ 上行 frameSent：   字段 2 = 方法名
       └─ 下行 frameReceived：字段 1 = 方法名

已观察到的方法
--------------
  titan.session               建连时上报会话（上行）
  titan.sync                  同步请求（上行）
  titan.notifyDataLite        服务端推送通知（下行）
  titan.notifyDataLite.ack    客户端确认通知（上行）

titan.notifyDataLite 的负载
  protobuf 字段 10 → 通知体 protobuf（负载较大时才 gzip 压缩，小块直接明文）
      ├─ 字段 5 = seq_id，22 字节 ASCII，形如 "1789893033491#fb13a7bd"
      ├─ 字段 4 = 数字序号，ACK 会回填同一个值
      └─ 字段 6 = UTF-8 的 JSON 字符串（真正的业务消息）

    上行 ACK 的字段 9 内含同一组 seq_id 与数字序号，
    因此可以在解码时把「通知」与「确认」成对关联起来。

用法
----
    python scripts/decode_pdd_ws.py temp/pdd_ws_20260920_163006.jsonl
    python scripts/decode_pdd_ws.py temp/pdd_ws_20260920_163615.jsonl --outdir temp/decoded

输出
----
    <outdir>/<名字>.decoded.json    逐帧完整解码
    <outdir>/<名字>.messages.json   只含聊天消息，按时间排序

注意
----
抓包里含买家昵称、商品 ID、图片地址等业务数据（access_token 已在抓包时脱敏）。
decoded.json 会保留这些字段，对外分享前请自行检查。
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

# 解码上限，防止畸形或超大负载拖垮脚本
MAX_GZIP_OUTPUT = 4 * 1024 * 1024
MAX_DEPTH = 6
MAX_FIELDS = 256
MAX_JSON_BYTES = 1 * 1024 * 1024

# 消息类型 -> 语义名，来源 Channel/pinduoduo/pdd_message.py 的 PDDMsgType
MSG_TYPE_NAMES = {
    0: "TEXT",
    1: "IMAGE",
    5: "EMOTION",
    14: "VIDEO",
    24: "TRANSFER",
    31: "ROBOT_SYSTEM_HINT",
    41: "USER_SOURCE",
    64: "GOODS_SPEC",
    1002: "WITHDRAW",
}

# type=0 时的子类型，来源 PDDSubType
SUB_TYPE_NAMES = {0: "GOODS_INQUIRY", 1: "ORDER_INFO"}


class DecodeError(Exception):
    """解码失败。携带原因，不做静默兜底。"""


def _decode_seq(raw: bytes | None) -> str | None:
    """把序号字节还原成可读字符串。

    实测 seq_id 是 ASCII 文本（"<毫秒时间戳>#<8位随机串>"）；
    若不是合法 ASCII 则回退成十六进制，避免丢掉信息。
    """
    if raw is None:
        return None
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        return raw.hex()


# ---------------------------------------------------------------------------
# protobuf 基础解析（wire format 子集）
# ---------------------------------------------------------------------------
def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """读取 varint，返回 (值, 新位置)。越界抛 DecodeError。"""
    result = 0
    shift = 0
    for _ in range(10):
        if pos >= len(buf):
            raise DecodeError("varint 截断")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7
    raise DecodeError("varint 过长")


def parse_message(buf: bytes) -> list[tuple[int, int, Any]]:
    """解析 protobuf 消息，返回 [(字段号, wire_type, 值)]。

    只支持 0/1/2/5 四种 wire type；遇到其他类型直接报错，
    避免把不认识的字节当成合法数据继续解析。
    """
    fields: list[tuple[int, int, Any]] = []
    pos = 0
    while pos < len(buf):
        if len(fields) >= MAX_FIELDS:
            raise DecodeError(f"字段数超过上限 {MAX_FIELDS}")
        key, pos = read_varint(buf, pos)
        field_no, wire = key >> 3, key & 0x7
        if field_no == 0:
            raise DecodeError("字段号为 0")
        if wire == 0:
            value, pos = read_varint(buf, pos)
        elif wire == 1:
            value = buf[pos:pos + 8]
            if len(value) < 8:
                raise DecodeError("fixed64 截断")
            pos += 8
        elif wire == 2:
            length, pos = read_varint(buf, pos)
            if length > len(buf) - pos:
                raise DecodeError("长度超出剩余字节")
            value = buf[pos:pos + length]
            pos += length
        elif wire == 5:
            value = buf[pos:pos + 4]
            if len(value) < 4:
                raise DecodeError("fixed32 截断")
            pos += 4
        else:
            raise DecodeError(f"不支持的 wire type {wire}")
        fields.append((field_no, wire, value))
    return fields


def _is_text(buf: bytes) -> bool:
    try:
        buf.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return all(32 <= b < 127 or b in (9, 10, 13) or b >= 128 for b in buf)


def render_bytes(buf: bytes, depth: int) -> Any:
    """把长度分隔字段渲染成可读结构。

    依次尝试：gzip → JSON → 嵌套 protobuf → 文本常量 → 字节摘要。
    """
    if buf.startswith(b"\x1f\x8b"):
        try:
            plain = zlib.decompress(buf, 16 + zlib.MAX_WBITS)
        except zlib.error as exc:
            return {"__gzip_error__": type(exc).__name__, "raw_bytes": len(buf)}
        if len(plain) > MAX_GZIP_OUTPUT:
            return {"__gzip_too_large__": len(plain)}
        return {
            "__gzip__": {"raw_bytes": len(buf), "plain_bytes": len(plain)},
            "content": render_bytes(plain, depth + 1),
        }

    if len(buf) <= MAX_JSON_BYTES:
        text = buf.decode("utf-8", errors="replace")
        if text.lstrip()[:1] in ("{", "["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

    if depth < MAX_DEPTH and buf:
        try:
            nested = parse_message(buf)
        except DecodeError:
            pass
        else:
            if nested:
                return {str(f): render_value(w, v, depth + 1) for f, w, v in nested}

    if _is_text(buf) and len(buf) <= 200:
        return buf.decode("utf-8")

    return {"__bytes__": len(buf), "head_hex": buf[:24].hex()}


def render_value(wire: int, value: Any, depth: int) -> Any:
    if wire == 0:
        return value
    if wire in (1, 5):
        return {"__fixed__": value.hex()}
    return render_bytes(value, depth)


def render_fields(fields: list[tuple[int, int, Any]], depth: int = 0) -> dict[str, Any]:
    """把字段列表渲染成 {字段号: 值}，同号重复字段自动合并成列表。"""
    out: dict[str, Any] = {}
    for field_no, wire, value in fields:
        key = str(field_no)
        rendered = render_value(wire, value, depth)
        if key in out:
            if not isinstance(out[key], list):
                out[key] = [out[key]]
            out[key].append(rendered)
        else:
            out[key] = rendered
    return out


# ---------------------------------------------------------------------------
# 帧解码
# ---------------------------------------------------------------------------
def decode_binary_frame(event: str, raw: bytes) -> dict[str, Any]:
    """解码一个 WebSocket 二进制帧。"""
    if len(raw) < 16:
        raise DecodeError(f"帧头不足 16 字节：{len(raw)}")

    version, channel, corr_id, flags, body_len = struct.unpack(">HHIII", raw[:16])
    body = raw[16:]
    if body_len != len(body):
        # 长度字段与实际不符时如实报错，不强行继续解析
        raise DecodeError(f"头部声明 {body_len} 字节，实际 {len(body)} 字节")

    result: dict[str, Any] = {
        "header": {
            "version_or_type": version,
            "channel_or_const": channel,
            "corr_id": corr_id,
            "flags": flags,
            "body_bytes": body_len,
        }
    }

    if version == 0:
        # 心跳帧：请求通常为空体，响应带少量字段
        result["method"] = "<heartbeat>"
        if body:
            result["body"] = render_fields(parse_message(body))
        return result

    fields = parse_message(body)

    # 方法名：上行在字段 2，下行在字段 1
    method_field = 2 if event == "frameSent" else 1
    for field_no, wire, value in fields:
        if field_no == method_field and wire == 2:
            result["method"] = value.decode("utf-8", errors="replace")
            break
    else:
        result["method"] = None

    method = result["method"]
    if method == "titan.notifyDataLite":
        result.update(_decode_notify(fields))
    elif method == "titan.notifyDataLite.ack":
        result.update(_decode_ack(fields))
    else:
        result["body"] = render_fields(fields)

    return result


def _decode_ack(fields: list[tuple[int, int, Any]]) -> dict[str, Any]:
    """解出 ACK 的回执内容。

    ACK 的字段 9 内嵌一条消息，字段 1 是被确认通知的 seq_id，
    字段 2 是同一个数字序号。据此可与通知配对。
    """
    out: dict[str, Any] = {}
    blob = next((v for f, w, v in fields if f == 9 and w == 2), None)
    if blob is None:
        out["body"] = render_fields(fields)
        return out
    try:
        inner = parse_message(blob)
    except DecodeError as exc:
        out["error"] = f"ACK 字段 9 解析失败: {exc}"
        return out
    out["ack_seq_id"] = _decode_seq(next((v for f, w, v in inner if f == 1 and w == 2), None))
    out["ack_msg_seq"] = next((v for f, w, v in inner if f == 2 and w == 0), None)
    return out


def _decode_notify(fields: list[tuple[int, int, Any]]) -> dict[str, Any]:
    """解出 titan.notifyDataLite 的通知体：seq_id + 内嵌 JSON。"""
    out: dict[str, Any] = {}
    blob = next((v for f, w, v in fields if f == 10 and w == 2), None)
    if blob is None:
        out["error"] = "未找到字段 10（通知负载）"
        return out

    # 实测：负载较大时字段 10 是 gzip，小块则是明文 protobuf，
    # 两种变体的内部字段结构完全相同。
    if blob.startswith(b"\x1f\x8b"):
        try:
            plain = zlib.decompress(blob, 16 + zlib.MAX_WBITS)
        except zlib.error as exc:
            out["error"] = f"gzip 解压失败: {type(exc).__name__}"
            return out
        if len(plain) > MAX_GZIP_OUTPUT:
            out["error"] = f"解压后过大: {len(plain)}"
            return out
        out["compressed"] = True
    else:
        plain = blob
        out["compressed"] = False

    try:
        inner = parse_message(plain)
    except DecodeError as exc:
        out["error"] = f"通知体解析失败: {exc}"
        return out
    # 字段 5 是 22 字节 ASCII 序号（wire type 2，不是 varint）
    out["seq_id"] = _decode_seq(next((v for f, w, v in inner if f == 5 and w == 2), None))
    out["msg_seq"] = next((v for f, w, v in inner if f == 4 and w == 0), None)
    out["raw_fields"] = render_fields(inner)

    payload = next((v for f, w, v in inner if f == 6 and w == 2), None)
    if payload is None:
        return out

    text = payload.decode("utf-8", errors="replace")
    try:
        out["notification"] = json.loads(text)
    except json.JSONDecodeError as exc:
        out["error"] = f"内嵌 JSON 解析失败: {exc.msg}"
        out["raw_text"] = text[:2000]
    return out


# ---------------------------------------------------------------------------
# 消息提取
# ---------------------------------------------------------------------------
def extract_messages(notification: Any) -> list[dict[str, Any]]:
    """从通知 JSON 中提取聊天消息。

    实际结构： push_data.data[] -> {chat_type_id, message:{...}}
    """
    found: list[dict[str, Any]] = []
    if not isinstance(notification, dict):
        return found
    push_data = notification.get("push_data")
    if isinstance(push_data, str):
        try:
            push_data = json.loads(push_data)
        except json.JSONDecodeError:
            return found
    if not isinstance(push_data, dict):
        return found
    for item in push_data.get("data") or []:
        if not isinstance(item, dict):
            continue
        message = item.get("message")
        if isinstance(message, dict):
            found.append({"chat_type_id": item.get("chat_type_id"), **message})
    return found


def describe_message(msg: dict[str, Any]) -> dict[str, Any]:
    """把消息整理成便于阅读的摘要字段。"""
    msg_type = msg.get("type")
    sub_type = msg.get("sub_type")
    info = msg.get("info")

    if msg_type == 0 and sub_type is not None:
        semantics = SUB_TYPE_NAMES.get(sub_type, f"TEXT(未知 sub_type {sub_type})")
    else:
        semantics = MSG_TYPE_NAMES.get(msg_type, f"未定义类型 {msg_type}")

    detail: Any = None
    if isinstance(info, dict):
        if "image_url" in info:
            detail = {"image_url": info.get("image_url"), "width": info.get("width"), "height": info.get("height")}
        elif "gifURL" in info:
            detail = {"gif_url": info.get("gifURL"), "description": info.get("description")}
        elif "goodsID" in info or "goodsName" in info:
            detail = {
                "goods_id": info.get("goodsID"),
                "goods_name": info.get("goodsName"),
                "goods_price": info.get("goodsPrice"),
                "link_url": info.get("linkUrl"),
            }
        elif "orderSequenceNo" in info:
            detail = {"order_id": info.get("orderSequenceNo"), "after_sales_status": info.get("afterSalesStatus")}
        else:
            detail = {"info_keys": sorted(info.keys())}

    def role_uid(side: Any) -> tuple[Any, Any]:
        if isinstance(side, dict):
            return side.get("role"), side.get("uid")
        return None, None

    from_role, from_uid = role_uid(msg.get("from"))
    to_role, to_uid = role_uid(msg.get("to"))

    return {
        "type": msg_type,
        "type_name": semantics,
        "sub_type": sub_type,
        "template_name": msg.get("template_name"),
        "from_role": from_role,
        "from_uid": from_uid,
        "to_role": to_role,
        "to_uid": to_uid,
        "msg_id": msg.get("msg_id"),
        "nickname": msg.get("nickname"),
        "content": msg.get("content"),
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def decode_capture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """解码一个抓包文件，返回 (逐帧结果, 聊天消息列表, 错误列表)。"""
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if line:
            rows.append((line_no, json.loads(line)))

    url_of = {r.get("requestId", ""): r.get("url", "") for _n, r in rows if r.get("url")}

    def host_of(url: str) -> str:
        if "m-ws" in url:
            return "m-ws"
        if "titan" in url:
            return "titan-ws"
        return url.split("/")[2] if "//" in url else (url or "?")

    timeline: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    errors: list[str] = []
    order = 0

    for line_no, row in rows:
        event = row.get("event", "?")
        if event not in ("frameSent", "frameReceived"):
            entry: dict[str, Any] = {
                "n": line_no,
                "ts": row.get("ts"),
                "event": event,
                "host": host_of(row.get("url", "")),
                "url": row.get("url"),
            }
            if "status" in row:
                entry["status"] = row["status"]
            timeline.append(entry)
            continue

        direction = "sent" if event == "frameSent" else "recv"
        opcode = row.get("opcode")
        entry = {
            "n": line_no,
            "ts": row.get("ts"),
            "event": event,
            "direction": direction,
            "host": host_of(url_of.get(row.get("requestId", ""), "")),
            "ws_opcode": opcode,
            "ws_frame": row.get("opcodeName"),
            "bytes": row.get("len"),
        }

        payload = row.get("payload", "")
        if opcode == 1:
            try:
                entry["payload"] = json.loads(payload)
            except json.JSONDecodeError:
                entry["payload"] = {"__text__": payload[:2000]}
        elif opcode == 2:
            try:
                raw = base64.b64decode(payload, validate=True)
            except Exception as exc:
                errors.append(f"第 {line_no} 行 base64 解码失败: {type(exc).__name__}")
                entry["error"] = "base64 解码失败"
                timeline.append(entry)
                continue
            try:
                entry.update(decode_binary_frame(event, raw))
            except DecodeError as exc:
                errors.append(f"第 {line_no} 行帧解码失败: {exc}")
                entry["error"] = str(exc)
            else:
                if entry.get("error"):
                    errors.append(f"第 {line_no} 行通知解码失败: {entry['error']}")
                notification = entry.get("notification")
                if notification is not None:
                    for msg in extract_messages(notification):
                        order += 1
                        messages.append(
                            {
                                "order": order,
                                "ts": row.get("ts"),
                                "direction": direction,
                                "seq_id": entry.get("seq_id"),
                                "msg_seq": entry.get("msg_seq"),
                                **describe_message(msg),
                                "raw": msg,
                            }
                        )
        else:
            entry["payload"] = {"__opcode__": row.get("opcodeName")}

        timeline.append(entry)

    # 关联通知与 ACK：ACK 回填的 seq_id 与它确认的通知一致
    pending: dict[tuple[Any, Any], dict[str, Any]] = {}
    for entry in timeline:
        if entry.get("method") == "titan.notifyDataLite" and entry.get("seq_id"):
            pending[(entry.get("host"), entry["seq_id"])] = entry
        elif entry.get("method") == "titan.notifyDataLite.ack" and entry.get("ack_seq_id"):
            source = pending.pop((entry.get("host"), entry["ack_seq_id"]), None)
            if source is not None:
                entry["acks_frame_n"] = source["n"]
                source["acked_by_frame_n"] = entry["n"]

    unacked = [
        entry["n"]
        for entry in timeline
        if entry.get("method") == "titan.notifyDataLite" and "acked_by_frame_n" not in entry
    ]
    mismatched = [
        entry["n"]
        for entry in timeline
        if entry.get("method") == "titan.notifyDataLite"
        and entry.get("msg_seq") is not None
        and entry.get("acked_by_frame_n") is None
    ]

    result = {
        "source": str(path),
        "frame_events": sum(1 for e in timeline if e.get("event") in ("frameSent", "frameReceived")),
        "notifications": sum(1 for e in timeline if e.get("method") == "titan.notifyDataLite"),
        "unacked_notifications": unacked,
        "timeline": timeline,
    }
    if mismatched:
        errors.append(f"有 {len(mismatched)} 条通知在同连接内找不到对应 ACK，帧号 {mismatched}")
    return result, messages, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="拼多多 WS 抓包离线解码器")
    parser.add_argument("captures", nargs="+", help="抓包 JSONL 文件")
    parser.add_argument("--outdir", default="temp/decoded", help="输出目录，默认 temp/decoded")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for raw_path in args.captures:
        path = Path(raw_path)
        if not path.exists():
            print(f"[!] 文件不存在: {path}")
            exit_code = 1
            continue

        result, messages, errors = decode_capture(path)

        decoded_path = outdir / f"{path.stem}.decoded.json"
        decoded_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        messages_path = outdir / f"{path.stem}.messages.json"
        messages_path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

        methods = Counter(e.get("method", "?") for e in result["timeline"] if "method" in e)
        notifies = [e for e in result["timeline"] if e.get("method") == "titan.notifyDataLite"]
        gzip_notifies = sum(1 for e in notifies if e.get("compressed"))
        plain_notifies = sum(1 for e in notifies if e.get("compressed") is False)
        print(f"\n=== {path.name} ===")
        print(
            f"  帧事件 {result['frame_events']}，通知 {result['notifications']}，"
            f"消息 {len(messages)}，未确认通知 {len(result['unacked_notifications'])}，错误 {len(errors)}"
        )
        for name, count in methods.most_common():
            print(f"    {name:<28} {count}")
        if notifies:
            print(f"    通知负载：gzip {gzip_notifies} 条，明文 {plain_notifies} 条")
        for err in errors:
            print(f"    [错误] {err}")
            exit_code = 1
        print(f"  -> {decoded_path}")
        print(f"  -> {messages_path}")

        if messages:
            print("  消息一览：")
            for msg in messages:
                label = "买家→客服" if msg.get("from_role") == "user" else "客服→买家"
                preview = msg.get("content")
                if isinstance(preview, str) and len(preview) > 40:
                    preview = preview[:40] + "…"
                if not preview:
                    preview = msg.get("detail")
                template = f" [{msg['template_name']}]" if msg.get("template_name") else ""
                print(
                    f"    #{msg['order']:<3} {msg['ts'][11:23]} {label:<10} "
                    f"{msg['type_name']:<18}{template} {preview}"
                )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
