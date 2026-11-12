# -*- coding: utf-8 -*-
"""
抖音飞鸽 HTTP 消息拉取模块 —— 纯协议接收主路径（与拼多多轮询同构）。
====================================================================
背景：pigeon WS 只推送"会话状态变化"(conv_unread_union)，不推消息正文；
消息正文必须主动通过 HTTP 拉取。本模块实现：

  get_user_message?pigeon_source=web&PIGEON_BIZ_TYPE=2&pigeon_sign=<sign>
  body(protobuf): f8.f12048.f1=起始时间戳(微秒) f4=结束时间戳(微秒)
  响应: f6 -> f12048 -> f1 -> f2(消息块) -> 每条消息 f2
    消息条目: f5=conv_id, f4=ts(微秒), f6=type(1000=文本), f8=text,
              f9 kv: s:sender_biz_role(Buyer=买家/Shop=商家/CurrentServer=商家侧)
              security_receiver_id / shop_id
  注意: f12048.f2 是"历史会话事件快照"(command_type/group_version 等，
        字段布局与实时消息不同，且没有正文)，不要当成消息块解析。

用法：
  h = PigeonHttp(session, template_path)
  msgs = h.pull_messages(since_us)   # 返回买家文本消息列表（含游标）
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, Any

from .pigeon_proto import parse_fields, rebuild, get_bytes_field

# 项目根目录（tools/reverse_probe/douyin/pigeon_http.py 向上四层）
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

MSG_TYPE_TEXT = 1000  # 文本消息
MSG_TYPE_SYS = 1005   # 系统消息（超时关闭等）
BUYER_ROLE = "Buyer"
SHOP_ROLES = ("Shop", "CurrentServer")


class PigeonHttpError(RuntimeError):
    """HTTP 拉取失败（网络异常 / 票据失效 / 非 200）。

    必须抛出来而不是返回空列表：调用方靠异常计数判断「凭据是不是过期了」，
    返回空列表会被当成「本轮没有新消息」，登录过期就再也不会触发自动重登，
    日志里也永远看不到失败。
    """


class PigeonHttp:
    """HTTP 主动拉取消息（时间窗口增量）。"""

    # 单次读超时。实测正常响应 0.2~0.4s；原来 20s 一旦偶发超时，
    # 拉取循环会整整空窗 20s，买家消息至少晚一个超时周期才可见。
    # 缩短后失败立刻返回、下一轮马上重试，最坏延迟压到个位数秒。
    HTTP_READ_TIMEOUT_SECONDS = 6.0

    def __init__(self, session, template_path: str):
        self.session = session
        t = json.load(open(template_path, encoding="utf-8"))
        self.url_base = t.get("url_base") or "https://fxg.jinritemai.com/pigeon_im/v3/message/get_user_message"
        self._body = bytearray(__import__("base64").b64decode(t["body_b64"]))
        self.last_status: Optional[int] = None
        self.last_error: Optional[str] = None

    # ---------- 请求构造 ----------
    def _build_body(self, start_us: int, end_us: int) -> bytes:
        """替换 body f8.f12048.f1(起始) / f4(结束) 时间戳（微秒）。"""
        def patch_12048(blob: bytes) -> bytes:
            fields = parse_fields(blob)
            new = []
            for f, wt, v, _ in fields:
                if f == 12048 and wt == 2:
                    inner = parse_fields(v)
                    new_inner = []
                    for f2, wt2, v2, _ in inner:
                        if f2 == 1 and wt2 == 0:
                            v2 = start_us
                        elif f2 == 4 and wt2 == 0:
                            v2 = end_us
                        new_inner.append((f2, wt2, v2, None))
                    new.append((f, wt, rebuild(new_inner), None))
                else:
                    new.append((f, wt, v, None))
            return rebuild(new)

        def patch_all(blob: bytes) -> bytes:
            fields = parse_fields(blob)
            new = []
            for f, wt, v, _ in fields:
                if f == 8 and wt == 2:
                    new.append((f, wt, patch_12048(v), None))
                else:
                    new.append((f, wt, v, None))
            return rebuild(new)

        return patch_all(bytes(self._body))

    def _post(self, body: bytes, timeout: Optional[float] = None) -> Optional[bytes]:
        if timeout is None:
            timeout = self.HTTP_READ_TIMEOUT_SECONDS
        self.last_status = None
        self.last_error = None
        sign = urllib.parse.quote(self.session.pigeon_sign, safe="")
        url = f"{self.url_base}?pigeon_source=web&PIGEON_BIZ_TYPE=2&pigeon_sign={sign}"
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/x-protobuf")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                self.last_status = getattr(resp, "status", None)
                return resp.read()
        except Exception as exc:
            self.last_status = getattr(exc, "code", None)
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    # ---------- 响应解码 ----------
    @staticmethod
    def _kvmap(b: bytes) -> Dict[str, str]:
        m = {}
        for f, wt, v, _ in parse_fields(b):
            if f == 9 and wt == 2:
                try:
                    sub = parse_fields(v)
                except Exception:
                    continue
                ks = [vv.decode("utf-8", "replace") for f, wt, vv, _ in sub if f == 1 and wt == 2]
                vs = [vv.decode("utf-8", "replace") for f, wt, vv, _ in sub if f == 2 and wt == 2]
                for k, vv in zip(ks, vs):
                    m[k] = vv
        return m

    @classmethod
    def _decode_response(cls, rb: bytes) -> List[Dict[str, Any]]:
        """解码 get_user_message 响应 → 原始消息条目列表。

        只解析 f12048.f1 下的消息块；f12048.f2 是历史会话事件快照，
        字段布局不同且无正文，交给它会被增量游标过滤成噪音。
        畸形响应返回空列表，避免 parse_fields 抛异常打断拉取循环。
        """
        try:
            return cls._decode_response_inner(rb)
        except (IndexError, ValueError, KeyError):
            return []

    @classmethod
    def _decode_response_inner(cls, rb: bytes) -> List[Dict[str, Any]]:
        f6 = get_bytes_field(rb, 6)
        if not f6:
            return []
        f12048 = get_bytes_field(f6, 12048)
        if not f12048:
            return []
        inner = get_bytes_field(f12048, 1)
        if not inner:
            return []

        # f12048.f1 下面可能有**多个** f2 消息块，必须全部解析。
        #
        # 真机实测（2026-11-11 00:38）：同一个响应里有 2 个 f2 块，
        #   block0 = 20 条，时间停在 00:02（旧快照）
        #   block1 = 20 条，一路到 00:36（含刚发出的回复）
        # 原来只取 get_bytes_field(inner, 2) 的第一个块，于是每次拿到的都是
        # 那份停在旧时间点的快照；游标早已越过它，增量过滤后永远是 0 条，
        # 表现就是「前面能收，后来越来越收不到，最后一条都收不到」。
        blocks = []
        for field, wire_type, value, _ in parse_fields(inner):
            if field == 2 and wire_type == 2:
                blocks.append(value)
        if not blocks:
            return []

        out = []
        seen_keys = set()
        for block in blocks:
            for item in cls._decode_block(block):
                # 跨块去重只看「会话 + 时间戳 + 正文」：两个块对同一条消息给出的
                # message_logid 未必一致（一块有、一块空），带上 msg_id 就会把
                # 同一条消息保留两份，派发时又变成「一条消息回两遍」。
                key = (
                    str(item.get("conv_id") or ""),
                    str(item.get("ts") or ""),
                    str(item.get("text") or ""),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append(item)
        return out

    @classmethod
    def _decode_block(cls, block: bytes) -> List[Dict[str, Any]]:
        """解析一个 f2 消息块里的所有消息条目。"""
        out = []
        for f, wt, v, _ in parse_fields(block):
            if f == 2 and wt == 2:
                try:
                    fs = parse_fields(v)
                except Exception:
                    continue
                conv = next((x for f, wt, x, _ in fs if f == 5 and wt == 0), None)
                mtype = next((x for f, wt, x, _ in fs if f == 6 and wt == 0), None)
                ts = next((x for f, wt, x, _ in fs if f == 4 and wt == 0), None)
                text = None
                for f, wt, x, _ in fs:
                    if f == 8 and wt == 2:
                        try:
                            text = x.decode("utf-8", "replace")
                        except Exception:
                            text = None
                kv = cls._kvmap(v)
                out.append({
                    "conv_id": str(conv) if conv else None,
                    "msg_id": kv.get("message_logid", ""),
                    "text": text,
                    "ts": ts,
                    "type": mtype,
                    # ext 里的语义类型（text / allocated_service /
                    # customer_enter_from_close ...），和上面 f6 的数字类型
                    # （1000/1005）不是一回事，不能混用。
                    "msg_type": kv.get("type", ""),
                    "sender_role": kv.get("im_sender_role", ""),
                    "sender_biz_role": kv.get("s:sender_biz_role", ""),
                    # 买家消息：接收方=shop_id，发送方(买家)=security_sender_id；回复时用它作 security_receiver_id
                    "security_receiver_id": kv.get("security_receiver_id", "") or kv.get("s:security_invisible", ""),
                    "security_sender_id": kv.get("security_sender_id", "") or kv.get("security_pigeon_uid", ""),
                    "security_conversation_id": kv.get("security_conversation_id", ""),
                    "shop_id": kv.get("shop_id", ""),
                    "talk_id": kv.get("talk_id", ""),
                    "client_message_id": kv.get("s:client_message_id", ""),
                    # 区分「真人坐席 / 平台智能客服 / 店铺机器人」靠这两个字段：
                    #   biz_sender_info = {"biz_role_type": "cs"    } → 真人客服
                    #   biz_sender_info = {"biz_role_type": "llm"   } → 抖音电商智能客服
                    #   biz_sender_info = {"biz_role_type": "robot" } → 店铺机器人
                    # 真机实测（2026-11-11 00:51）：平台智能客服那条回复是
                    #   s:sender_biz_role=Shop, sender_role=4, biz_sender=robot,
                    #   biz_role_type=llm, source_key=official_llm_cs
                    "biz_sender_info": kv.get("biz_sender_info", ""),
                    "biz_sender": kv.get("biz_sender", ""),
                    "sender_info": kv.get("sender_info", ""),
                })
        return out

    # ---------- 对外接口 ----------
    # 服务端时间窗口过滤存在已知问题：小回溯窗口（<24h）可能返回 0 条，
    # 即使窗口内确有消息。因此请求体始终用大回溯（REQUEST_BACKFILL_SECONDS），
    # 客户端靠 since_us 游标做增量过滤。
    REQUEST_BACKFILL_SECONDS = 86400  # 请求体固定回溯 24h

    def pull_messages(self, since_us: int, end_us: Optional[int] = None,
                      body_since_us: Optional[int] = None) -> List[Dict[str, Any]]:
        """拉取消息，返回归一化条目。

        Args:
            since_us: 客户端过滤游标（只返回 since_us < ts <= end_us 的消息）
            end_us: 客户端过滤结束时间（默认 now+60s）
            body_since_us: 请求体起始时间戳（默认 now-24h）。
                服务端小窗口可能返回 0 条，故固定用大回溯，客户端做增量过滤。
        """
        if end_us is None:
            # 抓包里 origin_timestamp_offset 可达 200 秒，说明消息 ts 可能来自
            # 买家客户端、与本机时钟有偏差。窗口上界放到 now+5min，避免
            # 「时间戳比本机快」的消息被 <= end_us 过滤掉而永远收不到。
            end_us = int(time.time() * 1e6) + int(300 * 1e6)
        if body_since_us is None:
            body_since_us = int(time.time() * 1e6) - int(self.REQUEST_BACKFILL_SECONDS * 1e6)
        body = self._build_body(body_since_us, end_us)
        rb = self._post(body)
        if rb is None:
            raise PigeonHttpError(
                "get_user_message 请求失败: status=%s err=%s"
                % (self.last_status, self.last_error)
            )
        if not rb:
            return []
        # 服务端有时会把整段历史一起返回，不能只相信请求体中的时间窗。
        # 若把已关闭会话的历史消息交给自动回复，飞鸽会 ACK 发送却静默丢弃，
        # 造成"日志成功、手机未收到"的假象。
        return [
            message
            for message in self._decode_response(rb)
            if isinstance(message.get("ts"), int)
            and since_us < message["ts"] <= end_us
        ]

    def pull_buyer_messages(self, since_us: int) -> List[Dict[str, Any]]:
        """拉取并过滤买家文本消息（type=1000 且 sender_biz_role=Buyer）。"""
        msgs = self.pull_messages(since_us)
        out = []
        for m in msgs:
            if m.get("type") == MSG_TYPE_TEXT and m.get("sender_biz_role") == BUYER_ROLE:
                out.append(m)
        return out


def extract_since_us(msgs: List[Dict[str, Any]]) -> Optional[int]:
    """从已拉取消息中提取最新时间戳（微秒）作为下次游标。"""
    ts = None
    for m in msgs:
        t = m.get("ts")
        if t:
            ts = max(ts, t) if ts is not None else t
    return ts


def update_user_msg_template(body_b64: str, url_base: Optional[str] = None,
                             path: str = str(_PROJECT_ROOT / "temp" / "dy_usermsg_template.json")):
    """更新模板文件（供重新抓包后刷新）。"""
    d = json.load(open(path, encoding="utf-8"))
    d["body_b64"] = body_b64
    if url_base:
        d["url_base"] = url_base
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return True
