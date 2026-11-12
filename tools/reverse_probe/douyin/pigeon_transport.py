# -*- coding: utf-8 -*-
"""
抖音飞鸽(抖店IM) 纯协议传输模块 —— 与拼多多同构的接口化收发。
================================================================
已实证(2026-09-04)：
  * 会话级凭据(token+pigeon_sign+device_id)可复用：原始 WS 直连被服务端接受
  * 发送：WS 帧 type=10018 重建(换文本+client_msg_id+时间戳) → 服务端 ACK code=0 → 消息真实投递到会话
  * 接收：持久 WS 连接，服务端推送消息/事件帧，结构 frame.f8.f6.f500.f5
  * security_receiver_id / conversation_id / shop_id 均从推送帧中直接取得

用法：
  from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport
  t = PigeonTransport(session_json_path, template_json_path)
  t.connect()
  for ev in t.poll(5):           # ev = {type, conv_id, text, security_receiver_id, shop_id, sender_role, msg_id, ts}
      if ev['type'] == 'message' and ev['sender_role'] != MERCHANT_ROLE:
          t.send_text(ev['conv_id'], "回复内容", ev['security_receiver_id'], ev['shop_id'])
  t.close()
"""
import asyncio, base64, json, os, threading, time, urllib.parse, urllib.request, uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from .pigeon_proto import parse_fields, get_bytes_field, rebuild, replace_bytes_field, enc_bytes_field

# 角色常量（依据抓包：发送帧 sender_role=2 为商家；3 为买家/服务端事件）
MERCHANT_ROLE = "2"
BUYER_ROLE = "3"


def _cmd_of(raw: bytes):
    """取帧顶层 f1 命令号。"""
    try:
        fs = parse_fields(raw)
        return next((v for f, wt, v, _ in fs if f == 1 and wt == 0), None)
    except Exception:
        return None

# 默认 WS 端点
WS_URL_TMPL = ("wss://ws.fxg.jinritemai.com/ws/v2?"
               "token={token}&aid={aid}&fpid=92&device_id={device_id}"
               "&access_key={access_key}&device_platform=web&version_code=10000"
               "&pigeon_source=web&PIGEON_BIZ_TYPE=2&pigeon_sign={pigeon_sign}")


class PigeonSession:
    """会话凭据（相当于拼多多的 cookie 快照）。"""

    def __init__(self, token, pigeon_sign, device_id="",
                 aid="1383", access_key="975a9f0608893b4a902dbe8a8532f6c8",
                 cookies=""):
        self.token = token
        self.pigeon_sign = pigeon_sign
        self.device_id = device_id
        self.aid = aid
        self.access_key = access_key
        # HTTP 发送/拉取时带上完整 cookie，与拼多多 BaseRequest 携带 cookie 同构。
        # WS 只认 sign，HTTP 网关两者都收，带上不亏。
        self.cookies = cookies or ""

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        cookie_map = d.get("cookies_full") or {}
        parts = []
        for name, value in cookie_map.items():
            parts.append("%s=%s" % (name, value))
        return cls(d["token"], d["pigeon_sign"], d.get("device_id", ""),
                   d.get("aid", "1383"), d.get("access_key", "975a9f0608893b4a902dbe8a8532f6c8"),
                   "; ".join(parts))

    def ws_url(self):
        return WS_URL_TMPL.format(
            token=self.token, aid=self.aid, device_id=self.device_id,
            access_key=self.access_key, pigeon_sign=self.pigeon_sign)


def _kvmap(blob: bytes) -> Dict[str, str]:
    m = {}
    for sub in [v for f, wt, v, _ in parse_fields(blob) if f == 5 and wt == 2]:
        ks = [v.decode("utf-8", "replace") for f, wt, v, _ in parse_fields(sub) if f == 1 and wt == 2]
        vs = [v.decode("utf-8", "replace") for f, wt, v, _ in parse_fields(sub) if f == 2 and wt == 2]
        for k, vv in zip(ks, vs):
            m[k] = vv
    return m


def _kvmap_field(blob: bytes, field=9) -> Dict[str, str]:
    """在 blob 中取所有 field 的 kv map（sub 内 f1=key / f2=value 直接解析）。"""
    m = {}
    for sub in [v for f, wt, v, _ in parse_fields(blob) if f == field and wt == 2]:
        ks = [v.decode("utf-8", "replace") for f, wt, v, _ in parse_fields(sub) if f == 1 and wt == 2]
        vs = [v.decode("utf-8", "replace") for f, wt, v, _ in parse_fields(sub) if f == 2 and wt == 2]
        for k, vv in zip(ks, vs):
            m[k] = vv
    return m


def _ack_payload(raw: bytes) -> bytes:
    """取 ACK 里的会话载荷。

    抓包模板里的载荷是 f8.f6.f610，实测服务端现在也会用 f8.f6.f616
    （f610 变成一个空壳）。两种都试，先到先用。
    """
    f8 = get_bytes_field(raw, 8)
    if not f8:
        return b""
    f6 = get_bytes_field(f8, 6)
    if not f6:
        return b""
    for field in (610, 616):
        blob = get_bytes_field(f6, field)
        if blob:
            return blob
    return b""


def _extract_ack_ticket(raw: bytes) -> str:
    """Extract the fresh conversation ticket returned with a start ACK."""
    try:
        payload = get_bytes_field(_ack_payload(raw), 1)
        ticket = get_bytes_field(payload, 4)
        return ticket.decode("utf-8", "replace") if ticket else ""
    except Exception:
        return ""


def _extract_ack_conversation_short_id(raw: bytes) -> int:
    """Extract the current conversation short ID returned with a 10015 ACK."""
    try:
        payload = get_bytes_field(_ack_payload(raw), 1)
        # 旧结构：短会话 ID 藏在 payload.f53.f2
        f53 = get_bytes_field(payload, 53)
        if f53:
            for field, wire, value, _ in parse_fields(f53):
                if field == 2 and wire == 0:
                    return value
        # 新结构：payload.f2 直接就是短会话 ID
        for field, wire, value, _ in parse_fields(payload):
            if field == 2 and wire == 0:
                return value
        return 0
    except Exception:
        return 0


def decode_inbound(raw: bytes) -> Optional[Dict[str, Any]]:
    """解码服务端推送帧 → 归一化事件。

    结构：frame.f8.payload -> f6 -> f500.msg -> f5.body
      body: f3=msg_id, f4=ts, f5=conversation_id, f8=content, f9=kv(security_receiver_id/shop_id/sender_role/type/...)
    若无 f8 负载但顶层 f5 kv 含 is_ack，则返回 ack 事件。
    """
    try:
        # 顶层 kv（ack 帧用）
        top_kv = _kvmap(raw)
        if top_kv.get("is_ack") == "1":
            return {
                "type": "is_ack",
                "conv_id": None,
                "msg_id": None,
                "text": None,
                "ts": None,
                "security_receiver_id": "",
                "shop_id": "",
                "sender_role": "",
                "sender_id": "",
                "client_message_id": "",
                "conversation_ticket": _extract_ack_ticket(raw),
                "conversation_short_id": _extract_ack_conversation_short_id(raw),
                "raw_kv": top_kv,
            }
        f8 = get_bytes_field(raw, 8)
        if not f8:
            return None
        f6 = get_bytes_field(f8, 6)
        if not f6:
            return None
        f500 = get_bytes_field(f6, 500)
        if not f500:
            return None
        body = get_bytes_field(f500, 5)
        if not body:
            return None
        kv = _kvmap_field(body, 9)
        msg_id = None
        conv_id = None
        content = None
        ts = None
        for f, wt, v, _ in parse_fields(body):
            if f == 3 and wt == 0:
                msg_id = v
            elif f == 5 and wt == 0:
                conv_id = v
            elif f == 8 and wt == 2:
                try:
                    content = v.decode("utf-8", "replace")
                except Exception:
                    content = v.hex()
            elif f == 4 and wt == 0:
                ts = v
        ev_type = kv.get("type", kv.get("msg_type", "message"))
        return {
            "type": ev_type,
            "conv_id": str(conv_id) if conv_id else None,
            "msg_id": str(msg_id) if msg_id else None,
            "text": content,
            "ts": ts,
            "security_receiver_id": kv.get("security_receiver_id", ""),
            "security_sender_id": kv.get("security_sender_id", "") or kv.get("security_pigeon_uid", ""),
            "shop_id": kv.get("shop_id", ""),
            "talk_id": kv.get("talk_id", ""),
            "sender_role": kv.get("sender_role", ""),
            # 来源判定字段：飞鸽用 s:sender_biz_role 标 Buyer / Shop / CurrentServer，
            # 平台机器人提示还会带 systemHint / biz_sender。缺了这几个字段，WS
            # 链路就无法区分「买家发言」和「平台提示」，会把系统消息交给 AI。
            "sender_biz_role": kv.get("s:sender_biz_role", ""),
            "system_hint": kv.get("systemHint", ""),
            "biz_sender": kv.get("biz_sender", ""),
            "sender_id": kv.get("sender_id", ""),
            "client_message_id": kv.get("s:client_message_id", ""),
            "raw_kv": kv,
        }
    except Exception:
        return None


class PigeonTransport:
    """持久 WS 传输：接收推送 + 发送文本。"""

    _instance: Optional["PigeonTransport"] = None

    @classmethod
    def instance(cls) -> Optional["PigeonTransport"]:
        return cls._instance

    @classmethod
    def set_instance(cls, t: Optional["PigeonTransport"]):
        cls._instance = t

    def __init__(self, session: PigeonSession, template_path: str):
        self.session = session
        self._load_templates(template_path)
        self._image_send_tmpl = self._load_image_send_template()
        self._ws = None
        self._reader_thread = None
        self._queue = asyncio.Queue()
        self._loop = None
        self._closed = False
        # conv_id -> {"security_receiver_id":..., "shop_id":...}
        self.conv_state: Dict[str, Dict[str, str]] = {}
        # 发送 ACK 等待（与 poll 的 queue 隔离，避免 poll 抢走 ACK）
        self._send_lock = threading.Lock()
        self._ack_event: Optional[asyncio.Event] = None
        self._echo_event: Optional[asyncio.Event] = None
        self._ack_payload: Optional[Dict[str, str]] = None
        self._ack_ticket = ""
        self._ack_conversation_short_id = 0
        self._echo_payload: Optional[Dict[str, Any]] = None
        # 最近收到的商家侧文本（回声候选）。用列表而不是单个 payload：
        # 抖音会在买家消息到达后自动转人工，推一条 role=2 的
        # 「人工客服XX为您服务」欢迎语，它和我们的回声同角色、先后到达。
        # 只留最后一个 payload 会被这条例外覆盖，导致误判 text_no_echo。
        self._echo_texts: List[str] = []

    def is_connected(self) -> bool:
        """Return whether the underlying websocket is still usable.

        ``websockets`` changed its public closed-state attribute between major
        versions.  Checking both representations prevents a dead reader task
        from leaving the channel marked connected forever.
        """
        ws = self._ws
        if ws is None or self._closed:
            return False
        if getattr(ws, "closed", False):
            return False
        return "CLOSED" not in str(getattr(ws, "state", "")).upper()

    def get_conv_state(self, conv_id) -> Dict[str, str]:
        return self.conv_state.get(str(conv_id), {})

    def _load_templates(self, template_path):
        d = json.load(open(template_path, encoding="utf-8"))
        frames = [__import__("base64").b64decode(f["b64"]) for f in d["ws_sent"] if "b64" in f]
        self._frames_by_cmd = {cmd: frame for frame in frames if (cmd := _cmd_of(frame))}
        self._handshake = frames[0:2]      # 10008 / 10009
        self._send_tmpl = frames[8]        # 10018 发送帧模板
        # 10011 注册帧（含 shop_id，让服务端把连接归属到店铺；缺它收不到任何推送）
        self._register = next((f for f in frames if _cmd_of(f) == 10011), None)
        self._send_tmpl = self._frames_by_cmd.get(10018)
        # 10013 打开会话帧：发送前必须针对当前会话补发，否则已关闭会话会 ACK
        # 但不会实际投递到买家端。
        self._open_conversation = next((f for f in frames if _cmd_of(f) == 10013), None)
        self._start_conversation = self._frames_by_cmd.get(10016)
        # 保存模板帧自身凭据（当外部传入的 session 与模板同源时使用）
        self._tmpl_session = self._extract_session(frames[0])

    @staticmethod
    def _load_image_send_template():
        """Load the most recent image-message frame captured from a normal Edge session.

        The frame is used as a structural protobuf template only.  Before every
        send, ``_rebuild_image_frame`` replaces the session, conversation, image
        and message-identity fields with fresh values.
        """
        path = Path(__file__).resolve().parent / "captures_cdp" / "events.jsonl"
        if not path.exists():
            return None
        try:
            for line in reversed(path.read_text(encoding="utf-8").splitlines()):
                event = json.loads(line)
                if (
                    event.get("kind") != "ws_frame"
                    or event.get("direction") != "sent"
                    or event.get("opcode") != 2
                ):
                    continue
                frame = base64.b64decode(event.get("payload", ""))
                f8 = get_bytes_field(frame, 8)
                inner = get_bytes_field(f8, 8) if f8 else None
                payload = get_bytes_field(inner, 100) if inner else None
                if payload and _kvmap(payload).get("type") == "file_image":
                    return frame
        except Exception:
            return None
        return None

    @staticmethod
    def _extract_session(frame0):
        top = _kvmap(frame0)
        f8 = get_bytes_field(frame0, 8)
        token = get_bytes_field(f8, 4).decode("utf-8", "replace") if f8 else ""
        sess = _kvmap(f8) if f8 else {}
        return {"token": token, "pigeon_sign": top.get("pigeon_sign", ""),
                "device_id": sess.get("session_did", ""), "aid": sess.get("session_aid", "1383")}

    # ---------- 连接 ----------
    def connect(self, timeout=15):
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()
            threading.Thread(target=self._loop.run_forever, daemon=True).start()
        fut = asyncio.run_coroutine_threadsafe(self._open(timeout), self._loop)
        return fut.result(timeout + 10)

    async def _open(self, timeout):
        import websockets
        self._ws = await websockets.connect(self.session.ws_url(), open_timeout=timeout, max_size=None)
        for idx in [0, 1]:
            await self._ws.send(self._refresh_session_frame(self._handshake[idx]))
            try:
                await asyncio.wait_for(self._ws.recv(), timeout=8)
            except asyncio.TimeoutError:
                pass
        # 补发 10011 注册帧：让服务端把连接归属到店铺（否则不推送任何消息/状态）
        if self._register:
            try:
                await self._ws.send(self._refresh_session_frame(self._register))
                await asyncio.wait_for(self._ws.recv(), timeout=8)
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass
        self._reader_task = asyncio.ensure_future(self._read_loop())
        return True

    async def _read_loop(self):
        try:
            while not self._closed:
                raw = await self._ws.recv()
                if isinstance(raw, bytes):
                    ev = decode_inbound(raw)
                    if ev:
                        # 缓存会话上下文（发送回填用）
                        if ev.get("conv_id"):
                            # 入站消息的 security_receiver_id 是店铺，出站要用买家
                            # security_sender_id；talk_id 是当前会话短 ID（帧内 f14）。
                            state = self.conv_state.setdefault(str(ev["conv_id"]), {})
                            buyer_id = ev.get("security_sender_id") or ev.get("security_receiver_id")
                            if buyer_id:
                                state["security_receiver_id"] = buyer_id
                            if ev.get("shop_id"):
                                state["shop_id"] = ev["shop_id"]
                            if ev.get("talk_id"):
                                state["talk_id"] = str(ev["talk_id"])
                        # 发送 ACK / 商家回声：用独立 event 通知发送方（不与 poll 抢 queue）
                        if ev.get("type") == "is_ack":
                            self._ack_payload = ev.get("raw_kv") or {}
                            if ev.get("conversation_ticket"):
                                self._ack_ticket = ev["conversation_ticket"]
                            if ev.get("conversation_short_id"):
                                self._ack_conversation_short_id = ev["conversation_short_id"]
                            if self._ack_event and not self._ack_event.is_set():
                                self._ack_event.set()
                        if ev.get("text") and ev.get("sender_role") == MERCHANT_ROLE:
                            self._echo_payload = ev
                            self._echo_texts.append(str(ev.get("text") or ""))
                            if len(self._echo_texts) > 30:
                                del self._echo_texts[: len(self._echo_texts) - 30]
                            if self._echo_event and not self._echo_event.is_set():
                                self._echo_event.set()
                            if self._ack_event and not self._ack_event.is_set():
                                self._ack_event.set()
                        await self._queue.put(ev)
        except Exception:
            # LifecycleMixin owns reconnect policy.  Do not silently reconnect
            # here: a failed retry used to leave the UI in a false "connected"
            # state and the message loop polling an abandoned socket.
            if not self._closed:
                self._ws = None

    # ---------- 接收 ----------
    def poll(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """同步轮询，取最近到达的事件列表。"""
        if not self._loop or not self._loop.is_running():
            return []
        fut = asyncio.run_coroutine_threadsafe(self._drain(timeout), self._loop)
        try:
            return fut.result(timeout + 2)
        except Exception:
            return []

    async def _drain(self, timeout):
        """取走当前事件；与拼多多一致，第一条事件到达就立刻返回。

        原实现固定等满 timeout 才返回，WS 已经推来的消息要白等最多 1 秒
        才被派发。这里改为：等到第一条就带上队列里已积压的事件一起返回，
        空队列时才等满 timeout（保持原有的空转频率）。
        """
        out = []
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            out.append(first)
        except asyncio.TimeoutError:
            return out
        while not self._queue.empty():
            out.append(self._queue.get_nowait())
        return out

    def send_text(self, conv_id: str, text: str, security_receiver_id: str = "",
                  shop_id: str = "", talk_id: str = "") -> bool:
        """按飞鸽完整协议发送文本，以商家侧回声确认真实投递。

        2026-11-10 真机实证：会话被系统关闭后重开时，服务端会生成新的
        会话短 ID。直接发单帧 10018（哪怕把 f14 换成历史 talk_id）只会
        拿到 ACK code=0，消息被静默丢弃。必须先跑
        10013/10014/10015 换取当前票据，再发 10016 开启会话。
        """
        if not self._loop or not self._loop.is_running() or self._ws is None:
            self.last_send_result = {
                "ok": False, "reason": "transport_not_connected"
            }
            return False
        state = self.conv_state.get(str(conv_id), {})
        if not security_receiver_id:
            security_receiver_id = state.get("security_receiver_id", "")
        if not shop_id:
            shop_id = state.get("shop_id", "")
        if not security_receiver_id or not shop_id:
            # 会话上下文还没从消息里补齐时，同步补拉一次历史。
            self._refresh_conv_context(conv_id)
            state = self.conv_state.get(str(conv_id), {})
            if not security_receiver_id:
                security_receiver_id = state.get("security_receiver_id", "")
            if not shop_id:
                shop_id = state.get("shop_id", "")
        if not security_receiver_id or not shop_id:
            self.last_send_result = {
                "ok": False,
                "reason": "missing_conversation_context",
                "conv_id": str(conv_id),
                "has_buyer": bool(security_receiver_id),
                "has_shop": bool(shop_id),
            }
            return False
        # 发送串行化：票据状态与回声匹配都是连接级全局量，不能并发。
        with self._send_lock:
            fut = asyncio.run_coroutine_threadsafe(
                self._send_async(conv_id, text, security_receiver_id, shop_id),
                self._loop,
            )
            try:
                return fut.result(25)
            except Exception:
                self.last_send_result = {"ok": False, "reason": "send_exception"}
                return False

    @staticmethod
    def _select_conversation_short_id(ack_short_id, talk_id) -> int:
        """选文本帧 f14 要用的会话短 ID。

        会话列表里的 subConversationId 是目标会话当前的子会话 ID，最可信。
        10015 的 ACK 给的是「服务端当前活跃会话」的短 ID：目标会话已关闭时
        它属于另一个买家，拿它发文本会被静默丢弃（表现为 text_no_echo）。
        """
        talk_text = str(talk_id or "")
        if talk_text.isdigit():
            return int(talk_text)
        try:
            return int(ack_short_id or 0)
        except (TypeError, ValueError):
            return 0

    async def _wait_text_echo(self, text, timeout: float) -> bool:
        """等待「文本完全一致」的商家侧回声。

        不能只等第一条 role=2 消息。真机（2026-11-11 01:12:58，会话
        <talk_id>）复现：买家发消息后平台会自动转人工，并推一条
        ``s:client_message_id = <talk>_Welcome_`` 的 role=2 欢迎语。
        它和我们的回复同角色，谁先到都可能；用 Event 等第一条会在这条欢迎语
        上立刻返回，于是判定 text_no_echo —— 而消息其实已经投递成功了
        （时间线里能看到）。改成轮询比对文本，只认自己发的那一句。
        """
        target = str(text or "")
        if not target:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for seen in list(self._echo_texts):
                if seen == target:
                    return True
            await asyncio.sleep(0.1)
        return False

    async def _send_protocol_flow(self, conv_id, text, security_receiver_id, shop_id):
        """完整发送流程：换票据 → 开会话 → 发文本 → 收尾帧 → 等回声。

        与图片发送 _send_image_async 保持一致，复用抓包里的浏览器帧序列。
        只有收到商家侧回声才判定成功，避免「日志成功、买家未收到」。

        2026-11-10 真机定位：会话关闭后 10013 会返回
        ``request.GetConversationId is error``，此时 10015 给的是服务端
        「当前活跃会话」的短 ID（属于别的买家）。拿它去发文本会被静默丢弃，
        表现就是 text_no_echo。会话列表里的 subConversationId 才是目标会话
        当前的子会话 ID，因此以它为准。
        """
        self._ack_event = asyncio.Event()
        self._echo_event = asyncio.Event()
        self._ack_payload = None
        self._echo_payload = None
        self._ack_ticket = ""
        self._ack_conversation_short_id = 0
        # 每轮发送重新收集回声候选；重试阶段不再清空，避免把已到达的回声丢掉。
        self._echo_texts = []

        state = self.conv_state.setdefault(str(conv_id), {})
        talk_id = str(state.get("talk_id") or "")

        async def send_control(command, pause=0.12):
            frame = self._rebuild_control_frame(
                command, conv_id, security_receiver_id, shop_id
            )
            if frame:
                await self._ws.send(frame)
            await asyncio.sleep(pause)

        async def send_start_and_text(short_id):
            start_frame = self._rebuild_start_conversation_frame(
                conv_id, security_receiver_id, shop_id, self._ack_ticket, short_id
            )
            if not start_frame:
                return "missing_start_template"
            await self._ws.send(start_frame)
            await asyncio.sleep(0.3)
            # 10017：抓包序列里紧跟 10016 的会话确认帧，缺它会丢正文。
            #
            # 2026-11-11 01:10 真机定位：浏览器的完整发送序列是
            #   10013 10014 10015 10016 **10017** 10018 10019 10020 10021
            # 原实现只发了 10016 就直接 10018。会话处于正常接待状态时服务端
            # 宽容，照样投递；一旦会话被平台智能客服接过、或超时关闭后重开，
            # 缺 10017 的正文帧会拿到 ACK code=0 但被静默丢弃（text_no_echo）。
            await send_control(10017, pause=0.12)
            text_frame = self._rebuild_message_frame(
                10018, conv_id, text, security_receiver_id, shop_id,
                self._ack_ticket, short_id,
            )
            if not text_frame:
                return "missing_text_template"
            await self._ws.send(text_frame)
            await asyncio.sleep(0.25)
            for command in (10019, 10020, 10021):
                await send_control(command, pause=0.08)
            return ""

        # 1) 换取当前会话票据与短 ID
        for command in (10013, 10014, 10015):
            await send_control(command)
        deadline = time.monotonic() + 4
        while (not self._ack_ticket or not self._ack_conversation_short_id) \
                and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        short_id = self._ack_conversation_short_id
        short_id = self._select_conversation_short_id(
            self._ack_conversation_short_id, talk_id
        )
        if not short_id:
            self.last_send_result = {
                "ok": False, "reason": "missing_conversation_ticket",
                "conv_id": str(conv_id),
                "has_ticket": bool(self._ack_ticket),
                "has_short_id": bool(short_id),
            }
            return False

        state["conversation_short_id"] = str(short_id)

        # 2) 开启会话 + 发文本（失败时用会话列表短 ID 再试一次）
        error = await send_start_and_text(short_id)
        if error:
            self.last_send_result = {
                "ok": False, "reason": error, "conv_id": str(conv_id),
            }
            return False

        # 3) 只认商家回声（按文本匹配，不按「第一条 role=2 消息」）
        if await self._wait_text_echo(text, 6.0):
            self.last_send_result = {
                "ok": True, "reason": "text_echo_confirmed",
                "conv_id": str(conv_id), "short_id": str(short_id),
            }
            return True

        # 4) 没回声时重试一次：会话刚被激活或票据过期都会导致首帧被丢。
        # 不清空 _echo_texts：首帧若已投递、只是回声晚到，清空会把它丢掉。
        self._echo_event = asyncio.Event()
        self._echo_payload = None
        await asyncio.sleep(0.3)
        error = await send_start_and_text(short_id)
        if not error:
            if await self._wait_text_echo(text, 5.0):
                self.last_send_result = {
                    "ok": True, "reason": "text_echo_confirmed_retry",
                    "conv_id": str(conv_id), "short_id": str(short_id),
                }
                return True

        self.last_send_result = {
            "ok": False, "reason": "text_no_echo", "conv_id": str(conv_id),
            "short_id": str(short_id),
            "talk_id": talk_id,
            "ack": bool(self._ack_payload and self._ack_payload.get("code") == "0"),
            "ack_code": (self._ack_payload or {}).get("code"),
            "ack_msg": (self._ack_payload or {}).get("msg"),
        }
        return False

    async def _send_async(self, conv_id, text, security_receiver_id, shop_id, talk_id=""):
        return await self._send_protocol_flow(
            conv_id, text, security_receiver_id, shop_id
        )

    def _refresh_conv_context(self, conv_id: str) -> bool:
        """用 HTTP 历史补齐单个会话的买家 ID / 店铺 ID / talk_id。

        每个会话独立存进 conv_state[conv_id]，不会串到别的买家。HTTP 补漏
        循环每 15 秒跑一次，这里只用于「WS 首条消息先到」的空窗期。
        """
        try:
            from tools.reverse_probe.douyin.pigeon_http import PigeonHttp
            from utils.runtime_path import get_temp_path
            template_path = str(get_temp_path("dy_usermsg_template.json"))
            http = PigeonHttp(self.session, template_path)
            # 两级窗口：首条一般就在最近几分钟；会话静默较久时再放宽到 3 小时。
            msgs = []
            for window_sec in (60 * 60, 3 * 60 * 60):
                since_us = int(time.time() * 1e6) - int(window_sec * 1e6)
                msgs = http.pull_messages(since_us)
                found = False
                for msg in msgs:
                    if str(msg.get("conv_id") or "") != str(conv_id):
                        continue
                    if msg.get("sender_biz_role") == "Buyer":
                        found = True
                        break
                if found:
                    break
        except Exception:
            return False

        latest = None
        latest_ts = -1
        for msg in msgs:
            if str(msg.get("conv_id") or "") != str(conv_id):
                continue
            if msg.get("sender_biz_role") != "Buyer":
                continue
            ts = msg.get("ts") or 0
            if ts >= latest_ts:
                latest_ts = ts
                latest = msg
        if latest is None:
            return False

        state = self.conv_state.setdefault(str(conv_id), {})
        if latest.get("security_sender_id"):
            state["security_receiver_id"] = latest["security_sender_id"]
        if latest.get("shop_id"):
            state["shop_id"] = latest["shop_id"]
        if latest.get("talk_id"):
            state["talk_id"] = str(latest["talk_id"])
        return True

    def _patch_short_id(self, frame: bytes, talk_id: str) -> bytes:
        """把帧内 f14 会话短 ID 换成当前会话 talk_id。"""
        try:
            f8 = get_bytes_field(frame, 8)
            inner = get_bytes_field(f8, 8)
            payload = get_bytes_field(inner, 100)
            payload = self._replace_varint_field(payload, 14, int(talk_id))
            inner = replace_bytes_field(inner, 100, payload)
            f8 = replace_bytes_field(f8, 8, inner)
            return replace_bytes_field(frame, 8, f8)
        except (TypeError, ValueError):
            return frame

    def send_image(
        self,
        conv_id: str,
        image_url: str,
        width: int = 0,
        height: int = 0,
        security_receiver_id: str = "",
        shop_id: str = "",
    ) -> bool:
        """Send an ImageX/TOS image URL through the verified Pigeon image frame.

        ``image_url`` must already be a Feige-resolvable image URL.  Uploading a
        local file is intentionally kept separate because ImageX applies a
        browser-session signature to every upload request.
        """
        if not self._loop or not self._loop.is_running() or self._ws is None:
            return False
        if not image_url:
            self.last_send_result = {"ok": False, "reason": "missing_image_url"}
            return False
        if not security_receiver_id or not shop_id:
            state = self.conv_state.get(str(conv_id), {})
            security_receiver_id = security_receiver_id or state.get("security_receiver_id", "")
            shop_id = shop_id or state.get("shop_id", "")
        if not security_receiver_id or not shop_id:
            self.last_send_result = {
                "ok": False,
                "reason": "missing_conversation_context",
                "conv_id": str(conv_id),
            }
            return False
        if not self._image_send_tmpl:
            self._image_send_tmpl = self._load_image_send_template()
        if not self._image_send_tmpl:
            self.last_send_result = {"ok": False, "reason": "missing_image_template"}
            return False
        with self._send_lock:
            future = asyncio.run_coroutine_threadsafe(
                self._send_image_async(
                    conv_id,
                    image_url,
                    width,
                    height,
                    security_receiver_id,
                    shop_id,
                ),
                self._loop,
            )
            try:
                return future.result(30)
            except Exception:
                self.last_send_result = {"ok": False, "reason": "image_send_exception"}
                return False

    async def _send_image_async(
        self,
        conv_id,
        image_url,
        width,
        height,
        security_receiver_id,
        shop_id,
    ):
        if self._ack_event is None:
            self._ack_event = asyncio.Event()
        if self._echo_event is None:
            self._echo_event = asyncio.Event()

        async def send_control(command, pause=0.12):
            frame = self._rebuild_control_frame(command, conv_id, security_receiver_id, shop_id)
            if frame:
                await self._ws.send(frame)
                await asyncio.sleep(pause)

        self._ack_event.clear()
        self._ack_payload = None
        self._ack_ticket = ""
        self._ack_conversation_short_id = 0
        for command in (10013, 10014, 10015):
            await send_control(command)

        deadline = time.monotonic() + 3
        while (
            not self._ack_ticket or not self._ack_conversation_short_id
        ) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        ticket = self._ack_ticket
        short_id = self._ack_conversation_short_id
        if not ticket or not short_id:
            self.last_send_result = {
                "ok": False,
                "reason": "missing_conversation_ticket",
                "conv_id": str(conv_id),
            }
            return False

        self._ack_event.clear()
        self._ack_payload = None
        start_frame = self._rebuild_start_conversation_frame(
            conv_id, security_receiver_id, shop_id, ticket, short_id
        )
        if not start_frame:
            self.last_send_result = {"ok": False, "reason": "missing_start_template"}
            return False
        await self._ws.send(start_frame)
        try:
            await asyncio.wait_for(self._ack_event.wait(), timeout=3)
        except asyncio.TimeoutError:
            pass

        self._ack_event.clear()
        self._echo_event.clear()
        self._ack_payload = None
        self._echo_payload = None
        image_frame = self._rebuild_image_frame(
            conv_id,
            image_url,
            width,
            height,
            security_receiver_id,
            shop_id,
            ticket,
            short_id,
        )
        if not image_frame:
            self.last_send_result = {"ok": False, "reason": "invalid_image_template"}
            return False
        await self._ws.send(image_frame)
        await asyncio.sleep(0.35)
        for command in (10019, 10020, 10021):
            await send_control(command, pause=0.08)

        ack_wait = asyncio.create_task(self._ack_event.wait())
        echo_wait = asyncio.create_task(self._echo_event.wait())
        done, pending = await asyncio.wait(
            (ack_wait, echo_wait), timeout=12, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if self._echo_payload and self._echo_payload.get("text") == "[图片]":
            self.last_send_result = {"ok": True, "reason": "image_ack_and_echo"}
            return True
        if self._ack_payload and self._ack_payload.get("code") == "0":
            self.last_send_result = {"ok": True, "reason": "image_ack_without_echo"}
            return True
        self.last_send_result = {
            "ok": False,
            "reason": "image_ack_timeout" if not done else "image_no_ack",
        }
        return False

    def _rebuild_image_frame(
        self,
        conv_id,
        image_url,
        width,
        height,
        security_receiver_id,
        shop_id,
        conversation_ticket="",
        conversation_short_id=0,
    ):
        frame = self._refresh_session_frame(self._image_send_tmpl)
        f8 = get_bytes_field(frame, 8)
        inner = get_bytes_field(f8, 8) if f8 else None
        payload = get_bytes_field(inner, 100) if inner else None
        if not payload or _kvmap(payload).get("type") != "file_image":
            return None
        timestamp_ms = int(time.time() * 1000)
        client_message_id = str(uuid.uuid4())
        payload = self._replace_varint_field(payload, 3, int(conv_id))
        if conversation_short_id:
            payload = self._replace_varint_field(payload, 14, int(conversation_short_id))
        payload = replace_bytes_field(payload, 4, "[图片]".encode("utf-8"))
        payload = self._replace_kv(payload, "security_receiver_id", security_receiver_id)
        payload = self._replace_kv(payload, "shop_id", shop_id)
        payload = self._replace_kv(payload, "imageUrl", image_url)
        if width:
            payload = self._replace_kv(payload, "imageWidth", str(int(width)))
        if height:
            payload = self._replace_kv(payload, "imageHeight", str(int(height)))
        payload = self._replace_kv(payload, "s:client_message_id", client_message_id)
        payload = self._replace_kv(payload, "uuid", str(uuid.uuid4()))
        # f7 会话长票据保留模板原值（同上）
        payload = self._replace_any_field(payload, 8, client_message_id.encode("utf-8"))
        payload = self._replace_any_field(
            payload,
            50,
            self._conversation_key(security_receiver_id, shop_id),
        )
        payload = self._refresh_track_info(payload, timestamp_ms)
        inner = replace_bytes_field(inner, 100, payload)
        f8 = replace_bytes_field(f8, 8, inner)
        frame = replace_bytes_field(frame, 8, f8)
        return self._with_frame_time(frame, timestamp_ms)
        """
        # 用独立 event 等 ACK（不消费 poll 的 queue）
        if self._ack_event is None:
            self._ack_event = asyncio.Event()
        if self._echo_event is None:
            self._echo_event = asyncio.Event()
        self._ack_event.clear()
        self._echo_event.clear()
        self._ack_payload = None
        self._echo_payload = None
        open_ack = None
        if self._open_conversation:
            await self._ws.send(self._rebuild_open_conversation_frame(conv_id))
            # 先等待 10013 的确认。此前固定等待 150ms 后直接清空，延迟到达的
            # 10013 ACK 会被误判成 10018 ACK，日志显示成功而手机端没有任何消息。
            try:
                await asyncio.wait_for(self._ack_event.wait(), timeout=3)
                open_ack = self._ack_payload
            except asyncio.TimeoutError:
                # 某些会话打开不回 ACK；保留兼容性，仍尝试发送文本，但不会拿它充当文本 ACK。
                pass
            self._ack_event.clear()
            self._echo_event.clear()
            self._ack_payload = None
            self._echo_payload = None
        new_frame = self._rebuild_send_frame(conv_id, text, security_receiver_id, shop_id)
        await self._ws.send(new_frame)
        try:
            await asyncio.wait_for(self._ack_event.wait(), timeout=12)
        except asyncio.TimeoutError:
            self.last_send_result = {
                "ok": False,
                "reason": "text_ack_timeout",
                "conv_id": str(conv_id),
                "open_ack": bool(open_ack),
            }
            return False
        if self._ack_payload and self._ack_payload.get("code") == "0":
            # 飞鸽对失效会话也可能返回 code=0。只有收到商家侧回声，才把
            # 发送结果交给上层标记为成功，避免“日志成功、买家未收到”。
            if not self._echo_payload:
                try:
                    await asyncio.wait_for(self._echo_event.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
            if self._echo_payload and self._echo_payload.get("text") == text:
                self.last_send_result = {
                    "ok": True,
                    "reason": "text_ack_and_echo",
                    "conv_id": str(conv_id),
                    "open_ack": bool(open_ack),
                }
                return True
            self.last_send_result = {
                "ok": False,
                "reason": "text_ack_without_echo",
                "conv_id": str(conv_id),
                "open_ack": bool(open_ack),
            }
            return False
        if (self._echo_payload and self._echo_payload.get("text") == text
                and self._echo_payload.get("sender_role") == MERCHANT_ROLE):
            self.last_send_result = {
                "ok": True,
                "reason": "echo",
                "conv_id": str(conv_id),
                "open_ack": bool(open_ack),
            }
            return True
        self.last_send_result = {
            "ok": False,
            "reason": "negative_text_ack",
            "conv_id": str(conv_id),
            "open_ack": bool(open_ack),
        }
        return False
        """

    def _rebuild_open_conversation_frame(self, conv_id: str) -> bytes:
        """以 10013 模板打开当前会话，并刷新时间戳与会话凭据。"""
        frame = self._refresh_session_frame(self._open_conversation)
        f8 = get_bytes_field(frame, 8)
        inner = get_bytes_field(f8, 8)
        payload = get_bytes_field(inner, 2000)
        payload = self._replace_varint_field(payload, 1, int(conv_id))
        inner = replace_bytes_field(inner, 2000, payload)
        f8 = replace_bytes_field(f8, 8, inner)
        now_ms = int(time.time() * 1000)
        f8 = self._replace_varint_field(f8, 2, now_ms)
        frame = replace_bytes_field(frame, 8, f8)
        return self._replace_varint_field(frame, 2, now_ms)

    @staticmethod
    def _conversation_key(security_receiver_id: str, shop_id: str, sender_role: str = "2") -> bytes:
        return ("%s:%s::%s:1:pigeon" % (security_receiver_id, shop_id, sender_role)).encode("utf-8")

    def _with_frame_time(self, frame: bytes, timestamp_ms: int) -> bytes:
        f8 = get_bytes_field(frame, 8)
        if f8:
            f8 = self._replace_varint_field(f8, 2, timestamp_ms)
            frame = replace_bytes_field(frame, 8, f8)
        return self._replace_varint_field(frame, 2, timestamp_ms)

    def _rebuild_control_frame(self, command, conv_id, security_receiver_id, shop_id):
        """Rebuild one non-text frame from the browser's conversation flow."""
        template = self._frames_by_cmd.get(command)
        if not template:
            return None
        frame = self._refresh_session_frame(template)
        f8 = get_bytes_field(frame, 8)
        inner = get_bytes_field(f8, 8)
        if not inner:
            return self._with_frame_time(frame, int(time.time() * 1000))
        conversation_key = self._conversation_key(security_receiver_id, shop_id)
        if command in (10013, 10017, 10019, 10020):
            payload = get_bytes_field(inner, 2000)
            if payload:
                payload = self._replace_varint_field(payload, 1, int(conv_id))
                payload = self._replace_any_field(payload, 10, conversation_key)
                inner = replace_bytes_field(inner, 2000, payload)
        elif command == 10014:
            payload = get_bytes_field(inner, 616)
            if payload:
                payload = self._replace_varint_field(payload, 1, int(conv_id))
                payload = self._replace_any_field(payload, 12, conversation_key)
                inner = replace_bytes_field(inner, 616, payload)
        elif command == 10015:
            payload = get_bytes_field(inner, 610)
            wrapper = get_bytes_field(payload, 1) if payload else None
            if wrapper:
                wrapper = self._replace_varint_field(wrapper, 2, int(conv_id))
                wrapper = self._replace_any_field(wrapper, 20, conversation_key)
                payload = replace_bytes_field(payload, 1, wrapper)
                inner = replace_bytes_field(inner, 610, payload)
        elif command == 10021:
            payload = get_bytes_field(inner, 618)
            wrapper = get_bytes_field(payload, 1) if payload else None
            if wrapper:
                wrapper = self._replace_varint_field(wrapper, 1, int(conv_id))
                wrapper = self._replace_any_field(wrapper, 10, conversation_key)
                payload = replace_bytes_field(payload, 1, wrapper)
                inner = replace_bytes_field(inner, 618, payload)
        f8 = replace_bytes_field(f8, 8, inner)
        frame = replace_bytes_field(frame, 8, f8)
        return self._with_frame_time(frame, int(time.time() * 1000))

    def _refresh_track_info(self, payload: bytes, timestamp_ms: int) -> bytes:
        raw = _kvmap(payload).get("track_info", "")
        try:
            track_info = json.loads(raw)
        except (TypeError, ValueError):
            return payload
        if not isinstance(track_info, dict):
            return payload
        track_info["send_time"] = str(timestamp_ms)
        return self._replace_kv(
            payload, "track_info", json.dumps(track_info, ensure_ascii=False, separators=(",", ":"))
        )

    def _rebuild_message_frame(
        self, command, conv_id, text, security_receiver_id, shop_id, conversation_ticket="",
        conversation_short_id=0,
    ):
        template = self._frames_by_cmd.get(command)
        if not template:
            return None
        timestamp_ms = int(time.time() * 1000)
        frame = self._refresh_session_frame(template)
        f8 = get_bytes_field(frame, 8)
        inner = get_bytes_field(f8, 8)
        payload = get_bytes_field(inner, 100)
        if not payload:
            return None
        template_kv = _kvmap(payload)
        client_message_id = str(uuid.uuid4())
        payload = self._replace_varint_field(payload, 3, int(conv_id))
        if conversation_short_id:
            payload = self._replace_varint_field(payload, 14, int(conversation_short_id))
        if text is not None:
            payload = replace_bytes_field(payload, 4, text.encode("utf-8"))
        payload = self._replace_kv(payload, "security_receiver_id", security_receiver_id)
        payload = self._replace_kv(payload, "shop_id", shop_id)
        payload = self._replace_kv(payload, "s:client_message_id", client_message_id)
        # f7 是「本次发送所属会话」的会话票据。模板里的值是抓包那个会话的，
        # 拿它去发别的会话，服务端照样回 ACK code=0 但静默丢弃消息
        # （2026-11-10 真机：msgList 里始终看不到自己的回复）。
        # 10015 的 ACK 会返回目标会话的当前票据（226 字节），必须写进来。
        ticket_text = str(conversation_ticket or "")
        if len(ticket_text) > 40:
            payload = replace_bytes_field(payload, 7, ticket_text.encode("utf-8"))
        if "p:check_Send" in template_kv:
            payload = self._replace_kv(payload, "p:check_Send", str(uuid.uuid4()))
        payload = self._replace_any_field(payload, 8, client_message_id.encode("utf-8"))
        sender_role = template_kv.get("sender_role") or MERCHANT_ROLE
        if command == 10016 and sender_role != MERCHANT_ROLE:
            # 10016 的 ext 里 sender_role=3 表示「系统从历史会话发起会话」。
            # 抓包环境是「客服本来就在接待中」，用 3 也能发；但会话被平台
            # 智能客服接待过、或超时关闭后重开时，用系统身份开出来的会话
            # 接待权不归坐席，后续正文帧只拿到 ACK code=0 被静默丢弃
            # （2026-11-11 01:06 真机：买家重开会话，发送 ok=False text_no_echo）。
            # 改成 2 = 商家坐席主动接待，接待权才会落到我们这边。
            payload = self._replace_kv(payload, "sender_role", MERCHANT_ROLE)
            sender_role = MERCHANT_ROLE
        payload = self._replace_any_field(
            payload, 50, self._conversation_key(security_receiver_id, shop_id, sender_role)
        )
        payload = self._refresh_track_info(payload, timestamp_ms)
        inner = replace_bytes_field(inner, 100, payload)
        f8 = replace_bytes_field(f8, 8, inner)
        frame = replace_bytes_field(frame, 8, f8)
        return self._with_frame_time(frame, timestamp_ms)

    def _rebuild_start_conversation_frame(
        self, conv_id, security_receiver_id, shop_id, conversation_ticket="", conversation_short_id=0
    ):
        return self._rebuild_message_frame(
            10016, conv_id, None, security_receiver_id, shop_id, conversation_ticket, conversation_short_id
        )

    def _refresh_session_frame(self, frame: bytes) -> bytes:
        """Refresh every session-bound value embedded in a captured frame."""
        frame = self._refresh_pigeon_sign(frame)
        f8 = get_bytes_field(frame, 8)
        if not f8:
            return frame
        if self.session.token:
            f8 = replace_bytes_field(f8, 4, self.session.token.encode("utf-8"))
        if self.session.device_id:
            f8 = replace_bytes_field(f8, 9, self.session.device_id.encode("utf-8"))
            f8 = self._replace_kv_field(f8, 15, "session_did", self.session.device_id)
        if self.session.aid:
            f8 = self._replace_kv_field(f8, 15, "session_aid", str(self.session.aid))
        return replace_bytes_field(frame, 8, f8)

    def _refresh_pigeon_sign(self, frame: bytes) -> bytes:
        """让抓包模板使用当前登录得到的会话级 pigeon_sign。"""
        if not self.session.pigeon_sign:
            return frame
        frame = self._replace_kv_field(frame, 5, "pigeon_sign", self.session.pigeon_sign)
        f8 = get_bytes_field(frame, 8)
        if not f8:
            return frame
        f8_with_sign = self._replace_kv_field(f8, 15, "pigeon_sign", self.session.pigeon_sign)
        return replace_bytes_field(frame, 8, f8_with_sign) if f8_with_sign != f8 else frame

    def _rebuild_send_frame(self, conv_id, text, security_receiver_id, shop_id):
        """基于模板帧重建发送帧（10018）。"""
        tmpl = self._send_tmpl
        new_ts = int(time.time() * 1000)
        # 网页端为这两个去重/追踪字段分别生成 RFC 4122 UUID 字符串
        # （36 位且含连字符）。不能复用抓包模板中的旧值；32 位 hex
        # 虽能得到传输 ACK，却会被业务层静默丢弃。
        new_cmid = str(uuid.uuid4())
        new_check_send = str(uuid.uuid4())

        # 注入当前会话 pigeon_sign（模板可能来自旧会话）：顶层 f5 + f8 内 f15
        tmpl = self._refresh_pigeon_sign(tmpl)

        f8 = get_bytes_field(tmpl, 8)
        inner = get_bytes_field(f8, 8)
        msg_payload = get_bytes_field(inner, 100)

        # 1) 文本 f4
        p2 = replace_bytes_field(msg_payload, 4, text.encode("utf-8"))
        # 2) 会话 id f3
        p2 = self._replace_varint_field(p2, 3, int(conv_id))
        # 3) kv map: security_receiver_id / shop_id / 每条消息专属 UUID
        p2 = self._replace_kv(p2, "security_receiver_id", security_receiver_id)
        p2 = self._replace_kv(p2, "shop_id", shop_id)
        p2 = self._replace_kv(p2, "s:client_message_id", new_cmid)
        p2 = self._replace_kv(p2, "p:check_Send", new_check_send)
        # 4) f8 = client_message_id, f50 = receiver 复合
        p2 = self._replace_any_field(p2, 8, new_cmid.encode())
        if security_receiver_id:
            p2 = self._replace_any_field(p2, 50, ("%s:%s::2:1:pigeon" % (security_receiver_id, shop_id)).encode())

        inner2 = replace_bytes_field(inner, 100, p2)
        f8_2 = replace_bytes_field(f8, 8, inner2)
        f8_2 = self._replace_varint_field(f8_2, 2, new_ts)
        new_frame = replace_bytes_field(tmpl, 8, f8_2)
        new_frame = self._replace_varint_field(new_frame, 2, new_ts)
        return new_frame

    @staticmethod
    def _replace_varint_field(blob, field, new_val):
        fields = parse_fields(blob)
        for i, (f, wt, v, _) in enumerate(fields):
            if f == field and wt == 0:
                fields[i] = (f, wt, new_val, None)
                return rebuild(fields)
        return blob

    @staticmethod
    def _replace_any_field(blob, field, new_val):
        """替换指定 field 的第一个 length-delimited 字段。"""
        fields = parse_fields(blob)
        for i, (f, wt, v, _) in enumerate(fields):
            if f == field and wt == 2:
                fields[i] = (f, wt, bytes(new_val), None)
                return rebuild(fields)
        return blob

    @staticmethod
    def _replace_kv(blob, key, new_val):
        """在 f5 的 kv 子消息中替换指定 key 的 value；无则追加。"""
        fields = parse_fields(blob)
        found = False
        out = []
        for f, wt, v, _ in fields:
            if f == 5 and wt == 2:
                kvs = parse_fields(v)
                is_target = any(sf == 1 and swt == 2 and sv.decode("utf-8", "replace") == key for sf, swt, sv, _ in kvs)
                if is_target:
                    rebuilt = []
                    for sf, swt, sv, _ in kvs:
                        if sf == 2 and swt == 2:
                            rebuilt.append((sf, swt, new_val.encode("utf-8"), None))
                        else:
                            rebuilt.append((sf, swt, sv, None))
                    out.append((f, wt, rebuild(rebuilt), None))
                    found = True
                else:
                    out.append((f, wt, v, None))
            else:
                out.append((f, wt, v, None))
        result = rebuild(out)
        if not found:
            # 追加新 kv
            sub = enc_bytes_field(1, key.encode()) + enc_bytes_field(2, new_val.encode())
            result = result + enc_bytes_field(5, sub)
        return result

    @classmethod
    def _replace_kv_field(cls, blob, kv_field, key, new_val):
        """在当前层指定 kv_field(5 或 15) 的 kv 子消息中替换 key 的值。不递归，安全。"""
        fields = parse_fields(blob)
        changed = False
        out = []
        for f, wt, v, _ in fields:
            if f == kv_field and wt == 2:
                kvs = parse_fields(v)
                is_target = any(sf == 1 and swt == 2 and sv.decode("utf-8", "replace") == key for sf, swt, sv, _ in kvs)
                if is_target:
                    rebuilt = []
                    for sf, swt, sv, _ in kvs:
                        if sf == 2 and swt == 2:
                            rebuilt.append((sf, swt, new_val.encode("utf-8"), None))
                        else:
                            rebuilt.append((sf, swt, sv, None))
                    out.append((f, wt, rebuild(rebuilt), None))
                    changed = True
                else:
                    out.append((f, wt, v, None))
            else:
                out.append((f, wt, v, None))
        return rebuild(out) if changed else blob

    def close(self):
        self._closed = True
        if PigeonTransport._instance is self:
            PigeonTransport._instance = None
        loop = self._loop
        if loop and loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(self._close_ws(), loop)
                fut.result(5)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
        self._ws = None
        self._loop = None

    async def _close_ws(self):
        reader_task = getattr(self, "_reader_task", None)
        if reader_task and not reader_task.done():
            reader_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
