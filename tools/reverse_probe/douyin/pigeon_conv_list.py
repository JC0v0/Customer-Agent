# -*- coding: utf-8 -*-
"""抖音飞鸽「当前会话列表」接收模块：覆盖所有会话，包括新接入的会话。

背景：v3 `get_user_message` 只返回最近一个活跃父会话的消息。新买家接入后
消息落在新的父会话里，它完全不返回，表现就是「买家发了消息，程序毫无反应」。
会话列表接口只靠登录 cookie 鉴权（不需要 a_bogus），响应里直接带每个会话
最近几条消息正文，因此用它做主接收路径即可覆盖新会话。

关键字段（真机 2026-11-10 22:35 实证）：
  data[i].parentConShortId                父会话 ID（派发 / 发送用的 conv_id）
  data[i].subConversationId               子会话 ID（talk_id，每次接入会变）
  data[i].coreInfoMap.service_entity_id   shop_id
  data[i].msgList[j].messageBody:
      msgType = 1000                      文本消息
      content                             正文
      createTime                          毫秒时间戳
      ext["s:sender_biz_role"] == "Buyer" 买家消息（其余为商家 / 系统）
      ext["security_sender_id"]           买家 ID，回复时作 security_receiver_id
      ext["message_logid"]                全局唯一消息 ID，跨接口去重
"""
import http.client
import json
import time
from typing import Any, Dict, List, Optional

from Channel.douyin.message_rules import (
    Origin,
    classify_merchant_sender,
    resolve_origin,
)

MSG_TYPE_TEXT = 1000
BUYER_ROLE = "Buyer"


class SessionExpiredError(RuntimeError):
    """登录态失效（接口返回 code=10005）。

    单独成类，让上层能区分「网络抖动，重试即可」和「必须重新登录」。
    混在 ValueError 里只会得到一句 conversation_list_error，看不出原因。
    """


class PigeonConvList:
    """拉取当前会话列表里的买家消息（纯 HTTP + cookie）。"""

    HOST = "pigeon.jinritemai.com"
    PATH = "/chat/api/backstage/conversation/get_current_conversation_list"
    QUERY = (
        "?biz_type=4&PIGEON_BIZ_TYPE=2&pageNo=0&pageSize=200"
        "&_pms=1&device_platform=web&FUSION=true"
    )
    # 实测 keep-alive 复用连接 0.2 秒级返回；新建 TLS 连接每次多 0.13 秒。
    # 拉取循环每轮都要请求一次，这个差额直接体现在「收到消息快不快」上。
    # 留 6 秒兜底，失败立刻退到下一个源。
    TIMEOUT_SECONDS = 6.0

    def __init__(self, session_path: str):
        self.session_path = str(session_path)
        self.last_status: Optional[int] = None
        self.last_error: Optional[str] = None
        self._conn: Optional[http.client.HTTPSConnection] = None

    def _close_conn(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _headers(self) -> Dict[str, str]:
        return {
            "Cookie": self._cookie_header(),
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Referer": "https://im.jinritemai.com/pc_seller_v2/main/workspace",
        }

    def _cookie_header(self) -> str:
        with open(self.session_path, encoding="utf-8") as fp:
            session = json.load(fp)
        cookies = session.get("cookies_full") or {}
        if not cookies:
            raise ValueError("session_cookies_missing")
        parts = []
        for name, value in cookies.items():
            parts.append("%s=%s" % (name, value))
        return "; ".join(parts)

    def fetch_conversations(self) -> List[Dict[str, Any]]:
        """返回原始会话列表；失败抛异常，由调用方决定是否降级。"""
        self.last_status = None
        self.last_error = None
        payload = self._get_json()
        if payload.get("code") != 0:
            code = payload.get("code")
            msg = str(payload.get("msg") or payload.get("extraMsg") or "")
            self.last_status = code
            self.last_error = "code=%s msg=%s" % (code, msg)
            if str(code) == "10005":
                raise SessionExpiredError(
                    "登录过期，请重新登录 (code=%s, msg=%s)" % (code, msg)
                )
            raise ValueError(
                "conversation_list_error: code=%s msg=%s" % (code, msg)
            )
        return payload.get("data") or []

    def _get_json(self) -> Dict[str, Any]:
        """带连接复用的 GET；连接失效时重连一次再放弃。"""
        headers = self._headers()
        path = self.PATH + self.QUERY
        attempt = 0
        while attempt < 2:
            attempt += 1
            try:
                if self._conn is None:
                    self._conn = http.client.HTTPSConnection(
                        self.HOST, timeout=self.TIMEOUT_SECONDS
                    )
                self._conn.request("GET", path, headers=headers)
                response = self._conn.getresponse()
                raw = response.read()
                status = response.status
                if status != 200:
                    raise ValueError("http_status_%s" % status)
                return json.loads(raw.decode("utf-8", "replace"))
            except Exception as exc:
                # 复用连接可能被服务端关掉（假死/超时），第一次失败重连再试，
                # 第二次仍失败才把异常交给上层记日志。
                self._close_conn()
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                if attempt >= 2:
                    raise
        raise ValueError("conversation_list_unreachable")

    @staticmethod
    def _to_us(value) -> Optional[int]:
        """毫秒时间戳转微秒；异常值返回 None。"""
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        if number < 10 ** 12:
            return number * 1000
        if number < 10 ** 15:
            return number * 1000
        return number

    def fetch_buyer_messages(self, since_us: int = 0) -> List[Dict[str, Any]]:
        """返回所有会话里 since_us 之后的买家文本消息。"""
        return self.fetch_all(since_us)["buyer"]

    def fetch_all(self, since_us: int = 0) -> Dict[str, List[Dict[str, Any]]]:
        """一次请求同时取回买家消息与商家侧消息。

        为什么必须一次拿两类：人工客服介入的判定依据就是「商家侧有新消息」，
        而这次请求本来就要发。分成两次请求只会让接收更慢，没有任何好处。
        """
        buyer_msgs: List[Dict[str, Any]] = []
        merchant_msgs: List[Dict[str, Any]] = []
        for conv in self.fetch_conversations():
            conv_id = str(conv.get("parentConShortId") or "")
            if not conv_id:
                continue
            talk_id = str(conv.get("subConversationId") or "")
            core_map = conv.get("coreInfoMap") or {}
            default_shop_id = str(core_map.get("service_entity_id") or "")
            for item in conv.get("msgList") or []:
                body = item.get("messageBody") or {}
                if body.get("msgType") != MSG_TYPE_TEXT:
                    continue
                ext = body.get("ext") or {}
                # 来源判定统一走 message_rules：平台机器人提示同样是
                # msgType=1000 且带正文，只看正文会把「平台可能主动介入」
                # 这类系统消息当成买家发言交给 AI。
                origin = resolve_origin(ext)
                if origin is Origin.MERCHANT:
                    merchant_item = self._build_merchant_item(
                        conv_id, talk_id, default_shop_id, item, body, ext, since_us
                    )
                    if merchant_item is not None:
                        merchant_msgs.append(merchant_item)
                    continue
                if origin is not Origin.CUSTOMER:
                    continue
                # 「客服XX接入」这类系统提示的 sender_biz_role 同样是 Buyer，
                # 但没有 security_sender_id；真实买家消息一定带这个买家 ID，
                # 而它又是回复时的 security_receiver_id，必须有值才能发出去。
                buyer_id = str(ext.get("security_sender_id") or "")
                if not buyer_id:
                    continue
                text = str(body.get("content") or "").strip()
                if not text or text.startswith("{"):
                    continue
                ts = self._to_us(body.get("createTime"))
                if ts is not None and ts <= since_us:
                    continue
                buyer_msgs.append({
                    "type": MSG_TYPE_TEXT,
                    "conv_id": conv_id,
                    "msg_id": str(ext.get("message_logid") or item.get("serverMessageId") or ""),
                    "text": text,
                    "ts": ts if ts is not None else int(time.time() * 1e6),
                    "sender_biz_role": BUYER_ROLE,
                    "security_sender_id": buyer_id,
                    "shop_id": str(ext.get("receiver_id") or default_shop_id),
                    "talk_id": talk_id,
                    "client_message_id": str(ext.get("s:client_message_id") or ""),
                })
        return {"buyer": buyer_msgs, "merchant": merchant_msgs}

    def _build_merchant_item(self, conv_id, talk_id, default_shop_id,
                             item, body, ext, since_us) -> Optional[Dict[str, Any]]:
        """商家侧文本消息 → 人工介入判定用的字段集合。"""
        text = str(body.get("content") or "").strip()
        if not text or text.startswith("{"):
            return None
        ts = self._to_us(body.get("createTime"))
        if ts is not None and ts <= since_us:
            return None
        sender_kind = classify_merchant_sender(ext)
        return {
            "conv_id": conv_id,
            "talk_id": talk_id,
            "text": text,
            "ts": ts if ts is not None else int(time.time() * 1e6),
            "msg_id": str(ext.get("message_logid") or item.get("serverMessageId") or ""),
            "shop_id": str(ext.get("shop_id") or default_shop_id),
            # 商家侧消息里的买家 ID：回复目标，也是转人工会话的归属人。
            "buyer_uid": str(ext.get("security_receiver_id") or ""),
            "sender_kind": sender_kind,
            "src_user_id": str(ext.get("src_user_id") or ""),
            "src": str(ext.get("src") or ""),
        }
