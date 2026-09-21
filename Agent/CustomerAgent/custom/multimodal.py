"""买家图片的多模态组装、校验与历史持久化。

买家发来的图片在下游表现为两种情况：当前这一轮的 ``Context.type`` 是
``IMAGE``、``content`` 是图片 URL；而写进会话历史后只剩一段文本。本模块
负责把 URL 还原成模型能直接理解的 image_url 内容块，并让这份信息跨越会话
历史存活下来。

三个约束决定了这里的设计：

1. 历史必须能还原图片。买家先发图、再追问「这个多少钱」时，图片只存在于
   历史里；只用 "[图片] URL" 这种纯文本，模型看不到图。因此带图的用户消息
   用一层显式信封（envelope）持久化，同时保留人可读的原文本。
2. 图片 URL 是不可信输入。把它交给供应商等于让对方去抓这个地址，所以先做
   一次协议层校验，挡掉内网、回环、链路本地与元数据地址。
3. 不预判供应商的视觉能力。LiteLLM 的能力表对 volcengine / qwen / zhipu 的
   视觉模型同样返回不支持（实测 doubao-1-5-vision-pro 与 qwen-vl-max 均为
   False），用它当开关会正好在支持视觉的模型上关掉图片。本模块只负责组装，
   是否降级由 CustomerAgent 依据真实请求结果决定。
"""
from __future__ import annotations

import ipaddress
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlsplit

from bridge.context import Context, ContextType


# 带图用户消息的持久化信封键。历史内容形如：
# {"__multimodal__": {"text": "...", "images": ["https://..."]}}
ENVELOPE_KEY = "__multimodal__"

# 历史里最多还原多少条带图消息。图片 token 成本远高于文本，且每次请求都会
# 重新下发；只保留最近几条，既覆盖「发完图紧接着追问」的常见节奏，又不会让
# 上下文被历史图片撑满。
DEFAULT_HISTORY_IMAGE_LIMIT = 2

# 单条 URL 长度上限，避免把畸形报文塞进请求
MAX_URL_LENGTH = 2048

# 明确指向本机/内网的域名，无需解析即可拒绝
_BLOCKED_HOSTS = frozenset({"localhost", "metadata.google.internal"})
_BLOCKED_HOST_SUFFIXES = (".local", ".localhost", ".internal")


def _is_safe_image_url(url: Any) -> bool:
    """校验一个图片 URL 是否可以交给模型供应商去抓取。

    只做协议层判断：必须是 http(s)、不带账号密码、主机不是字面量内网 IP、
    不是本机/内网域名。域名不做 DNS 解析——解析结果并不约束供应商那边的
    行为，却会引入网络依赖；字面量 IP 与元数据主机已覆盖主要风险面。
    """
    text = str(url or "").strip()
    if not text or len(text) > MAX_URL_LENGTH:
        return False

    try:
        parsed = urlsplit(text)
    except ValueError:
        return False

    if parsed.scheme.lower() not in ("http", "https"):
        return False
    # user:pass@host 形式既能藏身份也可能绕过主机判断
    if parsed.username or parsed.password:
        return False

    host = (parsed.hostname or "").rstrip(".")
    if not host:
        return False

    lowered = host.lower()
    if lowered in _BLOCKED_HOSTS or lowered.endswith(_BLOCKED_HOST_SUFFIXES):
        return False

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # 域名：放行，交由供应商解析
        return True

    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
        or not address.is_global
    )


def validate_image_urls(urls: Optional[Iterable[Any]]) -> List[str]:
    """过滤并去重图片 URL，保持原有顺序。"""
    if not urls:
        return []

    accepted: List[str] = []
    seen = set()
    for url in urls:
        if not isinstance(url, (str, bytes)):
            continue
        text = str(url).strip()
        if text in seen or not _is_safe_image_url(text):
            continue
        seen.add(text)
        accepted.append(text)
    return accepted


def _urls_from_content(content: Any) -> List[str]:
    """从消息内容里取图片 URL。

    拼多多的图片消息 content 就是 URL 字符串；同时兼容早期
    create_image_message 产出的 [{"type": "image", "url": ...}] 结构。
    """
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                return [text]
            if isinstance(parsed, list):
                return [
                    item.get("url")
                    for item in parsed
                    if isinstance(item, dict) and item.get("type") == "image"
                ]
        return [text]

    if isinstance(content, list):
        return [
            item.get("url")
            for item in content
            if isinstance(item, dict) and item.get("type") == "image"
        ]

    return []


def extract_context_images(context: Optional[Context]) -> List[str]:
    """取当前这条消息附带的图片 URL。

    只认 ContextType.IMAGE：买家完全可以在文本里粘一个链接，那是文本内容，
    不能因此触发一次图片抓取。
    """
    if context is None or getattr(context, "type", None) is not ContextType.IMAGE:
        return []
    return validate_image_urls(_urls_from_content(getattr(context, "content", None)))


def image_content_blocks(urls: Iterable[Any]) -> List[Dict[str, Any]]:
    """把图片 URL 转成 OpenAI 多模态内容块。"""
    return [
        {"type": "image_url", "image_url": {"url": url}}
        for url in validate_image_urls(urls)
    ]


def build_user_content(
    text: Any,
    urls: Optional[Iterable[Any]] = None,
) -> Union[str, List[Dict[str, Any]]]:
    """构建一条 user 消息的 content。

    没有可用图片时返回纯字符串——与引入多模态之前完全一致，避免为了新能力
    改变既有链路的请求形态。有图片时返回内容块列表，文本块在前。
    """
    safe_text = "" if text is None else str(text)
    blocks = image_content_blocks(urls or [])
    if not blocks:
        return safe_text
    if safe_text:
        blocks.insert(0, {"type": "text", "text": safe_text})
    return blocks


def encode_history_content(text: Any, urls: Optional[Iterable[Any]] = None) -> str:
    """持久化用户消息；带图时用信封包裹，否则原样保存。

    买家文本是作为 JSON 的值写进去的，无法逃逸出信封结构。
    """
    safe_text = "" if text is None else str(text)
    images = validate_image_urls(urls or [])
    if not images:
        return safe_text
    return json.dumps(
        {ENVELOPE_KEY: {"text": safe_text, "images": images}},
        ensure_ascii=False,
    )


def decode_history_content(content: Any) -> Tuple[str, List[str]]:
    """还原历史内容为 (文本, 图片列表)。

    非信封内容原样返回，因此引入多模态之前的历史记录不受影响。

    已知边界：买家若原样发来一段形如信封的 JSON 文本，会被当成信封解析。
    由于图片仍需通过 URL 校验，最坏结果只是供应商去抓一个买家指定的公网
    图片（内网地址已被挡掉），不涉及我们的凭据或内网服务。
    """
    if not isinstance(content, str):
        return ("" if content is None else str(content)), []

    text = content.strip()
    if not text.startswith("{"):
        return content, []

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return content, []

    if not isinstance(payload, dict):
        return content, []

    body = payload.get(ENVELOPE_KEY)
    if not isinstance(body, dict):
        return content, []

    inner_text = body.get("text")
    if not isinstance(inner_text, str):
        return content, []

    return inner_text, validate_image_urls(body.get("images"))


def history_image_plan(
    history: Optional[List[Dict[str, Any]]],
    limit: int = DEFAULT_HISTORY_IMAGE_LIMIT,
) -> Dict[int, List[str]]:
    """决定历史里哪几条消息还原图片，返回 {下标: 图片列表}。

    只取最近 limit 条带图消息：更早的图片既不常被追问，又会让每次请求都
    重复携带图片。
    """
    if not history or limit <= 0:
        return {}

    plan: Dict[int, List[str]] = {}
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        _, images = decode_history_content(message.get("content"))
        if not images:
            continue
        plan[index] = images
        if len(plan) >= limit:
            break
    return plan


def has_image_blocks(messages: Optional[List[Dict[str, Any]]]) -> bool:
    """消息列表里是否存在图片内容块。"""
    if not messages:
        return False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(block, dict) and block.get("type") == "image_url"
            for block in content
        ):
            return True
    return False


def strip_image_blocks(
    messages: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """返回去掉图片内容块的消息副本，其余字段（含 tool_calls）原样保留。

    只含图片的消息不能直接丢弃：那会让买家那一轮在对话里凭空消失。此时
    退化成一段文本占位，保留「买家发过图」这个事实。
    """
    stripped: List[Dict[str, Any]] = []
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(message)
            continue

        parts = [
            block
            for block in content
            if not (isinstance(block, dict) and block.get("type") == "image_url")
        ]
        if not parts:
            parts = [{"type": "text", "text": "[图片]"}]

        replacement = dict(message)
        if len(parts) == 1 and parts[0].get("type") == "text":
            replacement["content"] = parts[0].get("text", "")
        else:
            replacement["content"] = parts
        stripped.append(replacement)
    return stripped


__all__ = [
    "DEFAULT_HISTORY_IMAGE_LIMIT",
    "ENVELOPE_KEY",
    "MAX_URL_LENGTH",
    "build_user_content",
    "decode_history_content",
    "encode_history_content",
    "extract_context_images",
    "has_image_blocks",
    "history_image_plan",
    "image_content_blocks",
    "strip_image_blocks",
    "validate_image_urls",
]
