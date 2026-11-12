"""Live product collection from the authenticated Douyin merchant browser.

The merchant product-list request requires short-lived browser anti-bot
parameters.  Replaying it with copied cookies is unreliable, so this source
opens a normal Edge instance using the account's saved profile and reads the
JSON response that the merchant page itself receives.
"""
from __future__ import annotations

import hashlib
import json
import re
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from websockets.sync.client import connect

from utils.logger_loguru import get_logger
from utils.runtime_path import get_data_path, get_temp_path

logger = get_logger("DouyinProductSource")


class _CdpClient:
    """Small synchronous CDP client which keeps protocol events until read."""

    def __init__(self, url: str):
        self._ws = connect(url, open_timeout=5)
        self._next_id = 1
        self._events: List[Dict[str, Any]] = []

    def close(self) -> None:
        self._ws.close()

    def call(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 8):
        request_id = self._next_id
        self._next_id += 1
        self._ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = json.loads(self._ws.recv(timeout=max(0.1, deadline - time.monotonic())))
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(message["error"].get("message", "CDP command failed"))
                return message.get("result", {})
            self._events.append(message)
        raise TimeoutError(f"CDP command timed out: {method}")

    def next_event(self, predicate, timeout: float = 30) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for index, event in enumerate(self._events):
                if predicate(event):
                    return self._events.pop(index)
            message = json.loads(self._ws.recv(timeout=max(0.1, deadline - time.monotonic())))
            if predicate(message):
                return message
            self._events.append(message)
        raise TimeoutError("Expected merchant product response was not received")


class DouyinProductSource:
    """Fetch product pages through a fresh normal-Edge merchant session."""

    PRODUCT_PAGE_URL = "https://fxg.jinritemai.com/ffa/g/list"
    PRODUCT_LIST_PATH = "/product/tproduct/list"
    EDGE_CANDIDATES = (
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    )

    def __init__(self, shop_id: str, user_id: str, timeout: float = 45,
                 profile_dir: Optional[Path] = None, browser_path: Optional[Path] = None):
        self.shop_id = str(shop_id)
        self.user_id = str(user_id)
        self.timeout = timeout
        self.profile_dir = profile_dir or self._profile_dir()
        self.browser_path = browser_path or self._find_edge()

    def fetch_product_list(self, page: int, size: int) -> Dict[str, Any]:
        """Return a normalized product page or a non-sensitive failure reason."""
        try:
            payload = self._fetch_product_payload(page, size)
            rows = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ValueError("unexpected_product_list_response")
            products = [self.normalize_product(item) for item in rows if isinstance(item, dict)]
            products = [item for item in products if item is not None]
            return {
                "success": True,
                "products": products,
                "total": self._as_int(payload.get("total"), len(products)),
                "page": self._as_int(payload.get("page"), page),
                "size": self._as_int(payload.get("size"), len(products)),
            }
        except Exception as exc:
            return {
                "success": False,
                "products": [],
                "total": 0,
                "page": page,
                "size": size,
                "error_msg": f"抖店商品页读取失败: {type(exc).__name__}",
            }

    def _inject_session_cookies(self, client: _CdpClient) -> int:
        """把登录时抓到的完整 cookies 注入本次临时浏览器会话。

        商品页抓取用的是独立 Edge profile，与主程序登录用的浏览器不是同一个；
        不注入登录态，页面会停在登录页，等不到 /product/tproduct/list 响应，
        最终 45 秒超时。凭据统一取 temp/dy_session_live.json（登录时写入）。
        """
        session_path = get_temp_path("dy_session_live.json")
        if not session_path.exists():
            return 0
        try:
            with open(session_path, encoding="utf-8") as fp:
                session = json.load(fp)
        except (OSError, ValueError):
            return 0

        cookies = session.get("cookies_full")
        if not isinstance(cookies, dict) or not cookies:
            return 0

        entries = []
        for name, value in cookies.items():
            entries.append({
                "name": str(name),
                "value": str(value),
                "url": "https://fxg.jinritemai.com/",
            })
        try:
            client.call("Network.setCookies", {"cookies": entries}, timeout=10)
            logger.info(f"抖店商品页注入登录 cookies: {len(entries)} 个")
            return len(entries)
        except Exception as exc:
            logger.warning(
                f"抖店商品页注入登录 cookies 失败: error_type={type(exc).__name__}"
            )
            return 0

    def _fetch_product_payload(self, page: int, size: int) -> Dict[str, Any]:
        port = self._free_port()
        process = subprocess.Popen(
            [
                str(self.browser_path),
                f"--remote-debugging-port={port}",
                f"--user-data-dir={self.profile_dir}",
                "--no-first-run",
                "--no-default-browser-check",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = None
        try:
            target = self._wait_for_target(port)
            client = _CdpClient(target["webSocketDebuggerUrl"])
            client.call("Network.enable")
            self._inject_session_cookies(client)
            client.call("Page.navigate", {"url": self._page_url(page, size)})
            payload = self._read_product_response(client)
            # The merchant page does not honor page/pageSize URL parameters.
            # Use its own paginator so every request keeps a valid a_bogus token.
            for _ in range(1, max(1, int(page))):
                result = client.call("Runtime.evaluate", {
                    "expression": self._NEXT_PAGE_EXPRESSION,
                    "returnByValue": True,
                })
                clicked = result.get("result", {}).get("value", False)
                if not clicked:
                    raise ValueError("merchant_product_page_unavailable")
                payload = self._read_product_response(client)
            return payload
        finally:
            if client is not None:
                client.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    @classmethod
    def normalize_product(cls, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        goods_id = cls._as_int(cls._pick(item, "product_id", "productId", "goods_id", "goodsId", "id"), None)
        if goods_id is None:
            return None
        price = cls._pick(item, "price", "sale_price", "salePrice", "min_price", "minPrice")
        price_min = cls._as_int(cls._pick(item, "min_price", "minPrice", "price_min"), None)
        price_max = cls._as_int(cls._pick(item, "max_price", "maxPrice", "price_max"), None)
        specs = cls._pick(item, "specifications", "specs", "sku_spec", "skuSpec", default=[])
        if not isinstance(specs, list):
            specs = [str(specs)] if specs else []
        return {
            "goods_id": goods_id,
            "goods_name": str(cls._pick(item, "name", "product_name", "productName", "title", "goods_name", "goodsName", default="")),
            "thumb_url": str(cls._pick(item, "img", "image", "image_url", "imageUrl", "cover", "cover_url", "coverUrl", "thumb_url", default="")),
            "price": "" if price is None else str(price),
            "price_min": price_min,
            "price_max": price_max,
            "sold_quantity": cls._as_int(cls._pick(item, "sold_num", "soldNum", "sold_quantity", "soldQuantity"), 0),
            "quantity": cls._as_int(cls._pick(item, "stock_num", "stockNum", "quantity", "stock"), 0),
            "goods_type": str(cls._pick(item, "product_type", "productType", "goods_type", "goodsType", default="")),
            "tag": str(cls._pick(item, "tag", "tags", "label", "labels", default="")),
            "goods_url": str(cls._pick(item, "product_url", "productUrl", "goods_url", "goodsUrl", default="")),
            "is_spike": bool(cls._pick(item, "is_spike", "isSpike", default=False)),
            "support_customize": bool(cls._pick(item, "support_customize", "supportCustomize", default=False)),
            "_specifications": specs,
            "_description": str(cls._pick(item, "description", "desc", "detail", default="")),
        }

    @staticmethod
    def _pick(data: Dict[str, Any], *names: str, default=None):
        for name in names:
            value = data.get(name)
            if value not in (None, ""):
                return value
        return default

    @staticmethod
    def _as_int(value: Any, default: Optional[int]) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _profile_dir(self) -> Path:
        scope = f"douyin:{self.shop_id}:{self.user_id}"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", scope).strip("._")[:48] or "account"
        digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:12]
        return get_data_path() / "user_data" / f"{safe}-{digest}"

    @classmethod
    def _find_edge(cls) -> Path:
        for candidate in cls.EDGE_CANDIDATES:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Microsoft Edge was not found")

    @staticmethod
    def _wait_for_target(port: int) -> Dict[str, Any]:
        deadline = time.monotonic() + 15
        endpoint = f"http://127.0.0.1:{port}/json/list"
        while time.monotonic() < deadline:
            try:
                targets = json.loads(urllib.request.urlopen(endpoint, timeout=1).read().decode("utf-8"))
                target = next((item for item in targets if item.get("type") == "page"), None)
                if target:
                    return target
            except Exception:
                pass
            time.sleep(0.2)
        raise TimeoutError("normal Edge CDP endpoint was unavailable")

    def _page_url(self, page: int, size: int) -> str:
        # Pagination is driven by the page UI after navigation; see
        # ``_fetch_product_payload``.  Keeping this URL clean avoids sending
        # an invalid anti-bot signature for modified query parameters.
        return self.PRODUCT_PAGE_URL

    @staticmethod
    def _read_product_response(client: _CdpClient) -> Dict[str, Any]:
        response = client.next_event(DouyinProductSource._is_product_response, timeout=45)
        request_id = response["params"]["requestId"]
        client.next_event(
            lambda event: event.get("method") == "Network.loadingFinished"
            and event.get("params", {}).get("requestId") == request_id,
            timeout=45,
        )
        body = client.call("Network.getResponseBody", {"requestId": request_id})
        payload = json.loads(body.get("body", "{}"))
        if not isinstance(payload, dict) or payload.get("code") not in (0, "0"):
            raise ValueError("merchant_product_api_rejected")
        return payload

    @classmethod
    def _is_product_response(cls, event: Dict[str, Any]) -> bool:
        if event.get("method") != "Network.responseReceived":
            return False
        url = event.get("params", {}).get("response", {}).get("url", "")
        return urlsplit(url).path == cls.PRODUCT_LIST_PATH

    _NEXT_PAGE_EXPRESSION = """(() => {
        const direct = document.querySelector(
            '.arco-pagination-next:not(.arco-pagination-item-disabled),'
            + '.ant-pagination-next:not(.ant-pagination-disabled)'
        );
        const candidates = [...document.querySelectorAll('button, a, li')];
        const next = direct || candidates.find((node) => {
            const label = [node.innerText, node.getAttribute('aria-label'), node.title]
                .filter(Boolean).join(' ');
            return /\\u4e0b\\u4e00\\u9875|next/i.test(label) && !node.disabled
                && !node.className.includes('disabled');
        });
        if (!next) return false;
        next.click();
        return true;
    })()"""


__all__ = ["DouyinProductSource"]
