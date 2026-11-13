"""Regression coverage for live Douyin product collection and KB isolation."""
import asyncio
import json
from pathlib import Path
from unittest import mock

from Channel.douyin.douyin_login import DouyinLogin
from Channel.douyin.utils.API.product_manager import ProductManager
from Channel.douyin.utils.API.product_source import DouyinProductSource
from database.product_sync import ProductSyncService


def test_product_source_normalizes_merchant_product_fields():
    product = DouyinProductSource.normalize_product({
        "product_id": "998877",
        "product_name": "测试商品",
        "image_url": "https://example.invalid/product.jpg",
        "salePrice": "19.90",
        "stockNum": "7",
        "soldNum": "12",
        "specs": ["颜色: 红色"],
        "description": "商品描述",
    })

    assert product["goods_id"] == 998877
    assert product["goods_name"] == "测试商品"
    assert product["price"] == "19.90"
    assert product["quantity"] == 7
    assert product["_specifications"] == ["颜色: 红色"]


def test_product_manager_uses_live_source_for_required_platform_sync():
    source = mock.Mock()
    source.fetch_product_list.return_value = {
        "success": True,
        "products": [{
            "goods_id": 123456,
            "goods_name": "抖音商品",
            "price": "29.90",
            "thumb_url": "",
            "sold_quantity": 0,
            "_specifications": ["规格: 标准"],
            "_description": "平台商品详情",
        }],
        "total": 1,
        "page": 1,
        "size": 20,
    }
    with mock.patch(
        "Channel.douyin.utils.base_request.BaseRequest._init_account_info",
        lambda self: None,
    ):
        manager = ProductManager("douyin_shop", "douyin_user", product_source=source)

    page = manager.get_product_list(page=1, size=20, require_platform=True)
    detail = manager.get_product_detail(123456)

    source.fetch_product_list.assert_called_once_with(page=1, size=20)
    assert page["success"] is True
    assert detail["success"] is True
    assert detail["product_info"]["description"] == "平台商品详情"


def test_product_sync_requires_platform_source_for_douyin():
    manager = mock.Mock()
    manager.get_product_list.return_value = {
        "success": True, "products": [], "total": 0, "page": 1, "size": 20,
    }
    knowledge_service = mock.Mock()
    service = ProductSyncService(knowledge_service, request_delay=0)

    with mock.patch("database.product_sync._get_product_manager", return_value=manager):
        progress = asyncio.run(service.sync_shop(
            shop_id="douyin_shop",
            shop_db_id=1,
            user_id="douyin_user",
            channel_name="douyin",
        ))

    assert progress.total == 0
    assert manager.get_product_list.call_args.kwargs["require_platform"] is True


def test_visible_douyin_login_prefers_normal_edge_cdp():
    login = DouyinLogin("13800000000", "")
    sentinel = object()

    async def launch():
        with mock.patch.object(login, "_resolve_browser_path", return_value=("edge.exe", "msedge")), \
             mock.patch.object(login, "_clean_profile_locks"), \
             mock.patch.object(login, "_launch_via_cdp", new=mock.AsyncMock(return_value=sentinel)):
            return await login._launch_context(mock.Mock(), "profile", headless=False)

    assert asyncio.run(launch()) is sentinel


def test_product_manager_never_opens_browser_for_ai_queries():
    """AI 回复期间查商品必须只读知识库，不能拉起浏览器抓商品页。

    抓商品页要额外启动一个 Edge 实例，单次可达 40 秒，会把回复拖慢一整轮，
    还会因为共用登录凭据把主会话挤掉。只有「商品同步」显式传
    require_platform=True 时才允许走平台抓取。
    """
    source = mock.Mock()
    source.fetch_product_list.return_value = {
        "success": True, "products": [], "total": 0, "page": 1, "size": 20,
    }
    with mock.patch(
        "Channel.douyin.utils.base_request.BaseRequest._init_account_info",
        lambda self: None,
    ):
        manager = ProductManager("douyin_shop", "douyin_user", product_source=source)

    with mock.patch.object(ProductManager, "_fetch_list_from_knowledge", return_value=[]):
        result = manager.get_product_list(page=1, size=10)

    assert result["success"] is True
    source.fetch_product_list.assert_not_called()


class _RecordingClient:
    """只记录 CDP 调用，不真的连浏览器。"""

    def __init__(self, evaluate_value=None):
        self.calls = []
        self._evaluate_value = evaluate_value

    def call(self, method, params=None, timeout=None):
        self.calls.append((method, params))
        if method == "Runtime.evaluate":
            return {"result": {"value": self._evaluate_value}}
        return {}


def _source(tmp_path):
    return DouyinProductSource(
        "douyin_shop", "uid", profile_dir=tmp_path / "profile",
        browser_path=tmp_path / "msedge.exe",
    )


def test_merchant_cookies_are_injected_on_parent_domain(tmp_path):
    """登录态必须按 .jinritemai.com 注入。

    用 url= 交给 CDP 推断会写成 host-only cookie，抖店 SSO 跳转链上的
    sso.jinritemai.com 读不到，页面落到 /login/common；而未登录时 list
    接口仍返回 code=0 空列表，表现为「同步成功 0 个商品」（2026-11-13 实测）。
    """
    session_file = tmp_path / "dy_session_x.json"
    session_file.write_text(
        json.dumps({"cookies_full": {"sessionid": "abc", "sid_guard": "def"}}),
        encoding="utf-8",
    )
    client = _RecordingClient()

    with mock.patch("database.db_manager") as db, mock.patch(
        "Channel.douyin.session_paths.existing_pigeon_session_path",
        return_value=session_file,
    ):
        db.get_account.return_value = {"username": ""}
        injected = _source(tmp_path)._inject_session_cookies(client)

    assert injected == 2
    method, params = client.calls[0]
    assert method == "Network.setCookies"
    for entry in params["cookies"]:
        assert entry["domain"] == ".jinritemai.com"
        assert "url" not in entry


def test_login_page_detection_reads_location_path(tmp_path):
    landed = _RecordingClient(evaluate_value="/login/common")
    product_page = _RecordingClient(evaluate_value="/ffa/g/list")

    source = _source(tmp_path)
    assert source._landed_on_login(landed) is True
    assert source._landed_on_login(product_page) is False


def test_expired_login_is_reported_instead_of_zero_products(tmp_path):
    """登录态失效不能报成「读取成功 0 个商品」。"""
    source = _source(tmp_path)
    message = source._failure_message(ValueError("merchant_login_required"))
    assert "重新登录" in message


def test_product_page_browser_is_killed_as_process_tree():
    """Edge 是多进程的，terminate 只杀启动器，窗口会留在桌面上。"""
    source_text = Path("Channel/douyin/utils/API/product_source.py").read_text(
        encoding="utf-8"
    )
    assert "kill_process_tree" in source_text
    assert "process.terminate()" not in source_text
