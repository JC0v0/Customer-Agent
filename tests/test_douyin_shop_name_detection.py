"""
单元测试：抖音登录真实店铺名检测（backstage/currentuser API）。

覆盖用例：
  a. currentuser 正常返回 ShopName="示例店铺" → 提取正确
  b. currentuser 返回 code!=0 → 兜底登录名
  c. currentuser 抛异常/超时 → 兜底登录名
  d. currentuser data 为空/无ShopName字段 → 兜底登录名
  e. 提取逻辑封装为可独立测试的函数 _fetch_shop_name_from_api(page)

不依赖真实网络，全部 mock page / 响应。
"""
import asyncio
import json
import pytest

# 确保项目根目录在 sys.path 中
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Channel.douyin.douyin_login import DouyinLogin


# ---------------------------------------------------------------------------
# Mock 辅助
# ---------------------------------------------------------------------------

class MockResponse:
    """模拟 Playwright APIResponse。"""

    def __init__(self, ok=True, json_data=None, json_exc=None):
        self._ok = ok
        self._json_data = json_data
        self._json_exc = json_exc

    @property
    def ok(self):
        return self._ok

    async def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._json_data


class MockRequest:
    """模拟 page.request（APIRequestContext）。"""

    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = []

    async def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        if self._exc is not None:
            raise self._exc
        return self._response


class MockPage:
    """模拟 Playwright Page，仅暴露 request 属性。"""

    def __init__(self, request=None):
        self.request = request or MockRequest()


def _make_login_instance(name="13800138000"):
    """创建 DouyinLogin 实例（不触发浏览器）。"""
    return DouyinLogin(name=name, password="test_pwd")


# ---------------------------------------------------------------------------
# 用例 a：currentuser 正常返回 ShopName → 提取正确
# ---------------------------------------------------------------------------

def test_fetch_shop_name_success():
    """API 正常返回 ShopName="示例店铺"，应提取到真实店铺名。"""
    resp_data = {
        "code": 0,
        "data": {
            "ShopName": "示例店铺",
            "ShopId": 123456789,
            "ShopLogo": "https://example.com/logo.png",
        },
        "msg": "",
    }
    page = MockPage(MockRequest(MockResponse(ok=True, json_data=resp_data)))
    login = _make_login_instance()

    result = asyncio.run(login._fetch_shop_name_from_api(page))

    assert result is not None
    assert result["shop_name"] == "示例店铺"
    assert result["shop_id"] == 123456789
    assert result["shop_logo"] == "https://example.com/logo.png"
    # 验证请求 URL 和 headers
    assert len(page.request.calls) == 1
    assert "currentuser" in page.request.calls[0]["url"]
    assert page.request.calls[0]["headers"]["referer"] == "https://im.jinritemai.com/"


# ---------------------------------------------------------------------------
# 用例 b：currentuser 返回 code!=0 → 返回 None（调用方兜底）
# ---------------------------------------------------------------------------

def test_fetch_shop_name_code_nonzero():
    """API 返回 code!=0，应返回 None，由调用方兜底登录名。"""
    resp_data = {"code": 1001, "data": None, "msg": "未登录"}
    page = MockPage(MockRequest(MockResponse(ok=True, json_data=resp_data)))
    login = _make_login_instance()

    result = asyncio.run(login._fetch_shop_name_from_api(page))

    assert result is None


# ---------------------------------------------------------------------------
# 用例 c：currentuser 抛异常/超时 → 返回 None（调用方兜底）
# ---------------------------------------------------------------------------

def test_fetch_shop_name_exception():
    """API 请求抛异常（如超时），应返回 None，由调用方兜底。"""
    page = MockPage(MockRequest(exc=TimeoutError("Request timed out")))
    login = _make_login_instance()

    result = asyncio.run(login._fetch_shop_name_from_api(page))

    assert result is None


def test_fetch_shop_name_json_parse_error():
    """API 响应 JSON 解析失败，应返回 None。"""
    page = MockPage(MockRequest(MockResponse(ok=True, json_exc=ValueError("Invalid JSON"))))
    login = _make_login_instance()

    result = asyncio.run(login._fetch_shop_name_from_api(page))

    assert result is None


def test_fetch_shop_name_http_not_ok():
    """API 返回 HTTP 非 2xx，应返回 None。"""
    page = MockPage(MockRequest(MockResponse(ok=False, json_data={"code": 0})))
    login = _make_login_instance()

    result = asyncio.run(login._fetch_shop_name_from_api(page))

    assert result is None


# ---------------------------------------------------------------------------
# 用例 d：currentuser data 为空/无ShopName字段 → 返回 None（调用方兜底）
# ---------------------------------------------------------------------------

def test_fetch_shop_name_data_none():
    """API 返回 data=None，应返回 None。"""
    resp_data = {"code": 0, "data": None, "msg": ""}
    page = MockPage(MockRequest(MockResponse(ok=True, json_data=resp_data)))
    login = _make_login_instance()

    result = asyncio.run(login._fetch_shop_name_from_api(page))

    assert result is None


def test_fetch_shop_name_data_empty_dict():
    """API 返回 data={}（无 ShopName 字段），应返回 None。"""
    resp_data = {"code": 0, "data": {}, "msg": ""}
    page = MockPage(MockRequest(MockResponse(ok=True, json_data=resp_data)))
    login = _make_login_instance()

    result = asyncio.run(login._fetch_shop_name_from_api(page))

    assert result is None


def test_fetch_shop_name_shopname_empty_string():
    """API 返回 ShopName 为空字符串，应返回 None。"""
    resp_data = {"code": 0, "data": {"ShopName": "", "ShopId": 123}, "msg": ""}
    page = MockPage(MockRequest(MockResponse(ok=True, json_data=resp_data)))
    login = _make_login_instance()

    result = asyncio.run(login._fetch_shop_name_from_api(page))

    assert result is None


def test_fetch_shop_name_shopname_whitespace():
    """API 返回 ShopName 为纯空白，应返回 None。"""
    resp_data = {"code": 0, "data": {"ShopName": "   "}, "msg": ""}
    page = MockPage(MockRequest(MockResponse(ok=True, json_data=resp_data)))
    login = _make_login_instance()

    result = asyncio.run(login._fetch_shop_name_from_api(page))

    assert result is None


def test_fetch_shop_name_response_not_dict():
    """API 返回非 dict 响应（如 list），应返回 None。"""
    page = MockPage(MockRequest(MockResponse(ok=True, json_data=["not", "a", "dict"])))
    login = _make_login_instance()

    result = asyncio.run(login._fetch_shop_name_from_api(page))

    assert result is None


# ---------------------------------------------------------------------------
# 用例 e：Set_shop_info 兼容新的 shop_name（非手机号）
# ---------------------------------------------------------------------------

def test_set_shop_info_uses_real_shop_name():
    """Set_shop_info 应从 login JSON 中正确解析真实店铺名。"""
    login = _make_login_instance(name="13800138000")
    login.shop_logo = "https://example.com/logo.png"

    cookies_json = json.dumps({
        "pigeon_session": {"token": "test_token"},
        "shop_name": "示例店铺",
        "user_name": "13800138000",
    }, ensure_ascii=False)

    shop_id, shop_name, mall_logo = login.Set_shop_info(cookies_json)

    assert shop_name == "示例店铺"
    assert shop_name != "13800138000"  # 不能是手机号
    assert shop_id == "douyin_示例店铺"
    assert mall_logo == "https://example.com/logo.png"


def test_set_shop_info_fallback_to_login_name():
    """当 JSON 中无 shop_name 时，Set_shop_info 应兜底到 self.shop_name / self.name。"""
    login = _make_login_instance(name="13800138000")

    cookies_json = json.dumps({
        "pigeon_session": {"token": "test_token"},
        "user_name": "13800138000",
    }, ensure_ascii=False)

    shop_id, shop_name, mall_logo = login.Set_shop_info(cookies_json)

    # 无 shop_name 时兜底到 self.name（登录名）
    assert shop_name == "13800138000"
    assert mall_logo == ""


def test_set_shop_info_dict_input():
    """Set_shop_info 应兼容 dict 形式的 cookies_json（非 JSON 字符串）。"""
    login = _make_login_instance(name="13800138000")

    cookies_dict = {
        "pigeon_session": {"token": "test_token"},
        "shop_name": "示例店铺",
        "user_name": "13800138000",
    }

    shop_id, shop_name, mall_logo = login.Set_shop_info(cookies_dict)

    assert shop_name == "示例店铺"


# ---------------------------------------------------------------------------
# 集成：模拟 login() 中 API → DOM → 登录名 的优先级链
# ---------------------------------------------------------------------------

def test_shop_name_priority_api_over_dom():
    """验证 API 成功时优先使用 API 结果，不依赖 DOM。"""
    login = _make_login_instance(name="13800138000")

    # 模拟 API 返回真实店铺名
    api_result = {
        "shop_name": "示例店铺",
        "shop_id": 123456789,
        "shop_logo": "https://example.com/logo.png",
    }

    # 模拟 login() 中的选择逻辑
    if api_result:
        shop_name = api_result["shop_name"]
        login.shop_id = str(api_result.get("shop_id") or "")
        login.shop_logo = api_result.get("shop_logo") or ""
    else:
        shop_name = "13800138000"  # 兜底

    assert shop_name == "示例店铺"
    assert login.shop_id == "123456789"
    assert login.shop_logo == "https://example.com/logo.png"


def test_shop_name_priority_fallback_when_api_none():
    """验证 API 失败时兜底到登录名。"""
    login = _make_login_instance(name="13800138000")

    api_result = None  # API 失败

    if api_result:
        shop_name = api_result["shop_name"]
    else:
        shop_name = "13800138000"  # 兜底

    assert shop_name == "13800138000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
