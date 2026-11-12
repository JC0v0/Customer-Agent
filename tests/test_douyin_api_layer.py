"""
抖音 API 层补全测试：get_token / get_user_info / getAssignCsList

覆盖：
1. GetToken：从 cookies 提取 pigeon token，无 token 返回 None，绝不抛 NotImplementedError
2. GetUserInfo：currentuser 正常路径提取 CustomerServiceInfo；API 失败回退本地 DB
3. SendMessage.getAssignCsList：onlineservice 正常路径返回 dict；API 失败降级返回非空单元素 dict，绝不返回 None

所有 HTTP 调用均 mock，不依赖真实网络。
"""
import unittest
from unittest import mock

from Channel.douyin.utils.API.get_token import GetToken
from Channel.douyin.utils.API.get_user_info import GetUserInfo
from Channel.douyin.utils.API.send_message import SendMessage


# ============================================================================
# 测试辅助
# ============================================================================

def _make_no_db_init():
    """返回一个 patch，使 BaseRequest._init_account_info 不触发 DB 调用。"""
    return mock.patch("Channel.douyin.utils.base_request.BaseRequest._init_account_info", lambda self: None)


def _make_fake_db_manager():
    """模拟 db_manager，返回受控的账号信息"""
    fake = mock.MagicMock()
    fake.get_account.return_value = {
        "shop_id": "douyin_test_shop",
        "user_id": "123456789",
        "username": "示例店铺",
        "cookies": '{"pigeon_session_token": "fake_pigeon_token_abc123"}',
    }
    return fake


# ============================================================================
# 1. GetToken 测试
# ============================================================================

class TestGetToken(unittest.TestCase):
    """抖音 GetToken：pigeon token 兼容层，绝不抛 NotImplementedError"""

    def test_returns_token_from_cookies_pigeon_session_token(self):
        """cookies 含 pigeon_session_token → 提取返回"""
        with _make_no_db_init():
            gt = GetToken("shop1", "user1")
        gt.cookies = {"pigeon_session_token": "token_xyz"}
        token = gt.get_token()
        self.assertEqual(token, "token_xyz")

    def test_returns_token_from_cookies_access_token(self):
        """cookies 含 access_token（兜底候选键）→ 提取返回"""
        with _make_no_db_init():
            gt = GetToken("shop1", "user1")
        gt.cookies = {"access_token": "token_from_access"}
        token = gt.get_token()
        self.assertEqual(token, "token_from_access")

    def test_returns_token_from_nested_pigeon_session(self):
        """cookies 含嵌套 pigeon_session dict → 从嵌套提取 token"""
        with _make_no_db_init():
            gt = GetToken("shop1", "user1")
        gt.cookies = {"pigeon_session": {"token": "nested_token"}}
        token = gt.get_token()
        self.assertEqual(token, "nested_token")

    def test_no_token_returns_none(self):
        """cookies 无任何 token → 返回 None（不抛异常）"""
        with _make_no_db_init():
            gt = GetToken("shop1", "user1")
        gt.cookies = {"other_key": "value"}
        token = gt.get_token()
        self.assertIsNone(token)

    def test_empty_cookies_returns_none(self):
        """空 cookies → 返回 None"""
        with _make_no_db_init():
            gt = GetToken("shop1", "user1")
        gt.cookies = {}
        token = gt.get_token()
        self.assertIsNone(token)

    def test_never_raises_not_implemented(self):
        """绝不抛 NotImplementedError（核心修复点）"""
        with _make_no_db_init():
            gt = GetToken("shop1", "user1")
        gt.cookies = {}
        try:
            result = gt.get_token()
        except NotImplementedError:
            self.fail("get_token() 不应抛出 NotImplementedError")
        self.assertIsNone(result)

    def test_token_priority_order(self):
        """多 token 字段共存时，按优先级取 pigeon_session_token"""
        with _make_no_db_init():
            gt = GetToken("shop1", "user1")
        gt.cookies = {
            "pigeon_session_token": "primary_token",
            "access_token": "secondary_token",
            "token": "tertiary_token",
        }
        token = gt.get_token()
        self.assertEqual(token, "primary_token")


# ============================================================================
# 2. GetUserInfo 测试
# ============================================================================

class TestGetUserInfo(unittest.TestCase):
    """抖音 GetUserInfo：currentuser 真实 API + 本地 DB 降级"""

    def _make_currentuser_response(self):
        """构造抓包实证格式的 currentuser 响应"""
        return {
            "code": 0,
            "data": {
                "ShopName": "示例店铺",
                "ShopId": 123456789,
                "ShopLogo": "https://example.com/logo.png",
                "ToutiaoId": 1234567890123456,
                "CustomerServiceInfo": {
                    "id": "1234567890123456",
                    "screen_name": "示例店铺",
                    "avatar_url": "https://example.com/avatar.png",
                },
            },
        }

    def _make_gui(self):
        """构造 GetUserInfo（不触发 DB），手动设置属性"""
        with _make_no_db_init():
            gui = GetUserInfo("douyin_test_shop", "123456789")
        gui.account_name = "示例店铺"
        return gui

    def test_currentuser_normal_path_extracts_correctly(self):
        """currentuser 正常返回 → 从 CustomerServiceInfo 提取 user_id/user_name/mall_id"""
        gui = self._make_gui()
        with mock.patch.object(gui, "get", return_value=self._make_currentuser_response()):
            result = gui.get_user_info()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        user_id, user_name, mall_id = result
        self.assertEqual(user_id, "1234567890123456")
        self.assertEqual(user_name, "示例店铺")
        self.assertEqual(mall_id, "123456789")

    def test_currentuser_code_nonzero_falls_back_to_db(self):
        """currentuser 返回 code!=0 → 回退本地 DB（mock DB 返回账号信息）"""
        gui = self._make_gui()
        fake_db = mock.MagicMock()
        fake_db.get_account.return_value = {
            "shop_id": "douyin_test_shop",
            "user_id": "123456789",
            "username": "示例店铺",
        }
        with mock.patch.object(gui, "get", return_value={"code": 1, "data": {}}), \
             mock.patch("database.db_manager", fake_db):
            result = gui.get_user_info()
        self.assertIsInstance(result, tuple)
        user_id, user_name, mall_id = result
        self.assertEqual(user_id, "123456789")
        self.assertEqual(user_name, "示例店铺")

    def test_currentuser_exception_falls_back_to_db(self):
        """currentuser 请求抛异常 → 回退本地 DB"""
        gui = self._make_gui()
        fake_db = mock.MagicMock()
        fake_db.get_account.return_value = {
            "shop_id": "douyin_test_shop",
            "user_id": "123456789",
            "username": "示例店铺",
        }
        with mock.patch.object(gui, "get", side_effect=ConnectionError("network down")), \
             mock.patch("database.db_manager", fake_db):
            result = gui.get_user_info()
        self.assertIsInstance(result, tuple)
        user_id, user_name, _ = result
        self.assertEqual(user_id, "123456789")
        self.assertEqual(user_name, "示例店铺")

    def test_currentuser_missing_csi_falls_back(self):
        """currentuser 返回但缺 CustomerServiceInfo → 回退 DB"""
        gui = self._make_gui()
        fake_db = mock.MagicMock()
        fake_db.get_account.return_value = {
            "shop_id": "douyin_test_shop",
            "user_id": "123456789",
            "username": "示例店铺",
        }
        resp = {"code": 0, "data": {"ShopName": "示例店铺"}}  # 缺 CustomerServiceInfo
        with mock.patch.object(gui, "get", return_value=resp), \
             mock.patch("database.db_manager", fake_db):
            result = gui.get_user_info()
        self.assertIsInstance(result, tuple)

    def test_db_fallback_returns_false_when_no_account(self):
        """currentuser 失败且 DB 无账号且无构造入参 → 最终返回 False"""
        with _make_no_db_init():
            gui = GetUserInfo()  # 无 shop_id/user_id
        fake_db = mock.MagicMock()
        fake_db.get_account.return_value = None
        with mock.patch.object(gui, "get", side_effect=ConnectionError()), \
             mock.patch("database.db_manager", fake_db):
            result = gui.get_user_info()
        self.assertFalse(result)

    def test_constructor_accepts_cookies(self):
        """构造函数接受 cookies 参数并更新"""
        with _make_no_db_init():
            gui = GetUserInfo(cookies={"test": "cookie"})
        self.assertEqual(gui.cookies, {"test": "cookie"})


# ============================================================================
# 3. SendMessage.getAssignCsList 测试
# ============================================================================

class TestGetAssignCsList(unittest.TestCase):
    """抖音 SendMessage.getAssignCsList：onlineservice 真实 API + 非空 dict 降级"""

    def _make_onlineservice_response(self):
        """构造抓包实证格式的 onlineservice 响应"""
        return {
            "code": 0,
            "data": [
                {
                    "id": 61067207,
                    "shop_id": 123456789,
                    "toutiao_id": 1234567890123456,
                    "toutiao_id_str": "1234567890123456",
                    "service_toutiao_id": 1234567890123456,
                    "service_name": "示例店铺",
                    "online_status": 1,
                },
                {
                    "id": 61067208,
                    "shop_id": 123456789,
                    "toutiao_id": 999999999999999,
                    "toutiao_id_str": "999999999999999",
                    "service_name": "客服二号",
                    "online_status": 1,
                },
            ],
        }

    def _make_currentuser_response(self):
        return {
            "code": 0,
            "data": {
                "ShopName": "示例店铺",
                "ShopId": 123456789,
                "ToutiaoId": 1234567890123456,
                "CustomerServiceInfo": {
                    "id": "1234567890123456",
                    "screen_name": "示例店铺",
                },
            },
        }

    def _make_sender(self):
        """构造 SendMessage（不触发 DB），手动设置属性"""
        with _make_no_db_init():
            sm = SendMessage("douyin_test_shop", "123456789")
        sm.account_name = "示例店铺"
        return sm

    def test_onlineservice_normal_path_returns_dict(self):
        """onlineservice 正常返回 → 返回 dict（key=cs_uid, value=含username）"""
        sm = self._make_sender()
        with mock.patch.object(sm, "get", return_value=self._make_onlineservice_response()):
            result = sm.getAssignCsList()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        # 验证 key 格式 cs_{shop_id}_{toutiao_id}
        self.assertIn("cs_123456789_1234567890123456", result)
        self.assertIn("cs_123456789_999999999999999", result)
        # 验证 value 含 username
        self.assertEqual(result["cs_123456789_1234567890123456"]["username"], "示例店铺")
        self.assertEqual(result["cs_123456789_999999999999999"]["username"], "客服二号")

    def test_onlineservice_code_nonzero_falls_back_to_currentuser(self):
        """onlineservice code!=0 → 回退 currentuser，返回非空单元素 dict"""
        sm = self._make_sender()
        responses = [
            {"code": 1, "data": []},  # onlineservice 失败
            self._make_currentuser_response(),  # currentuser 成功
        ]
        with mock.patch.object(sm, "get", side_effect=responses):
            result = sm.getAssignCsList()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        cs_uid = list(result.keys())[0]
        self.assertIn("1234567890123456", cs_uid)
        self.assertEqual(result[cs_uid]["username"], "示例店铺")

    def test_onlineservice_exception_falls_back(self):
        """onlineservice 抛异常 → 回退 currentuser"""
        sm = self._make_sender()
        responses = [
            ConnectionError("network down"),
            self._make_currentuser_response(),
        ]
        with mock.patch.object(sm, "get", side_effect=responses):
            result = sm.getAssignCsList()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)

    def test_both_apis_fail_returns_fallback_dict(self):
        """onlineservice 和 currentuser 均失败 → 极兜底返回构造入参单元素 dict，绝不返回 None"""
        sm = self._make_sender()
        with mock.patch.object(sm, "get", side_effect=ConnectionError("all down")):
            result = sm.getAssignCsList()
        self.assertIsNotNone(result, "getAssignCsList 绝不返回 None")
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        cs_uid = list(result.keys())[0]
        self.assertIn("douyin_test_shop", cs_uid)
        self.assertIn("123456789", cs_uid)

    def test_never_returns_none(self):
        """核心修复点：任何情况下都不返回 None（None 会导致转人工工具报错）"""
        sm = self._make_sender()
        for side_effect in [
            {"code": 0, "data": []},  # 空列表
            {"code": 500, "data": None},
            ConnectionError(),
            {"code": 0, "data": "not_a_list"},
        ]:
            with mock.patch.object(sm, "get", side_effect=side_effect):
                result = sm.getAssignCsList()
            self.assertIsNotNone(result, f"side_effect={side_effect!r} 不应返回 None")

    def test_onlineservice_single_cs(self):
        """onlineservice 返回单个客服 → 返回单元素 dict"""
        sm = self._make_sender()
        resp = {"code": 0, "data": [self._make_onlineservice_response()["data"][0]]}
        with mock.patch.object(sm, "get", return_value=resp):
            result = sm.getAssignCsList()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)

    def test_onlineservice_skips_invalid_items(self):
        """onlineservice 返回含无效项 → 跳过无效项，保留有效项"""
        sm = self._make_sender()
        resp = {
            "code": 0,
            "data": [
                self._make_onlineservice_response()["data"][0],  # 有效
                "not_a_dict",  # 无效
                {"toutiao_id_str": "", "service_name": "no_id"},  # 缺 id
            ],
        }
        with mock.patch.object(sm, "get", return_value=resp):
            result = sm.getAssignCsList()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)  # 只有第一个有效


if __name__ == "__main__":
    unittest.main()
