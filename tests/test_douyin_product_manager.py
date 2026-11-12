"""
抖音 ProductManager + 商品工具统一架构测试

覆盖：
1. ProductManager.get_product_list：正常返回（mock KnowledgeService）/ 空知识库返回空列表
2. ProductManager.get_product_detail：正常返回 / 不存在返回 None
3. get_shop_products 工具：抖音统一走 ProductManager（不调拼多多、不直接调 KnowledgeService 分支）
4. send_goods_link 抖音：商品存在 → 校验通过返回提示；商品不存在 → 返回无效提示

所有 KnowledgeService / BaseRequest / DI 容器均 mock，不依赖真实 DB/网络。
"""
import unittest
from unittest import mock

# 统一 mock BaseRequest._init_account_info，避免构造时触发真实 DB
_init_patch = mock.patch(
    "Channel.douyin.utils.base_request.BaseRequest._init_account_info",
    lambda self: None,
)
_init_patch.start()

from Channel.douyin.utils.API.product_manager import ProductManager as DouyinProductManager
from Agent.CustomerAgent.tools.get_product_list import get_shop_products, GetShopProductsParams
from Agent.CustomerAgent.tools.send_goods_link import send_goods_link, SendGoodsLinkParams


# ============================================================================
# Mock 辅助
# ============================================================================

class _FakeProduct:
    """模拟 ProductKnowledge ORM 对象"""
    def __init__(self, goods_id=123456, goods_name="测试商品", price="29.90",
                 price_min=2990, price_max=2990, sold_quantity=100,
                 thumb_url="https://example.com/img.jpg", specifications="规格: 均码"):
        self.goods_id = goods_id
        self.goods_name = goods_name
        self.price = price
        self.price_min = price_min
        self.price_max = price_max
        self.sold_quantity = sold_quantity
        self.thumb_url = thumb_url
        self.specifications = specifications


class _FakeKnowledgeService:
    """模拟 KnowledgeService"""
    def __init__(self, products=None, detail_product=None):
        self._products = products or []
        self._detail_product = detail_product

    def _resolve_shop_id(self, shop_id):
        return 1  # 模拟解析成功

    def list_products_by_shop(self, db_shop_id):
        return self._products

    def get_product_by_goods_id(self, db_shop_id, goods_id):
        if self._detail_product and self._detail_product.goods_id == goods_id:
            return self._detail_product
        # 如果 products 列表中有匹配的，也返回
        for p in self._products:
            if p.goods_id == goods_id:
                return p
        return None


def _patch_di_container(fake_ks):
    """patch DI 容器的 get 方法，返回 fake KnowledgeService"""
    return mock.patch("core.di_container.container.get", return_value=fake_ks)


# ============================================================================
# 1. ProductManager.get_product_list
# ============================================================================

class TestProductManagerList(unittest.TestCase):
    """抖音 ProductManager.get_product_list"""

    def test_normal_returns_products(self):
        """正常：知识库有商品 → 返回商品列表 dict，格式与拼多多对齐"""
        fake_ks = _FakeKnowledgeService(products=[
            _FakeProduct(goods_id=111, goods_name="商品A", price="19.90"),
            _FakeProduct(goods_id=222, goods_name="商品B", price="39.90"),
        ])
        pm = DouyinProductManager(shop_id="douyin_test", user_id="user1")
        with _patch_di_container(fake_ks):
            result = pm.get_product_list(page=1, size=10)

        self.assertTrue(result["success"])
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["page"], 1)
        self.assertEqual(len(result["products"]), 2)
        # 验证商品字段格式与拼多多对齐
        p = result["products"][0]
        self.assertEqual(p["goods_id"], 111)
        self.assertEqual(p["goods_name"], "商品A")
        self.assertEqual(p["price"], "19.90")
        self.assertIn("quantity", p)  # 知识库无库存，默认0
        self.assertIn("goods_type", p)
        self.assertIn("tag", p)

    def test_empty_knowledge_returns_empty_list(self):
        """空知识库 → 返回空列表，success=True，不报错"""
        fake_ks = _FakeKnowledgeService(products=[])
        pm = DouyinProductManager(shop_id="douyin_test", user_id="user1")
        with _patch_di_container(fake_ks):
            result = pm.get_product_list(page=1, size=10)

        self.assertTrue(result["success"])
        self.assertEqual(result["products"], [])
        self.assertEqual(result["total"], 0)

    def test_pagination(self):
        """分页：size=1 时只返回第1个商品"""
        fake_ks = _FakeKnowledgeService(products=[
            _FakeProduct(goods_id=111, goods_name="商品A"),
            _FakeProduct(goods_id=222, goods_name="商品B"),
        ])
        pm = DouyinProductManager(shop_id="douyin_test", user_id="user1")
        with _patch_di_container(fake_ks):
            result = pm.get_product_list(page=1, size=1)

        self.assertTrue(result["success"])
        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["products"]), 1)
        self.assertEqual(result["products"][0]["goods_id"], 111)

    def test_does_not_call_pinduoduo_api(self):
        """渠道隔离：抖音 ProductManager 不调拼多多 API（不发起 HTTP 请求）"""
        fake_ks = _FakeKnowledgeService(products=[_FakeProduct()])
        pm = DouyinProductManager(shop_id="douyin_test", user_id="user1")
        with _patch_di_container(fake_ks), \
             mock.patch.object(pm, "post") as mock_post, \
             mock.patch.object(pm, "get") as mock_get:
            result = pm.get_product_list()
        # 不应发起任何 HTTP 请求（知识库数据源不走 HTTP）
        mock_post.assert_not_called()
        mock_get.assert_not_called()
        self.assertTrue(result["success"])


# ============================================================================
# 2. ProductManager.get_product_detail
# ============================================================================

class TestProductManagerDetail(unittest.TestCase):
    """抖音 ProductManager.get_product_detail"""

    def test_normal_returns_detail(self):
        """正常：商品存在 → 返回商品详情 dict，格式与拼多多对齐"""
        fake_product = _FakeProduct(
            goods_id=123456, goods_name="测试商品", price="29.90",
            specifications="规格: 均码; 颜色: 红色"
        )
        fake_ks = _FakeKnowledgeService(detail_product=fake_product)
        pm = DouyinProductManager(shop_id="douyin_test", user_id="user1")
        with _patch_di_container(fake_ks):
            result = pm.get_product_detail(123456)

        self.assertTrue(result["success"])
        self.assertIn("product_info", result)
        pi = result["product_info"]
        self.assertEqual(pi["goods_id"], 123456)
        self.assertEqual(pi["goods_name"], "测试商品")
        self.assertEqual(pi["price"], "29.90")
        self.assertIsInstance(pi["specifications"], list)

    def test_not_found_returns_failure(self):
        """商品不存在 → 返回 success=False，不抛异常"""
        fake_ks = _FakeKnowledgeService(detail_product=None)
        pm = DouyinProductManager(shop_id="douyin_test", user_id="user1")
        with _patch_di_container(fake_ks):
            result = pm.get_product_detail(999999)

        self.assertFalse(result["success"])
        self.assertIn("error_msg", result)
        self.assertNotIn("product_info", result)

    def test_empty_goods_id_returns_failure(self):
        """空 goods_id → 返回失败"""
        pm = DouyinProductManager(shop_id="douyin_test", user_id="user1")
        result = pm.get_product_detail(None)
        self.assertFalse(result["success"])
        self.assertIn("商品ID不能为空", result["error_msg"])


# ============================================================================
# 3. get_shop_products 工具（抖音统一走 ProductManager）
# ============================================================================

class TestGetShopProductsTool(unittest.TestCase):
    """get_shop_products 工具：抖音统一走 ProductManager，无特殊分支"""

    def test_douyin_uses_product_manager(self):
        """抖音渠道：通过 ProductManager 获取商品，输出格式不变"""
        fake_ks = _FakeKnowledgeService(products=[
            _FakeProduct(goods_id=111, goods_name="抖音商品", price="19.90"),
        ])
        params = GetShopProductsParams(
            channel_type="douyin", shop_id="douyin_test", user_id="user1"
        )
        with _patch_di_container(fake_ks):
            result = get_shop_products(params)

        self.assertIsInstance(result, str)
        self.assertIn("[untrusted_product_catalog]", result)
        self.assertIn("抖音商品", result)
        self.assertIn("111", result)
        self.assertIn("[/untrusted_product_catalog]", result)

    def test_douyin_empty_products(self):
        """抖音渠道：空知识库 → 返回"暂无商品"提示，不报错"""
        fake_ks = _FakeKnowledgeService(products=[])
        params = GetShopProductsParams(
            channel_type="douyin", shop_id="douyin_test", user_id="user1"
        )
        with _patch_di_container(fake_ks):
            result = get_shop_products(params)

        self.assertIsInstance(result, str)
        self.assertIn("暂无商品", result)

    def test_no_special_branch_in_code(self):
        """代码中不再有 _douyin_get_products 特殊分支"""
        import inspect
        from Agent.CustomerAgent.tools import get_product_list as gpl_module
        source = inspect.getsource(gpl_module)
        self.assertNotIn("_douyin_get_products", source)

    def test_missing_params_returns_error(self):
        """缺少 shop_id/user_id → 返回错误提示"""
        params = GetShopProductsParams(channel_type="douyin", shop_id=None, user_id=None)
        result = get_shop_products(params)
        self.assertIn("缺少必要的shop_id或user_id", result)


# ============================================================================
# 4. send_goods_link 抖音（商品校验）
# ============================================================================

class TestSendGoodsLinkDouyin(unittest.TestCase):
    """send_goods_link 抖音：商品存在→校验通过；不存在→无效提示"""

    def test_douyin_product_exists_returns_hint(self):
        """抖音：商品存在且校验通过 → 返回"商品已找到"提示（不调拼多多发送）"""
        fake_product = _FakeProduct(goods_id=123456, goods_name="抖音测试商品")
        fake_ks = _FakeKnowledgeService(detail_product=fake_product)
        params = SendGoodsLinkParams(
            channel_type="douyin",
            recipient_uid="buyer1",
            goods_id=123456,
            shop_id="douyin_test",
            user_id="user1",
        )
        with _patch_di_container(fake_ks), \
             mock.patch("bridge.sender.get_sender") as mock_get_sender:
            result = send_goods_link(params)

        self.assertIsInstance(result, str)
        self.assertIn("商品已找到", result)
        self.assertIn("抖音测试商品", result)
        # 抖音不调发送器（商品卡待接入）
        mock_get_sender.assert_not_called()

    def test_douyin_product_not_found_returns_invalid(self):
        """抖音：商品不存在 → 返回商品不属于当前店铺或已失效提示"""
        fake_ks = _FakeKnowledgeService(detail_product=None)
        params = SendGoodsLinkParams(
            channel_type="douyin",
            recipient_uid="buyer1",
            goods_id=999999,
            shop_id="douyin_test",
            user_id="user1",
        )
        with _patch_di_container(fake_ks):
            result = send_goods_link(params)

        self.assertIsInstance(result, str)
        self.assertIn("商品不属于当前店铺或已失效", result)

    def test_douyin_small_goods_id_rejected(self):
        """抖音：goods_id < 1000（列表序号误传）→ 直接拒绝"""
        params = SendGoodsLinkParams(
            channel_type="douyin",
            recipient_uid="buyer1",
            goods_id=3,  # 列表序号，不是真实商品ID
            shop_id="douyin_test",
            user_id="user1",
        )
        result = send_goods_link(params)
        self.assertIn("商品 ID 无效", result)

    def test_douyin_does_not_call_pinduoduo_product_manager(self):
        """渠道隔离：抖音 send_goods_link 不调拼多多 ProductManager"""
        fake_product = _FakeProduct(goods_id=123456, goods_name="抖音商品")
        fake_ks = _FakeKnowledgeService(detail_product=fake_product)
        params = SendGoodsLinkParams(
            channel_type="douyin",
            recipient_uid="buyer1",
            goods_id=123456,
            shop_id="douyin_test",
            user_id="user1",
        )
        with _patch_di_container(fake_ks), \
             mock.patch("Channel.pinduoduo.utils.API.product_manager.ProductManager") as mock_pdd_pm:
            result = send_goods_link(params)
        mock_pdd_pm.assert_not_called()
        self.assertIn("商品已找到", result)


if __name__ == "__main__":
    unittest.main()
