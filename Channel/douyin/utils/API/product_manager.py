"""
抖音商品管理 API（对齐拼多多 ProductManager 接口签名）。

【实现说明】
- 抖音商品真实 API（抖店商品列表/详情）尚未逆向，抓包中无相关端点。
- 当前数据源：知识库 KnowledgeService（product_knowledge 表），
  由 product_sync 从拼多多/抖音商品同步写入。
- 返回格式与拼多多 ProductManager 严格对齐：
  - get_product_list: {"success", "products", "total", "page", "error_msg"}
  - get_product_detail: {"success", "product_info", "error_msg"}
- 知识库无商品时返回空列表/None，不报错。
- 预留真实 API 接入点：未来逆向抖店商品 API 后，替换 _fetch_from_knowledge 为真实 HTTP 请求。
"""
import time
from typing import Optional, Dict, Any, List
from ..base_request import BaseRequest
from utils.logger_loguru import get_logger
from Channel.douyin.utils.log_throttle import throttled_warning, throttled_info

logger = get_logger("DouyinProductManager")

# 平台商品页抓取失败后的冷却时间（秒）：失败时拉起浏览器可能耗时 45s，
# 冷却期内直接返回失败，避免商品同步反复卡在同一个错误上。
PLATFORM_FETCH_COOLDOWN_SECONDS = 600
_last_platform_failure_at = 0.0

# 平台商品页抓取一次可能耗时 40 秒以上。成功结果按店铺缓存一段时间，
# 供连续翻页的商品同步复用，避免每页都重新拉起浏览器。
PLATFORM_FETCH_CACHE_SECONDS = 300
_platform_fetch_cache = {}


class ProductManager(BaseRequest):
    """抖音商品管理器（知识库数据源，与拼多多 ProductManager 对齐）。"""

    def __init__(self, shop_id: str = None, user_id: str = None, cookies=None,
                 product_source=None):
        """
        初始化商品管理器

        Args:
            shop_id: 店铺ID（本地稳定标识，如 douyin_<店铺名>）
            user_id: 用户ID（账号ID）
            cookies: 登录cookies（预留，当前知识库数据源不需要）
        """
        super().__init__(shop_id=shop_id, user_id=user_id, channel_name="douyin")
        if cookies:
            self.update_cookies(cookies)
        self._product_source = product_source
        self._live_products: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 公共 API（与拼多多对齐）
    # ------------------------------------------------------------------

    def get_product_list(self, page=1, size=10, require_platform=False):
        """
        获取店铺商品列表（知识库数据源）。

        Args:
            page (int): 页码，默认1
            size (int): 每页数量，默认10

        Returns:
            dict: 商品列表结果，格式与拼多多对齐：
                {
                    "success": True/False,
                    "products": [
                        {
                            "goods_id": int,
                            "goods_name": str,
                            "thumb_url": str,
                            "price": str,           # 价格（字符串，如 "29.90"）
                            "price_min": int,       # 最低价（分）
                            "price_max": int,       # 最高价（分）
                            "sold_quantity": int,   # 已售数量
                            "quantity": int,        # 库存（知识库无此字段，默认0）
                            "goods_type": str,      # 商品类型（知识库无，默认""）
                            "tag": str,             # 商品标签（知识库无，默认""）
                        },
                        ...
                    ],
                    "total": int,
                    "page": int,
                    "error_msg": str  # 仅失败时
                }
        """
        global _last_platform_failure_at
        try:
            # 只有「商品同步」这种显式动作才去平台抓商品页。
            # 日常问答一律读本地知识库：抓商品页要额外拉起一个浏览器，
            # 单次可达 40 秒，会把 AI 回复拖慢一整轮；临时浏览器与主程序
            # 共用同一份登录凭据，还可能把主会话挤掉（实测 22:53 拉起后
            # 22:55 会话即过期）。
            if require_platform:
                can_fetch = self._product_source is not None or bool(self.cookies)
                if not can_fetch:
                    return {
                        "success": False,
                        "error_msg": "抖音账号未登录，无法读取店铺商品",
                        "products": [],
                        "total": 0,
                        "page": page,
                    }
                now = time.monotonic()
                in_cooldown = (
                    (now - _last_platform_failure_at) < PLATFORM_FETCH_COOLDOWN_SECONDS
                )
                if in_cooldown:
                    return {
                        "success": False,
                        "error_msg": "商品页抓取处于失败冷却期，请稍后重试商品同步",
                        "products": [],
                        "total": 0,
                        "page": page,
                    }
                platform_result = self._fetch_list_from_platform(page, size)
                if platform_result.get("success"):
                    _last_platform_failure_at = 0.0
                    return platform_result
                _last_platform_failure_at = now
                return platform_result

            products = self._fetch_list_from_knowledge()
            if products is None:
                return {
                    "success": False,
                    "error_msg": "店铺未登记或知识库查询失败",
                    "products": [],
                    "total": 0,
                    "page": page,
                }

            total = len(products)
            # 简单分页（知识库数据量小，内存分页即可）
            start = (page - 1) * size
            end = start + size
            page_products = products[start:end]

            throttled_info(logger, "product.list_ok",
                f"抖音商品列表查询成功: shop_id={self.shop_id}, "
                f"total={total}, page={page}, size={size}, returned={len(page_products)}"
            )
            return {
                "success": True,
                "products": page_products,
                "total": total,
                "page": page,
            }
        except Exception as e:
            logger.error(
                f"抖音商品列表查询异常: error_type={type(e).__name__}, "
                f"shop_id={self.shop_id}"
            )
            return {
                "success": False,
                "error_msg": f"查询异常: {type(e).__name__}",
                "products": [],
                "total": 0,
                "page": page,
            }

    def get_product_detail(self, goods_id):
        """
        根据商品ID获取商品详情（知识库数据源）。

        Args:
            goods_id (int): 商品ID

        Returns:
            dict: 商品详情结果，格式与拼多多对齐：
                {
                    "success": True/False,
                    "product_info": {
                        "goods_id": int,
                        "goods_name": str,
                        "specifications": str/list,
                        "price": str,
                        "thumb_url": str,
                        "sold_quantity": int,
                    },
                    "error_msg": str  # 仅失败时
                }
        """
        if not goods_id:
            logger.error("商品ID不能为空")
            return {"success": False, "error_msg": "商品ID不能为空"}

        try:
            try:
                product = self._live_products.get(int(goods_id))
            except (TypeError, ValueError):
                product = None
            if product is not None:
                product = self._live_product_to_detail(product)
            else:
                product = self._fetch_detail_from_knowledge(goods_id)
            if product is None:
                throttled_warning(logger, "product.detail_missing",
                    f"抖音商品详情未找到: goods_id={goods_id}, shop_id={self.shop_id}"
                )
                return {
                    "success": False,
                    "error_msg": f"商品ID {goods_id} 不存在或不属于当前店铺",
                }

            throttled_info(logger, "product.detail_ok",
                f"抖音商品详情查询成功: goods_id={goods_id}, "
                f"goods_name={product.get('goods_name')}"
            )
            return {
                "success": True,
                "product_info": product,
            }
        except Exception as e:
            logger.error(
                f"抖音商品详情查询异常: error_type={type(e).__name__}, "
                f"goods_id={goods_id}"
            )
            return {
                "success": False,
                "error_msg": f"查询异常: {type(e).__name__}",
            }

    # ------------------------------------------------------------------
    # 知识库数据源（预留真实 API 接入点）
    # TODO(传输缝): 未来逆向抖店商品 API 后，替换以下方法为真实 HTTP 请求。
    #   - 商品列表: 抖店开放平台 /product/list 或飞鸽后台商品管理端点
    #   - 商品详情: 抖店开放平台 /product/detail
    #   替换后保持返回 dict 格式不变，上层工具无需修改。
    # ------------------------------------------------------------------

    def _fetch_list_from_platform(self, page: int, size: int) -> Dict[str, Any]:
        cache_key = (str(self.shop_id), page, size)
        cached = _platform_fetch_cache.get(cache_key)
        if cached is not None:
            cached_at = cached[0]
            if time.monotonic() - cached_at < PLATFORM_FETCH_CACHE_SECONDS:
                logger.debug(
                    f"抖店商品页使用缓存: shop_id={self.shop_id}, page={page}"
                )
                cached_result = cached[1]
                for product in cached_result.get("products", []):
                    goods_id = product.get("goods_id")
                    if goods_id is not None:
                        self._live_products[int(goods_id)] = product
                return cached_result

        if self._product_source is None:
            from .product_source import DouyinProductSource

            self._product_source = DouyinProductSource(self.shop_id, self.user_id)
        result = self._product_source.fetch_product_list(page=page, size=size)
        if result.get("success"):
            _platform_fetch_cache[cache_key] = (time.monotonic(), result)
            for product in result.get("products", []):
                goods_id = product.get("goods_id")
                if goods_id is not None:
                    self._live_products[int(goods_id)] = product
            logger.info(
                f"抖店商品页读取成功: shop_id={self.shop_id}, "
                f"page={page}, returned={len(result.get('products', []))}"
            )
        else:
            logger.warning(
                f"抖店商品页读取失败: shop_id={self.shop_id}, "
                f"reason={result.get('error_msg', 'unknown')}"
            )
        return result

    def _get_knowledge_service(self):
        """从 DI 容器获取 KnowledgeService 实例。"""
        from core.di_container import container
        from database.knowledge_service import KnowledgeService

        return container.get(KnowledgeService)

    def _resolve_db_shop_id(self, ks) -> Optional[int]:
        """将本地 shop_id（如 douyin_<店铺名>）解析为知识库 shops 表的 int id。"""
        try:
            return ks._resolve_shop_id(self.shop_id, channel_name="douyin")
        except TypeError:
            # Keep compatibility with lightweight test doubles and older
            # KnowledgeService implementations during rolling upgrades.
            return ks._resolve_shop_id(self.shop_id)
        except Exception as e:
            logger.warning(
                f"抖音店铺ID解析失败: shop_id={self.shop_id}, "
                f"error_type={type(e).__name__}"
            )
            return None

    def _fetch_list_from_knowledge(self) -> Optional[List[Dict[str, Any]]]:
        """从知识库获取商品列表，转换为与拼多多对齐的 dict 格式。"""
        ks = self._get_knowledge_service()
        db_shop_id = self._resolve_db_shop_id(ks)
        if db_shop_id is None:
            return None

        products = ks.list_products_by_shop(db_shop_id)
        if not products:
            return []

        return [self._product_to_dict(p) for p in products]

    def _fetch_detail_from_knowledge(self, goods_id) -> Optional[Dict[str, Any]]:
        """从知识库获取商品详情，转换为与拼多多对齐的 dict 格式。"""
        ks = self._get_knowledge_service()
        db_shop_id = self._resolve_db_shop_id(ks)
        if db_shop_id is None:
            return None

        try:
            goods_id_int = int(goods_id)
        except (ValueError, TypeError):
            logger.warning(f"商品ID格式无效: goods_id={goods_id!r}")
            return None

        product = ks.get_product_by_goods_id(db_shop_id, goods_id_int)
        if product is None:
            return None

        return self._product_to_detail_dict(product)

    @staticmethod
    def _live_product_to_detail(product: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "goods_id": product.get("goods_id"),
            "goods_name": product.get("goods_name", ""),
            "specifications": product.get("_specifications", []),
            "description": product.get("_description", ""),
            "price": product.get("price"),
            "thumb_url": product.get("thumb_url", ""),
            "sold_quantity": product.get("sold_quantity", 0),
        }

    @staticmethod
    def _product_to_dict(p) -> Dict[str, Any]:
        """将 ProductKnowledge ORM 对象转换为商品列表 dict（与拼多多对齐）。"""
        return {
            "goods_id": getattr(p, "goods_id", None),
            "goods_name": getattr(p, "goods_name", ""),
            "thumb_url": getattr(p, "thumb_url", "") or "",
            "price": getattr(p, "price", None),
            "price_min": getattr(p, "price_min", None),
            "price_max": getattr(p, "price_max", None),
            "sold_quantity": getattr(p, "sold_quantity", 0) or 0,
            "quantity": 0,  # 知识库无库存字段，默认0
            "goods_type": "",  # 知识库无商品类型，默认""
            "tag": "",  # 知识库无标签，默认""
            "goods_url": "",  # 知识库无商品链接，默认""
            "is_spike": False,
            "support_customize": False,
        }

    @staticmethod
    def _product_to_detail_dict(p) -> Dict[str, Any]:
        """将 ProductKnowledge ORM 对象转换为商品详情 dict（与拼多多对齐）。"""
        specs = getattr(p, "specifications", None)
        # 拼多多 specifications 是 list，知识库是 text，统一转为 list
        if isinstance(specs, str) and specs.strip():
            specifications = [specs]
        elif isinstance(specs, list):
            specifications = specs
        else:
            specifications = []

        return {
            "goods_id": getattr(p, "goods_id", None),
            "goods_name": getattr(p, "goods_name", ""),
            "specifications": specifications,
            "price": getattr(p, "price", None),
            "thumb_url": getattr(p, "thumb_url", "") or "",
            "sold_quantity": getattr(p, "sold_quantity", 0) or 0,
        }


__all__ = ["ProductManager"]
