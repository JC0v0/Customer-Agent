"""
产品知识自动同步服务
=================

从拼多多API拉取商品列表，调用多模态LLM分析提取产品知识存入知识库。
"""
import asyncio
import re
import threading
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass
import json
import time

from config import get_active_llm_profile

from Channel.pinduoduo.utils.API.product_manager import ProductManager
from database.knowledge_service import KnowledgeService
from utils.logger_loguru import get_logger
from utils.llm_provider import LLMProfile, ProfileValidationError
from utils.llm_transport import async_completion

logger = get_logger("ProductSync")


@dataclass
class SyncProgress:
    """同步进度"""
    total: int
    current: int
    success: int
    failed: int
    current_goods_name: str
    cancelled: bool = False
    phase: str = "fetching"  # "fetching": 抓取商品列表, "extracting": 提取知识
    # LLM 未产出有效知识：只补基本信息或保留旧内容，不计为提取成功。
    degraded: int = 0


class ProductSyncService:
    """产品知识自动同步服务"""

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        request_delay: float = 1.0,
    ):
        """
        初始化

        Args:
            knowledge_service: 知识库服务实例
            request_delay: API请求间隔（秒），避免限流
        """
        self.knowledge_service = knowledge_service
        self.request_delay = request_delay
        self._cancellation_event = threading.Event()
        logger.info("ProductSyncService 初始化成功")

    def cancel(self) -> None:
        """取消同步"""
        self._cancellation_event.set()
        logger.info("同步已取消")

    def is_cancelled(self) -> bool:
        """检查是否已取消"""
        return self._cancellation_event.is_set()

    def _reset_cancellation(self) -> None:
        """重置取消事件"""
        self._cancellation_event.clear()

    def _snapshot_llm_profile(self) -> Optional[LLMProfile]:
        """Capture one validated profile for a synchronization run."""
        try:
            return get_active_llm_profile(require_confirmation=False)
        except ProfileValidationError as exc:
            logger.warning(
                "LLM profile unavailable for product sync; "
                f"error_code={exc.code}"
            )
            return None

    async def sync_shop(
        self,
        shop_id: int,
        shop_db_id: int,
        user_id: str,
        is_full_sync: bool = False,
        progress_callback: Optional[Callable[[SyncProgress], None]] = None,
    ) -> SyncProgress:
        """
        同步店铺产品知识（两阶段同步）

        第一阶段：快速抓取所有商品基本信息（商品ID、名称等）并保存到数据库
        第二阶段：批量异步识别每个商品的详细信息

        Args:
            shop_id: 店铺ID（拼多多的shop_id）
            shop_db_id: 店铺在数据库中的ID
            user_id: 用户ID（用于拼多多API认证）
            is_full_sync: True=全量同步，False=增量同步（仅同步本地不存在的商品）
            progress_callback: 进度回调，每次更新进度调用

        Returns:
            最终同步进度
        """
        self._reset_cancellation()
        llm_profile = self._snapshot_llm_profile()
        pm = ProductManager(shop_id=str(shop_id), user_id=user_id)

        # ================== 第一阶段：快速抓取商品列表 ==================
        logger.info("=== 第一阶段：开始抓取商品列表 ===")

        # 第一页获取总数量
        first_page = await asyncio.to_thread(pm.get_product_list, page=1, size=20)
        if not first_page["success"]:
            logger.error("获取商品列表失败: provider_error=true")
            progress = SyncProgress(
                total=0, current=0, success=0, failed=0, current_goods_name="",
                phase="fetching"
            )
            progress.failed = 1
            return progress

        total = first_page["total"]
        logger.info(f"店铺共有 {total} 个商品，开始抓取商品列表...")

        progress = SyncProgress(
            total=total,
            current=0,
            success=0,
            failed=0,
            current_goods_name="",
            phase="fetching"
        )

        # 分页拉取所有商品
        current_page = 1
        all_products: List[Dict[str, Any]] = []

        while True:
            if self.is_cancelled():
                progress.cancelled = True
                logger.info("同步已被用户取消")
                break

            page_result = await asyncio.to_thread(
                pm.get_product_list, page=current_page, size=50
            )
            if not page_result["success"]:
                logger.error(
                    f"获取第 {current_page} 页失败: provider_error=true"
                )
                break

            products = page_result["products"]
            if not products:
                break

            all_products.extend(products)
            current_page += 1

            # 更新进度
            progress.current = len(all_products)
            if all_products:
                progress.current_goods_name = all_products[-1].get("goods_name", "")
            if progress_callback:
                progress_callback(progress)

            # 延迟避免限流
            await asyncio.sleep(self.request_delay)

        if self.is_cancelled():
            return progress

        logger.info(f"第一阶段完成：共获取 {len(all_products)} 个商品")

        # 增量同步筛选：只处理本地不存在的商品
        products_to_process: List[Dict[str, Any]] = []
        if not is_full_sync:
            original_count = len(all_products)
            filtered_products: List[Dict[str, Any]] = []
            for p in all_products:
                goods_id = p.get("goods_id")
                existing = await asyncio.to_thread(
                    self.knowledge_service.get_product_by_goods_id,
                    shop_db_id,
                    goods_id,
                )
                if not existing:
                    filtered_products.append(p)
            logger.info(f"增量同步: 总商品 {original_count}，需要同步 {len(filtered_products)} 个（已存在跳过）")
            products_to_process = filtered_products
        else:
            products_to_process = all_products

        # ================== 第一阶段B：快速保存商品基本信息 ==================
        logger.info("=== 开始快速保存商品基本信息 ===")
        progress.phase = "saving_basic"
        progress.total = len(products_to_process)
        progress.current = 0
        progress.success = 0
        progress.failed = 0
        progress.degraded = 0

        for idx, product in enumerate(products_to_process):
            if self.is_cancelled():
                progress.cancelled = True
                break

            goods_id = product.get("goods_id")
            goods_name = product.get("goods_name", f"goods_{goods_id}")
            progress.current = idx + 1
            progress.current_goods_name = goods_name

            try:
                # 先只保存基本信息，不调用LLM
                await asyncio.to_thread(
                    self.knowledge_service.add_or_update_product,
                    shop_id=shop_db_id,
                    goods_id=goods_id,
                    goods_name=goods_name,
                    price=product.get("price"),
                    price_min=product.get("price_min"),
                    price_max=product.get("price_max"),
                    sold_quantity=product.get("sold_quantity"),
                    thumb_url=product.get("thumb_url"),
                    specifications=None,
                    extracted_content=None,  # 留空，第二阶段填充
                )
                progress.success += 1
                logger.debug(f"商品基本信息已保存: {goods_name} (ID: {goods_id})")
            except Exception as e:
                logger.error(
                    f"保存商品基本信息失败: error_type={type(e).__name__}"
                )
                progress.failed += 1
                continue

            if progress_callback:
                progress_callback(progress)

        if self.is_cancelled():
            logger.info("同步已取消")
            return progress

        logger.info(f"商品基本信息保存完成: 成功 {progress.success}, 失败 {progress.failed}")

        # ================== 第二阶段：并发提取详细知识 ==================
        logger.info("=== 第二阶段：开始并发提取商品详细知识 ===")
        progress.phase = "extracting"
        progress.current = 0
        progress.success = 0
        progress.failed = 0
        progress.degraded = 0

        # 使用线程安全的计数器
        from threading import Lock
        counter_lock = Lock()

        async def process_single_product(product: Dict[str, Any]):
            """处理单个商品的知识提取"""
            if self.is_cancelled():
                return

            goods_id = product.get("goods_id")
            goods_name = product.get("goods_name", f"goods_{goods_id}")

            # 更新当前处理商品名称
            with counter_lock:
                progress.current_goods_name = goods_name
                if progress_callback:
                    progress_callback(progress)

            logger.debug(f"正在提取知识: {goods_name} (ID: {goods_id})")

            try:
                # 获取商品详情
                detail = await asyncio.to_thread(pm.get_product_detail, goods_id)
                await asyncio.sleep(self.request_delay)

                if not detail["success"]:
                    logger.error(
                        f"获取商品详情失败: error_type=provider_error"
                    )
                    with counter_lock:
                        progress.failed += 1
                        progress.current += 1
                        if progress_callback:
                            progress_callback(progress)
                    return

                product_info = detail["product_info"]

                # 调用LLM提取知识
                extracted, extraction_succeeded = await self._extract_product_knowledge(
                    product,
                    product_info,
                    profile=llm_profile,
                )

                # 由数据库层原子保护旧内容：降级只能填空，不能覆盖已有知识。
                saved = await asyncio.to_thread(
                    self.knowledge_service.update_product_extracted_content,
                    shop_id=shop_db_id,
                    goods_id=goods_id,
                    specifications=json.dumps(product_info.get("specifications", [])),
                    extracted_content=extracted,
                    extraction_succeeded=extraction_succeeded,
                )
                if not saved:
                    raise RuntimeError("product_knowledge_update_failed")

                with counter_lock:
                    if extraction_succeeded:
                        progress.success += 1
                        logger.info(f"商品知识提取成功: {goods_name} (ID: {goods_id})")
                    else:
                        progress.degraded += 1
                        logger.warning(
                            f"商品知识提取降级（仅补基础信息，已有内容不覆盖）: "
                            f"{goods_name} (ID: {goods_id})"
                        )
                    progress.current += 1
                    if progress_callback:
                        progress_callback(progress)

            except Exception as e:
                logger.error(
                    f"提取商品知识失败: error_type={type(e).__name__}"
                )
                with counter_lock:
                    progress.failed += 1
                    progress.current += 1
                    if progress_callback:
                        progress_callback(progress)

        # 并发处理，控制并发数量避免限流
        max_concurrent = 3  # 最多3个并发
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(product: Dict[str, Any]):
            async with semaphore:
                if not self.is_cancelled():
                    await process_single_product(product)

        # 创建所有任务
        tasks = [process_with_semaphore(product) for product in products_to_process]

        # 运行所有任务
        await asyncio.gather(*tasks)

        logger.info(
            f"同步完成: 总计 {progress.total}, 成功 {progress.success}, "
            f"降级 {progress.degraded}, 失败 {progress.failed}"
        )
        return progress

    async def _extract_product_knowledge(
        self,
        list_product: Dict[str, Any],
        detail_product: Dict[str, Any],
        profile: Optional[LLMProfile] = None,
    ) -> Tuple[str, bool]:
        """
        调用LLM提取产品知识

        Args:
            list_product: 商品列表中的商品信息
            detail_product: 商品详情信息

        Returns:
            (知识文本, 是否提取成功)。失败返回基本信息和 False，调用方必须
            将降级与成功分开计数，并防止覆盖已有知识。
        """
        profile = profile or self._snapshot_llm_profile()
        if profile is None:
            logger.warning("LLM profile unavailable, returning basic info only")
            return self._format_basic_info(list_product, detail_product), False

        # 构建prompt
        thumb_url = list_product.get("thumb_url", "")
        specifications = detail_product.get("specifications", [])

        system_prompt = """你是一个电商产品信息提取助手。请根据提供的商品信息和商品图片，提取详细的产品知识，方便客服回答顾客问题。

请务必先从商品名称和描述中提取以下独立字段，然后再生成其他内容：

请输出 JSON 格式，包含以下字段：
{
  "brand": "品牌（从商品名称或描述中提取，如"葵花"、"同仁堂"等）",
  "origin": "产地（从描述中提取，如"中国广东"、"日本"等）",
  "ingredients": "产品成分/材料/主要原料（从描述中提取，如"草本成分"、"植物精油"等）",
  "spec_quantity": "规格/数量/包装规格（从描述中提取，如"1盒8贴"、"50g/瓶"等）",
  "suitable_age": "适用年龄（从描述中提取，如"儿童成人通用"、"3岁以上"等）",
  "shelf_life": "保质期/有效期（从描述中提取，如"24个月"、"3年"等）",
  "description": "商品整体描述，包含卖点、特点、材质、用途等信息",
  "key_points": ["卖点1", "卖点2", ...],
  "usage": "使用方法或注意事项（如果有）",
  "faq": [{"question": "常见问题", "answer": "答案"}, ...]
}

重要提示：
1. 请优先从商品名称和描述文本中提取品牌、成分、规格、适用年龄等信息
2. 如果某个信息已经在描述中提到，请务必提取到对应的独立字段中
3. 如果无法分析图片，只基于文本信息提取即可
4. 如果某个信息确实无法提取，对应字段留空字符串"""

        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": f"""商品名称: {list_product.get('goods_name')}
商品价格: {list_product.get('price')}
已售数量: {list_product.get('sold_quantity')}
规格: {json.dumps(specifications, ensure_ascii=False)}
"""
            }
        ]

        # 如果有图片URL，添加图片
        if thumb_url:
            # OpenAI 多模态格式：image_url
            user_content.append({
                "type": "image_url",
                "image_url": {"url": thumb_url},
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await async_completion(
                profile,
                messages,
                temperature=0.3,
                use_tools=False,
                response_format={"type": "json_object"},
                timeout=120.0,
            )
            content = str(response.content or "").strip()
            logger.debug(f"LLM输出长度={len(content)}")

            # 尝试解析JSON
            try:
                data = self._parse_knowledge_json(content)
                # 记录提取到的规格信息
                logger.debug(f"提取到的规格字段 - brand: {data.get('brand')}, origin: {data.get('origin')}, ingredients: {data.get('ingredients')}, spec_quantity: {data.get('spec_quantity')}, suitable_age: {data.get('suitable_age')}, shelf_life: {data.get('shelf_life')}")

                # 格式化输出
                output_parts = [f"# {list_product.get('goods_name')}"]
                output_parts.append("")
                # 产品规格信息
                spec_info = []
                if data.get("brand") and data["brand"].strip():
                    spec_info.append(f"- **品牌**: {data['brand']}")
                if data.get("origin") and data["origin"].strip():
                    spec_info.append(f"- **产地**: {data['origin']}")
                if data.get("ingredients") and data["ingredients"].strip():
                    spec_info.append(f"- **产品成分**: {data['ingredients']}")
                if data.get("spec_quantity") and data["spec_quantity"].strip():
                    spec_info.append(f"- **规格/数量**: {data['spec_quantity']}")
                if data.get("suitable_age") and data["suitable_age"].strip():
                    spec_info.append(f"- **适用年龄**: {data['suitable_age']}")
                if data.get("shelf_life") and data["shelf_life"].strip():
                    spec_info.append(f"- **保质期**: {data['shelf_life']}")
                if spec_info:
                    output_parts.append("## 产品规格")
                    output_parts.extend(spec_info)
                    output_parts.append("")
                if data.get("description"):
                    output_parts.append("## 产品描述")
                    output_parts.append(data["description"])
                    output_parts.append("")
                if data.get("key_points") and isinstance(data["key_points"], list):
                    output_parts.append("## 产品卖点")
                    for i, point in enumerate(data["key_points"], 1):
                        output_parts.append(f"{i}. {point}")
                    output_parts.append("")
                if data.get("usage"):
                    output_parts.append("## 使用说明")
                    output_parts.append(data["usage"])
                    output_parts.append("")
                if data.get("faq") and isinstance(data["faq"], list):
                    output_parts.append("## 常见问题")
                    for faq in data["faq"]:
                        output_parts.append(f"**Q:** {faq.get('question')}")
                        output_parts.append(f"**A:** {faq.get('answer')}")
                        output_parts.append("")

                result = "\n".join(output_parts).strip()
                return result, True

            except (ValueError, TypeError) as exc:
                # 非法 JSON / 空结果 / 错误字段类型均为降级，不能把错误原文入库。
                logger.warning(
                    f"LLM输出不是有效商品知识，降级为基本信息: "
                    f"error_type={type(exc).__name__}"
                )
                return self._format_basic_info(list_product, detail_product), False

        except Exception as e:
            logger.error(f"LLM调用失败: error_type={type(e).__name__}")
            # 降级返回基本信息
            return self._format_basic_info(list_product, detail_product), False

    @staticmethod
    def _parse_knowledge_json(content: str) -> Dict[str, Any]:
        """只接受非空的商品知识对象；容忍完整 Markdown 围栏，不猜测残缺 JSON。"""
        # 某些兼容端点仍会返回围栏；只移除完整包裹，避免从说明文字中误取对象。
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```",
            content.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )
        data = json.loads(fenced.group(1) if fenced else content)
        if not isinstance(data, dict):
            raise ValueError("knowledge_must_be_object")

        normalized: Dict[str, Any] = {}
        for key in (
            "brand", "origin", "ingredients", "spec_quantity", "suitable_age",
            "shelf_life", "description", "usage",
        ):
            value = data.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError("invalid_knowledge_text_field")
            if value.strip():
                normalized[key] = value.strip()

        points = data.get("key_points")
        if points is not None:
            if not isinstance(points, list) or any(not isinstance(p, str) for p in points):
                raise ValueError("invalid_knowledge_key_points")
            if cleaned := [p.strip() for p in points if p.strip()]:
                normalized["key_points"] = cleaned

        faqs = data.get("faq")
        if faqs is not None:
            if not isinstance(faqs, list):
                raise ValueError("invalid_knowledge_faq")
            clean_faqs = []
            for faq in faqs:
                if not isinstance(faq, dict):
                    raise ValueError("invalid_knowledge_faq_item")
                question, answer = faq.get("question"), faq.get("answer")
                if not isinstance(question, str) or not isinstance(answer, str):
                    raise ValueError("invalid_knowledge_faq_text")
                if question.strip() and answer.strip():
                    clean_faqs.append({"question": question.strip(), "answer": answer.strip()})
            if clean_faqs:
                normalized["faq"] = clean_faqs

        # {} 或只有空字段不等于提取成功，不能用一个商品标题覆盖旧知识。
        if not normalized:
            raise ValueError("empty_knowledge")
        return normalized

    def _format_basic_info(
        self,
        list_product: Dict[str, Any],
        detail_product: Dict[str, Any],
    ) -> str:
        """LLM调用失败时，格式化基本信息"""
        output = [f"# {list_product.get('goods_name')}"]
        output.append("")
        if list_product.get("price"):
            output.append(f"**价格**: {list_product.get('price')}")
        if list_product.get("sold_quantity"):
            output.append(f"**已售**: {list_product.get('sold_quantity')} 件")
        specs = detail_product.get("specifications", [])
        if specs:
            output.append("")
            output.append("**规格信息**:")
            for spec in specs:
                output.append(f"- {spec}")
        return "\n".join(output).strip()
