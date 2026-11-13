"""
抖音店铺信息 API（对齐拼多多 GetShopInfo 接口签名）。

【实现说明】
- 主路径：调用飞鸽 backstage/currentuser 获取真实店铺信息（店铺名/ShopId）。
  backstage 鉴权依赖浏览器会话 cookie；程序内仅持有 pigeon 凭据（token/pigeon_sign），
  故请求可能鉴权失败——失败时优雅回退本地数据库店铺信息，绝不抛异常。
- 返回签名与拼多多 GetShopInfo.get_shop_info() 一致：(shop_id, shop_name, shop_logo)。
"""
from ..base_request import BaseRequest
from utils.logger_loguru import get_logger

logger = get_logger("GetShopInfo")


class GetShopInfo(BaseRequest):
    # 飞鸽后台：当前登录用户/店铺信息（浏览器抓包实证返回 ShopName/ShopId/ShopLogo）
    DOUYIN_CURRENT_USER_URL = (
        "https://pigeon.jinritemai.com/backstage/currentuser?biz_type=4&_pms=1"
    )

    def __init__(self, shop_id=None, user_id=None, cookies=None):
        super().__init__(shop_id=shop_id, user_id=user_id, channel_name="douyin")
        if cookies:
            self.update_cookies(cookies)

    def get_shop_info(self):
        """
        获取当前店铺信息。

        Returns:
            (shop_id, shop_name, shop_logo) 成功；False 失败（与拼多多接口签名一致）。
        """
        # 1) 主路径：尝试 backstage/currentuser（浏览器会话 cookie 鉴权）
        try:
            headers = {
                "user-agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
                ),
                "referer": "https://im.jinritemai.com/",
            }
            result = self.get(
                self.DOUYIN_CURRENT_USER_URL,
                params=self._backstage_params(),
                headers=headers,
                timeout=10,
            )
            if (
                isinstance(result, dict)
                and result.get("code") == 0
                and isinstance(result.get("data"), dict)
                and result["data"].get("ShopName")
            ):
                data = result["data"]
                shop_name = data.get("ShopName", "")
                real_shop_id = data.get("ShopId")
                shop_logo = data.get("ShopLogo", "")
                logger.info(
                    f"抖音店铺信息拉取成功: shop_name={shop_name}, "
                    f"real_shop_id={real_shop_id}"
                )
                # 本地 shop_id 用稳定标识（douyin_<店铺名>），
                # 与 login 的 _local_id 规则保持一致。
                shop_id = self._local_shop_id(shop_name, shop_logo)
                # 尝试把真实 ShopId 记录到日志供后续使用
                self._real_shop_id = real_shop_id
                return shop_id, shop_name, shop_logo
        except Exception as e:
            logger.warning(
                f"抖音 currentuser 拉取店铺信息异常，回退本地数据库: "
                f"error_type={type(e).__name__}"
            )

        # 2) 回退：本地数据库店铺信息（登录时已登记）
        try:
            from database import db_manager

            local = self._pick_shop(db_manager)
            if local:
                shop_id, shop_name, shop_logo = local
                logger.info(
                    f"抖音店铺信息（本地数据库）: shop_id={shop_id}, shop_name={shop_name}"
                )
                return shop_id, shop_name, shop_logo
        except Exception as e:
            logger.warning(
                f"抖音本地店铺信息读取失败: error_type={type(e).__name__}"
            )

        # 3) 极兜底：只回传调用方传入的 shop_id，店名留空由上层自行处理。
        #    这里刻意不写死任何店铺名，否则换一个店铺的环境会显示错误信息。
        logger.warning("抖音店铺信息拉取失败，返回空店名兜底")
        return self.shop_id or "douyin_shop", "", ""

    @staticmethod
    def _local_shop_id(shop_name: str, _shop_logo: str = "") -> str:
        """与 douyin_login._local_id 保持一致的本地店铺 ID 生成规则。"""
        import re

        raw = str(shop_name or "").strip()
        safe = re.sub(r"[^A-Za-z0-9_\u4e00-\u9fa5]+", "_", raw).strip("_")[:24]
        return f"douyin_{safe or 'shop'}"

    @staticmethod
    def _pick_shop(db_manager):
        """从 shops 表选一个抖音店铺（优先店名非账号名的规范店铺）。"""
        try:
            from bridge.context import ChannelType

            shops = db_manager.get_shops_by_channel(str(ChannelType.DOUYIN)) or []
            # 优先店名 != 登录名（店铺名通常非纯数字，登录名常是手机号）
            for s in shops:
                sn = str(s.get("shop_name") or "")
                sid = str(s.get("shop_id") or "")
                if sn and not sn.isdigit():
                    return sid, sn, s.get("shop_logo") or ""
            for s in shops:
                return (
                    s.get("shop_id"),
                    s.get("shop_name") or "",
                    s.get("shop_logo") or "",
                )
        except Exception:
            return None
        return None


__all__ = ["GetShopInfo"]
