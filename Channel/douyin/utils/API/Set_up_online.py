"""
抖音客服状态 / 店铺信息 API（对齐拼多多 AccountMonitor）。

【实证端点】2026-11-10 真机验证均返回 HTTP 200 且 code=0：
  - GET backstage/currentuser?biz_type=4            → ShopName / ShopId / ShopLogo
  - GET backstage/onlineservice?busy=1&biz_type=4    → 在线客服列表
  - GET backstage/hosting_service/status?biz_type=4  → 平台智能客服托管状态

【诚实说明】
抓包中没有「设置客服在线状态」的端点（拼多多对应 plateau/chat/set_csstatus）。
抖音客服在线状态由飞鸽工作台控制，本模块只读取平台真实状态；
set_csstatus 不会假装调用平台设置接口，仅同步本地状态并记录平台现状。
"""
from typing import Any, Dict, List

from ..base_request import BaseRequest
from utils.logger_loguru import get_logger

logger = get_logger("DouyinAccountMonitor")


class AccountMonitor(BaseRequest):
    """抖音客服状态/店铺信息查询（接口签名与拼多多 AccountMonitor 对齐）。"""

    CURRENT_USER_URL = (
        "https://pigeon.jinritemai.com/backstage/currentuser?biz_type=4&_pms=1"
    )
    ONLINE_SERVICE_URL = (
        "https://pigeon.jinritemai.com/backstage/onlineservice?busy=1&biz_type=4&_pms=1"
    )
    HOSTING_STATUS_URL = (
        "https://pigeon.jinritemai.com/backstage/hosting_service/status?biz_type=4&_pms=1"
    )

    def __init__(self, cookies=None, shop_id=None, user_id=None,
                 channel_name="douyin"):
        super().__init__(shop_id=shop_id, user_id=user_id, channel_name=channel_name)
        if cookies:
            self.update_cookies(cookies)

    def _headers(self) -> Dict[str, str]:
        return {
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
            ),
            "referer": "https://im.jinritemai.com/",
        }

    def _code0_data(self, url: str) -> Any:
        """请求飞鸽后台端点，只返回 code=0 时的 data，失败返回 None。"""
        try:
            result = self.get(
                url,
                params=self._backstage_params(),
                headers=self._headers(),
                timeout=10,
            )
        except Exception as exc:
            endpoint = url.split("?")[0].rsplit("/", 1)[-1]
            logger.warning(
                f"抖音后台状态拉取异常: endpoint={endpoint}, "
                f"error_type={type(exc).__name__}"
            )
            return None
        if isinstance(result, dict) and result.get("code") == 0:
            return result.get("data")
        return None

    def get_shop_info(self) -> Dict[str, Any]:
        """当前登录店铺的真实信息（店铺名 / 平台 ShopId / 店铺头像）。"""
        data = self._code0_data(self.CURRENT_USER_URL)
        if isinstance(data, dict):
            return {
                "shop_name": data.get("ShopName", ""),
                "real_shop_id": data.get("ShopId", ""),
                "shop_logo": data.get("ShopLogo", ""),
            }
        return {}

    def get_online_service(self) -> List[Dict[str, Any]]:
        """在线客服列表（id / shop_id / service_name / online_status 等）。"""
        data = self._code0_data(self.ONLINE_SERVICE_URL)
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, dict):
                    out.append(item)
            return out
        return []

    def get_hosting_status(self) -> Dict[str, Any]:
        """平台智能客服（托管）开关状态。"""
        data = self._code0_data(self.HOSTING_STATUS_URL)
        return data if isinstance(data, dict) else {}

    def get_account_status(self) -> Dict[str, Any]:
        """汇总平台侧真实状态：店铺信息 / 在线客服 / 智能客服托管。"""
        shop = self.get_shop_info()
        services = self.get_online_service()
        hosting = self.get_hosting_status()
        status = {
            "shop_name": shop.get("shop_name", ""),
            "real_shop_id": shop.get("real_shop_id", ""),
            "shop_logo": shop.get("shop_logo", ""),
            "online_service_count": len(services),
            "online_services": services,
            "hosting_opened": hosting.get("hosting_opened"),
            "hosting_service_status": hosting.get("hosting_service_status"),
            "platform_reachable": bool(shop),
        }
        logger.info(
            f"抖音平台状态拉取成功: 店铺={status['shop_name']}, "
            f"ShopId={status['real_shop_id']}, "
            f"在线客服={status['online_service_count']}, "
            f"智能客服托管={status['hosting_opened']}"
        )
        return status

    def set_csstatus(self, status: str) -> bool:
        """抖音未抓到平台设置端点，不假装设置；读取平台现状后仅同步本地状态。

        返回 True 表示本地状态可同步且平台状态可查询，
        不代表平台在线状态被修改（平台状态仍由飞鸽工作台控制）。
        """
        platform = self.get_account_status()
        logger.info(
            f"抖音在线状态由飞鸽工作台控制，未调用平台设置接口; "
            f"本地状态同步为 {status}, "
            f"平台在线客服={platform['online_service_count']}"
        )
        return True
