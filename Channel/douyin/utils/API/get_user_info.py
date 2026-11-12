"""
抖音用户信息 API（对齐拼多多 GetUserInfo 接口签名）。

【实现说明】
- 主路径：调用飞鸽 backstage/currentuser 获取真实用户信息
  （从 data.CustomerServiceInfo 取 id/screen_name/avatar，从 data.ToutiaoId 取用户ID）。
  backstage 鉴权依赖浏览器会话 cookie；程序内仅持有 pigeon 凭据，
  故请求可能鉴权失败——失败时优雅回退本地数据库，绝不抛异常。
- 返回签名与拼多多 GetUserInfo.get_user_info() 一致：(user_id, user_name, mall_id)。
"""
from ..base_request import BaseRequest
from utils.logger_loguru import get_logger
from Channel.douyin.utils.log_throttle import throttled_warning, throttled_info

logger = get_logger("GetUserInfo")


class GetUserInfo(BaseRequest):
    # 飞鸽后台：当前登录用户/店铺信息（与 get_shop_info 共用端点）
    # 抓包实证返回 data.CustomerServiceInfo{id,screen_name,avatar_url} + data.ToutiaoId
    DOUYIN_CURRENT_USER_URL = (
        "https://pigeon.jinritemai.com/backstage/currentuser?biz_type=4&_pms=1"
    )

    def __init__(self, shop_id=None, user_id=None, cookies=None):
        super().__init__(shop_id=shop_id, user_id=user_id, channel_name="douyin")
        if cookies:
            self.update_cookies(cookies)

    def get_user_info(self):
        """
        获取当前账号用户信息。

        Returns:
            (user_id, user_name, mall_id) 成功；False 失败（与拼多多接口签名一致）。
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
            ):
                data = result["data"]
                # 从 CustomerServiceInfo 取客服 id / 昵称
                csi = data.get("CustomerServiceInfo") or {}
                user_id = str(
                    csi.get("id")
                    or data.get("ToutiaoId")
                    or data.get("SubToutiaoId")
                    or ""
                )
                user_name = (
                    csi.get("screen_name")
                    or data.get("ShopName")
                    or ""
                )
                mall_id = str(data.get("ShopId") or self.shop_id or "")
                if user_id and user_name:
                    throttled_info(logger, "userinfo.api_ok",
                        f"抖音用户信息拉取成功（currentuser）: "
                        f"user_id={user_id}, user_name={user_name}, mall_id={mall_id}"
                    )
                    return user_id, user_name, mall_id
                throttled_warning(logger, "userinfo.api_incomplete",
                    f"抖音 currentuser 返回数据不完整: "
                    f"user_id={user_id!r}, user_name={user_name!r}, 回退本地DB"
                )
        except Exception as e:
            throttled_warning(logger, "userinfo.api_exception",
                f"抖音 currentuser 拉取用户信息异常，回退本地数据库: "
                f"error_type={type(e).__name__}"
            )

        # 2) 回退：本地数据库账号信息（登录时已登记）
        try:
            from database import db_manager

            acct = db_manager.get_account(
                self.channel_name, self.shop_id, self.user_id
            )
            if acct:
                user_id = self.user_id or acct.get("user_id") or ""
                user_name = acct.get("username") or acct.get("user_id") or ""
                mall_id = self.shop_id or ""
                throttled_info(logger, "userinfo.db_fallback",
                    f"抖音用户信息（本地数据库降级）: user_id={user_id}, user_name={user_name}"
                )
                return user_id, user_name, mall_id
        except Exception as e:
            throttled_warning(logger, "userinfo.db_exception",
                f"抖音本地用户信息读取失败: error_type={type(e).__name__}"
            )

        # 3) 极兜底：返回构造函数入参
        if self.user_id:
            throttled_warning(logger, "userinfo.input_fallback", "抖音用户信息拉取失败，返回构造入参兜底")
            return self.user_id, self.user_id, self.shop_id or ""

        logger.error("抖音用户信息获取完全失败（无 currentuser / 无本地DB / 无入参）")
        return False


__all__ = ["GetUserInfo"]
