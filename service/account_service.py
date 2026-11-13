"""账号/店铺/渠道服务层 - 封装账号管理与登录，解耦 UI 与 database/Channel。"""
from database.db_manager import db_manager


class AccountService:
    """账号服务的薄封装，转发到 db_manager；登录转发到拼多多登录模块。"""

    # ---- 渠道 ----
    def get_all_channels(self):
        return db_manager.get_all_channels()

    # ---- 店铺 ----
    def get_shops_by_channel(self, channel_name: str):
        return db_manager.get_shops_by_channel(channel_name)

    def get_shop(self, channel_name: str, shop_id: str):
        return db_manager.get_shop(channel_name, shop_id)

    def add_shop(self, channel_name, shop_id, shop_name, shop_logo, description=None) -> bool:
        return db_manager.add_shop(channel_name, shop_id, shop_name, shop_logo, description)

    # ---- 账号 ----
    def get_accounts_by_shop(self, channel_name, shop_id):
        return db_manager.get_accounts_by_shop(channel_name, shop_id)

    def get_all_accounts_with_details(self):
        return db_manager.get_all_accounts_with_details()

    def get_account(self, channel_name, shop_id, user_id):
        return db_manager.get_account(channel_name, shop_id, user_id)

    def add_account(self, channel_name, shop_id, user_id, username, password, cookies=None) -> bool:
        return db_manager.add_account(channel_name, shop_id, user_id, username, password, cookies)

    def update_account_info(self, channel_name, shop_id, user_id, username=None, password=None, cookies=None, status=None) -> bool:
        return db_manager.update_account_info(channel_name, shop_id, user_id, username, password, cookies, status)

    def update_account_status(self, channel_name, shop_id, user_id, status) -> bool:
        return db_manager.update_account_status(channel_name, shop_id, user_id, status)

    def update_account_cookies(self, channel_name, shop_id, user_id, cookies) -> bool:
        return db_manager.update_account_cookies(channel_name, shop_id, user_id, cookies)

    def delete_account(self, channel_name, shop_id, user_id) -> bool:
        return db_manager.delete_account(channel_name, shop_id, user_id)

    # ---- 登录 ----
    async def login(
        self,
        name: str,
        password: str,
        headless: bool = False,
        channel_name: str = "pinduoduo",
    ):
        """使用账号密码登录指定渠道，返回账号信息 dict 或 False。"""
        if str(channel_name).strip().lower() == "douyin":
            from Channel.douyin.douyin_login import login_douyin
            return await login_douyin(name, password, headless=headless)

        from Channel.pinduoduo.pdd_login import login_pdd
        return await login_pdd(name, password, headless=headless)


account_service = AccountService()
