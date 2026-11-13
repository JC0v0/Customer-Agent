import os
from pathlib import Path
import json
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Any, Optional, Union, Generator
from utils.logger_loguru import get_logger
from utils.db_pragma import setup_sqlite_pragmas
from utils.runtime_path import resolve_data_path
from utils.secret_store import protect_secret, unprotect_secret
from database.models import Base, Channel, Shop, Account, Keyword


class DatabaseManager:
    """数据库管理类，提供数据库操作的封装

    单例管理：通过 DI 容器注册为单例（推荐方式）。
    也支持通过 get_db_manager() 函数获取单例实例。
    """

    def __init__(self, db_path: str = './temp/channel_shop.db'):
        """初始化数据库连接

        Args:
            db_path: 数据库文件路径
        """
        # 确保数据库目录存在
        db_path = str(resolve_data_path(str(db_path or "./temp/channel_shop.db").strip()))
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 创建数据库引擎
        self.engine = create_engine(f'sqlite:///{db_path}')
        setup_sqlite_pragmas(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # 创建表结构
        Base.metadata.create_all(self.engine)

        self.logger = get_logger()
        # 初始化数据库
        self.init_db()

    def init_db(self):
        """初始化渠道信息（每次启动幂等补齐）

        这里必须把已支持的全部渠道登记上：add_shop / add_account 都先按
        channel_name 查 channels 表，查不到就直接返回 False。抖音接进来后
        这里漏登记，表现就是「添加抖音账号 → 登录成功 → 提示数据写入失败」。
        """
        channels = (
            ("pinduoduo", "拼多多"),
            ("douyin", "抖音"),
        )
        for channel_name, description in channels:
            self.add_channel(channel_name, description)


    def get_session(self):
        """获取数据库会话"""
        return self.Session()

    def dispose(self) -> None:
        """Close pooled SQLite connections during application shutdown/tests."""
        self.engine.dispose()

    @staticmethod
    def _serialize_cookies(cookies: Any) -> Optional[str]:
        if cookies is None:
            return None
        if isinstance(cookies, str):
            return cookies
        try:
            return json.dumps(cookies, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(cookies)

    @classmethod
    def _stored_cookies(cls, cookies: Any) -> Optional[str]:
        """Serialize and protect cookies before writing them to SQLite."""
        serialized = cls._serialize_cookies(cookies)
        return protect_secret(serialized) if serialized else serialized

    @staticmethod
    def _read_cookies(account: Account) -> Optional[str]:
        """Decrypt cookies and migrate legacy plaintext rows on Windows."""
        stored = account.cookies
        if not stored:
            return stored
        decoded = unprotect_secret(stored)
        if os.name == "nt" and not stored.startswith("dpapi:v1:"):
            # Keep the legacy value readable while upgrading it in place.  No
            # row is deleted, and protect_secret fails closed if DPAPI fails.
            account.cookies = protect_secret(stored)
        return decoded

    @staticmethod
    def _read_password(account: Account) -> str:
        """Decrypt a password and upgrade legacy plaintext rows in place.

        Older installations stored passwords as plaintext.  Keep those rows
        readable for compatibility, but write the protected representation
        during the same session so the migration is durable without deleting
        or replacing any account data.
        """
        stored = account.password
        if not stored:
            return ""
        decoded = unprotect_secret(stored) or ""
        if os.name == "nt" and decoded and not stored.startswith("dpapi:v1:"):
            account.password = protect_secret(decoded) or ""
        return decoded

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """数据库会话上下文管理器，自动处理 commit/rollback/close"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(
                f"数据库操作失败: error_type={type(e).__name__}"
            )
            raise
        finally:
            session.close()

    # ==================== 私有辅助方法 ====================
    def _get_channel(self, session: Session, channel_name: str) -> Optional[Channel]:
        """获取渠道对象"""
        return session.query(Channel).filter(Channel.channel_name == channel_name).first()

    def _get_shop(self, session: Session, channel: Channel, shop_id: str) -> Optional[Shop]:
        """获取店铺对象"""
        return session.query(Shop).filter(
            Shop.channel_id == channel.id,
            Shop.shop_id == shop_id
        ).first()

    def _get_account_by_user_id(self, session: Session, shop: Shop, user_id: str) -> Optional[Account]:
        """通过user_id获取账号对象"""
        return session.query(Account).filter(
            Account.shop_id == shop.id,
            Account.user_id == user_id
        ).first()

    def _get_account_by_username(self, session: Session, shop: Shop, username: str) -> Optional[Account]:
        """通过username获取账号对象"""
        return session.query(Account).filter(
            Account.shop_id == shop.id,
            Account.username == username
        ).first()

    # ==================== 渠道相关操作 ====================
    def add_channel(self, channel_name: str, description: str = None) -> bool:
        """添加渠道"""
        with self.session_scope() as session:
            existing = session.query(Channel).filter(Channel.channel_name == channel_name).first()
            if existing:
                return True
            channel = Channel(channel_name=channel_name, description=description)
            session.add(channel)
            return True

    def get_channel(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """获取渠道信息"""
        with self.session_scope() as session:
            channel = session.query(Channel).filter(Channel.channel_name == channel_name).first()
            if not channel:
                return None
            return {
                'id': channel.id,
                'channel_name': channel.channel_name,
                'description': channel.description
            }

    def get_all_channels(self) -> List[Dict[str, Any]]:
        """获取所有渠道"""
        with self.session_scope() as session:
            channels = session.query(Channel).all()
            return [
                {
                    'id': channel.id,
                    'channel_name': channel.channel_name,
                    'description': channel.description
                }
                for channel in channels
            ]

    def delete_channel(self, channel_name: str) -> bool:
        """删除渠道"""
        with self.session_scope() as session:
            channel = session.query(Channel).filter(Channel.channel_name == channel_name).first()
            if not channel:
                self.logger.warning(f"渠道 {channel_name} 不存在")
                return False
            session.delete(channel)
            self.logger.info(f"成功删除渠道: {channel_name}")
            return True
    
    # 店铺相关操作
    def add_shop(self, channel_name: str, shop_id: str, shop_name: str, shop_logo: str, description: str = None) -> bool:
        """添加店铺"""
        with self.session_scope() as session:
            channel = session.query(Channel).filter(Channel.channel_name == channel_name).first()
            if not channel:
                self.logger.error(f"添加店铺失败: 渠道 {channel_name} 不存在")
                return False
            existing = session.query(Shop).filter(
                Shop.channel_id == channel.id,
                Shop.shop_id == shop_id
            ).first()
            if existing:
                self.logger.warning(f"店铺 {shop_id} 已存在于渠道 {channel_name}")
                return False
            shop = Shop(
                channel_id=channel.id,
                shop_id=shop_id,
                shop_name=shop_name,
                shop_logo=shop_logo,
                description=description
            )
            session.add(shop)
            self.logger.info(f"成功添加店铺: {shop_name}({shop_id}) 到渠道 {channel_name}")
            return True

    def get_shop(self, channel_name: str, shop_id: str) -> Optional[Dict[str, Any]]:
        """获取店铺信息"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return None
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return None
            return {
                'id': shop.id,
                'channel_id': shop.channel_id,
                'channel_name': channel_name,
                'shop_id': shop.shop_id,
                'shop_name': shop.shop_name,
                'shop_logo': shop.shop_logo,
                'description': shop.description,
            }

    def get_shops_by_channel(self, channel_name: str) -> List[Dict[str, Any]]:
        """获取指定渠道下的所有店铺"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return []
            shops = session.query(Shop).filter(Shop.channel_id == channel.id).all()
            return [
                {
                    'id': shop.id,
                    'channel_id': shop.channel_id,
                    'channel_name': channel_name,
                    'shop_id': shop.shop_id,
                    'shop_name': shop.shop_name,
                    'shop_logo': shop.shop_logo,
                    'description': shop.description
                }
                for shop in shops
            ]

    def update_shop_info(self, channel_name: str, shop_id: str, shop_name: str = None, shop_logo: str = None, description: str = None) -> bool:
        """更新店铺信息"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return False
            if shop_name is not None:
                shop.shop_name = shop_name
            if shop_logo is not None:
                shop.shop_logo = shop_logo
            if description is not None:
                shop.description = description
            return True

    def delete_shop(self, channel_name: str, shop_id: str) -> bool:
        """删除店铺"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return False
            session.delete(shop)
            return True

    # ==================== 账号相关操作 ====================
    def add_account(self, channel_name: str, shop_id: str, user_id: str, username: str, password: str, cookies: str = None) -> bool:
        """添加账号"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                self.logger.error(f"添加账号失败: 渠道 {channel_name} 不存在")
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                self.logger.error(f"添加账号失败: 店铺 {shop_id} 不存在")
                return False
            existing_by_user = self._get_account_by_user_id(session, shop, user_id)
            if existing_by_user:
                self.logger.warning(f"账号用户ID {user_id} 已存在于店铺 {shop_id}")
                return False
            existing = self._get_account_by_username(session, shop, username)
            if existing:
                self.logger.warning(f"账号 {username} 已存在于店铺 {shop_id}")
                return False
            account = Account(
                shop_id=shop.id,
                user_id=user_id,
                username=username,
                password=protect_secret(password) or "",
                cookies=self._stored_cookies(cookies),
                status=None
            )
            session.add(account)
            self.logger.info(f"成功添加账号: {username} 到店铺 {shop_id}")
            return True

    def get_account(self, channel_name: str, shop_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """获取账号信息"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                self.logger.warning(f"未找到渠道: {channel_name}")
                return None
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                self.logger.warning(f"未找到店铺: {shop_id} (渠道: {channel_name})")
                return None
            account = self._get_account_by_user_id(session, shop, user_id)
            if not account:
                self.logger.warning(f"未找到账户: {user_id} (店铺 ID: {shop_id})")
                return None
            return {
                'id': account.id,
                'shop_id': account.shop_id,
                'user_id': account.user_id,
                'username': account.username,
                'password': self._read_password(account),
                'cookies': self._read_cookies(account),
                'status': account.status
            }

    def update_account_info(self, channel_name: str, shop_id: str, user_id: str, username: Optional[str] = None, password: Optional[str] = None, cookies: Optional[str] = None, status: Optional[int] = None) -> bool:
        """更新账号信息"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                self.logger.error(f"更新账号失败: 渠道 {channel_name} 不存在")
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                self.logger.error(f"更新账号失败: 店铺 {shop_id} 不存在于渠道 {channel_name}")
                return False
            account = self._get_account_by_user_id(session, shop, user_id)
            if not account:
                self.logger.error(f"更新账号失败: 账号 {user_id} 不存在于店铺 {shop_id}")
                return False
            if username is not None:
                account.username = username
            if password is not None:
                account.password = protect_secret(password) or ""
            if cookies is not None:
                account.cookies = self._stored_cookies(cookies)
            if status is not None:
                account.status = status
            self.logger.info(f"成功更新账号信息: {username} (用户ID: {user_id})")
            return True

    def get_accounts_by_shop(self, channel_name: str, shop_id: str) -> List[Dict[str, Any]]:
        """获取指定店铺下的所有账号"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return []
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return []
            accounts = session.query(Account).filter(Account.shop_id == shop.id).all()
            result = []
            for account in accounts:
                result.append({
                    'id': account.id,
                    'shop_id': account.shop_id,
                    'user_id': account.user_id,
                    'username': account.username,
                    'password': self._read_password(account),
                    'cookies': self._read_cookies(account),
                    'status': account.status
                })
            return result

    def get_all_accounts_with_details(self) -> List[Dict[str, Any]]:
        """
        批量获取所有账号及其关联的店铺和渠道信息（减少N+1查询）

        Returns:
            List[Dict]: 包含 channel_name, shop_id, shop_name, shop_logo, username, password, status, user_id, cookies
        """
        with self.session_scope() as session:
            # 使用 join 一次性查询所有数据
            results = (
                session.query(Account, Shop, Channel)
                .join(Shop, Account.shop_id == Shop.id)
                .join(Channel, Shop.channel_id == Channel.id)
                .all()
            )

            details = []
            for account, shop, channel in results:
                details.append({
                    'channel_name': channel.channel_name,
                    'shop_id': shop.shop_id,
                    'shop_name': shop.shop_name,
                    'shop_logo': shop.shop_logo,
                    'username': account.username,
                    'password': self._read_password(account),
                    'status': account.status,
                    'user_id': account.user_id,
                    'cookies': self._read_cookies(account)
                })
            return details

    def update_account_status(self, channel_name: str, shop_id: str, user_id: str, status: int) -> bool:
        """更新账号状态"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return False
            account = self._get_account_by_user_id(session, shop, user_id)
            if not account:
                return False
            account.status = status
            return True

    def update_account_cookies(self, channel_name: str, shop_id: str, user_id: str, cookies: str) -> bool:
        """更新账号cookies"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return False
            account = self._get_account_by_user_id(session, shop, user_id)
            if not account:
                return False
            account.cookies = self._stored_cookies(cookies)
            return True

    def delete_account(self, channel_name: str, shop_id: str, user_id: str) -> bool:
        """删除账号"""
        with self.session_scope() as session:
            channel = self._get_channel(session, channel_name)
            if not channel:
                return False
            shop = self._get_shop(session, channel, shop_id)
            if not shop:
                return False
            account = self._get_account_by_user_id(session, shop, user_id)
            if not account:
                return False
            session.delete(account)
            return True

    # 关键词相关操作
    def add_keyword(self, keyword: str) -> bool:
        """添加关键词"""
        with self.session_scope() as session:
            existing = session.query(Keyword).filter(Keyword.keyword == keyword).first()
            if existing:
                self.logger.warning(f"关键词 {keyword} 已存在")
                return False
            keyword_obj = Keyword(keyword=keyword)
            session.add(keyword_obj)
            self.logger.info(f"成功添加关键词: {keyword}")
            return True

    def get_keyword(self, keyword: str) -> Optional[Dict[str, Any]]:
        """获取关键词信息"""
        with self.session_scope() as session:
            keyword_obj = session.query(Keyword).filter(Keyword.keyword == keyword).first()
            if not keyword_obj:
                return None
            return {
                'id': keyword_obj.id,
                'keyword': keyword_obj.keyword
            }

    def get_all_keywords(self) -> List[Dict[str, Any]]:
        """获取所有关键词"""
        with self.session_scope() as session:
            keywords = session.query(Keyword).all()
            return [
                {
                    'id': keyword.id,
                    'keyword': keyword.keyword
                }
                for keyword in keywords
            ]

    def update_keyword(self, old_keyword: str, new_keyword: str) -> bool:
        """更新关键词"""
        with self.session_scope() as session:
            keyword_obj = session.query(Keyword).filter(Keyword.keyword == old_keyword).first()
            if not keyword_obj:
                self.logger.warning(f"关键词 {old_keyword} 不存在")
                return False
            if old_keyword != new_keyword:
                existing = session.query(Keyword).filter(Keyword.keyword == new_keyword).first()
                if existing:
                    self.logger.warning(f"关键词 {new_keyword} 已存在")
                    return False
            keyword_obj.keyword = new_keyword
            self.logger.info(f"成功更新关键词: {old_keyword} -> {new_keyword}")
            return True

    def delete_keyword(self, keyword: str) -> bool:
        """删除关键词"""
        with self.session_scope() as session:
            keyword_obj = session.query(Keyword).filter(Keyword.keyword == keyword).first()
            if not keyword_obj:
                self.logger.warning(f"关键词 {keyword} 不存在")
                return False
            session.delete(keyword_obj)
            self.logger.info(f"成功删除关键词: {keyword}")
            return True

_db_instance: Optional["DatabaseManager"] = None

def get_db_manager() -> "DatabaseManager":
    """获取 DatabaseManager 单例。

    优先从 DI 容器获取；若 DI 尚未注册（启动早期），回退到本地单例，
    以确保全应用始终共用同一个实例，避免出现多个 db 文件分裂。
    """
    global _db_instance
    try:
        from core.di_container import container
        if container.is_registered(DatabaseManager):
            return container.get(DatabaseManager)
    except ImportError:
        pass

    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance

class _LazyDBProxy:
    """延迟代理，用于兼容旧代码的全局 db_manager 实例，底层共用 DI 容器。"""

    def __getattr__(self, name: str):
        return getattr(get_db_manager(), name)

db_manager = _LazyDBProxy()
