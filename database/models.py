"""SQLAlchemy models for channel, account and knowledge data."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class Channel(Base):
    __tablename__ = "channels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_name = Column(String(50), unique=True, nullable=False)
    description = Column(String(255))
    shops = relationship("Shop", back_populates="channel", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Channel(channel_name='{self.channel_name}')>"


class Shop(Base):
    __tablename__ = "shops"
    __table_args__ = (
        UniqueConstraint("channel_id", "shop_id", name="uix_shop_channel_shop"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    shop_id = Column(String(100), nullable=False)
    shop_name = Column(String(100), nullable=False)
    shop_logo = Column(String(255), nullable=True)
    description = Column(String(255))
    channel = relationship("Channel", back_populates="shops")
    accounts = relationship("Account", back_populates="shop", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Shop(shop_id='{self.shop_id}', shop_name='{self.shop_name}')>"


class Account(Base):
    __tablename__ = "accounts"
    __table_args__ = (
        UniqueConstraint("shop_id", "user_id", name="uix_account_shop_user"),
        UniqueConstraint("shop_id", "username", name="uix_account_shop_username"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    shop_id = Column(Integer, ForeignKey("shops.id"), nullable=False)
    user_id = Column(String(100), nullable=False)
    username = Column(String(100), nullable=False)
    # Stored as a DPAPI-protected value on Windows; legacy plaintext rows are
    # still readable for backwards compatibility.
    password = Column(String(255), nullable=False)
    cookies = Column(Text)
    status = Column(Integer, default=None)
    shop = relationship("Shop", back_populates="accounts")

    def __repr__(self):
        return f"<Account(username='{self.username}')>"


class Keyword(Base):
    __tablename__ = "keywords"
    __table_args__ = (UniqueConstraint("keyword", name="uix_keyword_keyword"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword = Column(String(100), nullable=False)

    def __repr__(self):
        return f"<Keyword(keyword='{self.keyword}')>"


class ProductKnowledge(Base):
    __tablename__ = "product_knowledge"
    __table_args__ = (
        UniqueConstraint("shop_id", "goods_id", name="uix_product_knowledge_shop_goods"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    shop_id = Column(Integer, ForeignKey("shops.id", ondelete="CASCADE"), nullable=False)
    goods_id = Column(Integer, nullable=False)
    goods_name = Column(String(255), nullable=False)
    price = Column(String(50), nullable=True)
    price_min = Column(Integer, nullable=True)
    price_max = Column(Integer, nullable=True)
    sold_quantity = Column(Integer, nullable=True)
    thumb_url = Column(String(500), nullable=True)
    specifications = Column(Text, nullable=True)
    extracted_content = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    # 未成功提取时保持空；基本信息同步和降级均不能冒充成功提取。
    last_extracted_at = Column(DateTime, nullable=True)
    shop = relationship("Shop", backref="product_knowledge")

    def __repr__(self):
        return f"<ProductKnowledge(goods_id='{self.goods_id}', goods_name='{self.goods_name}')>"


class CustomerServiceKnowledge(Base):
    __tablename__ = "customer_service_knowledge"

    id = Column(Integer, primary_key=True, autoincrement=True)
    shop_id = Column(Integer, ForeignKey("shops.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(String(255), nullable=True)
    enabled = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    shop = relationship("Shop", backref="customer_service_knowledge")

    def __repr__(self):
        return f"<CustomerServiceKnowledge(title='{self.title}', enabled={self.enabled})>"
