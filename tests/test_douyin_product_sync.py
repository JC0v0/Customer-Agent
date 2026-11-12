"""Ensure product synchronization keeps the selected platform isolated."""
from unittest import mock

from database.product_sync import _get_product_manager


def test_douyin_product_sync_uses_douyin_manager():
    with mock.patch(
        "Channel.douyin.utils.API.product_manager.ProductManager"
    ) as douyin_manager, mock.patch(
        "Channel.pinduoduo.utils.API.product_manager.ProductManager"
    ) as pdd_manager:
        _get_product_manager("douyin", "douyin_shop", "account_1")

    douyin_manager.assert_called_once_with(shop_id="douyin_shop", user_id="account_1")
    pdd_manager.assert_not_called()


def test_default_product_sync_keeps_pinduoduo_manager():
    with mock.patch(
        "Channel.douyin.utils.API.product_manager.ProductManager"
    ) as douyin_manager, mock.patch(
        "Channel.pinduoduo.utils.API.product_manager.ProductManager"
    ) as pdd_manager:
        _get_product_manager("pinduoduo", "123", "account_1")

    pdd_manager.assert_called_once_with(shop_id="123", user_id="account_1")
    douyin_manager.assert_not_called()
