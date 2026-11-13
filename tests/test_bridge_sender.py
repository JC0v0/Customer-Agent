"""发送器抽象层（bridge.sender）单元测试。

`get_sender` 的文档约定「未注册渠道回退到默认渠道」，但此前对无法识别的
渠道字符串会因 `ChannelType(...)` 抛 `ValueError` 而崩溃。这里回归该修复。
"""

import unittest

from bridge.context import ChannelType
from bridge.sender import get_sender, PinduoduoSender


class GetSenderTests(unittest.TestCase):
    def test_none_falls_back_to_default(self):
        self.assertIsInstance(get_sender(), PinduoduoSender)

    def test_registered_channel_returns_its_sender(self):
        self.assertIsInstance(get_sender(ChannelType.PINDUODUO), PinduoduoSender)

    def test_registered_channel_string_returns_its_sender(self):
        self.assertIsInstance(get_sender("pinduoduo"), PinduoduoSender)

    def test_unregistered_enum_channel_falls_back_to_default(self):
        # JINGDONG 已在枚举中但未注册发送器，应回退而非崩溃。
        self.assertIsInstance(get_sender(ChannelType.JINGDONG), PinduoduoSender)

    def test_unrecognized_channel_string_falls_back_to_default(self):
        # 回归：原先 ChannelType("wechat") 会抛 ValueError 导致崩溃。
        self.assertIsInstance(get_sender("wechat"), PinduoduoSender)

    def test_non_string_channel_falls_back_to_default(self):
        self.assertIsInstance(get_sender(12345), PinduoduoSender)


if __name__ == "__main__":
    unittest.main()
