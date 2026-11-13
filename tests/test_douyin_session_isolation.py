"""抖音多账号隔离：会话文件与优先传输实例不能共用。"""
import unittest

from Channel.douyin.session_paths import pigeon_session_path
from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport


class DouyinSessionIsolationTests(unittest.TestCase):
    def tearDown(self):
        PigeonTransport._instances.clear()

    def test_session_path_is_account_scoped(self):
        path_a = pigeon_session_path("douyin:shop-a:user-a")
        path_b = pigeon_session_path("douyin:shop-b:user-b")
        self.assertNotEqual(path_a, path_b)
        self.assertEqual(path_a, pigeon_session_path("douyin:shop-a:user-a"))

    def test_transport_instances_are_keyed_by_account(self):
        transport_a = object()
        transport_b = object()
        PigeonTransport.set_instance(transport_a, "douyin|shop-a|user-a")
        PigeonTransport.set_instance(transport_b, "douyin|shop-b|user-b")

        self.assertIs(
            PigeonTransport.instance("douyin|shop-a|user-a"), transport_a
        )
        self.assertIs(
            PigeonTransport.instance("douyin|shop-b|user-b"), transport_b
        )
        self.assertIsNone(PigeonTransport.instance("douyin|shop-c|user-c"))

    def test_clearing_one_account_does_not_remove_another(self):
        transport_a = object()
        transport_b = object()
        PigeonTransport.set_instance(transport_a, "a")
        PigeonTransport.set_instance(transport_b, "b")
        PigeonTransport.set_instance(None, "a")
        self.assertIsNone(PigeonTransport.instance("a"))
        self.assertIs(PigeonTransport.instance("b"), transport_b)


if __name__ == "__main__":
    unittest.main()
