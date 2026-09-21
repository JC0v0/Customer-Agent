"""打包配置的回归测试。

背景：litellm 在 import 阶段就会读自身的数据文件
（get_model_cost_map 无条件读 model_prices_and_context_window_backup.json，
即使远端取数成功也要读本地那份做完整性校验），而 PyInstaller 只加
hiddenimports 不会带上数据文件。缺失时冻结版在 import litellm 抛
FileNotFoundError，导致全部 LLM 调用失效——该缺陷在 v1.4.0 / v1.5 的
发布包里存在过，直到日志才暴露。

这里做静态与轻量校验，防止 spec 里的数据收集被去掉。
"""

import ast
import sys
import unittest
from importlib.resources import files
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPEC_PATH = REPO_ROOT / "scripts" / "agent_customer.spec"


class PackagingSpecTest(unittest.TestCase):
    def setUp(self):
        self.source = SPEC_PATH.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def calls_in(self, func_name):
        """收集对指定函数的全部调用节点。"""
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name == func_name:
                    found.append(node)
        return found

    def test_litellm_data_files_are_collected(self):
        """必须收集 litellm 的数据文件，否则冻结版 import 就会失败。"""
        calls = [
            c for c in self.calls_in("collect_data_files")
            if c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "litellm"
        ]
        self.assertTrue(
            calls,
            "spec 里缺少 collect_data_files('litellm')；"
            "冻结后会在 import litellm 时因缺 model_prices_and_context_window_backup.json 而失败",
        )

    def test_litellm_data_collection_excludes_heavy_and_temp_files(self):
        """proxy/（约 24MB Web 控制台）与 .tmp 缓存不应进包。"""
        calls = [
            c for c in self.calls_in("collect_data_files")
            if c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "litellm"
        ]
        self.assertTrue(calls)
        excludes = []
        for keyword in calls[0].keywords:
            if keyword.arg == "excludes":
                excludes = [
                    elt.value for elt in getattr(keyword.value, "elts", [])
                    if isinstance(elt, ast.Constant)
                ]
        joined = " ".join(excludes)
        self.assertIn("proxy", joined)
        self.assertIn(".tmp", joined)

    def test_collected_data_is_passed_to_analysis(self):
        """收集结果必须真正并进 Analysis 的 datas，否则等于没收集。"""
        analysis = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "Analysis"
        ]
        self.assertTrue(analysis, "spec 里找不到 Analysis 调用")
        datas = None
        for keyword in analysis[0].keywords:
            if keyword.arg == "datas":
                datas = keyword.value
        self.assertIsNotNone(datas, "Analysis 缺少 datas 参数")
        names = {
            getattr(node, "id", None)
            for node in ast.walk(datas)
            if isinstance(node, ast.Name)
        }
        self.assertIn(
            "_llm_datas", names,
            "Analysis 的 datas 未包含 _llm_datas，收集到的 litellm 数据文件不会进包",
        )

    def test_tokenizers_subpackage_is_collected(self):
        """litellm.utils 会用 importlib.resources 访问该子包下的词表文件。"""
        calls = [
            c for c in self.calls_in("collect_submodules")
            if c.args and isinstance(c.args[0], ast.Constant)
            and c.args[0].value == "litellm.litellm_core_utils"
        ]
        self.assertTrue(
            calls,
            "缺少 collect_submodules('litellm.litellm_core_utils')；"
            "冻结后会报 No module named 'litellm.litellm_core_utils.tokenizers'",
        )

    def test_litellm_ships_the_backup_file_the_spec_collects(self):
        """收集目标文件必须真实存在于已安装的 litellm 里。

        路径与 litellm 源码一致：
        litellm/litellm_core_utils/get_model_cost_map.py 用
        files("litellm").joinpath("model_prices_and_context_window_backup.json")。
        """
        target = files("litellm").joinpath("model_prices_and_context_window_backup.json")
        self.assertTrue(
            target.is_file(),
            "已安装的 litellm 里找不到 model_prices_and_context_window_backup.json，"
            "collect_data_files 将收集不到它",
        )


if __name__ == "__main__":
    unittest.main()
