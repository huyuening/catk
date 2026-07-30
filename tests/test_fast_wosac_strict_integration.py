import ast
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class FastWOSACStrictIntegrationTest(unittest.TestCase):
    def test_fast_metric_constructor_exposes_required_gt_flag(self):
        tree = ast.parse(
            (
                ROOT / "src/smart/metrics/fast_wosac_metrics.py"
            ).read_text()
        )
        metric_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "FastWOSACMetrics"
        )
        constructor = next(
            node
            for node in metric_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        arguments = [argument.arg for argument in constructor.args.args]
        self.assertIn("require_preprocessed_gt", arguments)

    def test_smart_forwards_model_required_gt_flag(self):
        tree = ast.parse((ROOT / "src/smart/model/smart.py").read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "FastWOSACMetrics"
        ]
        self.assertEqual(len(calls), 1)
        keyword = next(
            item
            for item in calls[0].keywords
            if item.arg == "require_preprocessed_gt"
        )
        self.assertIn(
            "fast_wosac_require_preprocessed_gt",
            ast.unparse(keyword.value),
        )

    def test_smart_does_not_forward_external_trajtok_root(self):
        tree = ast.parse((ROOT / "src/smart/model/smart.py").read_text())
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "FastWOSACMetrics"
        )
        self.assertNotIn(
            "trajtok_root",
            {keyword.arg for keyword in call.keywords},
        )

    def test_model_default_preserves_optional_gt_behavior(self):
        config = yaml.safe_load(
            (ROOT / "configs/model/smart.yaml").read_text()
        )
        self.assertFalse(
            config["model_config"][
                "fast_wosac_require_preprocessed_gt"
            ]
        )


if __name__ == "__main__":
    unittest.main()
