import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.smart.modules.endpoint_interpolation import EndpointInterpolator
from tools.compare_endpoint_interpolation import (
    build_cfg,
    endpoint_delta_summary,
    motion_split_summary,
)


class EndpointComparisonTest(unittest.TestCase):
    def test_build_cfg_initializes_standalone_hydra_runtime(self):
        try:
            from hydra.core.hydra_config import HydraConfig
            from omegaconf import OmegaConf
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed in this test environment")

        with tempfile.TemporaryDirectory() as output_directory:
            args = SimpleNamespace(
                config_overrides=[],
                output_dir=output_directory,
                num_rollouts=1,
                trajtok_root="/root/workspace/TrajTok",
                sampling_num_k=1,
                sampling_temp=1.0,
            )
            cfg = build_cfg(args)
            resolved_paths = OmegaConf.to_container(cfg.paths, resolve=True)

            self.assertTrue(HydraConfig.initialized())
            self.assertEqual(
                Path(HydraConfig.get().runtime.output_dir),
                Path(output_directory).resolve(),
            )
            self.assertEqual(
                Path(resolved_paths["output_dir"]),
                Path(output_directory).resolve(),
            )

    def test_script_adds_repository_root_when_launched_elsewhere(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "tools" / "compare_endpoint_interpolation.py"
        code = (
            f"import runpy; runpy.run_path({str(script_path)!r}, "
            "run_name='catk_comparison_tool'); import src; print(src.__file__)"
        )

        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=working_directory,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn(str(repo_root / "src"), result.stdout)

    def test_endpoint_delta_uses_every_fifth_future_step(self):
        raw = np.zeros((1, 1, 80, 4), dtype=np.float32)
        post = raw.copy()
        post[..., 4::5, 0] = 2.0
        post[..., 4::5, 3] = 0.25

        summary = endpoint_delta_summary(raw, post)

        self.assertEqual(summary["indices"], list(range(4, 80, 5)))
        self.assertAlmostEqual(summary["mean_m"], 2.0)
        self.assertAlmostEqual(summary["max_m"], 2.0)
        self.assertAlmostEqual(summary["heading_mean_rad"], 0.25, places=6)

    def test_motion_modes_follow_endpoint_interpolator_policy(self):
        config = {
            "is_active": True,
            "low_speed_reconstruction": True,
            "low_speed_min_segment_threshold_mps": 0.1,
            "static_reconstruction": True,
            "static_max_segment_speed_threshold_mps": 0.5,
            "static_endpoint_span_threshold_m": 1.0,
            "static_pedestrian_max_segment_speed_threshold_mps": 0.2,
            "static_pedestrian_endpoint_span_threshold_m": 0.5,
            "static_cyclist_max_segment_speed_threshold_mps": 0.3,
            "static_cyclist_endpoint_span_threshold_m": 0.7,
        }
        interpolator = EndpointInterpolator(config=config, shift=5)
        tokenized_agent = {
            "gt_pos": torch.zeros(2, 2, 2),
            "type": torch.tensor([0, 0]),
        }
        decoder = SimpleNamespace(
            num_historical_steps=11,
            shift=5,
            endpoint_interpolator=interpolator,
        )
        model = SimpleNamespace(
            encoder=SimpleNamespace(agent_encoder=decoder),
            token_processor=lambda data: (None, tokenized_agent),
        )

        raw = np.zeros((1, 2, 80, 4), dtype=np.float32)
        moving_endpoints = np.arange(1, 17, dtype=np.float32)
        raw[0, 0, 4::5, 0] = moving_endpoints
        for segment in range(16):
            start = 0.0 if segment == 0 else moving_endpoints[segment - 1]
            end = moving_endpoints[segment]
            raw[0, 0, segment * 5 : (segment + 1) * 5, 0] = np.linspace(
                start, end, 6, dtype=np.float32
            )[1:]
        raw[0, 1, :, 0] = np.linspace(0.0, 0.8, 80, dtype=np.float32)

        summary = motion_split_summary(model, None, raw)

        self.assertEqual(summary["mode"][0, 0], "endpoint_interpolation")
        self.assertEqual(summary["mode"][0, 1], "static_reconstruction")
        self.assertEqual(summary["counts"]["endpoint_interpolation"], 1)
        self.assertEqual(summary["counts"]["static_reconstruction"], 1)


if __name__ == "__main__":
    unittest.main()
