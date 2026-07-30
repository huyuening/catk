import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "src/smart/metrics/fast_wosac_backend"


def test_embedded_backend_files_are_packaged():
    expected = {
        "scenario_gt_converter.py",
        "fast_sim_agents_metrics/metrics.py",
        "fast_sim_agents_metrics/challenge_2024_config.textproto",
        (
            "fast_sim_agents_metrics/"
            "challenge_2025_sim_agents_config.textproto"
        ),
    }

    assert expected == {
        str(path.relative_to(BACKEND))
        for path in BACKEND.rglob("*")
        if path.is_file() and str(path.relative_to(BACKEND)) in expected
    }


def test_embedded_backend_exports_metric_entry_points():
    pytest.importorskip("waymo_open_dataset")
    metrics = importlib.import_module(
        "src.smart.metrics.fast_wosac_backend."
        "fast_sim_agents_metrics.metrics"
    )
    converter = importlib.import_module(
        "src.smart.metrics.fast_wosac_backend.scenario_gt_converter"
    )

    assert callable(metrics.compute_scenario_metrics_for_bundle)
    assert callable(metrics.aggregate_metrics_to_buckets)
    assert callable(converter.extract_gt_scenario)
    assert callable(converter.gt_scenario_to_device)


@pytest.mark.parametrize(
    "filename",
    (
        "challenge_2024_config.textproto",
        "challenge_2025_sim_agents_config.textproto",
    ),
)
def test_embedded_wosac_configs_parse(filename):
    text_format = pytest.importorskip("google.protobuf.text_format")
    sim_agents_metrics_pb2 = pytest.importorskip(
        "waymo_open_dataset.protos.sim_agents_metrics_pb2"
    )
    path = BACKEND / "fast_sim_agents_metrics" / filename
    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    text_format.Parse(path.read_text(encoding="utf-8"), config)

    assert config.ByteSize() > 0
