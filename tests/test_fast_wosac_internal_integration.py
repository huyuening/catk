import inspect
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
COMMON_METRICS = (
    "metametric",
    "average_displacement_error",
    "min_average_displacement_error",
    "linear_speed_likelihood",
    "linear_acceleration_likelihood",
    "angular_speed_likelihood",
    "angular_acceleration_likelihood",
    "distance_to_nearest_object_likelihood",
    "collision_indication_likelihood",
    "time_to_collision_likelihood",
    "distance_to_road_edge_likelihood",
    "offroad_indication_likelihood",
    "simulated_collision_rate",
    "simulated_offroad_rate",
)


def _fast_metric_class():
    pytest.importorskip("waymo_open_dataset")
    from src.smart.metrics.fast_wosac_metrics import FastWOSACMetrics

    return FastWOSACMetrics


def test_fast_metric_constructs_without_external_trajtok(tmp_path):
    fast_metric = _fast_metric_class()
    before = list(sys.path)
    with patch.dict(
        os.environ,
        {"TRAJTOK_ROOT": str(tmp_path / "missing-trajtok")},
    ):
        metric = fast_metric(
            prefix="val_closed",
            version="2025",
            gt_scenario_dir=None,
            require_preprocessed_gt=False,
        )

    assert metric.version == "2025"
    assert list(sys.path) == before
    assert "trajtok_root" not in inspect.signature(fast_metric).parameters


def test_fast_metric_preserves_2024_and_2025_metric_sets():
    fast_metric = _fast_metric_class()
    metric_2024 = fast_metric(prefix="val", version="2024")
    metric_2025 = fast_metric(prefix="val", version="2025")

    assert tuple(metric_2024.metric_names) == COMMON_METRICS
    assert tuple(metric_2025.metric_names) == COMMON_METRICS + (
        "traffic_light_violation_likelihood",
        "simulated_traffic_light_violation_rate",
    )


def test_malformed_2025_gt_points_to_catk_preprocessing(tmp_path):
    torch = pytest.importorskip("torch")
    fast_metric = _fast_metric_class()
    gt_dir = tmp_path / "validation_gt"
    gt_dir.mkdir()
    with (gt_dir / "scenario-1.pkl").open("wb") as handle:
        pickle.dump({"scenario_id": "scenario-1"}, handle)
    metric = fast_metric(
        prefix="val",
        version="2025",
        gt_scenario_dir=str(gt_dir),
        require_preprocessed_gt=True,
    )

    with pytest.raises(
        KeyError,
        match=r"CatK's current src\.data_preprocess",
    ):
        metric.update(
            scenario_files=["unused.tfrecord"],
            scenario_ids=["scenario-1"],
            agent_id=torch.tensor([101]),
            agent_batch=torch.tensor([0]),
            simulated_states=torch.zeros(1, 1, 80, 4),
        )


def test_model_config_keeps_only_ignored_compatibility_value():
    config = yaml.safe_load(
        (ROOT / "configs/model/smart.yaml").read_text(encoding="utf-8")
    )

    assert config["model_config"]["trajtok_root"] is None


def test_inference_config_does_not_override_compatibility_value():
    config = yaml.safe_load(
        (ROOT / "configs/experiment/inference.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert "trajtok_root" not in config["model"]["model_config"]


def test_inference_config_does_not_resolve_trajtok_environment(tmp_path):
    hydra = pytest.importorskip("hydra")
    with patch.dict(
        os.environ,
        {"TRAJTOK_ROOT": str(tmp_path / "missing-trajtok")},
    ):
        with hydra.initialize_config_dir(
            config_dir=str(ROOT / "configs"),
            version_base=None,
        ):
            config = hydra.compose(
                config_name="run.yaml",
                overrides=["experiment=inference"],
            )

    assert config.model.model_config.trajtok_root is None
