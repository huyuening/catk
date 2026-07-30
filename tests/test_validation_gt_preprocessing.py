import ast
import pickle
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / "src/data_preprocess.py"


def _runtime_dependencies():
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
    except Exception as error:
        pytest.skip(f"TensorFlow/Waymo runtime is unavailable: {error}")

    from src.data_preprocess import batch_process9s_transformer
    from src.smart.metrics.fast_wosac_backend.scenario_gt_converter import (
        extract_gt_scenario,
    )

    return (
        tf,
        scenario_pb2,
        batch_process9s_transformer,
        extract_gt_scenario,
    )


def _scenario(scenario_pb2):
    scenario = scenario_pb2.Scenario(
        scenario_id="embedded-fast-wosac-test",
        current_time_index=10,
        sdc_track_index=0,
    )
    scenario.timestamps_seconds.extend([step * 0.1 for step in range(91)])
    scenario.objects_of_interest.append(101)

    track = scenario.tracks.add(id=101, object_type=1)
    for step in range(91):
        state = track.states.add()
        state.center_x = float(step) * 0.5
        state.center_y = 0.0
        state.center_z = 0.0
        state.length = 4.5
        state.width = 2.0
        state.height = 1.6
        state.heading = 0.0
        state.velocity_x = 5.0
        state.velocity_y = 0.0
        state.valid = True
    scenario.tracks_to_predict.add(track_index=0)

    road_edge = scenario.map_features.add(id=201)
    road_edge.road_edge.type = 1
    for x, y in ((-10.0, -3.0), (50.0, -3.0)):
        point = road_edge.road_edge.polyline.add()
        point.x = x
        point.y = y

    lane = scenario.map_features.add(id=301)
    lane.lane.type = 2
    for x, y in ((-10.0, 0.0), (20.0, 0.0), (50.0, 0.0)):
        point = lane.lane.polyline.add()
        point.x = x
        point.y = y

    for _ in range(91):
        dynamic_state = scenario.dynamic_map_states.add()
        dynamic_state.lane_states.add(lane=301, state=4)

    return scenario


def _write_shard(tf, path, scenario):
    path.parent.mkdir(parents=True)
    with tf.io.TFRecordWriter(str(path)) as writer:
        writer.write(scenario.SerializeToString())


def test_preprocessor_wires_validation_gt_to_workers():
    tree = ast.parse(PREPROCESS.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    worker = functions["wm2argo"]
    batch = functions["batch_process9s_transformer"]
    worker_arguments = {argument.arg for argument in worker.args.args}
    partial_call = next(
        node
        for node in ast.walk(batch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "partial"
    )

    assert "output_dir_gt" in worker_arguments
    assert "output_dir_gt" in {
        keyword.arg for keyword in partial_call.keywords
    }


def test_embedded_converter_produces_wosac_2025_fields():
    _, scenario_pb2, _, extract_gt_scenario = _runtime_dependencies()
    gt = extract_gt_scenario(_scenario(scenario_pb2))

    assert gt["scenario_id"] == "embedded-fast-wosac-test"
    assert gt["tracks"].shape == (1, 91, 9)
    assert gt["track_masks"].shape == (1, 91)
    assert gt["object_ids"].tolist() == [101]
    assert gt["lane_ids"] == [301]
    assert len(gt["traffic_signals"]) == 91
    assert {
        "scenario_id",
        "tracks",
        "track_masks",
        "object_ids",
        "object_types",
        "road_edges",
        "predict_index",
        "sim_agent_ids",
        "lane_ids",
        "lane_polylines",
        "traffic_signals",
    }.issubset(gt)


def test_validation_preprocessing_writes_compatible_gt(tmp_path):
    (
        tf,
        scenario_pb2,
        batch_process9s_transformer,
        _,
    ) = _runtime_dependencies()
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_shard(
        tf,
        input_root / "validation" / "validation.tfrecord-00000-of-00001",
        _scenario(scenario_pb2),
    )

    batch_process9s_transformer(
        input_dir=input_root,
        output_dir=output_root,
        split="validation",
        num_workers=1,
    )

    gt_path = (
        output_root
        / "validation_gt"
        / "embedded-fast-wosac-test.pkl"
    )
    with gt_path.open("rb") as handle:
        gt = pickle.load(handle)

    assert gt["scenario_id"] == "embedded-fast-wosac-test"
    assert gt["tracks"].shape == (1, 91, 9)
    assert (
        output_root
        / "validation_tfrecords_splitted"
        / "embedded-fast-wosac-test.tfrecords"
    ).is_file()
    assert (
        output_root
        / "validation"
        / "embedded-fast-wosac-test.pkl"
    ).is_file()


@pytest.mark.parametrize("split", ("training", "testing"))
def test_non_validation_preprocessing_does_not_write_validation_gt(
    tmp_path,
    split,
):
    (
        tf,
        scenario_pb2,
        batch_process9s_transformer,
        _,
    ) = _runtime_dependencies()
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_shard(
        tf,
        input_root / split / f"{split}.tfrecord-00000-of-00001",
        _scenario(scenario_pb2),
    )

    batch_process9s_transformer(
        input_dir=input_root,
        output_dir=output_root,
        split=split,
        num_workers=1,
    )

    assert not (output_root / "validation_gt").exists()
