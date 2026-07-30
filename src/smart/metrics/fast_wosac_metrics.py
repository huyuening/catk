# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sys
import warnings
from pathlib import Path
from typing import Iterable, List

import torch
from google.protobuf import text_format
from torch import Tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import scenario_pb2, sim_agents_metrics_pb2

from src.smart.metrics.preprocessed_scenario_gt import PreprocessedScenarioGT


_COMMON_METRIC_NAMES = (
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

_2025_METRIC_NAMES = (
    "traffic_light_violation_likelihood",
    "simulated_traffic_light_violation_rate",
)

_2025_REQUIRED_GT_KEYS = {
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
}


def _load_trajtok_modules(trajtok_root: str):
    root = Path(trajtok_root).expanduser().resolve()
    package_dir = root / "wosac_fast_eval_tool"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            "TrajTok fast WOSAC package not found at "
            f"{package_dir}. Set model.model_config.trajtok_root or the "
            "TRAJTOK_ROOT environment variable."
        )

    root_str = str(root)
    if root_str not in sys.path:
        # Append instead of prepend so CatK's own top-level `src` package keeps
        # priority over TrajTok's package with the same name.
        sys.path.append(root_str)

    from wosac_fast_eval_tool.fast_sim_agents_metrics import (  # noqa: PLC0415
        metrics as fast_metrics,
    )
    from wosac_fast_eval_tool.scenario_gt_converter import (  # noqa: PLC0415
        extract_gt_scenario,
        gt_scenario_to_device,
    )

    return fast_metrics, extract_gt_scenario, gt_scenario_to_device


class FastWOSACMetrics(Metric):
    """GPU-accelerated WOSAC metrics backed by TrajTok's evaluator."""

    full_state_update = False

    def __init__(
        self,
        prefix: str,
        trajtok_root: str,
        version: str = "2025",
        gt_scenario_dir: str | None = None,
        require_preprocessed_gt: bool = False,
    ) -> None:
        super().__init__()
        if version not in {"2024", "2025"}:
            raise ValueError(f"Unsupported WOSAC version: {version}")

        self.prefix = prefix
        self.trajtok_root = trajtok_root
        self.version = version
        self.preprocessed_gt = PreprocessedScenarioGT(
            gt_scenario_dir,
            required=require_preprocessed_gt,
        )
        self.gt_scenario_dir = self.preprocessed_gt.directory
        self.metric_names = list(_COMMON_METRIC_NAMES)
        if version == "2025":
            self.metric_names.extend(_2025_METRIC_NAMES)

        fast_metrics, _, _ = _load_trajtok_modules(trajtok_root)
        config_name = (
            "challenge_2025_sim_agents_config.textproto"
            if version == "2025"
            else "challenge_2024_config.textproto"
        )
        config_path = Path(fast_metrics.__file__).resolve().parent / config_name
        if not config_path.is_file():
            raise FileNotFoundError(f"Fast WOSAC config not found: {config_path}")

        self.wosac_config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        with config_path.open("r") as config_file:
            text_format.Parse(config_file.read(), self.wosac_config)

        # TensorFlow is used only to read CatK's per-scenario TFRecords. Keep it
        # away from CUDA; the actual fast metric runs on the PyTorch device.
        import tensorflow as tf  # noqa: PLC0415

        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as error:
            warnings.warn(
                "TensorFlow was initialized before Fast WOSAC and could not be "
                f"restricted to CPU: {error}",
                stacklevel=2,
            )

        for metric_name in self.metric_names:
            self.add_state(
                f"{metric_name}_sum",
                default=torch.tensor(0.0, dtype=torch.float64),
                dist_reduce_fx="sum",
            )
        self.add_state(
            "scenario_counter",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )

    def _load_scenario(
        self,
        scenario_file: str,
        scenario_id: str,
        extract_gt_scenario,
    ) -> dict:
        gt_scenario = self.preprocessed_gt.load(scenario_id)
        if gt_scenario is not None:
            return gt_scenario

        import tensorflow as tf  # noqa: PLC0415

        record = next(
            iter(
                tf.data.TFRecordDataset(
                    [str(scenario_file)],
                    compression_type="",
                )
            ),
            None,
        )
        if record is None:
            raise ValueError(f"No scenario found in TFRecord: {scenario_file}")

        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(record.numpy()))
        return extract_gt_scenario(scenario)

    @staticmethod
    def _unbatch_agents(
        src: Tensor,
        agent_batch: Tensor,
        num_scenarios: int,
        dim: int,
    ) -> List[Tensor]:
        sizes = torch.bincount(
            agent_batch,
            minlength=num_scenarios,
        ).tolist()
        if len(sizes) != num_scenarios or any(size == 0 for size in sizes):
            raise ValueError(
                "Fast WOSAC expects at least one agent for every scenario in a batch."
            )
        return list(src.split(sizes, dim=dim))

    @torch.no_grad()
    def update(
        self,
        scenario_files: Iterable[str],
        scenario_ids: Iterable[str],
        agent_id: Tensor,
        agent_batch: Tensor,
        simulated_states: Tensor,
    ) -> None:
        """
        Args:
            scenario_files: Per-scenario CatK TFRecord paths.
            scenario_ids: Scenario identifiers in batch order.
            agent_id: Agent IDs with shape ``[n_agent]``.
            agent_batch: Scenario index for every agent, shape ``[n_agent]``.
            simulated_states: Global ``x, y, z, yaw`` states with shape
                ``[n_rollout, n_agent, n_step, 4]``.
        """
        if simulated_states.ndim != 4 or simulated_states.shape[-1] != 4:
            raise ValueError(
                "simulated_states must have shape "
                "[n_rollout, n_agent, n_step, 4], got "
                f"{tuple(simulated_states.shape)}"
            )

        files = list(scenario_files)
        ids = list(scenario_ids)
        if len(files) != len(ids):
            raise ValueError(
                f"Got {len(files)} TFRecords for {len(ids)} scenario IDs."
            )

        fast_metrics, extract_gt_scenario, gt_scenario_to_device = (
            _load_trajtok_modules(self.trajtok_root)
        )
        num_scenarios = len(ids)
        agent_ids = self._unbatch_agents(
            agent_id,
            agent_batch,
            num_scenarios,
            dim=0,
        )
        states = self._unbatch_agents(
            simulated_states,
            agent_batch,
            num_scenarios,
            dim=1,
        )
        device = simulated_states.device

        for scenario_file, expected_id, scenario_agent_ids, scenario_states in zip(
            files,
            ids,
            agent_ids,
            states,
        ):
            gt_scenario = self._load_scenario(
                scenario_file,
                expected_id,
                extract_gt_scenario,
            )
            if self.version == "2025":
                missing_keys = _2025_REQUIRED_GT_KEYS.difference(gt_scenario)
                if missing_keys:
                    raise KeyError(
                        "TrajTok validation GT is missing WOSAC 2025 fields "
                        f"{sorted(missing_keys)} for scenario {expected_id}. "
                        "Regenerate it with TrajTok's current "
                        "scenario_gt_converter."
                    )
            if gt_scenario["scenario_id"] != expected_id:
                raise ValueError(
                    "Scenario ID mismatch: "
                    f"batch has {expected_id}, GT has "
                    f"{gt_scenario['scenario_id']} ({scenario_file})."
                )
            gt_scenario = gt_scenario_to_device(gt_scenario, device=device)
            prediction = {
                "agent_id": scenario_agent_ids.to(device=device),
                "simulated_states": scenario_states.to(device=device),
            }
            scenario_metrics = fast_metrics.compute_scenario_metrics_for_bundle(
                self.wosac_config,
                gt_scenario,
                prediction,
                self.version,
            )

            for metric_name in self.metric_names:
                value = torch.as_tensor(
                    scenario_metrics[metric_name],
                    dtype=torch.float64,
                    device=device,
                )
                getattr(self, f"{metric_name}_sum").add_(value)
            self.scenario_counter.add_(1)

    def compute(self) -> dict[str, Tensor]:
        if self.scenario_counter.item() == 0:
            raise RuntimeError(
                "Fast WOSAC received no scenarios. Check "
                "model.model_config.n_batch_wosac_metric and "
                "trainer.limit_val_batches."
            )

        mean_metrics = {
            metric_name: getattr(self, f"{metric_name}_sum")
            / self.scenario_counter
            for metric_name in self.metric_names
        }
        mean_proto = sim_agents_metrics_pb2.SimAgentMetrics(
            scenario_id="",
            **{
                metric_name: value.detach().cpu().item()
                for metric_name, value in mean_metrics.items()
            },
        )
        fast_metrics, _, _ = _load_trajtok_modules(self.trajtok_root)
        bucketed = fast_metrics.aggregate_metrics_to_buckets(
            self.wosac_config,
            mean_proto,
        )

        def as_metric_tensor(value: float) -> Tensor:
            return torch.as_tensor(
                value,
                dtype=torch.float64,
                device=self.scenario_counter.device,
            )

        output = {
            f"{self.prefix}/wosac/realism_meta_metric": as_metric_tensor(
                bucketed.realism_meta_metric
            ),
            f"{self.prefix}/wosac/kinematic_metrics": as_metric_tensor(
                bucketed.kinematic_metrics
            ),
            f"{self.prefix}/wosac/interactive_metrics": as_metric_tensor(
                bucketed.interactive_metrics
            ),
            f"{self.prefix}/wosac/map_based_metrics": as_metric_tensor(
                bucketed.map_based_metrics
            ),
            f"{self.prefix}/wosac/min_ade": as_metric_tensor(bucketed.min_ade),
            f"{self.prefix}/wosac/simulated_collision_rate": as_metric_tensor(
                bucketed.simulated_collision_rate
            ),
            f"{self.prefix}/wosac/simulated_offroad_rate": as_metric_tensor(
                bucketed.simulated_offroad_rate
            ),
            f"{self.prefix}/wosac/scenario_counter": self.scenario_counter,
        }
        if self.version == "2025":
            output[
                f"{self.prefix}/wosac/simulated_traffic_light_violation_rate"
            ] = as_metric_tensor(bucketed.simulated_traffic_light_violation_rate)

        for metric_name, value in mean_metrics.items():
            output[f"{self.prefix}/wosac_likelihood/{metric_name}"] = value
        return output
