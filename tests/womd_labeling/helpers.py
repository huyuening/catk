from __future__ import annotations

from pathlib import Path
import struct
from typing import Iterable

from src.womd_labeling.proto import scenario_pb2


def make_scenario(scenario_id: str, *, frame_count: int = 1):
    scenario = scenario_pb2.Scenario()
    scenario.scenario_id = scenario_id
    scenario.current_time_index = min(10, frame_count - 1)
    scenario.sdc_track_index = 0
    scenario.timestamps_seconds.extend(0.1 * index for index in range(frame_count))

    track = scenario.tracks.add()
    track.id = 1
    track.object_type = 1
    for index in range(frame_count):
        state = track.states.add()
        state.center_x = float(index)
        state.center_y = 0.0
        state.length = 4.8
        state.width = 2.0
        state.height = 1.6
        state.velocity_x = 2.0
        state.valid = True

    lane = scenario.map_features.add()
    lane.id = 100
    lane.lane.type = 2
    lane.lane.speed_limit_mph = 30.0
    for x in (-20.0, 0.0, 20.0):
        point = lane.lane.polyline.add()
        point.x = x
        point.y = 0.0
    return scenario


def write_tfrecord(path: Path, payloads: Iterable[bytes]) -> None:
    with path.open("wb") as stream:
        for payload in payloads:
            stream.write(struct.pack("<Q", len(payload)))
            stream.write(b"\0\0\0\0")
            stream.write(payload)
            stream.write(b"\0\0\0\0")
