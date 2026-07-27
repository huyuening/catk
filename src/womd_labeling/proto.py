"""Load CatK's bundled WOMD protobuf bindings without duplicating them."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_PB2_ROOT = (
    Path(__file__).resolve().parents[1]
    / "smart"
    / "tokens"
    / "womd_proto"
    / "pb2"
)
if str(_PB2_ROOT) not in sys.path:
    sys.path.insert(0, str(_PB2_ROOT))

map_pb2 = importlib.import_module("map_pb2")
scenario_pb2 = importlib.import_module("scenario_pb2")

__all__ = ["map_pb2", "scenario_pb2"]
