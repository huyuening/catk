from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_shell_wrapper_forwards_full_dataset_options(tmp_path):
    fake_python = tmp_path / "fake-python"
    capture = tmp_path / "args.txt"
    fake_python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    repository_root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "WOMD_ROOT": str(tmp_path / "raw"),
        "LABEL_OUTPUT_ROOT": str(tmp_path / "labels"),
        "NUM_WORKERS": "24",
        "SPLITS": "training validation",
        "STAGES": "annotations statistics",
        "VISUALIZE_MAX_SCENARIOS": "0",
        "RESUME": "false",
        "PYTHON_BIN": str(fake_python),
        "CAPTURE_PATH": str(capture),
    }

    subprocess.run(
        ["bash", "scripts/label_womd_dataset.sh"],
        cwd=repository_root,
        env=environment,
        check=True,
    )

    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == [
        "-m",
        "src.womd_labeling.run_dataset",
        "--input-root",
    ]
    assert "--workers" in arguments
    assert arguments[arguments.index("--workers") + 1] == "24"
    assert "--visualize-max-scenarios" in arguments
    assert arguments[
        arguments.index("--visualize-max-scenarios") + 1
    ] == "0"
    assert "--no-resume" in arguments
