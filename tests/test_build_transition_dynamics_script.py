import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class BuildTransitionDynamicsScriptTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    SCRIPT = (
        ROOT
        / "scripts"
        / "build_original_vocab_reconstructed_dynamics.sh"
    )

    def test_wrapper_has_valid_bash_and_expands_default_paths(self):
        subprocess.run(["bash", "-n", str(self.SCRIPT)], check=True)
        with TemporaryDirectory() as temp_dir:
            reconstruction_root = Path(temp_dir) / "reconstruction"
            environment = os.environ.copy()
            environment.update(
                {
                    "CATK_ROOT": str(self.ROOT),
                    "RECON_OUTPUT": str(reconstruction_root),
                    "PYTHON_BIN": "/bin/echo",
                    "BATCH_SIZE": "3",
                    "NUM_WORKERS": "4",
                    "SHRINKAGE_COUNT": "5",
                    "MAX_SCENARIOS": "6",
                }
            )
            result = subprocess.run(
                ["bash", str(self.SCRIPT)],
                check=True,
                text=True,
                capture_output=True,
                env=environment,
            )

        arguments = result.stdout
        self.assertIn(
            "-m src.smart.tokens.build_transition_dynamics",
            arguments,
        )
        self.assertIn(
            "--assignment-training-dir "
            f"{reconstruction_root}/datasets/original/training",
            arguments,
        )
        self.assertIn(
            "--dynamics-training-dir "
            f"{reconstruction_root}/datasets/reconstructed/training",
            arguments,
        )
        self.assertIn(
            f"--agent-token-file {self.ROOT}/src/smart/tokens/"
            "agent_vocab_555_s2.pkl",
            arguments,
        )
        self.assertIn(
            "--output "
            f"{reconstruction_root}/"
            "agent_transition_dynamics_original_vocab_reconstructed.pt",
            arguments,
        )
        self.assertIn("--batch-size 3", arguments)
        self.assertIn("--num-workers 4", arguments)
        self.assertIn("--shrinkage-count 5", arguments)
        self.assertIn("--max-scenarios 6", arguments)

    def test_wrapper_requires_reconstruction_output(self):
        environment = os.environ.copy()
        environment.pop("RECON_OUTPUT", None)
        environment.update(
            {
                "CATK_ROOT": str(self.ROOT),
                "PYTHON_BIN": "/bin/echo",
            }
        )

        result = subprocess.run(
            ["bash", str(self.SCRIPT)],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RECON_OUTPUT", result.stderr)

    def test_wrapper_honors_vocabulary_and_output_overrides(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            environment = os.environ.copy()
            environment.update(
                {
                    "CATK_ROOT": str(self.ROOT),
                    "RECON_OUTPUT": str(root / "reconstruction"),
                    "VOCAB_FILE": str(root / "original_vocab.pkl"),
                    "LOOKUP_FILE": str(root / "hybrid_lookup.pt"),
                    "PYTHON_BIN": "/bin/echo",
                }
            )

            result = subprocess.run(
                ["bash", str(self.SCRIPT)],
                check=True,
                text=True,
                capture_output=True,
                env=environment,
            )

        self.assertIn(
            f"--agent-token-file {root}/original_vocab.pkl",
            result.stdout,
        )
        self.assertIn(
            f"--output {root}/hybrid_lookup.pt",
            result.stdout,
        )
        self.assertNotIn("--max-scenarios", result.stdout)


if __name__ == "__main__":
    unittest.main()
