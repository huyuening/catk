import unittest

from src.smart.inference.audit_text_control import validate_hard_ce_contract


def config(
    *,
    active=True,
    mode="cached_reconstructed",
    spatial=False,
    smoothing=0.0,
):
    return {
        "history_dynamics": {
            "is_active": active,
            "mode": mode,
        },
        "training_loss": {
            "spatial_aware_smoothing": spatial,
            "label_smoothing": smoothing,
        },
    }


class HardCECheckpointContractTest(unittest.TestCase):
    def test_accepts_cached_history_hard_ce(self):
        history, loss = validate_hard_ce_contract(config())

        self.assertEqual(history["mode"], "cached_reconstructed")
        self.assertEqual(loss["label_smoothing"], 0.0)

    def test_accepts_legacy_history_checkpoint_without_mode_as_cached(self):
        legacy = config()
        del legacy["history_dynamics"]["mode"]

        history, loss = validate_hard_ce_contract(legacy)

        self.assertEqual(history["mode"], "cached_reconstructed")
        self.assertEqual(loss["label_smoothing"], 0.0)

    def test_rejects_disabled_or_online_history(self):
        with self.assertRaisesRegex(RuntimeError, "history dynamics disabled"):
            validate_hard_ce_contract(config(active=False))
        with self.assertRaisesRegex(RuntimeError, "cached_reconstructed"):
            validate_hard_ce_contract(config(mode="online_raw"))

    def test_rejects_spatial_smoothing(self):
        with self.assertRaisesRegex(RuntimeError, "spatial smoothing"):
            validate_hard_ce_contract(config(spatial=True))

    def test_rejects_nonzero_label_smoothing(self):
        with self.assertRaisesRegex(RuntimeError, "label_smoothing=0.0"):
            validate_hard_ce_contract(config(smoothing=0.1))

    def test_rejects_missing_hard_ce_metadata(self):
        missing_spatial = config()
        del missing_spatial["training_loss"]["spatial_aware_smoothing"]
        with self.assertRaisesRegex(RuntimeError, "spatial_aware_smoothing"):
            validate_hard_ce_contract(missing_spatial)

        missing_label = config()
        del missing_label["training_loss"]["label_smoothing"]
        with self.assertRaisesRegex(RuntimeError, "label_smoothing=0.0"):
            validate_hard_ce_contract(missing_label)


if __name__ == "__main__":
    unittest.main()
