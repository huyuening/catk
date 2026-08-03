import json
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "smart"
    / "datasets"
    / "text_prompts.py"
)
SPEC = importlib.util.spec_from_file_location("catk_text_prompts", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load module spec for {MODULE_PATH}")
TEXT_PROMPTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TEXT_PROMPTS
SPEC.loader.exec_module(TEXT_PROMPTS)
ECoSimTagPromptStore = TEXT_PROMPTS.ECoSimTagPromptStore
scene_bucket = TEXT_PROMPTS.scene_bucket


class ECoSimTagPromptStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mapping_path = self.root / "mapping.json"
        self.mapping_path.write_text(
            json.dumps({"waymo-a": "scene_101"}), encoding="utf-8"
        )
        self.tag_dir = (
            self.root
            / "tag_prompts"
            / "waymo_train_v_action"
            / "tags"
            / "1"
        )
        self.tag_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def write_tags(self, entries):
        (self.tag_dir / "scene_101.json").write_text(
            json.dumps(entries), encoding="utf-8"
        )

    def make_store(self, **overrides):
        kwargs = {
            "root": self.root,
            "mapping_path": self.mapping_path,
            "split": "train",
            "sentence_style": "ordered",
        }
        kwargs.update(overrides)
        return ECoSimTagPromptStore(**kwargs)

    def test_ordered_prompts_follow_catk_agent_order(self):
        self.write_tags(
            [
                "Straight(10 at 11-31)",
                "Accelerate(10 at 21-44)",
                "LeftTurn(10 at 31-70)",
                "Parked(10 at 11-20)",
                "KeepSpeed(20 at 11-15)",
                "UnknownAction(20 at 11-90)",
            ]
        )

        prompts, mask = self.make_store().prompts_for(
            scenario_id="waymo-a",
            agent_ids=[20, 10, 30],
            agent_types=[0, 0, 1],
            ego_id=30,
        )

        self.assertEqual(prompts[0], "")
        self.assertEqual(
            prompts[1],
            "The target vehicle is moving straight. "
            "Then, the target vehicle is accelerating. "
            "Then, the target vehicle is turning left.",
        )
        self.assertEqual(prompts[2], "")
        torch.testing.assert_close(mask, torch.tensor([False, True, False]))

    def test_equal_start_actions_are_joined_in_stable_priority_order(self):
        self.write_tags(
            [
                "KeepSpeed(10 at 11-41)",
                "Accelerate(10 at 11-31)",
                "LeftTurn(10 at 11-51)",
            ]
        )

        prompts, mask = self.make_store().prompts_for(
            scenario_id="waymo-a",
            agent_ids=[10],
            agent_types=[2],
            ego_id=None,
        )

        self.assertEqual(
            prompts,
            [
                "The target cyclist is turning left, accelerating, and "
                "keeping a steady speed."
            ],
        )
        self.assertTrue(mask.item())

    def test_gap_merge_and_minimum_duration_follow_ecosim_rules(self):
        self.write_tags(
            [
                "AccelerateTemporal(10 at 11-18)",
                "Accelerate(10 at 22-35)",
                "RightTurn(10 at 40-49)",
            ]
        )

        prompts, mask = self.make_store().prompts_for(
            scenario_id="waymo-a",
            agent_ids=[10],
            agent_types=[0],
            ego_id=None,
        )

        self.assertEqual(prompts, ["The target vehicle is accelerating."])
        self.assertTrue(mask.item())

    def test_direction_priority_removes_overlapping_straight_but_keeps_acceleration(self):
        self.write_tags(
            [
                "Straight(10 at 11-61)",
                "LeftTurn(10 at 21-51)",
                "Accelerate(10 at 21-51)",
            ]
        )

        prompts, _ = self.make_store().prompts_for(
            scenario_id="waymo-a",
            agent_ids=[10],
            agent_types=[0],
            ego_id=None,
        )

        self.assertEqual(
            prompts,
            [
                "The target vehicle is moving straight. "
                "Then, the target vehicle is turning left and accelerating."
            ],
        )

    def test_ego_alias_is_replaced_before_real_id_alignment(self):
        self.write_tags(["Decelerate(ego at 11-31)"])

        prompts, mask = self.make_store().prompts_for(
            scenario_id="waymo-a",
            agent_ids=[77, 10],
            agent_types=[1, 0],
            ego_id=77,
        )

        self.assertEqual(
            prompts,
            ["The target pedestrian is decelerating.", ""],
        )
        torch.testing.assert_close(mask, torch.tensor([True, False]))

    def test_missing_mapping_or_tag_file_returns_auto_for_every_agent(self):
        store = self.make_store()

        prompts, mask = store.prompts_for(
            scenario_id="unmapped",
            agent_ids=[1, 2],
            agent_types=[0, 2],
            ego_id=1,
        )

        self.assertEqual(prompts, ["", ""])
        self.assertFalse(mask.any())

    def test_agent_metadata_lengths_must_match(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            self.make_store().prompts_for(
                scenario_id="waymo-a",
                agent_ids=[1, 2],
                agent_types=[0],
                ego_id=None,
            )

    def test_scene_bucket_supports_numeric_and_opaque_waymo_ids(self):
        self.assertEqual(scene_bucket("scene_101"), 1)
        first = scene_bucket("5f4dcc3b5aa765d6")
        second = scene_bucket("5f4dcc3b5aa765d6")
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 100)

    def test_invalid_split_and_sentence_style_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "split"):
            self.make_store(split="test")
        with self.assertRaisesRegex(ValueError, "sentence_style"):
            self.make_store(sentence_style="timed")


if __name__ == "__main__":
    unittest.main()
