import importlib.machinery
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
import types
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


class _DatasetStub:
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter


class _LoggerStub:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _load_dataset_modules():
    torch_geometric = types.ModuleType("torch_geometric")
    torch_geometric.__path__ = []
    torch_geometric.__spec__ = importlib.machinery.ModuleSpec(
        "torch_geometric", loader=None, is_package=True
    )
    geometric_data = types.ModuleType("torch_geometric.data")
    geometric_data.Dataset = _DatasetStub
    torch_geometric.data = geometric_data
    sys.modules.setdefault("torch_geometric", torch_geometric)
    sys.modules.setdefault("torch_geometric.data", geometric_data)

    utils = types.ModuleType("src.utils")
    utils.RankedLogger = lambda *args, **kwargs: _LoggerStub()
    sys.modules["src.utils"] = utils

    datasets = types.ModuleType("src.smart.datasets")
    datasets.__path__ = [str(ROOT / "src" / "smart" / "datasets")]
    datasets.__spec__ = importlib.machinery.ModuleSpec(
        "src.smart.datasets", loader=None, is_package=True
    )
    sys.modules["src.smart.datasets"] = datasets

    text_path = ROOT / "src" / "smart" / "datasets" / "text_prompts.py"
    text_spec = importlib.util.spec_from_file_location(
        "src.smart.datasets.text_prompts", text_path
    )
    text_module = importlib.util.module_from_spec(text_spec)
    sys.modules[text_spec.name] = text_module
    text_spec.loader.exec_module(text_module)

    dataset_path = ROOT / "src" / "smart" / "datasets" / "scalable_dataset.py"
    dataset_spec = importlib.util.spec_from_file_location(
        "src.smart.datasets.scalable_dataset", dataset_path
    )
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    sys.modules[dataset_spec.name] = dataset_module
    dataset_spec.loader.exec_module(dataset_module)
    datasets.MultiDataset = dataset_module.MultiDataset
    datasets.ECoSimTagPromptStore = text_module.ECoSimTagPromptStore
    return dataset_module, datasets


SCALABLE_DATASET, DATASETS_PACKAGE = _load_dataset_modules()


def _load_datamodule_module():
    lightning = types.ModuleType("lightning")
    lightning.LightningDataModule = type("LightningDataModule", (), {})
    sys.modules["lightning"] = lightning

    utilities_types = types.ModuleType("lightning.pytorch.utilities.types")
    utilities_types.EVAL_DATALOADERS = object
    utilities_types.TRAIN_DATALOADERS = object
    sys.modules["lightning.pytorch"] = types.ModuleType("lightning.pytorch")
    sys.modules["lightning.pytorch.utilities"] = types.ModuleType(
        "lightning.pytorch.utilities"
    )
    sys.modules["lightning.pytorch.utilities.types"] = utilities_types

    loader_module = types.ModuleType("torch_geometric.loader")
    loader_module.DataLoader = type("DataLoader", (), {})
    sys.modules["torch_geometric.loader"] = loader_module

    package = types.ModuleType("src.smart.datamodules")
    package.__path__ = [str(ROOT / "src" / "smart" / "datamodules")]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "src.smart.datamodules", loader=None, is_package=True
    )
    sys.modules["src.smart.datamodules"] = package

    target_builder = types.ModuleType("src.smart.datamodules.target_builder")
    target_builder.WaymoTargetBuilderTrain = lambda maximum: ("train", maximum)
    target_builder.WaymoTargetBuilderVal = lambda: ("val",)
    sys.modules[target_builder.__name__] = target_builder

    path = ROOT / "src" / "smart" / "datamodules" / "scalable_datamodule.py"
    spec = importlib.util.spec_from_file_location(
        "src.smart.datamodules.scalable_datamodule", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCALABLE_DATAMODULE = _load_datamodule_module()


class TextPromptDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw_dir = self.root / "training"
        self.raw_dir.mkdir()
        sample = {
            "scenario_id": "waymo-a",
            "agent": {
                "id": torch.tensor([20, 10]),
                "type": torch.tensor([0, 2]),
                "role": torch.tensor(
                    [[True, False, False], [False, True, False]]
                ),
            },
        }
        with (self.raw_dir / "waymo-a.pkl").open("wb") as stream:
            pickle.dump(sample, stream)

        self.mapping_path = self.root / "mapping.json"
        self.mapping_path.write_text(
            json.dumps({"waymo-a": "scene_101"}), encoding="utf-8"
        )
        tag_dir = (
            self.root
            / "tag_prompts"
            / "waymo_train_v_action"
            / "tags"
            / "1"
        )
        tag_dir.mkdir(parents=True)
        (tag_dir / "scene_101.json").write_text(
            json.dumps(["LeftTurn(10 at 11-31)"]), encoding="utf-8"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_enabled_dataset_attaches_prompt_in_agent_order(self):
        dataset = SCALABLE_DATASET.MultiDataset(
            raw_dir=str(self.raw_dir),
            transform=lambda value: value,
            text_prompt_root=str(self.root),
            text_mapping_path=str(self.mapping_path),
            text_split="auto",
        )

        data = dataset.get(0)

        self.assertEqual(
            data["agent"]["text_prompt"],
            ["", "The target cyclist is turning left."],
        )
        torch.testing.assert_close(
            data["agent"]["text_prompt_mask"],
            torch.tensor([False, True]),
        )

    def test_disabled_dataset_does_not_add_or_open_prompt_artifacts(self):
        self.mapping_path.unlink()
        dataset = SCALABLE_DATASET.MultiDataset(
            raw_dir=str(self.raw_dir), transform=lambda value: value
        )

        data = dataset.get(0)

        self.assertNotIn("text_prompt", data["agent"])
        self.assertNotIn("text_prompt_mask", data["agent"])


class TextPromptDataModuleTest(unittest.TestCase):
    def test_fit_setup_routes_split_specific_mappings(self):
        calls = []

        class RecordingDataset:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                calls.append(self)

        original = SCALABLE_DATAMODULE.MultiDataset
        SCALABLE_DATAMODULE.MultiDataset = RecordingDataset
        self.addCleanup(setattr, SCALABLE_DATAMODULE, "MultiDataset", original)
        module = SCALABLE_DATAMODULE.MultiDataModule(
            train_batch_size=4,
            val_batch_size=4,
            test_batch_size=4,
            train_raw_dir="/cache/training",
            val_raw_dir="/cache/validation",
            test_raw_dir="/cache/testing",
            val_tfrecords_splitted="/cache/tfrecords",
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            train_max_num=32,
            text_prompt_root="/tags",
            train_text_mapping_path="/tags/train.json",
            val_text_mapping_path="/tags/val.json",
            tag_prompt_subdir="auto",
            tag_sentence_style="ordered",
        )

        module.setup("fit")

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["text_split"], "train")
        self.assertEqual(
            calls[0].kwargs["text_mapping_path"], "/tags/train.json"
        )
        self.assertEqual(calls[1].kwargs["text_split"], "val")
        self.assertEqual(calls[1].kwargs["text_mapping_path"], "/tags/val.json")

    def test_test_setup_never_attaches_future_derived_prompts(self):
        calls = []

        class RecordingDataset:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                calls.append(self)

        original = SCALABLE_DATAMODULE.MultiDataset
        SCALABLE_DATAMODULE.MultiDataset = RecordingDataset
        self.addCleanup(setattr, SCALABLE_DATAMODULE, "MultiDataset", original)
        module = SCALABLE_DATAMODULE.MultiDataModule(
            train_batch_size=4,
            val_batch_size=4,
            test_batch_size=4,
            train_raw_dir="/cache/training",
            val_raw_dir="/cache/validation",
            test_raw_dir="/cache/testing",
            val_tfrecords_splitted="/cache/tfrecords",
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            train_max_num=32,
            text_prompt_root="/tags",
            train_text_mapping_path="/tags/train.json",
            val_text_mapping_path="/tags/val.json",
        )

        module.setup("test")

        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0].kwargs.get("text_prompt_root"))
        self.assertIsNone(calls[0].kwargs.get("text_mapping_path"))


if __name__ == "__main__":
    unittest.main()
