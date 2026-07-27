from pathlib import Path


def test_labeling_imports_do_not_reference_source_checkout():
    import src.womd_labeling.map_annotation as module

    module_path = Path(module.__file__).resolve()
    assert "WOMD-Traffic-Signal-Data-Improvement" not in str(module_path)
    assert "CatK" in module_path.parts
