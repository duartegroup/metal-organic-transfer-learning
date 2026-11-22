"""Regression tests for pre-trained checkpoint loading.

The failure these cover was silent and expensive: a checkpoint that matched only
part of the model was reported as "successfully loaded", pretraining was skipped,
and training continued with randomly initialised encoders. Nothing in the logs
said so.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_manager():
    """Load model_utils.py by path, bypassing the package __init__.

    model_utils itself needs only torch, pandas and the standard library, so this
    keeps the test runnable without the full training environment.
    """
    pytest.importorskip("torch")
    pytest.importorskip("pandas")

    path = REPO / "src" / "transfer_learning_src" / "train" / "model_utils.py"
    spec = importlib.util.spec_from_file_location("_model_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ModelComponentManager


def _make_model(mol_width: int = 8):
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mol_encoder = nn.Linear(mol_width, 4)
            self.protein_encoder = nn.Linear(6, 4)
            self.classification_head = nn.Linear(4, 1)

    return Tiny()


def test_complete_checkpoint_loads_every_component(tmp_path):
    manager = _load_manager()
    manager.save_components(_make_model(), tmp_path, "pretrain_src")

    status = manager.load_components_from_directory(
        model=_make_model(), load_dir=tmp_path, strict=False
    )

    assert status, "expected a non-empty load status"
    assert all(status.values()), f"a complete checkpoint failed to load: {status}"
    assert set(status) == {"mol_encoder", "protein_encoder", "classification_head"}


def test_absent_component_file_is_reported_as_failure(tmp_path):
    """A component the model needs but the directory lacks must report False.

    Previously such a component was simply missing from the returned dict, so a
    caller checking the statuses saw nothing wrong.
    """
    manager = _load_manager()
    manager.save_components(_make_model(), tmp_path, "pretrain_src")
    (tmp_path / "protein_encoder.pth").unlink()

    status = manager.load_components_from_directory(
        model=_make_model(), load_dir=tmp_path, strict=False
    )

    assert status.get("protein_encoder") is False, (
        f"missing component file was not reported as a failure: {status}"
    )
    assert not all(status.values())


def test_partially_matching_component_is_not_reported_as_loaded(tmp_path):
    """Shape/key mismatch leaves parameters random, so it is not a success.

    The checkpoint here is saved from a model whose mol_encoder has a different
    input width. load_state_dict(strict=False) does not raise, it just declines to
    copy the mismatched tensors -- which used to be recorded as True.
    """
    manager = _load_manager()
    import torch.nn as nn

    class Partial(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mol_encoder = nn.Sequential(nn.Linear(8, 4))  # nested: different keys
            self.protein_encoder = nn.Linear(6, 4)
            self.classification_head = nn.Linear(4, 1)

    manager.save_components(Partial(), tmp_path, "pretrain_src")

    status = manager.load_components_from_directory(
        model=_make_model(), load_dir=tmp_path, strict=False
    )

    assert status.get("mol_encoder") is False, (
        f"a component that was not restored was reported as loaded: {status}"
    )
    assert not all(status.values())
