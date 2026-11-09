"""Regression test for the single-sample trailing batch.

The model uses BatchNorm1d, which cannot compute statistics from one sample:

    ValueError: Expected more than 1 value per channel when training,
                got input size torch.Size([1, 384])

The published runs never reached this. With 214 caged compounds over five folds the
training sets were 175/168/178/159/176, and none leaves a remainder of 1 against a
batch size of 64. Carving out an inner validation fold shrinks them, and some of the
new sizes do -- three of eight pilot runs of the nested sweep died this way.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_utils_data():
    """Load utils_data with its HDF5/descriptor dependencies stubbed."""
    pytest.importorskip("torch")
    pytest.importorskip("numpy")

    anything = type("Anything", (), {})
    pkg_name = "_ud_pkg"
    data_dir = REPO / "src" / "transfer_learning_src" / "data"

    package = types.ModuleType(pkg_name)
    package.__path__ = [str(data_dir)]
    sys.modules[pkg_name] = package

    # utils_data uses relative imports, so the stubs are registered as submodules of
    # a synthetic package rather than as top-level names.
    for name, attrs in {
        "dataset": {
            "BindingAffinitiesDataset": anything,
            "BindingAffinitiesDatasetWithKeys": anything,
            "WrappedDataLoader": anything,
        },
        "load_data": {"load_data_hdf5": lambda *a, **k: None},
        "split_data": {
            "split_data_by_group": lambda *a, **k: None,
            "split_data_by_group_random": lambda *a, **k: None,
        },
    }.items():
        module = types.ModuleType(f"{pkg_name}.{name}")
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[f"{pkg_name}.{name}"] = module
        setattr(package, name, module)

    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.utils_data", data_dir / "utils_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = pkg_name
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Sized:
    """Minimal dataset. create_dataloaders also logs the class balance."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.neg_count = n // 2
        self.pos_count = n - n // 2

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        import torch

        return torch.zeros(4), torch.zeros(1)


@pytest.fixture(scope="module")
def utils_data():
    return _load_utils_data()


@pytest.mark.parametrize("n", [65, 129, 193])
def test_trailing_single_sample_batch_is_dropped(utils_data, n):
    """n % 64 == 1, which is the case that crashes BatchNorm."""
    loaders = utils_data.create_dataloaders(
        None, None, _Sized(n), None, {"batch_size": 64}
    )
    loader = loaders["tgt_train_loader"]
    assert loader.drop_last, f"a training set of {n} must drop its trailing batch"
    sizes = [len(batch) for batch in loader.batch_sampler]
    assert 1 not in sizes, f"{n} still yields a batch of one: {sizes}"


@pytest.mark.parametrize("n", [128, 159, 175, 200])
def test_other_sizes_keep_every_sample(utils_data, n):
    """Dropping is conditional, so nothing else loses data."""
    loaders = utils_data.create_dataloaders(
        None, None, _Sized(n), None, {"batch_size": 64}
    )
    loader = loaders["tgt_train_loader"]
    assert not loader.drop_last
    assert sum(len(b) for b in loader.batch_sampler) == n


def test_validation_loaders_never_drop(utils_data):
    """Every held-out sample must be evaluated."""
    loaders = utils_data.create_dataloaders(
        None, None, None, _Sized(129), {"batch_size": 64}
    )
    loader = loaders["tgt_val_loader"]
    assert not loader.drop_last
    assert sum(len(b) for b in loader.batch_sampler) == 129
