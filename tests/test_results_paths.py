"""Regression tests for where a run writes its results.

`create_results_base_dir` keyed the path on descriptor / modality / scaler /
affinity / scaling_config only. Two runs differing solely in --threshold or
--seed therefore resolved to the same directory and overwrote each other's
raw_results.json, so a sweep over three seeds would have produced one result per
configuration rather than three. `pretrain_src` lives under the same directory,
so an SSL checkpoint trained for one seed would also have been reused for
another. And --output_dir was parsed but never read here: the root was a
hardcoded relative Path("results"), so output landed inside the checkout.

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
PKG = "_phases_paths_under_test"


def _stub(name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _load_phases():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")

    anything = type("Anything", (), {})

    _stub("wandb", finish=lambda *a, **k: None, init=lambda *a, **k: None)
    _stub("prettytable", PrettyTable=anything)
    _stub("pytorch_lightning", Trainer=anything, LightningModule=anything)
    _stub("pytorch_lightning.callbacks", EarlyStopping=anything,
          LearningRateMonitor=anything)
    _stub("pytorch_lightning.utilities")
    _stub("pytorch_lightning.utilities.combined_loader", CombinedLoader=anything)
    _stub("lightning")
    _stub("lightning.pytorch")
    _stub("lightning.pytorch.loggers", WandbLogger=anything)
    _stub("data")
    _stub("data.dataset", WrappedDataLoader=anything)
    _stub("data.utils_data", get_datasets=lambda *a, **k: None,
          create_dataloaders=lambda *a, **k: None)
    _stub("model")
    _stub("model.multitask", MultiTaskPocket=anything, CCSAFinetune=anything)
    _stub("utils")
    _stub("utils.checkpoint_utils", PrintOnBestModelCheckpoint=anything,
          save_phase_stats=lambda *a, **k: None)

    package = types.ModuleType(PKG)
    package.__path__ = [str(REPO / "src" / "transfer_learning_src" / "train")]
    sys.modules[PKG] = package
    _stub(f"{PKG}.config", TrainingConfig=type("TrainingConfig", (), {}))
    _stub(f"{PKG}.model_utils", ModelComponentManager=anything,
          transfer_weights=lambda *a, **k: None)

    path = REPO / "src" / "transfer_learning_src" / "train" / "phases.py"
    spec = importlib.util.spec_from_file_location(f"{PKG}.phases", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = PKG
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def phases():
    return _load_phases()


def _base_dir(phases, tmp_path, **overrides):
    params = {
        "inhibitor_descriptor": "SOAP",
        "use_mol": True,
        "use_protein": False,
        "use_pocket": False,
        "data_scaler": "minmax",
        "affinity_type": "pk",
        "source_smote": True,
        "target_smote": True,
        "cross_domain_scaling": False,
        "threshold": [6.0],
        "seed": 42,
        "output_dir": str(tmp_path),
    }
    params.update(overrides)
    phase = phases.TrainingPhase.__new__(phases.TrainingPhase)
    phase.params = params
    phase.config = None
    return phase.create_results_base_dir()


def test_output_dir_is_honoured(phases, tmp_path):
    path = _base_dir(phases, tmp_path)
    assert tmp_path in path.parents, f"{path} is not under the requested {tmp_path}"


def test_threshold_separates_runs(phases, tmp_path):
    a = _base_dir(phases, tmp_path, threshold=[6.0])
    b = _base_dir(phases, tmp_path, threshold=[7.0])
    assert a != b, f"thresholds 6.0 and 7.0 both resolved to {a}"


def test_seed_separates_runs(phases, tmp_path):
    a = _base_dir(phases, tmp_path, seed=42)
    b = _base_dir(phases, tmp_path, seed=43)
    assert a != b, f"seeds 42 and 43 both resolved to {a}"


def test_configuration_still_separates_runs(phases, tmp_path):
    """The original keys must keep working."""
    a = _base_dir(phases, tmp_path, inhibitor_descriptor="SOAP")
    b = _base_dir(phases, tmp_path, inhibitor_descriptor="MACE")
    c = _base_dir(phases, tmp_path, affinity_type="pic50")
    assert len({a, b, c}) == 3


def test_identical_runs_resolve_identically(phases, tmp_path):
    """Checkpoint reuse depends on the path being stable for the same settings."""
    assert _base_dir(phases, tmp_path) == _base_dir(phases, tmp_path)


def test_scalar_and_list_thresholds_agree(phases, tmp_path):
    """The threshold arrives as a list from the CLI and a float from defaults."""
    assert _base_dir(phases, tmp_path, threshold=6.0) == _base_dir(
        phases, tmp_path, threshold=[6.0]
    )
