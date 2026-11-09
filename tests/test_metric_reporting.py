"""Regression tests for how reported metrics are obtained.

Two silent failure modes are covered, both of which produced a number shaped
exactly like a valid result:

1. Falling back to the inner validation loader when no test loader was built, so
   the set used for model selection was reported as held-out performance.
2. Returning the last epoch's logged metrics, or a dict of zeros, when the best
   checkpoint could not be evaluated. MCC 0.0 is a legitimate value, so such a
   row was indistinguishable from a genuinely chance-level model.

phases.py imports the whole training stack, so the heavy dependencies are
stubbed and the module is loaded by path. The code under test is the production
code, not a reimplementation of it.

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
PKG = "_phases_under_test"


def _stub(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _DummyCheckpointCallback:
    """Stands in for PrintOnBestModelCheckpoint; only isinstance matters."""

    def __init__(self, best_model_path: str = "", monitor: str = "val/loss") -> None:
        self.best_model_path = best_model_path
        self.monitor = monitor


def _load_phases():
    """Load train/phases.py with its heavy imports stubbed out."""
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")

    anything = type("Anything", (), {})

    _stub("wandb", finish=lambda *a, **k: None, init=lambda *a, **k: None)
    _stub("prettytable", PrettyTable=anything)
    _stub("pytorch_lightning", Trainer=anything, LightningModule=anything)
    _stub("pytorch_lightning.callbacks", EarlyStopping=anything,
          LearningRateMonitor=anything)
    _stub("pytorch_lightning.utilities", )
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
    _stub("utils.checkpoint_utils",
          PrintOnBestModelCheckpoint=_DummyCheckpointCallback,
          save_phase_stats=lambda *a, **k: None)

    package = types.ModuleType(PKG)
    package.__path__ = [str(REPO / "src" / "transfer_learning_src" / "train")]
    sys.modules[PKG] = package
    _stub(f"{PKG}.config", TrainingConfig=type("TrainingConfig", (), {}))
    _stub(f"{PKG}.model_utils", ModelComponentManager=anything)

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


# --- the test loader must never be the validation loader ---------------------


def test_present_test_loader_is_returned(phases):
    val, test = object(), object()
    loaders = {"tgt_val_loader": val, "tgt_test_loader": test}
    assert phases._require_test_loader(loaders, val) is test


def test_missing_test_loader_raises_instead_of_using_validation(phases):
    val = object()
    loaders = {"tgt_val_loader": val, "tgt_test_loader": None}
    with pytest.raises(RuntimeError, match="no held-out fold"):
        phases._require_test_loader(loaders, val)


def test_test_loader_identical_to_validation_loader_raises(phases):
    val = object()
    loaders = {"tgt_val_loader": val, "tgt_test_loader": val}
    with pytest.raises(RuntimeError, match="same object"):
        phases._require_test_loader(loaders, val)


# --- a failed evaluation must not return a number ----------------------------


class _FakeTrainer:
    def __init__(self, callbacks, validate_result=None, raise_on_validate=False):
        self.callbacks = callbacks
        self.lightning_module = _FakeModule()
        self._validate_result = validate_result
        self._raise = raise_on_validate
        # Populated in the real thing; the old fallback read it and reported it.
        self.logged_metrics = {
            "val/loss": 0.1, "val/accuracy": 0.9, "val/f1": 0.9,
            "val/auroc": 0.9, "val/mcc": 0.85,
        }

    def validate(self, model, dataloaders=None, verbose=False):
        if self._raise:
            raise RuntimeError("validation exploded")
        return self._validate_result


class _FakeModule:
    def load_state_dict(self, state_dict):
        return None


def _checkpoint_file(tmp_path):
    import torch

    path = tmp_path / "best.ckpt"
    torch.save({"state_dict": {}}, path)
    return str(path)


def test_successful_evaluation_returns_the_metrics(phases, tmp_path):
    metrics = {"val/loss": 0.25, "val/accuracy": 0.8, "val/f1": 0.7,
               "val/auroc": 0.75, "val/mcc": 0.46}
    trainer = _FakeTrainer(
        [_DummyCheckpointCallback(_checkpoint_file(tmp_path))],
        validate_result=[metrics],
    )

    got = phases._metrics_from_best_checkpoint(trainer, object(), "TestPhase")

    assert got == pytest.approx(metrics)


def test_no_checkpoint_path_raises(phases):
    """Previously returned trainer.logged_metrics -- the validation numbers."""
    trainer = _FakeTrainer([_DummyCheckpointCallback(best_model_path="")])

    with pytest.raises(RuntimeError, match="no best_model_path"):
        phases._metrics_from_best_checkpoint(trainer, object(), "TestPhase")


def test_no_checkpoint_callback_raises(phases):
    trainer = _FakeTrainer([])

    with pytest.raises(RuntimeError, match="no PrintOnBestModelCheckpoint"):
        phases._metrics_from_best_checkpoint(trainer, object(), "TestPhase")


def test_failed_validation_raises_rather_than_returning_zeros(phases, tmp_path):
    trainer = _FakeTrainer(
        [_DummyCheckpointCallback(_checkpoint_file(tmp_path))],
        raise_on_validate=True,
    )

    with pytest.raises(RuntimeError, match="could not evaluate the best checkpoint"):
        phases._metrics_from_best_checkpoint(trainer, object(), "TestPhase")


def test_incomplete_metrics_raise_rather_than_defaulting(phases, tmp_path):
    """A result missing val/mcc used to be reported as val/mcc = 0.0."""
    trainer = _FakeTrainer(
        [_DummyCheckpointCallback(_checkpoint_file(tmp_path))],
        validate_result=[{"val/loss": 0.3, "val/accuracy": 0.8}],
    )

    with pytest.raises(RuntimeError, match="val/mcc"):
        phases._metrics_from_best_checkpoint(trainer, object(), "TestPhase")
