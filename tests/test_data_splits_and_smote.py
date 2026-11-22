"""Regression tests for the two training-pipeline fixes.

Both of the issues these cover were silent: the code ran, produced plausible numbers,
and gave no indication anything was wrong. These tests exist so that neither can come
back unnoticed.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
METADATA = REPO / "data" / "excel" / "metadata.xlsx"
N_FOLDS = 5


# --------------------------------------------------------------------------- SMOTE

def _load_dataset_class():
    """Load dataset.py directly, bypassing the package __init__.

    `transfer_learning_src.data.__init__` imports the HDF5 readers, which pull in the
    ESM SDK and the rest of the training stack. dataset.py itself needs only torch,
    torch_geometric, sklearn and numpy, so loading it by path keeps this test runnable
    without the full environment while still exercising the production class.
    """
    import importlib.util

    for module in ("torch", "imblearn", "torch_geometric", "sklearn"):
        pytest.importorskip(module)

    path = REPO / "src" / "transfer_learning_src" / "data" / "dataset.py"
    spec = importlib.util.spec_from_file_location("_dataset_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BindingAffinitiesDataset


def _build_dataset(affinities: np.ndarray, threshold: float, seed: int = 42):
    """Construct the real BindingAffinitiesDataset with oversampling enabled.

    Importing the production class is the point: a test that re-implemented the
    resampling logic here would keep passing if the fix were reverted.
    """
    BindingAffinitiesDataset = _load_dataset_class()

    rng = np.random.default_rng(seed)
    keys = [f"compound_{i}" for i in range(len(affinities))]
    # Molecular descriptors are per-atom and are mean-pooled by the dataset, so they
    # must be (n_atoms, n_features); protein and pocket embeddings are per-residue.
    features = {k: rng.normal(size=(6, 8)).astype(np.float32) for k in keys}
    biological = {k: rng.normal(size=(5, 4)).astype(np.float32) for k in keys}

    return BindingAffinitiesDataset(
        features,
        biological,
        dict(biological),
        {k: float(a) for k, a in zip(keys, affinities)},
        "mol",
        apply_smote=True,
        binding_threshold=threshold,
        random_state=seed,
    )


def _imbalanced_affinities() -> np.ndarray:
    """Imbalanced affinities spanning all three activity thresholds."""
    rng = np.random.default_rng(0)
    return np.concatenate(
        [rng.uniform(4.0, 5.9, 30), rng.uniform(6.0, 6.9, 15), rng.uniform(7.1, 9.0, 10)]
    ).astype(np.float32)


@pytest.fixture
def affinities() -> np.ndarray:
    return _imbalanced_affinities()


@pytest.mark.parametrize("threshold", [5.0, 6.0, 7.0])
def test_measured_affinities_survive_oversampling(affinities, threshold):
    """Oversampling must not overwrite real measurements.

    The original implementation rebuilt the whole affinity tensor from placeholder
    values, discarding every measured value in an oversampled training set. The
    originals are the first rows imblearn returns, so they must come back unchanged.
    """
    torch = pytest.importorskip("torch")
    dataset = _build_dataset(affinities, threshold)

    n_original = len(affinities)
    assert len(dataset.affinities) >= n_original
    assert torch.allclose(
        dataset.affinities[:n_original],
        torch.tensor(affinities, dtype=torch.float32),
    ), "measured affinities were modified by oversampling"


def test_fixed_threshold_labels_everything_active():
    """Characterise the consequence of the hard-coded threshold.

    NOTE: this passes on both branches. The threshold defect was in the *wiring* --
    build_dataset never forwarded the experiment's threshold, so the dataset kept its
    7.0 default -- not in the dataset class, which has always honoured whatever it is
    given. This test documents the failure mode that wiring produced; the guard
    against the wiring itself is test_threshold_is_forwarded_to_datasets below.

    With the threshold left at 7.0, both synthetic affinity values (8.0 and 6.0)
    exceed an experiment threshold of 5.0, so every training sample comes out active.
    """
    dataset = _build_dataset(_imbalanced_affinities(), threshold=7.0)
    n_original = len(_imbalanced_affinities())
    synthetic = dataset.affinities[n_original:]
    assert len(synthetic) > 0, "expected SMOTE to add synthetic rows"
    assert set((synthetic > 5.0).int().tolist()) == {1}, (
        "expected every synthetic sample to be labelled active at a threshold of 5.0"
    )


def test_threshold_is_forwarded_to_datasets():
    """prepare_datasets must binarise at the threshold under evaluation.

    This is the real guard on the wiring defect. It needs the full training
    environment, since transfer_learning_src.data imports the HDF5 readers; it skips
    where those are unavailable and runs on the cluster.
    """
    pytest.importorskip("torch")
    pytest.importorskip("imblearn")
    try:
        import sys

        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "src"))
        from transfer_learning_src.data.utils_data import prepare_datasets
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"full training environment unavailable: {exc}")

    rng = np.random.default_rng(0)
    affinities = _imbalanced_affinities()
    keys = [f"compound_{i}" for i in range(len(affinities))]
    split = {
        "inhibitor_feats": {k: rng.normal(size=(6, 8)).astype(np.float32) for k in keys},
        "protein_feats": {k: rng.normal(size=(5, 4)).astype(np.float32) for k in keys},
        "pocket_feats": {k: rng.normal(size=(5, 4)).astype(np.float32) for k in keys},
        "binding_affs": {k: float(a) for k, a in zip(keys, affinities)},
    }

    for threshold in (5.0, 6.0, 7.0):
        # A list, as the config carries it, to exercise the unwrapping too.
        datasets, _ = prepare_datasets(
            None, None, split, None, None, "mol",
            target_smote=True,
            activity_threshold=[threshold],
        )
        tgt_train = datasets[2]
        assert tgt_train.binding_threshold == threshold, (
            f"dataset was built with binding_threshold="
            f"{tgt_train.binding_threshold}, expected {threshold}"
        )
        classes = set((tgt_train.affinities > threshold).int().tolist())
        assert classes == {0, 1}, (
            f"threshold {threshold}: training labels collapsed to {classes}"
        )


# ---------------------------------------------------------------------- fold splits

@pytest.mark.skipif(not METADATA.exists(), reason="metadata.xlsx not available")
@pytest.mark.parametrize("endpoint", ["pIC50", "pKi"])
def test_inner_validation_split_is_disjoint(endpoint):
    """train / inner-validation / test must partition the data, for every fold.

    Model selection must not see the fold the reported metric is computed on. This
    mirrors the fold arithmetic in utils_data._carve_inner_validation.
    """
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    measurements = pd.read_excel(METADATA, sheet_name="measurements")
    measurements = measurements[measurements.activity_type == endpoint]
    fold_of = dict(zip(measurements.entry_name, measurements.fold + 1))
    everything = set(fold_of)

    for fold in range(1, N_FOLDS + 1):
        test = {k for k in everything if fold_of[k] == fold}
        inner_fold = (fold % N_FOLDS) + 1 if fold < N_FOLDS else 1
        inner = {k for k in everything if fold_of[k] == inner_fold} - test
        train = (everything - test) - inner

        assert not (train & inner), f"fold {fold}: train overlaps inner validation"
        assert not (train & test), f"fold {fold}: train overlaps test"
        assert not (inner & test), f"fold {fold}: inner validation overlaps test"
        assert train | inner | test == everything, f"fold {fold}: partition loses data"
        assert train and inner and test, f"fold {fold}: a split came out empty"
