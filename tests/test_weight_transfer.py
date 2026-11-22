"""Regression tests for the phase-to-phase weight handoff.

Every transfer in the pipeline -- source pretraining to fold model, fold model to
CCSA base, CCSA base to fine-tuning -- used
`load_state_dict(..., strict=False)` and threw away the returned key lists. A
name mismatch anywhere in that chain left the target randomly initialised while
the following log line announced the transfer as done, which would produce a
transfer-learning run that transferred nothing and scored like the from-scratch
baseline it is compared against.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    pytest.importorskip("torch")
    pytest.importorskip("pandas")

    path = REPO / "src" / "transfer_learning_src" / "train" / "model_utils.py"
    spec = importlib.util.spec_from_file_location("_model_utils_wt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encoder_model(with_projection_head: bool = False):
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mol_encoder = nn.Linear(8, 4)
            self.protein_encoder = nn.Linear(6, 4)
            self.classification_head = nn.Linear(4, 1)
            if with_projection_head:
                self.projection_head = nn.Linear(4, 4)

    return Model()


def test_full_transfer_copies_every_parameter():
    import torch

    mod = _load_module()
    source, target = _encoder_model(), _encoder_model()

    mod.transfer_weights(target, source, "source -> target")

    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value), key


def test_self_ssl_extras_are_allowed():
    """A source built with self_ssl=True carries a projection head the target lacks."""
    mod = _load_module()
    source = _encoder_model(with_projection_head=True)
    target = _encoder_model(with_projection_head=False)

    mod.transfer_weights(target, source, "ssl source -> plain target")  # must not raise


def test_head_absent_from_an_ssl_source_is_allowed():
    """MultiTaskPocket builds a classification head XOR a projection head.

    An SSL-pretrained source therefore has no classification head to give, and the
    supervised target's head is meant to be trained from scratch. This is the real
    shape of every pretrained -> finetune transfer in the pipeline, and an earlier
    version of the guard rejected it, failing the first S4C run at CCSA fold 1.
    """
    import torch.nn as nn

    mod = _load_module()

    class SSLSource(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mol_encoder = nn.Linear(8, 4)
            self.protein_encoder = nn.Linear(6, 4)
            self.projection_head = nn.Linear(4, 4)  # and no classification_head

    target = _encoder_model()  # has classification_head
    mod.transfer_weights(
        target,
        SSLSource(),
        "ssl source -> supervised target",
        allow_missing=("classification_head",),
    )


def test_the_head_exemption_does_not_excuse_an_encoder():
    """The exemption is named, so a missing encoder is still fatal."""
    import torch.nn as nn

    mod = _load_module()

    class NoMolEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.protein_encoder = nn.Linear(6, 4)
            self.projection_head = nn.Linear(4, 4)

    with pytest.raises(RuntimeError, match="mol_encoder"):
        mod.transfer_weights(
            _encoder_model(),
            NoMolEncoder(),
            "ssl source missing an encoder",
            allow_missing=("classification_head",),
        )


def test_missing_parameters_raise_instead_of_staying_random():
    """Without an exemption, any component the source cannot supply is fatal."""
    mod = _load_module()
    source = _encoder_model(with_projection_head=False)
    target = _encoder_model(with_projection_head=True)

    with pytest.raises(RuntimeError, match="projection_head"):
        mod.transfer_weights(target, source, "plain source -> ssl target")


def test_renamed_layer_is_caught():
    """A structural rename matches no keys at all, which strict=False tolerated."""
    import torch.nn as nn

    mod = _load_module()

    class Renamed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.molecule_encoder = nn.Linear(8, 4)  # was mol_encoder
            self.protein_encoder = nn.Linear(6, 4)
            self.classification_head = nn.Linear(4, 1)

    with pytest.raises(RuntimeError, match="mol_encoder"):
        mod.transfer_weights(_encoder_model(), Renamed(), "renamed -> target")
