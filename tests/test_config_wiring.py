"""Regression tests for command-line arguments reaching the trainer.

Twelve hyperparameters -- max_epochs, patience, the three learning rates, the
three dropouts, gamma, margin and friends -- were parsed, stored in params,
printed in the run's logged configuration, and then ignored, because the trainer
reads TrainingConfig and nothing ever copied the parsed values into it. Passing
`--max_epochs 20` ran 500 epochs.

MultitaskTrainingPipeline now copies them across. These tests pin the invariant
that makes that safe: a CLI default must agree with the dataclass default, or
wiring it through would silently change what every existing run does. Two of them
did not agree (dropout_pretrain 0.4 vs 0.05, patience 15 vs 35) and the dataclass
value is the one the published runs used.

Both modules are dependency-free, so they load by path without the training
stack.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import importlib.util
from dataclasses import fields
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "transfer_learning_src"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config_module():
    return _load(SRC / "train" / "config.py", "_tl_config")


@pytest.fixture(scope="module")
def loader_module():
    return _load(SRC / "utils" / "config_loader.py", "_tl_config_loader")


def _cli_defaults(loader_module) -> dict:
    parser_args = {}
    args = loader_module.parse_args.__wrapped__ if hasattr(
        loader_module.parse_args, "__wrapped__"
    ) else None
    # parse_args builds and consumes its parser internally, so parse an empty
    # command line and read the resulting namespace instead.
    import sys

    saved = sys.argv
    try:
        sys.argv = ["prog", "--use_mol", "--use_protein"]
        namespace = loader_module.parse_args()
    finally:
        sys.argv = saved
    for key, value in vars(namespace).items():
        parser_args[key] = value
    return parser_args


def test_cli_defaults_match_the_dataclass(config_module, loader_module):
    """A disagreement here means wiring a flag through changes existing runs."""
    config = config_module.TrainingConfig()
    cli = _cli_defaults(loader_module)

    mismatches = {}
    for field in fields(config):
        if field.name in cli:
            configured = getattr(config, field.name)
            if cli[field.name] != configured:
                mismatches[field.name] = (cli[field.name], configured)

    # data_scaler is normalised from "none" to None on purpose, and the pipeline
    # assigns it explicitly after the generic copy.
    mismatches.pop("data_scaler", None)
    # enable_progress_bar is a display setting, not a hyperparameter.
    mismatches.pop("enable_progress_bar", None)

    assert not mismatches, (
        "CLI default disagrees with TrainingConfig (cli, config): " f"{mismatches}"
    )


def test_wired_hyperparameters_are_present_in_params(config_module, loader_module):
    """The pipeline copies params -> config by field name, so they must be in params."""
    config = config_module.TrainingConfig()
    cli = _cli_defaults(loader_module)

    import sys

    saved = sys.argv
    try:
        sys.argv = ["prog", "--use_mol", "--use_protein"]
        params = loader_module.build_params_from_args(loader_module.parse_args())
    finally:
        sys.argv = saved

    shared = {f.name for f in fields(config)} & set(cli)
    # These are read straight from params rather than from the config object.
    shared -= {"ssl_noise_scale", "ssl_node_drop_prob"}

    absent = sorted(name for name in shared if name not in params)
    assert not absent, (
        f"TrainingConfig fields with a CLI flag that never reach params: {absent}"
    )


def test_max_epochs_and_patience_are_wired(config_module, loader_module):
    """The two that mattered most: a short smoke run must actually be short."""
    import sys

    saved = sys.argv
    try:
        sys.argv = ["prog", "--use_mol", "--use_protein", "--max_epochs", "7",
                    "--patience", "2"]
        params = loader_module.build_params_from_args(loader_module.parse_args())
    finally:
        sys.argv = saved

    assert params["max_epochs"] == 7
    assert params["patience"] == 2

    config = config_module.TrainingConfig()
    for field in fields(config):
        if field.name in params and params[field.name] is not None:
            setattr(config, field.name, params[field.name])

    assert config.max_epochs == 7
    assert config.patience == 2


def test_run_types_is_parsed_into_a_list(loader_module):
    import sys

    saved = sys.argv
    try:
        sys.argv = ["prog", "--use_mol", "--use_protein",
                    "--run_types", "no_pretrain_encoder_task, pretrain_task"]
        params = loader_module.build_params_from_args(loader_module.parse_args())
    finally:
        sys.argv = saved

    assert params["run_types"] == ["no_pretrain_encoder_task", "pretrain_task"]


def test_run_types_defaults_to_the_full_pipeline(loader_module):
    import sys

    saved = sys.argv
    try:
        sys.argv = ["prog", "--use_mol", "--use_protein"]
        params = loader_module.build_params_from_args(loader_module.parse_args())
    finally:
        sys.argv = saved

    assert params["run_types"] == ["full_pipeline"]
