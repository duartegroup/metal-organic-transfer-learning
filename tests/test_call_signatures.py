"""Static check that in-repo calls supply every required argument.

This exists because a required parameter was added to prepare_datasets --
`tgt_test`, for the inner-validation split -- without updating two callers in
pipeline.py, which would have raised TypeError on the SSL-pretraining and
domain-adaptation paths, i.e. every transfer-learning run. The dynamic test that
covers prepare_datasets skipped because `esm` was missing, so nothing caught it
until a real job failed ten minutes in.

The check is purely syntactic: it parses the tree, so it needs none of the
training stack and cannot be skipped for a missing dependency. It does not
resolve imports or aliases, so it only compares calls against a same-named
function defined in the repo -- enough for the refactoring mistake above.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

# Functions whose arity has changed, or is likely to, during this revision.
WATCHED = {
    "prepare_datasets",
    "get_datasets",
    "split_data_by_group",
    "transfer_weights",
    "_require_test_loader",
    "_metrics_from_best_checkpoint",
    "_carve_inner_validation",
}


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _definitions() -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in WATCHED:
                    found[node.name] = node
    return found


def _required_parameters(node: ast.FunctionDef) -> list[str]:
    """Positional parameters with no default, excluding self."""
    args = node.args
    positional = args.posonlyargs + args.args
    names = [a.arg for a in positional]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
        positional = positional[1:]
    n_defaulted = len(args.defaults)
    required = names[: len(names) - n_defaulted] if n_defaulted else names
    return required


def _calls(name: str):
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if called == name:
                yield path, node


def test_watched_functions_are_found():
    """Guards against the watch list silently going stale after a rename."""
    definitions = _definitions()
    missing = sorted(WATCHED - set(definitions))
    assert not missing, f"watched functions no longer defined in src/: {missing}"


def test_every_call_supplies_the_required_arguments():
    definitions = _definitions()
    problems = []

    for name, node in definitions.items():
        required = _required_parameters(node)
        if not required:
            continue
        for path, call in _calls(name):
            if any(isinstance(a, ast.Starred) for a in call.args) or any(
                kw.arg is None for kw in call.keywords
            ):
                continue  # *args / **kwargs: cannot be checked statically
            supplied = set(required[: len(call.args)])
            supplied |= {kw.arg for kw in call.keywords if kw.arg}
            absent = [p for p in required if p not in supplied]
            if absent:
                problems.append(
                    f"{path.relative_to(REPO)}:{call.lineno} calls {name}() "
                    f"without {absent}"
                )

    assert not problems, "calls missing required arguments:\n  " + "\n  ".join(problems)
