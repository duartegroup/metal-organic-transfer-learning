# io_utils.py
"""Filesystem helpers shared by the analysis, reporting and plotting modules.

Every artefact this pipeline produces goes through :func:`atomic_write` or
:func:`atomic_savefig`.

Opening a path with mode ``"w"`` truncates it *before* the new bytes are
written, so a failure part-way through -- a full filesystem, an exceeded quota,
a killed job -- leaves a zero-byte file where a good one used to be. Writing to
a sibling temporary and renaming it into place means the destination is only
ever replaced by a file that was written in full; on any failure the previous
version survives untouched.

The rename is ``os.replace``, which is atomic within a filesystem, and the
temporary is created in the destination's own directory so that holds.
"""
import os
from typing import Union

from console import printer


def _write_tmp(path: str, payload: bytes) -> str:
    """Write ``payload`` to ``<path>.tmp``, flushed to disk, and return that path."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())

    written = os.path.getsize(tmp)
    if written != len(payload):
        raise IOError(
            f"short write to {tmp}: {written} of {len(payload)} bytes "
            "(filesystem full or over quota?)"
        )
    return tmp


def atomic_write(path: str, data: Union[str, bytes], encoding: str = "utf-8") -> None:
    """Write ``data`` to ``path``, leaving the original intact if anything fails.

    Parameters
    ----------
    path : str
        Destination path. Parent directories are created if missing.
    data : str or bytes
        Content to write.
    encoding : str, optional
        Encoding used when ``data`` is a str. Default ``"utf-8"``.

    Raises
    ------
    OSError
        Propagated from the underlying write, after the temporary is cleaned up.
        The destination is unchanged.
    """
    payload = data.encode(encoding) if isinstance(data, str) else data
    tmp = None
    try:
        tmp = _write_tmp(path, payload)
        os.replace(tmp, path)
    except OSError as e:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        printer.error(f"Failed to write {path}: {e}")
        raise


def atomic_savefig(figure, path: str, **savefig_kwargs) -> None:
    """Save a matplotlib figure through the same write-then-rename dance.

    Parameters
    ----------
    figure : matplotlib figure or the pyplot module
        Anything exposing ``savefig``.
    path : str
        Destination path. Parent directories are created if missing.
    **savefig_kwargs
        Passed straight through to ``savefig``.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # matplotlib infers the output format from the extension, and "<name>.png.tmp"
    # has none it recognises -- so state it explicitly.
    tmp = f"{path}.tmp"
    savefig_kwargs.setdefault("format", os.path.splitext(path)[1].lstrip(".") or "png")
    try:
        figure.savefig(tmp, **savefig_kwargs)
        if os.path.getsize(tmp) == 0:
            raise IOError(f"savefig produced an empty file at {tmp}")
        os.replace(tmp, path)
    except OSError as e:
        if os.path.exists(tmp):
            os.unlink(tmp)
        printer.error(f"Failed to save figure {path}: {e}")
        raise
