import h5py
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple


class CompoundH5Reader:
    """
    Lightweight, memory-friendly reader for HDF5 compound datasets.

    Provides efficient random access to compound data stored in HDF5 format,
    with optional preloading for faster repeated access.
    """

    DATASETS = (
        "mol_gfn2_atype",
        "mol_gfn2_xyz",
        "mol_mace_embedding",
        "mol_mbtr_embedding",
        "mol_soap_embedding",
        "mol_acsf_embedding",
        "pocket_embeddings",
        "pocket_residues",
        "target_value",
    )

    def __init__(self, h5_path: Union[str, Path], preload: bool = False) -> None:
        """
        Initialize the HDF5 reader for compound data.

        Parameters
        ----------
        h5_path : str or Path
            Path to the HDF5 file to read.
        preload : bool, optional
            If True, load all data into memory at initialization for faster access.
            Default is False for memory efficiency.
        """
        self._file = h5py.File(h5_path, "r")
        self._keys = sorted(list(self._file.keys()))
        self._preload = preload

        if preload:
            self._cache = {k: self._read_group(self._file[k]) for k in self._keys}

    def __len__(self) -> int:
        """
        Get the number of compounds in the dataset.

        Returns
        -------
        int
            Number of compounds in the HDF5 file.
        """
        return len(self._keys)

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        """
        Get compound data by index or key.

        Parameters
        ----------
        idx : int or str
            Either an integer index or string key identifying the compound.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the compound's datasets (embeddings, coordinates, etc.).
        """
        if self._preload:
            return self._cache[idx if isinstance(idx, str) else self._keys[idx]]
        grp = self._file[idx] if isinstance(idx, str) else self._file[self._keys[idx]]
        return self._read_group(grp)

    @staticmethod
    def _read_group(grp: h5py.Group) -> Dict[str, Any]:
        """
        Read all datasets from an HDF5 group.

        Handles different data types appropriately: strings, scalars, and arrays.

        Parameters
        ----------
        grp : h5py.Group
            HDF5 group to read data from.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping dataset names to their values.
        """

        def _grab(name: str) -> Any:
            ds = grp[name]
            if ds.dtype.kind == "O":
                return ds.asstr()[...]
            if ds.shape == ():
                return ds[()]
            return ds[...]

        return {name: _grab(name) for name in CompoundH5Reader.DATASETS}

    def close(self) -> None:
        """
        Close the HDF5 file.

        Call this when done reading to release file resources.
        """
        self._file.close()

    def __enter__(self) -> "CompoundH5Reader":
        """
        Enable context manager support.

        Returns
        -------
        CompoundH5Reader
            Returns self for use in 'with' statements.
        """
        return self

    def __exit__(self, *exc: Any) -> None:
        """
        Clean up when exiting context manager.

        Parameters
        ----------
        *exc : Any
            Exception information if an error occurred.
        """
        self.close()
