import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat
from multiprocessing import Pool, cpu_count
import logging
import sys
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get the directory where this file is located
current_file_dir = Path(__file__).resolve().parent.parent

# Add the project root to Python path to enable absolute imports
# We need to go up one more level to get to MetalTransferLearning/
project_root = current_file_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.data_src.compound_h5_reader import CompoundH5Reader


def load_data_hdf5(
    dataset_name: str,
    inhibitor_type: str,
    target_type: str,
    percent_of_data: float = 1.0,
) -> dict:  # Return type is a single dictionary now
    """
    Load the data for the given dataset from HDF5 files using efficient single-pass loading.
    Can load 'pk', 'pic50', or a combined 'pk+pic50' dataset, ensuring no duplicates.

    Parameters
    ----------
    dataset_name : str
        The type of data (e.g., 'caged', 'organic').
    inhibitor_type : str
        The type of inhibitor descriptor (e.g., 'MACE').
    target_type : str
        The type of affinity ('pic50', 'pk', or 'pk+pic50').
    percent_of_data : float
        The percentage of data to load.
    Returns
    -------
    dict
        A dictionary containing the inhibitor features, protein/pocket embeddings, and binding affinities.
    """
    if target_type == "pic50+pk" and dataset_name == "organic":
        logger.info("Loading combined 'pic50' and 'pk' datasets for pre-training.")

        # Load pic50 data
        pic50_data = load_data_hdf5(
            dataset_name, inhibitor_type, "pic50", percent_of_data
        )

        # Load pk data
        pk_data = load_data_hdf5(dataset_name, inhibitor_type, "pk", percent_of_data)

        # # Merge the datasets, automatically handling duplicates.
        # # If a key exists in both, the value from pk_data will be kept.
        # combined_inhibitor_feats = {**pic50_data['inhibitor_feats'], **pk_data['inhibitor_feats']}
        # combined_protein_feats = {**pic50_data['protein_feats'], **pk_data['protein_feats']}
        # combined_pocket_feats = {**pic50_data['pocket_feats'], **pk_data['pocket_feats']}
        # combined_binding_affs = {**pic50_data['binding_affs'], **pk_data['binding_affs']}

        # logger.info(f"Total unique data points after merging 'pk' and 'pic50': {len(combined_inhibitor_feats)}")

        if percent_of_data < 1.0:
            pic50_keys = np.array(list(pic50_data["inhibitor_feats"].keys()))
            sample_size = int(len(pic50_keys) * percent_of_data)
            pic50_random_keys = np.random.choice(
                pic50_keys, size=sample_size, replace=False
            )
            pic50_data["inhibitor_feats"] = {
                k: v
                for k, v in pic50_data["inhibitor_feats"].items()
                if k in pic50_random_keys
            }
            pic50_data["protein_feats"] = {
                k: v
                for k, v in pic50_data["protein_feats"].items()
                if k in pic50_random_keys
            }
            pic50_data["pocket_feats"] = {
                k: v
                for k, v in pic50_data["pocket_feats"].items()
                if k in pic50_random_keys
            }
            pic50_data["binding_affs"] = {
                k: v
                for k, v in pic50_data["binding_affs"].items()
                if k in pic50_random_keys
            }

            logger.info(
                f"Sampled {len(pic50_data['inhibitor_feats'])} data points ({percent_of_data*100}%)"
            )

            pk_keys = np.array(list(pk_data["inhibitor_feats"].keys()))
            sample_size = int(len(pk_keys) * percent_of_data)
            pk_random_keys = np.random.choice(pk_keys, size=sample_size, replace=False)
            pk_data["inhibitor_feats"] = {
                k: v
                for k, v in pk_data["inhibitor_feats"].items()
                if k in pk_random_keys
            }
            pk_data["protein_feats"] = {
                k: v for k, v in pk_data["protein_feats"].items() if k in pk_random_keys
            }
            pk_data["pocket_feats"] = {
                k: v for k, v in pk_data["pocket_feats"].items() if k in pk_random_keys
            }
            pk_data["binding_affs"] = {
                k: v for k, v in pk_data["binding_affs"].items() if k in pk_random_keys
            }

            logger.info(
                f"Sampled {len(pk_data['inhibitor_feats'])} data points ({percent_of_data*100}%)"
            )

        return pic50_data, pk_data

    # Get the directory where this file is located
    current_file_dir = Path(__file__).resolve().parent
    # Navigate to the project root and then to the data directory
    data_dir = current_file_dir.parent.parent.parent / "data"
    if dataset_name == "organic":
        h5_path = (
            Path(f"{data_dir}/hf5_dataset/organic")
            / f"{dataset_name}_{target_type}_dataset.h5"
        )
    elif dataset_name == "caged":
        h5_dir = Path(
            f"{data_dir}/hf5_dataset/caged"
        )
        h5_path = h5_dir / f"{dataset_name}_{target_type}_dataset.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")

    logger.info(f"Loading data for {dataset_name} {target_type} from {h5_path}")

    (
        inhibitor_feats,
        protein_feats,
        pocket_feats,
        binding_affs,
    ) = load_data_parallel(h5_path, inhibitor_type)

    logger.info(f"Loaded {len(inhibitor_feats)} inhibitor features")
    logger.info(f"Loaded {len(protein_feats)} protein embeddings")
    logger.info(f"Loaded {len(pocket_feats)} pocket embeddings")
    logger.info(f"Loaded {len(binding_affs)} binding affinities")

    loaded_data = {
        "inhibitor_feats": inhibitor_feats,
        "protein_feats": protein_feats,
        "pocket_feats": pocket_feats,
        "binding_affs": binding_affs,
    }

    if percent_of_data < 1.0:
        random_keys = random.sample(
            list(loaded_data["inhibitor_feats"].keys()),
            int(len(loaded_data["inhibitor_feats"]) * percent_of_data),
        )
        loaded_data["inhibitor_feats"] = {
            k: v for k, v in loaded_data["inhibitor_feats"].items() if k in random_keys
        }
        loaded_data["protein_feats"] = {
            k: v for k, v in loaded_data["protein_feats"].items() if k in random_keys
        }
        loaded_data["pocket_feats"] = {
            k: v for k, v in loaded_data["pocket_feats"].items() if k in random_keys
        }
        loaded_data["binding_affs"] = {
            k: v for k, v in loaded_data["binding_affs"].items() if k in random_keys
        }

        logger.info(
            f"Loaded {len(loaded_data['inhibitor_feats'])} items after random sampling {percent_of_data*100}%"
        )

    return loaded_data


def process_chunk(
    h5_path: Path,
    keys_chunk: list[str],
    inhibitor_type: str,
) -> tuple[dict, dict, dict, dict]:
    """
    Worker function to process a chunk of keys.
    Each worker opens its own file handle.

    Parameters
    ----------
    h5_path : Path
        The path to the HDF5 file.
    keys_chunk : list[str]
        A list of keys (compound identifiers) to process.
    inhibitor_type : str
        The type of inhibitor descriptor (e.g., 'MACE').

    Returns
    -------
    tuple[dict, dict, dict, dict]
        A tuple containing:
        - inhibitor_feats: Dictionary mapping keys to inhibitor feature arrays
        - protein_feats: Dictionary mapping keys to protein embedding arrays
        - pocket_feats: Dictionary mapping keys to pocket embedding arrays
        - binding_affs: Dictionary mapping keys to binding affinity values
    """
    inhibitor_feats = {}
    protein_feats = {}
    pocket_feats = {}
    binding_affs = {}

    with h5py.File(h5_path, "r") as h5_file:
        for key in keys_chunk:
            h5_group = h5_file[key]

            # Check what datasets are actually available
            required_datasets = [
                f"mol_{inhibitor_type.lower()}_embedding",
                "protein_embeddings",
                "pocket_embeddings",
                "target_value",
            ]

            # 2. Check if the target value is valid.
            try:
                val = h5_group["target_value"][()]
                if pd.isna(val) or np.isnan(val):
                    continue
            except Exception:
                continue  # Skip if target can't be read

            # 3. All checks passed, now read the data into memory.
            inhibitor_feats[key] = h5_group[f"mol_{inhibitor_type.lower()}_embedding"][
                ...
            ]

            protein_feats[key] = h5_group["protein_embeddings"][...]
            pocket_feats[key] = h5_group["pocket_embeddings"][...]
            binding_affs[key] = float(val)
    return (
        inhibitor_feats,
        protein_feats,
        pocket_feats,
        binding_affs,
    )


def load_data_parallel(
    h5_path: Path, inhibitor_type: str
) -> tuple[dict, dict, dict, dict]:
    """
    Loads data from the HDF5 file in parallel using multiprocessing.

    Parameters
    ----------
    h5_path : Path
        The path to the HDF5 file.
    inhibitor_type : str
        The type of inhibitor descriptor (e.g., 'MACE').

    Returns
    -------
    tuple[dict, dict, dict, dict]
        A tuple containing:
        - inhibitor_feats: Dictionary mapping compound names to inhibitor feature arrays
        - protein_feats: Dictionary mapping compound names to protein embedding arrays
        - pocket_feats: Dictionary mapping compound names to pocket embedding arrays
        - binding_affs: Dictionary mapping compound names to binding affinity values
    """
    # Use the reader just to get the sorted list of keys
    with CompoundH5Reader(h5_path) as reader:
        keys = reader._keys

    # Determine the number of processes and chunk the keys
    num_processes = cpu_count()
    size = len(keys)
    chunk_size = int(np.ceil(size / num_processes))  # 100000
    key_chunks = [keys[i : i + chunk_size] for i in range(0, size, chunk_size)]

    # --- Parallel Processing ---
    with Pool(processes=num_processes) as pool:
        # Use starmap to pass multiple arguments to the worker function
        results = pool.starmap(
            process_chunk,
            zip(
                repeat(h5_path),
                key_chunks,
                repeat(inhibitor_type),
            ),
        )

    # --- Merge results from all workers ---
    inhibitor_feats = {}
    protein_feats = {}
    pocket_feats = {}
    binding_affs = {}
    for inhib, protein, pocket, bind in results:
        inhibitor_feats.update(inhib)
        protein_feats.update(protein)
        pocket_feats.update(pocket)
        binding_affs.update(bind)

    # Get common keys across all dictionaries
    common_keys = (
        set(inhibitor_feats.keys())
        & set(binding_affs.keys())
        & set(pocket_feats.keys())
        & set(protein_feats.keys())
    )
    common_keys = sorted(list(common_keys))

    # Re-create final dictionaries with sorted keys if order is important
    inhibitor_feats = {k: inhibitor_feats[k] for k in common_keys}
    pocket_feats = {k: pocket_feats[k] for k in common_keys}
    protein_feats = {k: protein_feats[k] for k in common_keys}
    binding_affs = {k: binding_affs[k] for k in common_keys}

    return (
        inhibitor_feats,
        protein_feats,
        pocket_feats,
        binding_affs,
    )
