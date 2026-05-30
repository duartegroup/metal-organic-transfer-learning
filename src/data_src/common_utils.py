import pickle
import re
import pandas as pd
import numpy as np
import os
import h5py
from typing import Tuple, List, Dict, Any, Set, Optional
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_names(csv_file: str, data_type: str) -> List[str]:
    """
    Create a list of names for the dataset.

    Parameters
    ----------
    csv_file : str
        Path to the csv file
    data_type : str
        Type of data ('organic' or 'caged')

    Returns
    -------
    names : List[str]
        List of names for the data
    """
    df = pd.read_csv(csv_file)

    if data_type == "organic":
        pdb_id = df["PDBID"].tolist()
        ligand_name = df["Ligand Name"].tolist()
        chain_id = df["Ligand Chain"].tolist()
        ligand_residue_number = df["Ligand Residue Number"].tolist()
        names = [
            f"{pdb_id[i]}_{ligand_name[i]}_{chain_id[i]}_{ligand_residue_number[i]}"
            for i in range(len(pdb_id))
        ]
    elif data_type == "caged":
        protein_name = df["protein"].tolist()
        protein_name = [f"{ent}-{idx}" for idx, ent in enumerate(protein_name)]
        compound_n = df["compound_n"].tolist()
        names = [
            f"compound_{compound_n[i]}_{protein_name[i]}"
            for i in range(len(protein_name))
        ]
    else:
        raise ValueError(
            f"Unknown data_type: {data_type}. Must be 'organic' or 'caged'."
        )

    return names


def load_coordinate_data_from_h5(h5_path: str, names: List[str]) -> Dict[str, Dict]:
    """
    Load coordinate data from HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file
    names : List[str]
        List of names to load

    Returns
    -------
    coordinate_data : Dict[str, Dict]
        Dictionary mapping keys to coordinate data with 'coords' and 'atype'
    """
    logger.info(f"Loading coordinate data from {h5_path}")
    logger.info(f"Looking for {len(names)} specific entries")

    coordinate_data = {}
    names_set = set(names)
    found_names = set()

    if not os.path.exists(h5_path):
        logger.error(f"H5 file {h5_path} not found")
        return coordinate_data

    with h5py.File(h5_path, "r") as h5_file:
        # Only iterate through the requested names
        for group_name in names:
            if group_name in h5_file and isinstance(h5_file[group_name], h5py.Group):
                group = h5_file[group_name]

                # Check if required datasets exist
                if "mol_gfn2_xyz" in group and "mol_gfn2_atype" in group:
                    # Read the coordinate and atom type data
                    coords = group["mol_gfn2_xyz"][()]
                    atype_data = group["mol_gfn2_atype"][()]

                    # Decode atom types from bytes to strings
                    atype = [atom.decode("utf-8") for atom in atype_data]

                    # Store in the expected format
                    coordinate_data[group_name] = {
                        "coords": coords.tolist(),
                        "atype": atype,
                    }
                    found_names.add(group_name)
                else:
                    logger.warning(f"Missing gfn2 data in group {group_name}")
            elif group_name not in h5_file:
                logger.debug(f"Group {group_name} not found in HDF5 file")

    # Report which names were not found
    missing_names = names_set - found_names
    if missing_names:
        logger.warning(
            f"{len(missing_names)} requested entries not found: {sorted(list(missing_names)[:5])}{'...' if len(missing_names) > 5 else ''}"
        )

    logger.info(f"Loaded {len(coordinate_data)} coordinate entries from HDF5 file")
    return coordinate_data


def load_sequence_data_from_h5(
    sequence_path: str, names: List[str]
) -> List[Tuple[str, str]]:
    """
    Load protein sequence data from an HDF5 file.

    Parameters
    ----------
    sequence_path : str
        The path to the HDF5 file containing the protein sequences.
    names : List[str]
        A list of names for the protein sequences to load.

    Returns
    -------
    sequence_list : List[Tuple[str, str]]
        A list of tuples, where each tuple contains a PDB ID and a protein sequence.
    """
    logger.info(f"Loading sequence data from {sequence_path}")
    logger.info(f"Looking for {len(names)} specific sequences")

    sequence_list = []
    names_set = set(names)  # Convert to set for faster lookup
    found_names = set()

    with h5py.File(sequence_path, "r") as h5_file:
        # Only iterate through the requested names
        for group_name in names:
            if group_name in h5_file and isinstance(h5_file[group_name], h5py.Group):
                # Extract protein sequence from the group
                group = h5_file[group_name]
                if "protein_sequence" in group:
                    # Read the protein sequence dataset
                    protein_seq = group["protein_sequence"][()].decode("utf-8")
                    sequence_list.append((group_name, protein_seq))
                    found_names.add(group_name)
                else:
                    logger.warning(f"No protein_sequence found in group {group_name}")
            elif group_name not in h5_file:
                logger.debug(f"Group {group_name} not found in HDF5 file")

    # Report which names were not found
    missing_names = names_set - found_names
    if missing_names:
        logger.warning(
            f"{len(missing_names)} requested sequences not found: {sorted(missing_names)}"
        )

    logger.info(f"Loaded {len(sequence_list)} protein sequences from HDF5 file")
    return sequence_list


def extract_organic_csv(csv_file: str) -> pd.DataFrame:
    """
    Extract the organic data from the csv file.

    Parameters
    ----------
    csv_file : str
        Path to the csv file

    Returns
    -------
    organic_df : pd.DataFrame
        DataFrame containing the organic data
    """
    # import the csv file
    organic_df = pd.read_csv(csv_file)

    # set all columns to lower case
    organic_df.columns = organic_df.columns.str.lower()

    # Drop rows with missing sequence
    organic_df = organic_df.dropna(subset=["seq"])

    return organic_df


def extract_caged_csv(csv_file: str) -> Tuple[List[str], List[str]]:
    """
    Extract the pdb names and ligand names from the csv file.

    Parameters
    ----------
    csv_file : str
        Path to the csv file

    Returns
    -------
    pdb_names : List[str]
        List of pdb names
    ligand_ids : List[str]
        List of ligand ids
    """
    # import the csv file
    caged_df = pd.read_csv(csv_file)

    # set all columns to lower case
    caged_df.columns = caged_df.columns.str.lower()

    caged_df = caged_df.dropna(subset=["protein_seq"])

    pdb_names = caged_df["protein"].values
    pdb_names = [f"{ent}-{idx}" for idx, ent in enumerate(pdb_names)]

    # remove \xa0 from the protein name
    pdb_names = [re.sub(r"\xa0", "", ent) for ent in pdb_names]

    ligand_id = caged_df["compound_n"].values
    ligand_ids = [f"compound_{ent}" for ent in ligand_id]

    return pdb_names, ligand_ids


def get_all_atom_types(coordinate_data: Dict[str, Dict]) -> Set[str]:
    """
    Get all unique atom types from the coordinate data.

    Parameters
    ----------
    coordinate_data : Dict[str, Dict]
        Dictionary mapping keys to coordinate data

    Returns
    -------
    atom_types : Set[str]
        Set of unique atom types
    """
    atom_types = set()
    for key, data in coordinate_data.items():
        atom_types.update(data["atype"])
    return atom_types


def get_max_atoms(coordinate_data: Dict[str, Dict]) -> int:
    """
    Get the maximum number of atoms in any molecule from the coordinate data.

    Parameters
    ----------
    coordinate_data : Dict[str, Dict]
        Dictionary mapping keys to coordinate data

    Returns
    -------
    max_atoms : int
        Maximum number of atoms in any molecule
    """
    max_atoms = 0
    for key, data in coordinate_data.items():
        n_atoms = len(data["atype"])
        if n_atoms > max_atoms:
            max_atoms = n_atoms
    return max_atoms


def extract_target_vals_organic(csv_file: str, hf5_file: str) -> Dict[str, float]:
    """
    Extract target values for organic dataset ensuring CSV ordering is preserved.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing organic compound data.
    hf5_file : str
        Path to the HDF5 file containing molecular data.

    Returns
    -------
    dict[str, float]
        Dictionary mapping compound names to binding affinity values.
    """
    # Extract from CSV (maintains original order)
    organic_df = pd.read_csv(csv_file)
    organic_df.columns = organic_df.columns.str.lower()

    # Create entry names from CSV
    csv_names = []
    for _, row in organic_df.iterrows():
        pdb_id = row["pdbid"]
        ligand_name = row["ligand name"]
        chain_id = row["ligand chain"]
        ligand_residue_number = row["ligand residue number"]
        key = f"{pdb_id}_{ligand_name}_{chain_id}_{ligand_residue_number}"
        csv_names.append(key)

    # Extract target values in CSV order
    tgt_vals = organic_df["log binding affinity"].values
    tgt_vals = [
        float(ent) * -1 for ent in tgt_vals
    ]  # convert to positive values HiQBind reports negative log values

    # Get HF5 names to determine which entries to keep
    with h5py.File(hf5_file, "r") as f:
        hf5_names = [name.decode("utf-8") for name in f["names"][:]]

    # Find common entries but maintain CSV ordering
    hf5_names_set = set(hf5_names)

    # Filter CSV data to only include entries that exist in HF5
    # CRUCIAL: This maintains CSV ordering while filtering
    filtered_names = []
    filtered_values = []

    for i, csv_name in enumerate(csv_names):
        if csv_name in hf5_names_set:
            filtered_names.append(csv_name)
            filtered_values.append(tgt_vals[i])

    # Create dictionary with filtered data in CSV order
    target_dict = {}
    for name, val in zip(filtered_names, filtered_values):
        target_dict[name] = val

    return target_dict


def extract_target_vals_caged(csv_file: str, hf5_file: str) -> Dict[str, float]:
    """
    Extract target values for caged dataset ensuring CSV ordering is preserved.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing caged compound data.
    hf5_file : str
        Path to the HDF5 file containing molecular data.

    Returns
    -------
    dict[str, float]
        Dictionary mapping compound names to binding affinity values.
    """
    # Extract from CSV (maintains original order)
    caged_df = pd.read_csv(csv_file)
    caged_df.columns = caged_df.columns.str.lower()

    # Create entry names from CSV
    csv_names = []
    for _, row in caged_df.iterrows():
        compound_id = row["compound_id"]
        target_name = row["target_name"]
        key = f"{compound_id}_{target_name}"
        csv_names.append(key)

    # Extract target values in CSV order
    tgt_vals = caged_df["pic50"].values
    tgt_vals = [float(ent) for ent in tgt_vals]

    # Get HF5 names to determine which entries to keep
    with h5py.File(hf5_file, "r") as f:
        hf5_names = [name.decode("utf-8") for name in f["names"][:]]

    # Find common entries but maintain CSV ordering
    hf5_names_set = set(hf5_names)

    # Filter CSV data to only include entries that exist in HF5
    # CRUCIAL: This maintains CSV ordering while filtering
    filtered_names = []
    filtered_values = []

    for i, csv_name in enumerate(csv_names):
        if csv_name in hf5_names_set:
            filtered_names.append(csv_name)
            filtered_values.append(tgt_vals[i])

    # Create dictionary with filtered data in CSV order
    target_dict = {}
    for name, val in zip(filtered_names, filtered_values):
        target_dict[name] = val

    return target_dict


def save_embeddings(
    embeddings_dict: Dict, output_path: str, average: bool = False
) -> None:
    """
    Save embeddings to a pickle file.

    Parameters
    ----------
    embeddings_dict : Dict
        Dictionary containing embeddings
    output_path : str
        Path to save the embeddings
    average : bool, optional
        Whether embeddings are averaged (for naming), by default False
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to pickle file
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_dict, f)

    logger.info(
        f"Saved {'averaged' if average else 'full'} embeddings to {output_path}"
    )
    logger.info(f"Number of entries: {len(embeddings_dict)}")


def process_molecules_parallel(
    coordinate_data: Dict[str, Dict],
    descriptor: Any,
    process_func: callable,
    num_processes: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Process molecules with a given descriptor in parallel.

    Parameters
    ----------
    coordinate_data : Dict[str, Dict]
        Dictionary mapping molecule names to coordinate data
    descriptor : Any
        The descriptor object to use for processing
    process_func : callable
        Function to process a single molecule (should take tuple of (name, data, descriptor))
    num_processes : Optional[int]
        Number of processes to use (default: CPU count - 1)

    Returns
    -------
    embeddings_dict : Dict[str, np.ndarray]
        Dictionary mapping molecule names to their descriptor embeddings
    """
    if num_processes is None:
        num_processes = mp.cpu_count() - 1

    logger.info(
        f"Processing {len(coordinate_data)} molecules with {num_processes} processes"
    )

    # Prepare data for multiprocessing
    items_to_process = [
        (name, data, descriptor) for name, data in coordinate_data.items()
    ]

    # Create a pool of workers
    embeddings_dict = {}
    with mp.Pool(processes=num_processes) as pool:
        # Process the data in parallel and track progress with tqdm
        results = list(
            tqdm(
                pool.imap(process_func, items_to_process),
                total=len(items_to_process),
                desc="Processing molecules",
            )
        )

    # Combine results
    for name, embedding in results:
        embeddings_dict[name] = embedding

    return embeddings_dict


def get_combined_atom_types_from_datasets(
    data_type: str, affinity_types: List[str], h5_dir: str
) -> Set[str]:
    """
    Get combined atom types across multiple datasets.

    Parameters
    ----------
    data_type : str
        Type of data ('organic', 'caged', or 'both')
    affinity_types : List[str]
        List of affinity types to process
    h5_path : str
        Path to the HDF5 file

    Returns
    -------
    combined_atom_types : Set[str]
        Set of all unique atom types across datasets
    """
    combined_atom_types = set()

    datasets_to_process = []
    if data_type in ["organic", "both"]:
        for affinity_type in affinity_types:
            datasets_to_process.append(("organic", affinity_type))
    if data_type in ["caged", "both"]:
        for affinity_type in affinity_types:
            datasets_to_process.append(("caged", affinity_type))

    for dtype, affinity_type in datasets_to_process:
        h5_path = f"{h5_dir}/{dtype}/{dtype}_{affinity_type}_dataset.h5"

        if os.path.exists(h5_path):
            logger.info(f"Scanning atom types from {h5_path}")
            with h5py.File(h5_path, "r") as h5_file:
                for group_name in h5_file.keys():
                    if isinstance(h5_file[group_name], h5py.Group):
                        group = h5_file[group_name]
                        if "mol_gfn2_atype" in group:
                            atype_data = group["mol_gfn2_atype"][()]
                            atom_types = [atom.decode("utf-8") for atom in atype_data]
                            combined_atom_types.update(atom_types)
        else:
            logger.warning(f"H5 file {h5_path} not found")

    return combined_atom_types


def get_max_atoms_across_datasets(
    data_type: str, affinity_types: List[str], h5_dir: str
) -> int:
    """
    Get the maximum number of atoms across multiple datasets.

    Parameters
    ----------
    data_type : str
        Type of data ('organic', 'caged', or 'both')
    affinity_types : List[str]
        List of affinity types to process
    h5_dir : str
        Directory containing HDF5 files

    Returns
    -------
    max_atoms : int
        Maximum number of atoms found across all datasets
    """
    max_atoms = 0

    datasets_to_process = []
    if data_type in ["organic", "both"]:
        for affinity_type in affinity_types:
            datasets_to_process.append(("organic", affinity_type))
    if data_type in ["caged", "both"]:
        for affinity_type in affinity_types:
            datasets_to_process.append(("caged", affinity_type))

    for dtype, affinity_type in datasets_to_process:
        h5_path = f"{h5_dir}/{dtype}/{dtype}_{affinity_type}_dataset.h5"

        if os.path.exists(h5_path):
            logger.info(f"Scanning max atoms from {h5_path}")
            with h5py.File(h5_path, "r") as h5_file:
                for group_name in h5_file.keys():
                    if isinstance(h5_file[group_name], h5py.Group):
                        group = h5_file[group_name]
                        if "mol_gfn2_atype" in group:
                            atype_data = group["mol_gfn2_atype"][()]
                            n_atoms = len(atype_data)
                            if n_atoms > max_atoms:
                                max_atoms = n_atoms
        else:
            logger.warning(f"H5 file {h5_path} not found")

    return max_atoms
