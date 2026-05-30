import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import logging

# RDKit imports for MCS analysis
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.append("./../../../src/transfer_learning_src/data")
from utils_data import load_data_hdf5
from split_data import load_fold_indices

excel_pic50_path: str = "./../../excel/caged/pic50/caged_pic50.csv"
excel_pk_path: str = "./../../excel/caged/pk/caged_pk.csv"


def SmilesMCStoGridImage(
    smiles, align_substructure: bool = True, verbose: bool = False, **kwargs
) -> object:
    """
    Convert a list or dictionary of SMILES strings to an RDKit grid image of the maximum common substructure (MCS).

    Parameters
    ----------
    smiles : list or dict
        The SMILES molecules to be compared and drawn. Can be a list of SMILES strings
        or a dictionary mapping SMILES strings to legend strings.
    align_substructure : bool, default=True
        Whether to align the MCS substructures when plotting the molecules.
    verbose : bool, default=False
        Whether to return verbose output (MCS SMARTS string and molecule, and list of molecules).
    **kwargs : dict
        Additional arguments passed to rdFMCS.FindMCS.

    Returns
    -------
    drawing : RDKit grid image
        Grid image of molecules with MCS highlighted.
    mcs_smarts : str (only if verbose=True)
        MCS SMARTS string.
    mcs_mol : RDKit molecule (only if verbose=True)
        MCS molecule.
    valid_mols : list of RDKit molecules (only if verbose=True)
        List of molecules for input SMILES strings.
    """
    # Handle both list and dict inputs
    if isinstance(smiles, dict):
        smiles_list = list(smiles.keys())
        mol_legends = list(smiles.values())
    else:
        smiles_list = smiles
        mol_legends = ["" for _ in smiles]

    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(smile, sanitize=False) for smile in smiles_list]

    # Filter out None molecules (invalid SMILES)
    valid_mols = [mol for mol in mols if mol is not None]
    if not valid_mols:
        logger.warning("No valid molecules found from SMILES")
        return None

    # Find MCS
    res = rdFMCS.FindMCS(valid_mols, **kwargs)
    mcs_smarts = res.smartsString
    mcs_mol = Chem.MolFromSmarts(res.smartsString)

    smarts_and_mols = [mcs_mol] + valid_mols
    smarts_legend = "Max. substructure match"
    legends = [smarts_legend] + mol_legends[: len(valid_mols)]

    matches = [""] + [mol.GetSubstructMatch(mcs_mol) for mol in valid_mols]

    subms = [
        x for x in smarts_and_mols if x is not None and x.HasSubstructMatch(mcs_mol)
    ]

    Chem.Compute2DCoords(mcs_mol)

    if align_substructure:
        for m in subms:
            _ = Chem.GenerateDepictionMatching2DStructure(m, mcs_mol)

    drawing = Draw.MolsToGridImage(
        smarts_and_mols, highlightAtomLists=matches, legends=legends
    )

    if verbose:
        return drawing, mcs_smarts, mcs_mol, valid_mols
    else:
        return drawing


def SdfMCStoGridImage(
    sdf_files, align_substructure: bool = True, verbose: bool = False, **kwargs
) -> object:
    """
    Convert a list or dictionary of SDF file paths to an RDKit grid image of the maximum common substructure (MCS).

    Parameters
    ----------
    sdf_files : list or dict
        The SDF file paths to be compared and drawn. Can be a list of SDF file path strings
        or a dictionary mapping SDF file path strings to legend strings.
    align_substructure : bool, default=True
        Whether to align the MCS substructures when plotting the molecules.
    verbose : bool, default=False
        Whether to return verbose output (MCS SMARTS string and molecule, and list of molecules).
    **kwargs : dict
        Additional arguments passed to rdFMCS.FindMCS.

    Returns
    -------
    drawing : RDKit grid image
        Grid image of molecules with MCS highlighted.
    mcs_smarts : str (only if verbose=True)
        MCS SMARTS string.
    mcs_mol : RDKit molecule (only if verbose=True)
        MCS molecule.
    valid_mols : list of RDKit molecules (only if verbose=True)
        List of molecules for input SDF files.
    """
    # Handle both list and dict inputs
    if isinstance(sdf_files, dict):
        sdf_file_list = list(sdf_files.keys())
        mol_legends = list(sdf_files.values())
    else:
        sdf_file_list = sdf_files
        mol_legends = ["" for _ in sdf_files]

    # Load molecules from SDF files
    mols = []
    valid_legends = []

    for i, sdf_file in enumerate(sdf_file_list):
        try:
            # Use RDKit to load the SDF file
            supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
            mol = next(iter(supplier), None)

            if mol is not None:
                mols.append(mol)
                valid_legends.append(mol_legends[i])
            else:
                logger.warning(f"Could not load molecule from SDF file: {sdf_file}")
        except Exception as e:
            logger.error(f"Error loading SDF file {sdf_file}: {e}")

    # Filter out None molecules
    valid_mols = [mol for mol in mols if mol is not None]
    if not valid_mols:
        logger.warning("No valid molecules found from SDF files")
        return None

    # Find MCS
    res = rdFMCS.FindMCS(valid_mols, **kwargs)
    mcs_smarts = res.smartsString
    mcs_mol = Chem.MolFromSmarts(res.smartsString)

    smarts_and_mols = [mcs_mol] + valid_mols
    smarts_legend = "Max. substructure match"
    legends = [smarts_legend] + valid_legends[: len(valid_mols)]

    matches = [""] + [mol.GetSubstructMatch(mcs_mol) for mol in valid_mols]

    subms = [
        x for x in smarts_and_mols if x is not None and x.HasSubstructMatch(mcs_mol)
    ]

    Chem.Compute2DCoords(mcs_mol)

    if align_substructure:
        for m in subms:
            _ = Chem.GenerateDepictionMatching2DStructure(m, mcs_mol)

    drawing = Draw.MolsToGridImage(
        smarts_and_mols, highlightAtomLists=matches, legends=legends
    )

    if verbose:
        return drawing, mcs_smarts, mcs_mol, valid_mols
    else:
        return drawing


def extract_caged_csv_with_smiles_and_proteins(
    csv_file: str,
    affinity_type: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Extract data from caged CSV file with metal SMILES and protein information.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file.
    affinity_type : str
        Type of affinity data ('pic50' or 'pk').

    Returns
    -------
    Tuple[Dict[str, str], Dict[str, str]]
        A tuple containing:
        - Dictionary mapping compound names to metal SMILES
        - Dictionary mapping compound names to protein names
    """
    caged_df = pd.read_csv(csv_file)
    caged_df.columns = caged_df.columns.str.lower()

    # Drop rows with missing protein sequence (same as in HDF5 creation)
    caged_df = caged_df.dropna(subset=["protein_seq"])

    # Generate names exactly like in the HDF5 creation script
    protein_names = [
        f"{name}-{idx}" for idx, name in enumerate(caged_df["protein"].values)
    ]
    # Clean protein names (remove non-breaking spaces)
    protein_names = [re.sub(r"\xa0", "", name) for name in protein_names]

    compound_numbers = caged_df["compound_n"].values.tolist()

    # Create the final names that match HDF5 format
    names = []
    for protein_name, compound_n in zip(protein_names, compound_numbers):
        names.append(f"compound_{compound_n}_{protein_name}")

    # Extract the metal_smiles
    metal_smiles = caged_df["metal_smiles"].values.tolist()

    # Create a key-value pair of names and metal_smiles
    names_metal_smiles = dict(zip(names, metal_smiles))

    # Create a key-value pair of names and protein names (without compound info)
    # Extract base protein name from the indexed protein names
    base_protein_names = [name.split("-")[0] for name in protein_names]
    names_proteins = dict(zip(names, base_protein_names))

    return names_metal_smiles, names_proteins


def extract_caged_csv_with_sdf_paths_and_proteins(
    csv_file: str,
    affinity_type: str,
    compound_base_dir: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Extract data from caged CSV file with SDF file paths and protein information.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file.
    affinity_type : str
        Type of affinity data ('pic50' or 'pk').
    compound_base_dir : str
        Base directory containing compound SDF files.

    Returns
    -------
    Tuple[Dict[str, str], Dict[str, str]]
        A tuple containing:
        - Dictionary mapping compound names to SDF file paths
        - Dictionary mapping compound names to protein names
    """
    caged_df = pd.read_csv(csv_file)
    caged_df.columns = caged_df.columns.str.lower()

    # Drop rows with missing protein sequence (same as in HDF5 creation)
    caged_df = caged_df.dropna(subset=["protein_seq"])

    # Generate names exactly like in the HDF5 creation script
    protein_names = [
        f"{name}-{idx}" for idx, name in enumerate(caged_df["protein"].values)
    ]
    # Clean protein names (remove non-breaking spaces)
    protein_names = [re.sub(r"\xa0", "", name) for name in protein_names]

    compound_numbers = caged_df["compound_n"].values.tolist()

    # Create the final names that match HDF5 format
    names = []
    sdf_paths = []
    for protein_name, compound_n in zip(protein_names, compound_numbers):
        names.append(f"compound_{compound_n}_{protein_name}")
        # Create the SDF file path
        sdf_path = (
            Path(compound_base_dir)
            / f"compound_{compound_n}"
            / f"compound_{compound_n}.sdf"
        )
        sdf_paths.append(str(sdf_path))

    # Create a key-value pair of names and SDF file paths
    names_sdf_paths = dict(zip(names, sdf_paths))

    # Create a key-value pair of names and protein names (without compound info)
    # Extract base protein name from the indexed protein names
    base_protein_names = [name.split("-")[0] for name in protein_names]
    names_proteins = dict(zip(names, base_protein_names))

    return names_sdf_paths, names_proteins


def extract_caged_csv_with_smiles(
    csv_file: str,
    affinity_type: str,
) -> Dict[str, str]:
    """
    Extract data from caged CSV file with metal SMILES.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file.
    affinity_type : str
        Type of affinity data ('pic50' or 'pk').

    Returns
    -------
    Dict[str, str]
        Dictionary mapping compound names to metal SMILES.
    """
    names_metal_smiles, _ = extract_caged_csv_with_smiles_and_proteins(
        csv_file, affinity_type
    )
    return names_metal_smiles


def split_metal_smiles_and_proteins_by_fold(
    dataset: str,
    affinity_type: str,
    csv_file: str,
    fold: int = 1,
    n_folds: int = 4,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Split metal SMILES and protein data by fold groups using pre-computed fold indices.

    Parameters
    ----------
    dataset : str
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str
        The type of affinity ('pic50' or 'pk').
    csv_file : str
        Path to the CSV file containing the data.
    fold : int, default=1
        The fold number (1-indexed).
    n_folds : int, default=4
        The total number of folds.

    Returns
    -------
    Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]
        A tuple containing:
        - train_smiles: Train data metal SMILES mapping (name -> SMILES)
        - test_smiles: Test data metal SMILES mapping (name -> SMILES)
        - train_proteins: Train data protein mapping (name -> protein)
        - test_proteins: Test data protein mapping (name -> protein)
    """
    # Load the pre-computed fold indices
    fold_indices = load_fold_indices(dataset, affinity_type, n_folds)

    # Convert from 1-indexed to 0-indexed for accessing fold_indices
    fold_idx = fold - 1

    if fold_idx not in fold_indices:
        raise ValueError(
            f"fold {fold} (index {fold_idx}) not found in fold_indices. Available: {list(fold_indices.keys())}"
        )

    # Get the train and test names for this fold
    train_names = fold_indices[fold_idx]["train_names"]
    test_names = fold_indices[fold_idx]["test_names"]

    # Extract metal SMILES and protein data from CSV
    names_metal_smiles, names_proteins = extract_caged_csv_with_smiles_and_proteins(
        csv_file, affinity_type
    )

    # Split SMILES data by fold
    train_smiles = {
        name: names_metal_smiles[name]
        for name in train_names
        if name in names_metal_smiles
    }

    test_smiles = {
        name: names_metal_smiles[name]
        for name in test_names
        if name in names_metal_smiles
    }

    # Split protein data by fold
    train_proteins = {
        name: names_proteins[name] for name in train_names if name in names_proteins
    }

    test_proteins = {
        name: names_proteins[name] for name in test_names if name in names_proteins
    }

    return train_smiles, test_smiles, train_proteins, test_proteins


def split_sdf_paths_and_proteins_by_fold(
    dataset: str,
    affinity_type: str,
    csv_file: str,
    compound_base_dir: str,
    fold: int = 1,
    n_folds: int = 5,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Split SDF file paths and protein data by fold groups using pre-computed fold indices.

    Parameters
    ----------
    dataset : str
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str
        The type of affinity ('pic50' or 'pk').
    csv_file : str
        Path to the CSV file containing the data.
    compound_base_dir : str
        Base directory containing compound SDF files.
    fold : int, default=1
        The fold number (1-indexed).
    n_folds : int, default=5
        The total number of folds.

    Returns
    -------
    Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]
        A tuple containing:
        - train_sdf_paths: Train data SDF file paths mapping (name -> SDF path)
        - test_sdf_paths: Test data SDF file paths mapping (name -> SDF path)
        - train_proteins: Train data protein mapping (name -> protein)
        - test_proteins: Test data protein mapping (name -> protein)
    """
    # Load the pre-computed fold indices
    fold_indices = load_fold_indices(dataset, affinity_type, n_folds)

    # Convert from 1-indexed to 0-indexed for accessing fold_indices
    fold_idx = fold - 1

    if fold_idx not in fold_indices:
        raise ValueError(
            f"fold {fold} (index {fold_idx}) not found in fold_indices. Available: {list(fold_indices.keys())}"
        )

    # Get the train and test names for this fold
    train_names = fold_indices[fold_idx]["train_names"]
    test_names = fold_indices[fold_idx]["test_names"]

    # Extract SDF paths and protein data from CSV
    names_sdf_paths, names_proteins = extract_caged_csv_with_sdf_paths_and_proteins(
        csv_file, affinity_type, compound_base_dir
    )

    # Split SDF paths data by fold
    train_sdf_paths = {
        name: names_sdf_paths[name] for name in train_names if name in names_sdf_paths
    }

    test_sdf_paths = {
        name: names_sdf_paths[name] for name in test_names if name in names_sdf_paths
    }

    # Split protein data by fold
    train_proteins = {
        name: names_proteins[name] for name in train_names if name in names_proteins
    }

    test_proteins = {
        name: names_proteins[name] for name in test_names if name in names_proteins
    }

    return train_sdf_paths, test_sdf_paths, train_proteins, test_proteins


def split_metal_smiles_by_fold(
    dataset: str,
    affinity_type: str,
    csv_file: str,
    fold: int = 1,
    n_folds: int = 4,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Split metal SMILES data by fold groups using pre-computed fold indices.

    Parameters
    ----------
    dataset : str
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str
        The type of affinity ('pic50' or 'pk').
    csv_file : str
        Path to the CSV file containing the data.
    fold : int, default=1
        The fold number (1-indexed).
    n_folds : int, default=4
        The total number of folds.

    Returns
    -------
    Tuple[Dict[str, str], Dict[str, str]]
        A tuple containing:
        - train_smiles: Train data metal SMILES mapping
        - test_smiles: Test data metal SMILES mapping
    """
    train_smiles, test_smiles, _, _ = split_metal_smiles_and_proteins_by_fold(
        dataset, affinity_type, csv_file, fold, n_folds
    )
    return train_smiles, test_smiles


def split_data_by_group_with_smiles(
    inhibitor_feats: dict,
    protein_feats: dict,
    pocket_feats: dict,
    binding_affs: dict,
    dataset: str,
    affinity_type: str,
    csv_file: str,
    fold: int = 1,
    n_folds: int = 4,
) -> Tuple[Dict, Dict]:
    """
    Split the data using pre-computed fold indices, including metal SMILES.

    Parameters
    ----------
    inhibitor_feats : dict
        The inhibitor features.
    pocket_feats : dict
        The pocket embeddings.
    protein_feats : dict
        The protein embeddings.
    binding_affs : dict
        The binding affinities.
    dataset : str
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str
        The type of affinity (e.g., 'pic50', 'pk').
    csv_file : str
        Path to the CSV file containing the data.
    fold : int
        The split to return (1-indexed).
    n_folds : int
        The number of folds.

    Returns
    -------
    train_data : Dict
        The train data including metal SMILES.
    test_data : Dict
        The test data including metal SMILES.
    """
    # Load the pre-computed fold indices
    fold_indices = load_fold_indices(dataset, affinity_type, n_folds)

    # Convert from 1-indexed to 0-indexed for accessing fold_indices
    fold_idx = fold - 1

    if fold_idx not in fold_indices:
        raise ValueError(
            f"fold {fold} (index {fold_idx}) not found in fold_indices. Available: {list(fold_indices.keys())}"
        )

    # Get the train and test names for this fold
    train_names = fold_indices[fold_idx]["train_names"]
    test_names = fold_indices[fold_idx]["test_names"]

    # Extract regular data for this fold
    train_inhibitor_features = {
        name: inhibitor_feats[name] for name in train_names if name in inhibitor_feats
    }
    train_protein_embeddings = {
        name: protein_feats[name] for name in train_names if name in protein_feats
    }
    train_pocket_embeddings = {
        name: pocket_feats[name] for name in train_names if name in pocket_feats
    }
    train_binding_affs = {
        name: binding_affs[name] for name in train_names if name in binding_affs
    }

    test_inhibitor_features = {
        name: inhibitor_feats[name] for name in test_names if name in inhibitor_feats
    }
    test_protein_embeddings = {
        name: protein_feats[name] for name in test_names if name in protein_feats
    }
    test_pocket_embeddings = {
        name: pocket_feats[name] for name in test_names if name in pocket_feats
    }
    test_binding_affs = {
        name: binding_affs[name] for name in test_names if name in binding_affs
    }

    # Extract metal SMILES data
    train_smiles, test_smiles = split_metal_smiles_by_fold(
        dataset, affinity_type, csv_file, fold, n_folds
    )

    return (
        {
            "inhibitor_feats": train_inhibitor_features,
            "protein_feats": train_protein_embeddings,
            "pocket_feats": train_pocket_embeddings,
            "binding_affs": train_binding_affs,
            "metal_smiles": train_smiles,
        },
        {
            "inhibitor_feats": test_inhibitor_features,
            "protein_feats": test_protein_embeddings,
            "pocket_feats": test_pocket_embeddings,
            "binding_affs": test_binding_affs,
            "metal_smiles": test_smiles,
        },
    )


def analyze_metal_smiles_distribution(
    dataset: str,
    affinity_type: str,
    csv_file: str,
    n_folds: int = 4,
) -> Dict:
    """
    Analyze the distribution of unique metal SMILES across all folds.

    Parameters
    ----------
    dataset : str
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str
        The type of affinity ('pic50' or 'pk').
    csv_file : str
        Path to the CSV file containing the data.
    n_folds : int, default=4
        The number of folds.

    Returns
    -------
    Dict
        Dictionary containing analysis results for each fold.
    """
    analysis = {}

    for fold in range(1, n_folds + 1):
        train_smiles, test_smiles = split_metal_smiles_by_fold(
            dataset, affinity_type, csv_file, fold, n_folds
        )

        # Get unique SMILES
        unique_train_smiles = set(train_smiles.values())
        unique_test_smiles = set(test_smiles.values())

        # Analyze overlap
        overlap = unique_train_smiles & unique_test_smiles
        train_only = unique_train_smiles - unique_test_smiles
        test_only = unique_test_smiles - unique_train_smiles

        analysis[f"fold_{fold}"] = {
            "train_total_compounds": len(train_smiles),
            "test_total_compounds": len(test_smiles),
            "train_unique_smiles": len(unique_train_smiles),
            "test_unique_smiles": len(unique_test_smiles),
            "overlapping_smiles": len(overlap),
            "train_only_smiles": len(train_only),
            "test_only_smiles": len(test_only),
            "unique_train_smiles_list": list(unique_train_smiles),
            "unique_test_smiles_list": list(unique_test_smiles),
            "overlapping_smiles_list": list(overlap),
        }

    return analysis


def analyze_metal_smiles_and_proteins_distribution_with_mcs(
    dataset: str,
    affinity_type: str,
    csv_file: str,
    n_folds: int = 5,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Analyze the distribution of unique metal SMILES and proteins across all folds and generate MCS images.

    Parameters
    ----------
    dataset : str
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str
        The type of affinity ('pic50' or 'pk').
    csv_file : str
        Path to the CSV file containing the data.
    n_folds : int, default=5
        The number of folds.
    output_dir : Optional[str], default=None
        Directory to save MCS images. If None, images won't be saved.

    Returns
    -------
    Dict
        Dictionary containing analysis results for each fold.
    """
    analysis = {}

    # Create output directory if specified - include affinity type in the name
    if output_dir:
        output_path = Path(f"{output_dir}_{affinity_type}")
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_path}")

    for fold in range(1, n_folds + 1):
        train_smiles, test_smiles, train_proteins, test_proteins = (
            split_metal_smiles_and_proteins_by_fold(
                dataset, affinity_type, csv_file, fold, n_folds
            )
        )

        # Get unique SMILES
        unique_train_smiles = list(set(train_smiles.values()))
        unique_test_smiles = list(set(test_smiles.values()))

        # Get unique proteins
        unique_train_proteins = set(train_proteins.values())
        unique_test_proteins = set(test_proteins.values())

        # Analyze SMILES overlap
        smiles_overlap = set(train_smiles.values()) & set(test_smiles.values())
        smiles_train_only = set(train_smiles.values()) - set(test_smiles.values())
        smiles_test_only = set(test_smiles.values()) - set(train_smiles.values())

        # Analyze protein overlap
        protein_overlap = unique_train_proteins & unique_test_proteins
        protein_train_only = unique_train_proteins - unique_test_proteins
        protein_test_only = unique_test_proteins - unique_train_proteins

        fold_analysis = {
            # SMILES analysis
            "train_total_compounds": len(train_smiles),
            "test_total_compounds": len(test_smiles),
            "train_unique_smiles": len(unique_train_smiles),
            "test_unique_smiles": len(unique_test_smiles),
            "overlapping_smiles": len(smiles_overlap),
            "train_only_smiles": len(smiles_train_only),
            "test_only_smiles": len(smiles_test_only),
            "unique_train_smiles_list": unique_train_smiles,
            "unique_test_smiles_list": unique_test_smiles,
            "overlapping_smiles_list": list(smiles_overlap),
            # Protein analysis
            "train_unique_proteins": len(unique_train_proteins),
            "test_unique_proteins": len(unique_test_proteins),
            "overlapping_proteins": len(protein_overlap),
            "train_only_proteins": len(protein_train_only),
            "test_only_proteins": len(protein_test_only),
            "unique_train_proteins_list": list(unique_train_proteins),
            "unique_test_proteins_list": list(unique_test_proteins),
            "overlapping_proteins_list": list(protein_overlap),
            "train_only_proteins_list": list(protein_train_only),
            "test_only_proteins_list": list(protein_test_only),
        }

        # Generate MCS images for train and test sets
        try:
            if len(unique_train_smiles) > 1:
                logger.info(f"Generating MCS image for fold {fold} train set...")
                train_drawing, train_mcs_smarts, _, _ = SmilesMCStoGridImage(
                    unique_train_smiles, verbose=True
                )
                fold_analysis["train_mcs_smarts"] = train_mcs_smarts
                fold_analysis["train_mcs_image"] = train_drawing

                if output_dir and train_drawing:
                    train_img_path = (
                        output_path / f"fold_{fold}_train_mcs_{affinity_type}.png"
                    )
                    train_drawing.save(str(train_img_path))
                    logger.info(f"Saved train MCS image to {train_img_path}")
            else:
                logger.warning(
                    f"Fold {fold} train set has only {len(unique_train_smiles)} unique SMILES, skipping MCS"
                )
                fold_analysis["train_mcs_smarts"] = None
                fold_analysis["train_mcs_image"] = None

            if len(unique_test_smiles) > 1:
                logger.info(f"Generating MCS image for fold {fold} test set...")
                test_drawing, test_mcs_smarts, _, _ = SmilesMCStoGridImage(
                    unique_test_smiles, verbose=True
                )
                fold_analysis["test_mcs_smarts"] = test_mcs_smarts
                fold_analysis["test_mcs_image"] = test_drawing

                if output_dir and test_drawing:
                    test_img_path = (
                        output_path / f"fold_{fold}_test_mcs_{affinity_type}.png"
                    )
                    test_drawing.save(str(test_img_path))
                    logger.info(f"Saved test MCS image to {test_img_path}")
            else:
                logger.warning(
                    f"Fold {fold} test set has only {len(unique_test_smiles)} unique SMILES, skipping MCS"
                )
                fold_analysis["test_mcs_smarts"] = None
                fold_analysis["test_mcs_image"] = None

        except Exception as e:
            logger.error(f"Error generating MCS for fold {fold}: {e}")
            fold_analysis["train_mcs_smarts"] = None
            fold_analysis["train_mcs_image"] = None
            fold_analysis["test_mcs_smarts"] = None
            fold_analysis["test_mcs_image"] = None

        analysis[f"fold_{fold}"] = fold_analysis

    return analysis


def analyze_sdf_and_proteins_distribution_with_mcs(
    dataset: str,
    affinity_type: str,
    csv_file: str,
    compound_base_dir: str,
    n_folds: int = 4,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Analyze the distribution of unique SDF structures and proteins across all folds and generate MCS images.

    Parameters
    ----------
    dataset : str
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str
        The type of affinity ('pic50' or 'pk').
    csv_file : str
        Path to the CSV file containing the data.
    compound_base_dir : str
        Base directory containing the compound SDF files.
    n_folds : int, default=4
        The number of folds.
    output_dir : Optional[str], default=None
        Directory to save MCS images. If None, images won't be saved.

    Returns
    -------
    Dict
        Dictionary containing analysis results for each fold.
    """
    analysis = {}

    # Create output directory if specified - include affinity type in the name
    if output_dir:
        output_path = Path(f"{output_dir}_{affinity_type}")
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_path}")

    for fold in range(1, n_folds + 1):
        train_sdf_paths, test_sdf_paths, train_proteins, test_proteins = (
            split_sdf_paths_and_proteins_by_fold(
                dataset, affinity_type, csv_file, compound_base_dir, fold, n_folds
            )
        )

        # Get unique SDF file paths (representing unique structures)
        unique_train_sdf_paths = list(set(train_sdf_paths.values()))
        unique_test_sdf_paths = list(set(test_sdf_paths.values()))

        # Get unique proteins
        unique_train_proteins = set(train_proteins.values())
        unique_test_proteins = set(test_proteins.values())

        # Analyze SDF overlap
        sdf_overlap = set(train_sdf_paths.values()) & set(test_sdf_paths.values())
        sdf_train_only = set(train_sdf_paths.values()) - set(test_sdf_paths.values())
        sdf_test_only = set(test_sdf_paths.values()) - set(train_sdf_paths.values())

        # Analyze protein overlap
        protein_overlap = unique_train_proteins & unique_test_proteins
        protein_train_only = unique_train_proteins - unique_test_proteins
        protein_test_only = unique_test_proteins - unique_train_proteins

        fold_analysis = {
            # SDF analysis
            "train_total_compounds": len(train_sdf_paths),
            "test_total_compounds": len(test_sdf_paths),
            "train_unique_structures": len(unique_train_sdf_paths),
            "test_unique_structures": len(unique_test_sdf_paths),
            "overlapping_structures": len(sdf_overlap),
            "train_only_structures": len(sdf_train_only),
            "test_only_structures": len(sdf_test_only),
            "unique_train_sdf_paths": unique_train_sdf_paths,
            "unique_test_sdf_paths": unique_test_sdf_paths,
            "overlapping_sdf_paths": list(sdf_overlap),
            # Protein analysis
            "train_unique_proteins": len(unique_train_proteins),
            "test_unique_proteins": len(unique_test_proteins),
            "overlapping_proteins": len(protein_overlap),
            "train_only_proteins": len(protein_train_only),
            "test_only_proteins": len(protein_test_only),
            "unique_train_proteins_list": list(unique_train_proteins),
            "unique_test_proteins_list": list(unique_test_proteins),
            "overlapping_proteins_list": list(protein_overlap),
            "train_only_proteins_list": list(protein_train_only),
            "test_only_proteins_list": list(protein_test_only),
        }

        # Generate MCS images for train and test sets using SDF files
        try:
            if len(unique_train_sdf_paths) > 1:
                logger.info(
                    f"Generating MCS image for fold {fold} train set from SDF files..."
                )
                train_drawing, train_mcs_smarts, _, _ = SdfMCStoGridImage(
                    unique_train_sdf_paths, verbose=True
                )
                fold_analysis["train_mcs_smarts"] = train_mcs_smarts
                fold_analysis["train_mcs_image"] = train_drawing

                if output_dir and train_drawing:
                    train_img_path = (
                        output_path / f"fold_{fold}_train_mcs_{affinity_type}_sdf.png"
                    )
                    train_drawing.save(str(train_img_path))
                    logger.info(f"Saved train MCS image to {train_img_path}")
            else:
                logger.warning(
                    f"Fold {fold} train set has only {len(unique_train_sdf_paths)} unique structures, skipping MCS"
                )
                fold_analysis["train_mcs_smarts"] = None
                fold_analysis["train_mcs_image"] = None

            if len(unique_test_sdf_paths) > 1:
                logger.info(
                    f"Generating MCS image for fold {fold} test set from SDF files..."
                )
                test_drawing, test_mcs_smarts, _, _ = SdfMCStoGridImage(
                    unique_test_sdf_paths, verbose=True
                )
                fold_analysis["test_mcs_smarts"] = test_mcs_smarts
                fold_analysis["test_mcs_image"] = test_drawing

                if output_dir and test_drawing:
                    test_img_path = (
                        output_path / f"fold_{fold}_test_mcs_{affinity_type}_sdf.png"
                    )
                    test_drawing.save(str(test_img_path))
                    logger.info(f"Saved test MCS image to {test_img_path}")
            else:
                logger.warning(
                    f"Fold {fold} test set has only {len(unique_test_sdf_paths)} unique structures, skipping MCS"
                )
                fold_analysis["test_mcs_smarts"] = None
                fold_analysis["test_mcs_image"] = None

        except Exception as e:
            logger.error(f"Error generating MCS for fold {fold}: {e}")
            fold_analysis["train_mcs_smarts"] = None
            fold_analysis["train_mcs_image"] = None
            fold_analysis["test_mcs_smarts"] = None
            fold_analysis["test_mcs_image"] = None

        analysis[f"fold_{fold}"] = fold_analysis

    return analysis


# Keep the old function for backward compatibility
def analyze_metal_smiles_distribution_with_mcs(
    dataset: str,
    affinity_type: str,
    csv_file: str,
    n_folds: int = 4,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Analyze the distribution of unique metal SMILES across all folds and generate MCS images.

    This function is kept for backward compatibility. It calls
    analyze_metal_smiles_and_proteins_distribution_with_mcs.

    Parameters
    ----------
    dataset : str
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str
        The type of affinity ('pic50' or 'pk').
    csv_file : str
        Path to the CSV file containing the data.
    n_folds : int, default=4
        The number of folds.
    output_dir : Optional[str], default=None
        Directory to save MCS images. If None, images won't be saved.

    Returns
    -------
    Dict
        Dictionary containing analysis results for each fold.
    """
    return analyze_metal_smiles_and_proteins_distribution_with_mcs(
        dataset, affinity_type, csv_file, n_folds, output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--affinity_type", type=str, default="pic50", choices=["pic50", "pk"]
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mcs_images",
        help="Directory to save MCS images (affinity type will be appended)",
    )
    args = parser.parse_args()

    if args.affinity_type == "pic50":
        excel_path = excel_pic50_path
    elif args.affinity_type == "pk":
        excel_path = excel_pk_path

    # Test the updated function
    logger.info(f"Loading {args.affinity_type} data from {excel_path}")
    logger.info("Loading HDF5 dataset...")
    tgt_dataset = load_data_hdf5(
        dataset_name="caged",
        inhibitor_type="MACE",
        target_type=args.affinity_type,
    )

    # Analyze all folds with MCS generation and protein analysis
    logger.info(
        "Analyzing metal SMILES and protein distribution across all folds with MCS generation..."
    )
    analysis = analyze_metal_smiles_and_proteins_distribution_with_mcs(
        dataset="caged",
        affinity_type=args.affinity_type,
        csv_file=excel_path,
        n_folds=5,
        output_dir=args.output_dir,
    )

    for fold_name, stats in analysis.items():
        logger.info(f"{fold_name.upper()}:")
        logger.info(f"  SMILES Analysis:")
        logger.info(
            f"    Train: {stats['train_total_compounds']} compounds, {stats['train_unique_smiles']} unique SMILES"
        )
        logger.info(
            f"    Test:  {stats['test_total_compounds']} compounds, {stats['test_unique_smiles']} unique SMILES"
        )
        logger.info(
            f"    SMILES Overlap: {stats['overlapping_smiles']} SMILES appear in both train and test"
        )

        logger.info(f"  Protein Analysis:")
        logger.info(f"    Train: {stats['train_unique_proteins']} unique proteins")
        logger.info(f"    Test:  {stats['test_unique_proteins']} unique proteins")
        logger.info(
            f"    Protein Overlap: {stats['overlapping_proteins']} proteins appear in both train and test"
        )

        if stats["overlapping_proteins"] > 0:
            logger.info(
                f"    Overlapping proteins: {', '.join(stats['overlapping_proteins_list'])}"
            )
        if stats["train_only_proteins"] > 0:
            logger.info(
                f"    Train-only proteins: {', '.join(stats['train_only_proteins_list'])}"
            )
        if stats["test_only_proteins"] > 0:
            logger.info(
                f"    Test-only proteins: {', '.join(stats['test_only_proteins_list'])}"
            )

        if stats.get("train_mcs_smarts"):
            logger.info(f"  Train MCS SMARTS: {stats['train_mcs_smarts']}")
        if stats.get("test_mcs_smarts"):
            logger.info(f"  Test MCS SMARTS: {stats['test_mcs_smarts']}")
        logger.info("")
