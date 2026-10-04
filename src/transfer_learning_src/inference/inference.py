import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from ase.io import read, write
from ase.optimize import BFGS
from xtb.ase.calculator import XTB
from ase import Atoms
import subprocess
import pandas as pd

# Add the src directory to the path to import modules
path = str(Path(__file__).parent.parent.parent.parent / "src" / "data_src")
sys.path.append(path)

from create_hf5_common import (
    compute_soap_embedding,
    compute_acsf_embedding,
    compute_mbtr_embedding,
    generate_embeddings_esmc,
    create_chain_position_mapping,
    extract_pocket_embeddings,
    extract_pocket_edges,
    create_pocket_mask,
    create_mace_calculator,
    add_mace_embeddings_to_entry,
)

# Add transfer_learning_src to the path so data can be imported as a package
# (required because data modules use relative imports internally)
data_path = str(Path(__file__).parent.parent)
sys.path.append(data_path)

from data.load_data import load_data_hdf5
from data.split_data import split_data_by_group
from data.utils_data import prepare_datasets, compute_scaling_stats

# Add the model directory to the path
model_path = str(Path(__file__).parent.parent / "model")
sys.path.append(model_path)

from multitask import MultiTaskPocket, CCSAFinetune

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_train_config(
    separate_domains: bool, source_smote: bool, target_smote: bool
) -> str:
    """
    Build training configuration directory name from boolean flags.

    This function constructs a standardized directory name that encodes the
    training configuration for domain scaling and SMOTE application.

    Parameters
    ----------
    separate_domains : bool
        If True, use separate domain scaling; if False, use cross-domain scaling.
    source_smote : bool
        If True, SMOTE was applied to source domain.
    target_smote : bool
        If True, SMOTE was applied to target domain.

    Returns
    -------
    str
        Training configuration string (e.g., 'cross_domain_no_source_smote_no_target_smote').
    """
    # Determine domain type
    if separate_domains:
        domain_type = "separate_domain"
    else:
        domain_type = "cross_domain"

    # Build SMOTE configuration
    smote_parts = []
    if source_smote:
        smote_parts.append("source_smote")
    else:
        smote_parts.append("no_source_smote")

    if target_smote:
        smote_parts.append("target_smote")
    else:
        smote_parts.append("no_target_smote")

    # Combine domain type and SMOTE configuration
    train_config = f"{domain_type}_{smote_parts[0]}_{smote_parts[1]}"

    return train_config


def build_params(
    threshold: float,
    inhibitor_descriptor: str,
    biological_descriptor: str,
    affinity_type: str,
    modality: str,
    data_scaler: str,
    cross_domain_scaling: bool,
    source_smote: bool,
    target_smote: bool,
    embed_dim: int = 128,
    dropout: float = 0.4,
    learning_rate: float = 1e-4,
    batch_size: int = 64,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Construct the parameters dictionary for model inference.

    This function creates a comprehensive parameter dictionary that matches
    the configuration used during model training.

    Parameters
    ----------
    threshold : float
        Binding affinity threshold for binary classification.
    inhibitor_descriptor : str
        Type of molecular descriptor (e.g., 'MACE', 'SOAP', 'ACSF').
    biological_descriptor : str
        Type of biological descriptor (e.g., 'ESM-C').
    embed_dim : int
        Embedding dimension for model layers.
    dropout : float
        Dropout rate.
    learning_rate : float
        Learning rate (included for consistency).
    batch_size : int
        Batch size for inference.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed for reproducibility.
    affinity_type : str
        Type of binding affinity ('pic50', 'pk', or 'pic50+pk').
    modality : str
        Feature modalities to use (e.g., 'molecule_protein_pocket').
    data_scaler : str
        Scaling method ('minmax', 'standard', etc.).
    cross_domain_scaling : bool
        Whether cross-domain scaling was used.
    source_smote : bool
        Whether SMOTE was applied to source domain.
    target_smote : bool
        Whether SMOTE was applied to target domain.

    Returns
    -------
    Dict[str, Any]
        Complete parameters dictionary for model initialization.
    """
    features_dim_dict = {"MACE": 256, "SOAP": 5376, "ACSF": 360, "ESM-C": 1152}

    return {
        "threshold": threshold,
        "inhibitor_descriptor": inhibitor_descriptor,
        "biological_descriptor": biological_descriptor,
        "embed_dim": embed_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_folds": n_folds,
        "seed": seed,
        "affinity_type": affinity_type,
        "modality": modality,
        "use_mol": "molecule" in modality or "mol" in modality,
        "use_protein": "protein" in modality,
        "use_pocket": "pocket" in modality,
        "inhibitor_input_dim": features_dim_dict.get(inhibitor_descriptor, 256),
        "biological_input_dim": features_dim_dict.get(biological_descriptor, 1152),
        "batch_size": batch_size,
        "n_folds": n_folds,
        "seed": seed,
        "data_scaler": data_scaler,
        "source_smote": source_smote,
        "target_smote": target_smote,
        "cross_domain_scaling": cross_domain_scaling,
    }


def optimize_with_gfn2_xtb(
    xyz_in: str,
    xyz_out: str,
    charge: int = 0,
    opt_fmax: float = 0.05,
    force: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Perform geometry optimization using GFN2-xTB via ASE.

    This function optimizes molecular geometry using the GFN2-xTB method.
    Results are cached in an organized directory structure and reused unless
    force re-optimization is requested.

    Parameters
    ----------
    xyz_in : str
        Path to input .xyz file.
    xyz_out : str
        Path to output optimized .xyz file.
    charge : int, optional
        Molecular charge, by default 0.
    opt_fmax : float, optional
        Convergence criterion for forces in eV/Å, by default 0.05.
    force : bool, optional
        If True, force re-optimization even if cached result exists, by default False.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        A tuple containing:
        - Optimized atomic coordinates (N x 3 array)
        - List of atomic symbols
    """
    # Honour the caller's output path in full. Using only its basename and
    # rebuilding the directory relative to the CWD sent every optimized
    # structure to wherever the job happened to be launched from, left the path
    # recorded in the result JSON pointing at a non-existent file, and keyed the
    # reuse-cache below on the input basename alone -- so two inputs with the
    # same name in different directories would silently share one geometry even
    # if they were optimized at different charges.
    xyz_base_name = os.path.splitext(os.path.basename(xyz_in))[0]

    if xyz_out:
        organized_xyz_out = os.path.abspath(xyz_out)
    else:
        # No path given: keep the historical layout, relative to the input file
        # rather than to the CWD.
        organized_xyz_out = os.path.join(
            os.path.dirname(os.path.abspath(xyz_in)),
            "GFN2-xTB_optimized",
            xyz_base_name,
            "optimized.xyz",
        )

    os.makedirs(os.path.dirname(organized_xyz_out), exist_ok=True)

    # Reuse existing optimized structure if available and not forcing
    # re-optimization -- but only when it is actually a geometry of the molecule
    # we were asked about. Editing an input .xyz without noticing a stale cached
    # result would otherwise silently featurize the previous molecule.
    if organized_xyz_out and os.path.exists(organized_xyz_out) and not force:
        cached = read(organized_xyz_out)
        wanted = read(xyz_in)
        if sorted(cached.get_chemical_symbols()) == sorted(
            wanted.get_chemical_symbols()
        ):
            logger.info(f"Reusing existing optimized structure: {organized_xyz_out}")
            return cached.get_positions(), cached.get_chemical_symbols()
        logger.warning(
            f"Cached geometry {organized_xyz_out} has a different composition "
            f"({cached.get_chemical_formula()}) than the input "
            f"({wanted.get_chemical_formula()}); re-optimizing."
        )

    # Read input structure and optimize
    atoms = read(xyz_in)

    # Attach the xTB calculator (ASE needs xtb installed in PATH)
    atoms.calc = XTB(method="GFN2-xTB", charge=charge)

    # Run BFGS optimization
    dyn = BFGS(atoms, logfile="opt.log")
    dyn.run(fmax=opt_fmax)

    # Write optimized structure to organized location
    write(organized_xyz_out, atoms)
    logger.info(f"Optimized structure saved to: {organized_xyz_out}")

    # obtain coords and atom_types
    coords = atoms.get_positions()
    atom_types = atoms.get_chemical_symbols()

    return coords, atom_types


def run_p2rank(pdb_file_path: str, protein_name: str) -> bool:
    """
    Run P2Rank pocket prediction for a given PDB file.

    This function executes P2Rank to predict binding pockets in the protein structure.
    Results are saved in an organized directory structure. Requires 'prank.sh' to be
    available in the system PATH.

    Parameters
    ----------
    pdb_file_path : str
        Path to the PDB file for pocket prediction.
    protein_name : str
        Name identifier for the protein (used for organizing output).

    Returns
    -------
    bool
        True if P2Rank executed successfully, False otherwise.
    """

    # Resolve absolute paths
    pdb_file_abs = os.path.abspath(pdb_file_path)
    pdb_dir = os.path.dirname(pdb_file_abs)

    # Create organized output directory structure
    p2rank_base_dir = "P2Rank_pocket_predictions"
    protein_output_dir = os.path.join(p2rank_base_dir, protein_name)
    os.makedirs(protein_output_dir, exist_ok=True)

    # Construct the command
    cmd = [
        "prank.sh",
        "predict",
        "-f",
        pdb_file_abs,
        "-o",
        protein_output_dir,
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        run_cwd = pdb_dir if pdb_dir else None
        subprocess.run(cmd, capture_output=True, text=True, cwd=run_cwd, check=True)
        logger.info(f"✓ Successfully processed {protein_name}")
        return True

    except FileNotFoundError:
        logger.error(
            "✗ 'prank.sh' not found. Please make sure P2Rank is installed and in your PATH."
        )
        return False

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Error processing {protein_name}: {e.stderr}")
        return False

    except Exception as e:
        logger.error(f"✗ Exception while processing {protein_name}: {str(e)}")
        return False


def load_p2rank_predictions(p2rank_path: str) -> List[str]:
    """
    Load pocket residue IDs from P2Rank predictions CSV file.

    This function reads the P2Rank output CSV and extracts residue IDs for the
    first (best-ranked) predicted pocket.

    Parameters
    ----------
    p2rank_path : str
        Path to P2Rank predictions CSV file.

    Returns
    -------
    List[str]
        List of residue IDs in the predicted pocket, or empty list if file not found/invalid.
    """
    try:
        p2rank_df = pd.read_csv(p2rank_path)
        p2rank_df.columns = p2rank_df.columns.str.replace(" ", "")
        if p2rank_df.empty or "residue_ids" not in p2rank_df.columns:
            return []
        residues_string = str(p2rank_df["residue_ids"].iloc[0]).strip()
        return residues_string.split()
    except Exception as e:
        logger.debug(f"Error loading p2rank predictions from {p2rank_path}: {e}")
        return []


def _infer_p2rank_csv_path(pdb_file: str, protein_name: str) -> str:
    """
    Infer P2Rank CSV path in the organized P2Rank output structure.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file.
    protein_name : str
        Name identifier for the protein.

    Returns
    -------
    str
        Expected path to P2Rank predictions CSV file.
    """
    pdb_base = os.path.basename(pdb_file)
    csv_name = f"{pdb_base}_predictions.csv"
    p2rank_base_dir = "P2Rank_pocket_predictions"
    protein_output_dir = os.path.join(p2rank_base_dir, protein_name)
    return os.path.join(protein_output_dir, csv_name)


def get_pocket_residues(pdb_file: str) -> List[str]:
    """
    Ensure P2Rank predictions exist for the PDB and return residue IDs.

    This function checks if P2Rank predictions exist for the given PDB file.
    If not, it runs P2Rank to generate predictions, then returns the predicted
    pocket residue IDs.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file.

    Returns
    -------
    List[str]
        List of residue IDs in the predicted pocket.
    """
    protein_name = os.path.splitext(os.path.basename(pdb_file))[0]
    csv_path = _infer_p2rank_csv_path(pdb_file, protein_name)
    if not os.path.exists(csv_path):
        logger.info("P2Rank predictions not found; running P2Rank...")
        run_p2rank(pdb_file, protein_name)
    if not os.path.exists(csv_path):
        logger.warning(f"P2Rank output not found at {csv_path}")
        return []
    return load_p2rank_predictions(csv_path)


def convert_xyz_to_embedding(
    descriptor_type: str,
    coords: np.ndarray,
    atom_types: List[str],
    mace_calculator: Any = None,
) -> np.ndarray:
    """
    Convert XYZ coordinates to molecular embedding using specified descriptor.

    This function computes molecular descriptors from atomic coordinates using
    various methods (SOAP, MACE, ACSF, or MBTR).

    Parameters
    ----------
    descriptor_type : str
        Type of molecular descriptor ('SOAP', 'MACE', 'ACSF', or 'MBTR').
    coords : np.ndarray
        Atomic coordinates (N x 3 array).
    atom_types : List[str]
        List of atomic symbols.
    mace_calculator : Any, optional
        MACE calculator instance (required if descriptor_type is 'MACE'), by default None.

    Returns
    -------
    np.ndarray
        Molecular embedding vector.

    Raises
    ------
    ValueError
        If descriptor_type is unsupported or MACE calculator is missing for MACE descriptors.
    """
    if descriptor_type == "SOAP":
        return compute_soap_embedding(coords=coords, atom_types=atom_types)
    elif descriptor_type == "MACE":
        if mace_calculator is None:
            raise ValueError("MACE calculator is required for MACE embeddings")
        # Create ASE Atoms object
        atoms = Atoms(symbols=atom_types, positions=coords)
        # Use MACE calculator to get embeddings
        atoms.calc = mace_calculator
        return atoms.calc.get_property("mace_embedding", atoms)
    elif descriptor_type == "ACSF":
        return compute_acsf_embedding(coords=coords, atom_types=atom_types)
    elif descriptor_type == "MBTR":
        return compute_mbtr_embedding(coords=coords, atom_types=atom_types)
    else:
        raise ValueError(f"Unsupported molecular descriptor: {descriptor_type}")


def load_model_checkpoint(
    model_path: str,
    fold: int,
    phase: str,
    params: Dict[str, Any],
    descriptor_type: str,
    modality: str,
    scaler_type: str,
    affinity_type: str,
    train_config: str,
) -> torch.nn.Module:
    """
    Load a trained model checkpoint for a specific fold and phase.

    This function locates and loads a trained model checkpoint from the organized
    directory structure used during training.

    Parameters
    ----------
    model_path : str
        Base path to the models directory.
    fold : int
        Cross-validation fold number.
    phase : str
        Training phase (e.g., 'no_pretrain_ccsa', 'no_pretrain_encoder_task').
    params : Dict[str, Any]
        Model parameters dictionary.
    descriptor_type : str
        Type of molecular descriptor used.
    modality : str
        Feature modalities used (e.g., 'molecule_protein_pocket').
    scaler_type : str
        Scaling method used (e.g., 'minmax', 'standard').
    affinity_type : str
        Type of binding affinity ('pic50', 'pk', or 'pic50+pk').
    train_config : str
        Training configuration string.

    Returns
    -------
    torch.nn.Module
        Loaded model with trained weights.

    Raises
    ------
    FileNotFoundError
        If checkpoint directory or files are not found.
    """
    # Build the path based on the actual directory structure
    checkpoint_dir = (
        Path(model_path)
        / affinity_type
        / f'threshold_{params["threshold"]:1.0f}'
        / f"fold_{fold}"
        / phase
        / "checkpoints"
    )

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find the best checkpoint file
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # Use the first checkpoint (assuming they're named consistently)
    checkpoint_path = checkpoint_files[0]
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Checkpoints were written by the Lightning wrapper, so every tensor is
    # prefixed with "base_model."; strip it so the keys match MultiTaskPocket.
    raw_sd = checkpoint["state_dict"]
    state_dict = {
        k[len("base_model."):]: v
        for k, v in raw_sd.items()
        if k.startswith("base_model.")
    }
    if not state_dict:
        state_dict = raw_sd

    model = MultiTaskPocket(domain="tgt", **params)
    model.load_state_dict(state_dict, strict=True)

    return model


def load_all_models(
    model_path: str,
    params: Dict[str, Any],
    descriptor_type: str,
    modality: str,
    scaler_type: str,
    affinity_type: str,
    train_config: str,
    phases: Optional[List[str]] = None,
) -> Dict[str, Dict[int, torch.nn.Module]]:
    """
    Load all trained models for ensemble prediction.

    This function loads all available models across different folds and training
    phases for use in ensemble prediction.

    Parameters
    ----------
    model_path : str
        Base path to the models directory.
    params : Dict[str, Any]
        Model parameters dictionary.
    descriptor_type : str
        Type of molecular descriptor used.
    modality : str
        Feature modalities used (e.g., 'molecule_protein_pocket').
    scaler_type : str
        Scaling method used (e.g., 'minmax', 'standard').
    affinity_type : str
        Type of binding affinity ('pic50', 'pk', or 'pic50+pk').
    train_config : str
        Training configuration string.
    phases : Optional[List[str]], optional
        List of training phases to load. If None, defaults to all no-pretrain phases,
        by default None.

    Returns
    -------
    Dict[str, Dict[int, torch.nn.Module]]
        Nested dictionary mapping phase -> fold -> model.

    Raises
    ------
    FileNotFoundError
        If base model path is not found.
    """
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default phases if not specified
    if phases is None:
        phases = [
            "no_pretrain_ccsa",
            "no_pretrain_ccsa_encoder_task",
            "no_pretrain_ccsa_task",
            "no_pretrain_encoder_task",
        ]

    # First, let's discover available phases and folds
    base_path = (
        Path(model_path)
        / affinity_type
        / f'threshold_{params["threshold"]:1.0f}'
    )

    if not base_path.exists():
        raise FileNotFoundError(f"Base model path not found: {base_path}")

    # Discover available folds
    available_folds = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("fold_"):
            try:
                fold_num = int(item.name.split("_")[1])
                available_folds.append(fold_num)
            except ValueError:
                continue

    available_folds.sort()
    logger.info(f"Found available folds: {available_folds}")

    # Discover available phases for each fold
    for fold in available_folds:
        fold_path = base_path / f"fold_{fold}"
        if not fold_path.exists():
            continue

        available_phases = []
        for item in fold_path.iterdir():
            if item.is_dir() and item.name != "pretrain_src":  # Skip pretrain_src
                available_phases.append(item.name)

        logger.info(f"Found available phases for fold {fold}: {available_phases}")

        # Load models for this fold
        for phase in available_phases:
            if phases and phase not in phases:
                continue

            try:
                model = load_model_checkpoint(
                    model_path,
                    fold,
                    phase,
                    params,
                    descriptor_type,
                    modality,
                    scaler_type,
                    affinity_type,
                    train_config,
                )
                model = model.to(device)
                model.eval()

                if phase not in models:
                    models[phase] = {}
                models[phase][fold] = model
                logger.info(f"Successfully loaded {phase} model for fold {fold}")
            except Exception as e:
                logger.warning(f"Failed to load {phase} model for fold {fold}: {e}")
                continue

    return models


def load_target_dataset_and_compute_fold_stats(
    dataset_name: str,
    affinity_type: str,
    modality: str,
    scaling_method: str,
    cross_domain_scaling: bool,
    n_folds: int,
) -> Dict[int, Any]:
    """
    Load the target dataset and compute fold-specific normalization statistics.

    This function loads the target dataset and computes scaling statistics for each
    cross-validation fold using the same data preparation pipeline as training.

    Parameters
    ----------
    dataset_name : str
        Name of the target dataset (e.g., 'caged', 'organic').
    affinity_type : str
        Type of binding affinity ('pic50', 'pk', or 'pic50+pk').
    modality : str
        Feature modalities to use (e.g., 'molecule_protein_pocket').
    scaling_method : str
        Scaling method to use ('minmax', 'standard', or None).
    cross_domain_scaling : bool
        Whether cross-domain scaling is used (for compatibility).
    n_folds : int
        Number of cross-validation folds.

    Returns
    -------
    Dict[int, Any]
        Dictionary mapping fold number to scaling statistics for that fold.
    """
    logger.info(f"Loading target dataset '{dataset_name}' for {affinity_type}...")

    # Load the target dataset (same as training)
    data_path = f"./../../../data/hf5_dataset/{dataset_name}/{dataset_name}_{affinity_type}_dataset.h5"

    # Load data using the same function as training
    target_data = load_data_hdf5(
        dataset_name, "SOAP", affinity_type, percent_of_data=1.0
    )

    print(target_data.keys())

    logger.info(f"Loaded {len(target_data['inhibitor_feats'])} inhibitor features")
    logger.info(f"Loaded {len(target_data['protein_feats'])} protein embeddings")
    logger.info(f"Loaded {len(target_data['pocket_feats'])} pocket embeddings")
    logger.info(f"Loaded {len(target_data['binding_affs'])} binding affinities")

    fold_stats = {}

    # Process each fold (same as training)
    for fold in range(1, n_folds + 1):
        logger.info(f"Processing fold {fold}/{n_folds}...")

        # Split data by fold (same as training)
        train_data, test_data = split_data_by_group(
            target_data["inhibitor_feats"],
            target_data["protein_feats"],
            target_data["pocket_feats"],
            target_data["binding_affs"],
            dataset_name,
            affinity_type,
            fold=fold,
            n_folds=n_folds,
        )

        logger.info(
            f"Successfully split data for fold {fold}. Training set size: {len(train_data['binding_affs'])}"
        )

        # Compute scaling stats from training data (same as training)
        if scaling_method is not None:
            stats = compute_scaling_stats(train_data, modality, scaling_method)
            fold_stats[fold] = stats
            logger.info(f"Successfully computed stats for fold {fold}.")
        else:
            fold_stats[fold] = None
            logger.info(f"No scaling for fold {fold}.")

    return fold_stats


def prepare_input_with_fold_stats(
    mol_embedding: np.ndarray,
    protein_embedding: np.ndarray,
    pocket_embedding: np.ndarray,
    fold_stats: Dict[str, Any],
    modality: str,
    device: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare input batch using the same scaling as a specific fold's training data.

    This function processes and scales molecular, protein, and pocket embeddings
    to match the format and normalization used during model training.

    Parameters
    ----------
    mol_embedding : np.ndarray
        Molecular descriptor embedding.
    protein_embedding : np.ndarray
        Protein embedding from ESM or similar model.
    pocket_embedding : np.ndarray
        Pocket embedding (can be None if not using pocket modality).
    fold_stats : Dict[str, Any]
        Scaling statistics from the fold's training data.
    modality : str
        Feature modalities being used (e.g., 'molecule_protein_pocket').
    device : Any, optional
        PyTorch device to use (CPU or CUDA), by default None.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - Scaled molecular embedding tensor
        - Scaled protein embedding tensor
        - Scaled pocket embedding tensor (or empty tensor)
        - Dummy labels tensor (for compatibility)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process embeddings to match training format
    logger.info(
        f"Original molecular embedding shape: {mol_embedding.shape if hasattr(mol_embedding, 'shape') else 'N/A'}"
    )
    if isinstance(mol_embedding, np.ndarray) and len(mol_embedding.shape) > 1:
        mol_embedding = np.mean(mol_embedding, axis=0)
        logger.info(f"Averaged molecular embedding shape: {mol_embedding.shape}")

    # Process protein embedding: take the <cls> token (first token)
    if hasattr(protein_embedding, "cpu"):
        protein_embedding = protein_embedding.float().cpu().numpy()
    logger.info(
        f"Original protein embedding shape: {protein_embedding.shape if hasattr(protein_embedding, 'shape') else 'N/A'}"
    )
    if isinstance(protein_embedding, np.ndarray) and len(protein_embedding.shape) > 1:
        protein_embedding = protein_embedding[0]
        logger.info(f"<cls> token protein embedding shape: {protein_embedding.shape}")

    # Process pocket embedding if available
    logger.info(
        f"Original pocket embedding shape: {pocket_embedding.shape if hasattr(pocket_embedding, 'shape') else 'N/A'}"
    )
    if pocket_embedding is not None:
        if isinstance(pocket_embedding, np.ndarray) and len(pocket_embedding.shape) > 1:
            pocket_embedding = np.mean(pocket_embedding, axis=0)
        logger.info(f"Averaged pocket embedding shape: {pocket_embedding.shape}")

    # Apply fold-specific scaling (same as training)
    if fold_stats is not None:
        mol_scaled, protein_scaled, pocket_scaled = apply_scaling_like_training(
            mol_embedding, protein_embedding, pocket_embedding, modality, fold_stats
        )
    else:
        mol_scaled, protein_scaled, pocket_scaled = (
            mol_embedding,
            protein_embedding,
            pocket_embedding,
        )

    # Convert to tensors
    mol_tensor = (
        torch.tensor(mol_scaled, dtype=torch.float32)
        .detach()
        .clone()
        .unsqueeze(0)
        .to(device)
    )
    protein_tensor = (
        torch.as_tensor(protein_scaled, dtype=torch.float32)
        .detach()
        .clone()
        .unsqueeze(0)
        .to(device)
    )

    if pocket_scaled is not None:
        pocket_tensor = (
            torch.tensor(pocket_scaled, dtype=torch.float32)
            .detach()
            .clone()
            .unsqueeze(0)
            .to(device)
        )
    else:
        pocket_tensor = torch.empty(0).to(device)

    # Dummy labels (not used in inference)
    labels = torch.tensor([0.0], dtype=torch.float32).to(device)

    return (mol_tensor, protein_tensor, pocket_tensor, labels)


def apply_scaling_like_training(
    mol_vec: np.ndarray,
    protein_vec: np.ndarray,
    pocket_vec: np.ndarray,
    modality: str,
    stats: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply scaling exactly like in training using the same approach as utils_data.py.

    This function combines features according to modality, applies scaling using
    saved statistics, and splits them back into individual components.

    Parameters
    ----------
    mol_vec : np.ndarray
        Molecular descriptor vector.
    protein_vec : np.ndarray
        Protein embedding vector.
    pocket_vec : np.ndarray
        Pocket embedding vector (can be None).
    modality : str
        Feature modalities being used (e.g., 'molecule_protein_pocket').
    stats : Dict[str, Any]
        Scaling statistics (min/max or mean/std) from training.

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
        A tuple containing:
        - Scaled molecular embedding (or None)
        - Scaled protein embedding (or None)
        - Scaled pocket embedding (or None)

    Raises
    ------
    ValueError
        If no features are available to combine.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Combine features according to modality (same as training)
    parts = []
    if ("molecule" in modality or "mol" in modality) and mol_vec is not None:
        if isinstance(mol_vec, np.ndarray) and len(mol_vec.shape) > 1:
            parts.append(mol_vec.astype(np.float32).mean(axis=0))
        else:
            parts.append(mol_vec.astype(np.float32))
    if "protein" in modality and protein_vec is not None:
        if isinstance(protein_vec, np.ndarray) and len(protein_vec.shape) > 1:
            parts.append(protein_vec.astype(np.float32)[0])
        else:
            parts.append(protein_vec.astype(np.float32))
    if "pocket" in modality and pocket_vec is not None:
        if isinstance(pocket_vec, np.ndarray) and len(pocket_vec.shape) > 1:
            parts.append(pocket_vec.astype(np.float32).mean(axis=0))
        else:
            parts.append(pocket_vec.astype(np.float32))

    if not parts:
        raise ValueError("No features available to combine")
    combined = np.concatenate(parts, axis=0)

    # Apply scaling using the same approach as training
    scaler_type = stats.get("scaling_method", "minmax")
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
        # Set the scaler attributes from saved stats
        scaler.data_min_ = stats["combined_min"]
        scaler.data_max_ = stats["combined_max"]
        scaler.data_range_ = stats["combined_max"] - stats["combined_min"]
        safe_range = np.where(scaler.data_range_ == 0, 1.0, scaler.data_range_)
        scaler.scale_ = 1.0 / safe_range
        scaler.min_ = -stats["combined_min"] * scaler.scale_

        # A feature that never varied in training carries no information; map it
        # to a constant 0 instead of letting the substituted range rescale it.
        zero_var_mask = scaler.data_range_ == 0
        if np.any(zero_var_mask):
            scaler.scale_[zero_var_mask] = 1.0
            scaler.min_[zero_var_mask] = 0.0

        scaler.n_features_in_ = len(stats["combined_min"])
        scaler.feature_range = (0, 1)
        scaled = scaler.transform(combined.reshape(1, -1)).reshape(-1)

    elif scaler_type == "standard":
        scaler = StandardScaler()
        # Set the scaler attributes from saved stats
        scaler.mean_ = stats["combined_mean"]
        scaler.var_ = stats["combined_std"] ** 2
        scaler.scale_ = 1.0 / stats["combined_std"]
        scaler.n_features_in_ = len(stats["combined_mean"])
        scaled = scaler.transform(combined.reshape(1, -1)).reshape(-1)
    else:
        scaled = combined

    # Split back into components (same as training)
    mol_dim = stats.get("mol_dim")
    protein_dim = stats.get("protein_dim")
    pocket_dim = stats.get("pocket_dim")

    offset = 0
    mol_scaled = protein_scaled = pocket_scaled = None

    if ("molecule" in modality or "mol" in modality) and mol_dim:
        mol_scaled = scaled[offset : offset + mol_dim]
        offset += mol_dim
    if "protein" in modality and protein_dim:
        protein_scaled = scaled[offset : offset + protein_dim]
        offset += protein_dim
    if "pocket" in modality and pocket_dim:
        pocket_scaled = scaled[offset : offset + pocket_dim]
        offset += pocket_dim

    return mol_scaled, protein_scaled, pocket_scaled


def ensemble_predict_with_fold_specific_stats(
    models: Dict[str, Dict[int, torch.nn.Module]],
    mol_embedding: np.ndarray,
    protein_embedding: np.ndarray,
    pocket_embedding: np.ndarray,
    fold_stats: Dict[int, Any],
    modality: str,
    threshold: float = 7.0,
) -> Dict[str, Any]:
    """
    Make ensemble prediction using fold-specific normalization statistics.

    This function runs inference with multiple models (across different folds and phases)
    and combines their predictions via averaging. Each model uses its fold-specific
    scaling statistics.

    Parameters
    ----------
    models : Dict[str, Dict[int, torch.nn.Module]]
        Nested dictionary mapping phase -> fold -> model.
    mol_embedding : np.ndarray
        Molecular descriptor embedding.
    protein_embedding : np.ndarray
        Protein embedding from ESM or similar model.
    pocket_embedding : np.ndarray
        Pocket embedding (can be None if not using pocket modality).
    fold_stats : Dict[int, Any]
        Dictionary mapping fold number to its scaling statistics.
    modality : str
        Feature modalities being used (e.g., 'molecule_protein_pocket').
    threshold : float, optional
        Binding affinity threshold (for reference), by default 7.0.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'ensemble_probability': Average probability across all models
        - 'ensemble_logit': Average logit across all models
        - 'binary_prediction': Binary classification (0 or 1)
        - 'is_active': Boolean indicating if compound is predicted as active
        - 'individual_predictions': List of individual model probabilities
        - 'n_models_used': Number of models that successfully made predictions

    Raises
    ------
    RuntimeError
        If no models were able to make predictions.
    """
    all_logits = []
    all_probabilities = []

    with torch.no_grad():
        for phase, phase_models in models.items():
            for fold, model in phase_models.items():
                try:
                    # Get fold-specific stats
                    stats = fold_stats.get(fold)
                    if stats is None:
                        logger.warning(f"No stats available for fold {fold}; skipping")
                        continue

                    # Prepare input batch with fold-specific scaling
                    device = (
                        model.device
                        if hasattr(model, "device")
                        else torch.device("cpu")
                    )
                    input_batch = prepare_input_with_fold_stats(
                        mol_embedding,
                        protein_embedding,
                        pocket_embedding,
                        stats,
                        modality,
                        device,
                    )

                    # Move batch to model device
                    batch = tuple(
                        t.to(device) if torch.is_tensor(t) else t for t in input_batch
                    )

                    # Make prediction
                    logits = model.forward(batch)
                    prob = torch.sigmoid(logits).cpu().numpy()[0]

                    all_logits.append(logits.cpu().numpy()[0])
                    all_probabilities.append(prob)
                    logger.info(f"Prediction from {phase} fold {fold}: prob={prob:.4f}")

                except Exception as e:
                    logger.warning(
                        f"Failed prediction for phase={phase}, fold={fold}: {e}"
                    )
                    continue

    if not all_probabilities:
        raise RuntimeError("No models were able to make predictions")

    # Ensemble prediction (average probabilities)
    ensemble_prob = np.mean(all_probabilities)
    ensemble_logit = np.mean(all_logits)

    # Binary prediction based on averaged probability
    binary_prediction = 1 if ensemble_prob > 0.5 else 0

    return {
        "ensemble_probability": float(ensemble_prob),
        "ensemble_logit": float(ensemble_logit),
        "binary_prediction": int(binary_prediction),
        "is_active": binary_prediction == 1,
        "individual_predictions": all_probabilities,
        "n_models_used": len(all_probabilities),
    }


def main(args: argparse.Namespace) -> None:
    """
    Main inference function for running predictions on new molecular structures.

    This function orchestrates the complete inference pipeline including geometry
    optimization, descriptor generation, model loading, and ensemble prediction.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - inhibitor_descriptor: Type of molecular descriptor
        - biological_descriptor: Type of biological descriptor
        - modality: Feature modalities to use
        - affinity_type: Type of binding affinity
        - data_scaler: Scaling method
        - source_smote: Whether source SMOTE was used
        - target_smote: Whether target SMOTE was used
        - model_path: Path to trained models
        - phases: Comma-separated training phases
        - threshold: Binding affinity threshold
        - separate_domains: Whether separate domain scaling was used
        - xyz_file: Path to molecular structure file
        - pdb_file: Path to protein structure file
        - charge: Molecular charge
        - force_optimization: Force re-optimization

    Returns
    -------
    None
    """
    logger.info("Starting inference pipeline...")

    inhibitor_descriptor = args.inhibitor_descriptor
    biological_descriptor = args.biological_descriptor
    modality = args.modality
    affinity_type = args.affinity_type
    data_scaler = args.data_scaler
    source_smote = args.source_smote
    target_smote = args.target_smote
    model_path = args.model_path
    phases = args.phases.split(",") if args.phases else None
    threshold = args.threshold

    if args.separate_domains:
        separate_domains = True
        cross_domain_scaling = False
    else:
        separate_domains = False
        cross_domain_scaling = True

    if args.source_smote:
        source_smote = True
    else:
        source_smote = False

    if args.target_smote:
        target_smote = True
    else:
        target_smote = False

    # Build training configuration from boolean flags
    train_config = build_train_config(
        separate_domains=separate_domains,
        source_smote=source_smote,
        target_smote=target_smote,
    )

    logger.info(f"Training configuration: {train_config}")

    # Construct the parameters dictionary
    params = build_params(
        threshold=args.threshold,
        inhibitor_descriptor=inhibitor_descriptor,
        biological_descriptor=biological_descriptor,
        affinity_type=affinity_type,
        modality=modality,
        data_scaler=data_scaler,
        cross_domain_scaling=cross_domain_scaling,
        source_smote=source_smote,
        target_smote=target_smote,
    )

    logger.info(f"Configuration: {params}")

    # Load all models
    logger.info("Loading trained models...")
    models = load_all_models(
        model_path,
        params,
        inhibitor_descriptor,
        modality,
        data_scaler,
        affinity_type,
        train_config,
        phases,
    )

    if not any(models.values()):
        raise RuntimeError("No models were successfully loaded")

    # Load the target dataset and compute the normalization statistics per fold,
    # rather than reading a saved normalization_stats.pkl. The stored file is the
    # same object everywhere it appears, so reusing it would silently apply one
    # fold's statistics to all five and leak the other folds' training data into
    # this prediction. Recomputing here costs one dataset load and keeps each
    # fold's model paired with the statistics it was actually fitted under.
    # RECONSTRUCTED COMMENT: the original wording was lost with the file; the
    # code below is byte-exact against the surviving bytecode.
    logger.info(
        "Loading target dataset and computing fold-specific normalization stats..."
    )
    fold_stats = load_target_dataset_and_compute_fold_stats(
        dataset_name="caged",  # Target dataset name
        affinity_type=affinity_type,
        modality=modality,
        scaling_method=data_scaler if data_scaler != "none" else None,
        cross_domain_scaling=cross_domain_scaling,
        n_folds=params["n_folds"],
    )

    if not fold_stats:
        raise RuntimeError("No normalization stats could be computed for the folds")

    # Initialize ESM-C model if needed
    esm_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if biological_descriptor == "ESM-C":
        logger.info("Initializing ESM-C model...")
        from esm.models.esmc import ESMC

        esm_model = ESMC.from_pretrained("esmc_600m").to(device)
        esm_model.eval()

    # Initialize MACE calculator if needed
    mace_calculator = None
    if inhibitor_descriptor == "MACE":
        logger.info("Initializing MACE calculator...")
        mace_calculator = create_mace_calculator(args.mace_model_path)
        if mace_calculator is None:
            raise RuntimeError("Failed to initialize MACE calculator")

    # Process input files
    logger.info(f"Processing XYZ file: {args.xyz_file}")
    optimized_coords, optimized_atom_types = optimize_with_gfn2_xtb(
        args.xyz_file,
        args.output_xyz if args.output_xyz else "optimized.xyz",
        charge=args.charge,
        opt_fmax=args.opt_fmax,
        force=getattr(args, "force_opt", False),
    )

    # Generate molecular embedding
    logger.info(f"Generating {inhibitor_descriptor} molecular embedding...")
    mol_embedding = convert_xyz_to_embedding(
        inhibitor_descriptor, optimized_coords, optimized_atom_types, mace_calculator
    )

    # Generate protein embedding and (optionally) pocket embedding
    logger.info(f"Generating {biological_descriptor} biological embedding...")
    pocket_embedding_np = None
    if biological_descriptor == "ESM-C":
        from esm.sdk.api import ProteinComplex, ESMProtein

        # Parse PDB and create ESM inputs
        protein_complex = ProteinComplex.from_pdb(args.pdb_file)
        protein_multi = ESMProtein.from_protein_complex(protein_complex)
        # Embeddings (token layout: [CLS] residues... [EOS] ...)
        protein_embedding = generate_embeddings_esmc(protein_multi, esm_model, device)
        # If pocket modality is used, slice pocket tokens from full embedding
        if "pocket" in modality:
            chain_mapping = create_chain_position_mapping(
                protein_complex, protein_multi.sequence
            )
            pocket_residues = get_pocket_residues(args.pdb_file)
            if pocket_residues:
                pocket_tensor = extract_pocket_embeddings(
                    protein_embedding, pocket_residues, chain_mapping
                )
                if hasattr(pocket_tensor, "numel") and pocket_tensor.numel() > 0:
                    # Average over pocket residues to a single vector for the model input
                    # Convert from BFloat16 to Float32 before converting to NumPy
                    pocket_embedding_np = (
                        pocket_tensor.float().mean(dim=0).cpu().numpy()
                    )
                else:
                    logger.warning(
                        "No valid pocket residues mapped to embeddings; proceeding without pocket"
                    )
            else:
                logger.warning(
                    "P2Rank returned no pocket residues; proceeding without pocket"
                )
        # If modality expects pocket but none available, provide zero vector to avoid shape errors
        if "pocket" in modality and pocket_embedding_np is None:
            hidden_dim = (
                protein_embedding.shape[-1]
                if hasattr(protein_embedding, "shape")
                else 1152
            )
            logger.info("No pocket embedding available; using zero vector as fallback")
            pocket_embedding_np = np.zeros((hidden_dim,), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported biological descriptor: {biological_descriptor}")

    # Convert embeddings to numpy
    mol_vec_np = (
        mol_embedding
        if isinstance(mol_embedding, np.ndarray)
        else mol_embedding.detach().cpu().numpy()
    )
    if hasattr(protein_embedding, "float"):
        protein_embedding = protein_embedding.float()
    prot_vec_np = (
        protein_embedding
        if isinstance(protein_embedding, np.ndarray)
        else protein_embedding.detach().cpu().numpy()
    )
    pocket_vec_np = pocket_embedding_np if pocket_embedding_np is not None else None

    # Make ensemble prediction using fold-specific stats computed on-the-fly
    logger.info("Making ensemble prediction with fold-specific normalization...")
    prediction = ensemble_predict_with_fold_specific_stats(
        models,
        mol_vec_np,
        prot_vec_np,
        pocket_vec_np,
        fold_stats,
        modality,
        params["threshold"],
    )

    # Print results
    logger.info("=" * 50)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 50)
    logger.info(f"Input XYZ file: {args.xyz_file}")
    logger.info(f"Input PDB file: {args.pdb_file}")
    logger.info(
        f"Optimized XYZ saved to: {args.output_xyz if args.output_xyz else 'optimized.xyz'}"
    )
    logger.info(f"Molecular descriptor: {inhibitor_descriptor}")
    logger.info(f"Biological descriptor: {biological_descriptor}")
    logger.info(f"Modality: {modality}")
    logger.info(f"Threshold: {params['threshold']}")
    logger.info(f"Number of models used: {prediction['n_models_used']}")
    logger.info(f"Ensemble probability: {prediction['ensemble_probability']:.4f}")
    logger.info(f"Ensemble logit: {prediction['ensemble_logit']:.4f}")
    logger.info(f"Binary prediction: {prediction['binary_prediction']}")
    logger.info(
        f"Predicted class: {'ACTIVE' if prediction['is_active'] else 'INACTIVE'}"
    )
    logger.info("=" * 50)

    # Save results to file if requested
    if args.output_file:
        # Convert numpy float32 values to Python floats for JSON serialization
        prediction_json = {
            "n_models_used": int(prediction["n_models_used"]),
            "ensemble_probability": float(prediction["ensemble_probability"]),
            "ensemble_logit": float(prediction["ensemble_logit"]),
            "binary_prediction": bool(prediction["binary_prediction"]),
            "is_active": bool(prediction["is_active"]),

            # Keep the per-fold probabilities, not just their mean: the folds
            # disagree enough that the spread is part of the result.
            "fold_probabilities": [
                float(p) for p in prediction["individual_predictions"]
            ],
        }

        results = {
            "input_files": {
                "xyz_file": args.xyz_file,
                "pdb_file": args.pdb_file,
                "optimized_xyz": (
                    args.output_xyz if args.output_xyz else "optimized.xyz"
                ),
            },
            "configuration": {
                "inhibitor_descriptor": inhibitor_descriptor,
                "biological_descriptor": biological_descriptor,
                "modality": modality,
                "affinity_type": affinity_type,
                "threshold": float(params["threshold"]),
            },
            "prediction": prediction_json,
        }

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer Learning Model Inference Script"
    )

    # Input files
    parser.add_argument(
        "--xyz_file", type=str, required=True, help="Path to input XYZ file"
    )
    parser.add_argument(
        "--pdb_file", type=str, required=True, help="Path to input PDB file"
    )

    # Model configuration
    parser.add_argument(
        "--threshold", type=float, default=6.0, help="Threshold for prediction"
    )
    parser.add_argument(
        "--inhibitor_descriptor",
        type=str,
        default="SOAP",
        choices=["SOAP", "MACE", "ACSF", "MBTR"],
        help="Molecular descriptor type",
    )
    parser.add_argument(
        "--biological_descriptor",
        type=str,
        default="ESM-C",
        choices=["ESM-C"],
        help="Biological descriptor type",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="molecule",
        choices=[
            "molecule",
            "protein",
            "pocket",
            "molecule_protein",
            "molecule_pocket",
            "protein_pocket",
            "molecule_protein_pocket",
        ],
        help="Model modality",
    )
    parser.add_argument(
        "--affinity_type",
        type=str,
        default="pic50",
        choices=["pic50", "pk"],
        help="Affinity type",
    )
    parser.add_argument(
        "--data_scaler",
        type=str,
        default="minmax",
        choices=["minmax", "standard", "none"],
        help="Data scaling method",
    )
    parser.add_argument(
        "--separate_domains", action="store_true", help="Use separate domain training"
    )
    parser.add_argument(
        "--source_smote", action="store_true", help="Use SMOTE for source data"
    )
    parser.add_argument(
        "--target_smote", action="store_true", help="Use SMOTE for target data"
    )

    # Model paths
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--mace_model_path",
        type=str,
        default="../mace_model/MACE-MP-0b3.model",
        help="Path to MACE model file",
    )

    # Training configuration
    parser.add_argument(
        "--phases",
        type=str,
        default="no_pretrain_ccsa,no_pretrain_ccsa_encoder_task,no_pretrain_ccsa_task,no_pretrain_encoder_task",
        help="Comma-separated list of phases to use",
    )

    # Optimization parameters
    parser.add_argument(
        "--charge", type=int, default=0, help="Molecular charge for optimization"
    )
    parser.add_argument(
        "--opt_fmax",
        type=float,
        default=0.05,
        help="Force convergence criterion for optimization",
    )

    # Output options
    parser.add_argument("--output_xyz", type=str, help="Path for optimized XYZ output")
    parser.add_argument("--output_file", type=str, help="Path for JSON results output")
    parser.add_argument(
        "--force_opt",
        action="store_true",
        help="Force GFN2-xTB re-optimization even if output exists",
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise
