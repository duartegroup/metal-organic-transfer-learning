import h5py
import os
import glob
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
import pandas as pd
import logging
from tqdm import tqdm
import sys
from Bio.PDB import PDBParser, Polypeptide
import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, ProteinComplex, LogitsConfig

import gc

path = str(Path(__file__).resolve().parent.parent.parent.parent / "src" / "data_src")
sys.path.append(path)

from create_hf5_common import (
    generate_embeddings_esmc,
    create_chain_position_mapping,
    extract_pocket_embeddings,
    extract_pocket_edges,
    extract_sdf,
    create_mace_calculator,
    add_mace_embeddings_to_entry,
    compute_mbtr_embedding,
    compute_soap_embedding,
    compute_acsf_embedding,
    generate_embeddings_contact_esm_plus_plus,
    generate_distance_matrix,
    generate_spatial_position_matrix,
    extract_xyz,
    create_pocket_mask,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set default pocket threshold
DEFAULT_POCKET_THRESHOLD = 6.0


def extract_organic_csv(csv_file: str) -> List[str]:
    """
    Extract data from organic CSV file to get entry names.

    Parameters
    ----------
    csv_file : str
        Path to the organic CSV file

    Returns
    -------
    List[str]
        List of entry names
    """
    try:
        df = pd.read_csv(csv_file)

        # Create names following the same pattern as in protein_embeddings.py
        pdb_id = df["PDBID"].tolist()
        ligand_name = df["Ligand Name"].tolist()
        chain_id = df["Ligand Chain"].tolist()
        ligand_residue_number = df["Ligand Residue Number"].tolist()

        names = [
            f"{pdb_id[i]}_{ligand_name[i]}_{chain_id[i]}_{ligand_residue_number[i]}"
            for i in range(len(pdb_id))
        ]

        # Extract target values in CSV order
        tgt_vals = df["Log Binding Affinity"].values
        tgt_vals = [
            float(ent) * -1 for ent in tgt_vals
        ]  # convert to positive values HiQBind reports negative log values

        logger.info(f"Extracted {len(names)} entry names from CSV")
        return names, tgt_vals
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_file}: {e}")
        return []


def find_pocket_residues(
    pdb_file: str,
    ligand_coords: np.ndarray,
    pocket_threshold: float = 6.0,
) -> List:
    """
    Find pocket residues using distance calculations.

    Parameters
    ----------
    structure : Structure
        Protein structure
    ligand_coords : np.ndarray
        Ligand coordinates
    residue_positions : Dict[str, int]
        Residue positions
    pocket_threshold : float
        Distance threshold
    close_chains : Optional[Set[str]]
        Set of chain IDs to consider. If None, considers all chains.

    Returns
    -------
    List[PocketResidue]
        List of pocket residues
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    pocket_residues = []

    for model in structure:
        for chain in model:
            chain_id = chain.id

            # Collect residue information
            for res in chain.get_residues():
                # Skip water, ligands, or non-standard residues
                if (
                    res.get_resname() == "HOH"
                    or res.get_id()[0] != " "
                    or not Polypeptide.is_aa(res, standard=True)
                ):
                    continue

                # Get atom coordinates for this residue
                atom_coords = np.array([atom.coord for atom in res.get_atoms()])
                if len(atom_coords) == 0:
                    continue

                # Calculate minimum distance between any atom in residue and any atom in ligand
                # Reshape for broadcasting: (n_res_atoms, 1, 3) - (1, n_lig_atoms, 3)
                res_coords = atom_coords.reshape(-1, 1, 3)
                lig_coords = ligand_coords.reshape(1, -1, 3)

                # Calculate pairwise distances
                distances = np.linalg.norm(res_coords - lig_coords, axis=2)

                # If any atom pair is within threshold, add to pocket
                if np.any(distances < pocket_threshold):
                    # Map to sequence position
                    residue_key = f"{chain_id}_{res.id[1]}"
                    pocket_residues.append(residue_key)

    return pocket_residues


# def create_pocket_mask(
#     pocket_residues: List[str],
#     chain_mapping: Dict[str, Dict[int, int]],
#     sequence_length: int,
# ) -> np.ndarray:
#     """
#     Create a binary mask indicating which residues are part of the binding pocket.
    
#     The mask corresponds to the actual protein sequence positions (excluding special tokens).
#     The embedding tensor structure is: [CLS] residue_1 residue_2 ... residue_n [EOS] [PAD] ...
#                                         0    1        2              n        n+1   n+2
    
#     But the pocket mask corresponds to:    residue_1 residue_2 ... residue_n
#                                            0        1              n-1
    
#     Parameters
#     ----------
#     pocket_residues : List[str]
#         List of residue IDs like ['A_116', 'B_127', ...]
#     chain_mapping : Dict[str, Dict[int, int]]
#         Mapping from chain+residue to ESM sequence position (includes CLS offset)
#     sequence_length : int
#         Total length of the protein sequence (excluding special tokens)
    
#     Returns
#     -------
#     pocket_mask : np.ndarray
#         Binary array of shape (sequence_length,) where 1 indicates pocket residue
#     """
#     # Initialize mask with zeros
#     pocket_mask = np.zeros(sequence_length, dtype=np.int32)
    
#     logger.debug(f"Creating pocket mask for sequence length: {sequence_length}")
#     logger.debug(f"Processing {len(pocket_residues)} pocket residues")
    
#     # Set pocket positions to 1
#     for residue_id in pocket_residues:
#         try:
#             # Parse residue ID (e.g., 'A_116' -> chain='A', res_num=116)
#             chain_id, res_num_str = residue_id.split("_")
            
#             # Handle cases like 'C_50B' where there's a letter suffix
#             res_num = int("".join(filter(str.isdigit, res_num_str)))
            
#             # Get ESM sequence position (includes CLS offset)
#             if chain_id in chain_mapping and res_num in chain_mapping[chain_id]:
#                 esm_pos = chain_mapping[chain_id][res_num]
                
#                 # Convert ESM position (which includes CLS offset) to mask position
#                 # ESM position 1 -> mask position 0, ESM position 2 -> mask position 1, etc.
#                 mask_pos = esm_pos - 1  # Subtract 1 to account for CLS token
                
#                 # Check bounds to avoid index errors
#                 if 0 <= mask_pos < sequence_length:
#                     pocket_mask[mask_pos] = 1
#                     logger.debug(f"Mapped residue {residue_id}: ESM pos {esm_pos} -> mask pos {mask_pos}")
#                 else:
#                     logger.debug(f"Warning: Mask position {mask_pos} (ESM pos {esm_pos}) out of bounds for residue {residue_id}")
#             else:
#                 logger.debug(f"Warning: Could not map residue {residue_id}")
                
#         except (ValueError, IndexError) as e:
#             logger.debug(f"Warning: Could not parse residue ID {residue_id}: {e}")
#             continue
    
#     logger.debug(f"Created pocket mask: {np.sum(pocket_mask)} pocket residues out of {sequence_length} total")
#     return pocket_mask


# === ORGANIC PROCESSING FUNCTIONS === #
# Add global error tracking
failed_entries = []


def process_single_entry(
    subdir_path: Path,
    model,
    device,
    include_protein: bool = True,
    include_gfn2: bool = True,
    include_sdf: bool = True,
    include_pocket: bool = False,
    include_mace: bool = False,
    include_mbtr: bool = False,
    include_soap: bool = False,
    include_acsf: bool = False,
    pocket_threshold: float = 6.0,
    mace_calculator=None,
) -> Optional[Dict]:
    """
    Process a single subdirectory entry with improved structure and error handling.

    Parameters
    ----------
    subdir_path : Path
        Path to the subdirectory to process
    include_protein : bool
        Whether to include protein PDB files
    include_gfn2 : bool
        Whether to include GFN2-xTB XYZ files
    include_sdf : bool
        Whether to include SDF files from pocket geometry
    include_pocket : bool
        Whether to extract pocket residues
    include_mace : bool
        Whether to include MACE embeddings
    include_mbtr : bool
        Whether to include MBTR embeddings
    include_soap : bool
        Whether to include SOAP embeddings
    include_acsf : bool
        Whether to include ACSF embeddings
    pocket_threshold : float
        Distance threshold for pocket definition
    mace_calculator : Any or None
        The MACE calculator instance

    Returns
    -------
    Optional[Dict]
        Entry data dictionary if successful, None if failed
    """
    subdir = Path(subdir_path)
    entry_name = subdir.name
    entry_data = {"name": entry_name}

    # Initialize components
    # Find chains close to ligand first if we need protein data
    if include_protein or include_pocket:
        pdb_files = list(subdir.glob("*_protein_refined.pdb"))
        sdf_files = list(subdir.glob("*_ligand_refined.sdf"))

        if not pdb_files:
            logger.debug(f"No protein PDB file found for {entry_name}")
            return None

        if not sdf_files:
            logger.debug(f"No ligand SDF file found for {entry_name}")
            return None

        pdb_file = pdb_files[0]
        sdf_file = sdf_files[0]

        # Process protein PDB file if requested
        if include_protein or include_pocket:
            try:
                protein_complex = ProteinComplex.from_pdb(pdb_file)
                protein_multi = ESMProtein.from_protein_complex(protein_complex)
                chain_mapping = create_chain_position_mapping(
                    protein_complex, protein_multi.sequence
                )

                protein_embeddings = generate_embeddings_esmc(
                    protein_multi, model, device
                )
                # embeddings, _ = generate_embeddings_contact_esm_plus_plus(
                #     protein_multi, model, device
                # )
                # protein_embeddings = embeddings

                if include_protein:
                    entry_data["protein_sequence"] = protein_multi.sequence
                    entry_data["protein_embeddings"] = protein_embeddings
                    # entry_data["protein_attentions"] = (
                        # attentions  # 33 layers of attention maps
                    # )
            except Exception as e:
                # Track failed entries with detailed error information
                error_info = {
                    "entry_name": entry_name,
                    "pdb_file": str(pdb_file),
                    "error_type": "PDB_PROCESSING",
                    "error_message": str(e),
                    "error_details": f"{type(e).__name__}: {str(e)}",
                }
                failed_entries.append(error_info)
                logger.error(
                    f"Error reading PDB file for {entry_name}: {error_info['error_details']}"
                )
                return None

    # Process GFN2-xTB XYZ file if requested
    if include_gfn2:
        gfn2_dir = subdir / "GFN2-xTB"
        if not gfn2_dir.exists():
            logger.debug(f"GFN2-xTB directory does not exist: {gfn2_dir}")
            return None

        xyz_files = list(gfn2_dir.glob("*.xyz"))
        xyz_files = [f for f in xyz_files if "_trj" not in f.name]

        if not xyz_files:
            logger.debug(f"No XYZ file found for {entry_name}")
            return None

        xyz_file = xyz_files[0]
        mol_gfn2_coords, mol_gfn2_atype = extract_xyz(str(xyz_file))

        if mol_gfn2_coords.shape[0] == 0 or len(mol_gfn2_atype) == 0:
            logger.debug(f"Empty XYZ data for {entry_name}")
            return None

        if include_gfn2:
            entry_data["mol_gfn2_coords"] = mol_gfn2_coords
            entry_data["mol_gfn2_atype"] = mol_gfn2_atype

    if include_sdf:
        sdf_coords, sdf_atype, sdf_bonds = extract_sdf(sdf_file)
        entry_data["sdf_coords"] = sdf_coords
        entry_data["sdf_atype"] = sdf_atype
        entry_data["sdf_bonds"] = sdf_bonds
        
    # Process pocket extraction if requested
    if include_pocket:
        try:
            pocket_residues = find_pocket_residues(
                pdb_file, sdf_coords, pocket_threshold
            )
            pocket_embeddings = extract_pocket_embeddings(
                protein_embeddings, pocket_residues, chain_mapping
            )
            pocket_edges = extract_pocket_edges(pocket_residues, pdb_file)
            
            # Create binary pocket mask
            sequence_length = len(protein_multi.sequence)
            pocket_mask = create_pocket_mask(
                pocket_residues, chain_mapping, sequence_length
            )
            
            entry_data["pocket_residues"] = pocket_residues
            entry_data["pocket_embeddings"] = pocket_embeddings
            entry_data["pocket_edges"] = pocket_edges
            entry_data["pocket_mask"] = pocket_mask  # Add this line
            
        except Exception as e:
            error_info = {
                "entry_name": entry_name,
                "pdb_file": str(pdb_file),
                "error_type": "POCKET_EXTRACTION",
                "error_message": str(e),
                "error_details": f"{type(e).__name__}: {str(e)}",
            }
            failed_entries.append(error_info)
            logger.error(
                f"Error extracting pocket for {entry_name}: {error_info['error_details']}"
            )
            return None


    # Process molecular embeddings that depend on GFN2 data
    if include_mbtr and include_gfn2:
        entry_data["mol_mbtr_embedding"] = compute_mbtr_embedding(
            mol_gfn2_coords, mol_gfn2_atype
        )

    if include_soap and include_gfn2:
        entry_data["mol_soap_embedding"] = compute_soap_embedding(
            mol_gfn2_coords, mol_gfn2_atype
        )

    if include_acsf and include_gfn2:
        entry_data["mol_acsf_embedding"] = compute_acsf_embedding(
            mol_gfn2_coords, mol_gfn2_atype
        )

    # Add MACE embeddings if requested (depends on GFN2 data)
    if include_mace and include_gfn2:
        entry_data = add_mace_embeddings_to_entry(
            entry_data, mol_gfn2_coords, mol_gfn2_atype, mace_calculator, include_mace
        )

    # Debug: Show what data was collected before validation
    logger.debug(f"DEBUG for {entry_name}: Collected keys: {list(entry_data.keys())}")
    for key, value in entry_data.items():
        if key != "name":
            if hasattr(value, "shape"):
                logger.debug(f"  {key}: shape={value.shape}, type={type(value)}")
            elif hasattr(value, "__len__"):
                logger.debug(f"  {key}: length={len(value)}, type={type(value)}")
            else:
                logger.debug(f"  {key}: value={value}, type={type(value)}")

    # Validate entry
    if validate_entry_data(
        entry_data,
        include_protein,
        include_gfn2,
        include_pocket,
        include_mace,
        include_mbtr,
        include_soap,
        include_acsf,
        include_sdf,
    ):
        return entry_data
    else:
        logger.debug(f"Invalid data for {entry_name}")
        return None


def validate_entry_data(
    entry: Dict,
    include_protein: bool,
    include_gfn2: bool,
    include_pocket: bool,
    include_mace: bool,
    include_mbtr: bool,
    include_soap: bool,
    include_acsf: bool,
    include_sdf: bool,
) -> bool:
    """Validate that entry contains required data."""
    entry_name = entry.get("name", "unknown")

    if include_protein:
        # Check for protein sequence
        protein_sequence = entry.get("protein_sequence")
        if not protein_sequence or not protein_sequence.strip():
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Missing or empty protein_sequence"
            )
            return False
        # Check for protein embeddings
        protein_embeddings = entry.get("protein_embeddings")
        if protein_embeddings is None:
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Missing protein_embeddings"
            )
            return False

    if include_sdf:
        sdf_coords = entry.get("sdf_coords")
        if sdf_coords is None or (hasattr(sdf_coords, "size") and sdf_coords.size == 0):
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Missing or empty sdf_coords"
            )
            return False

    if include_gfn2:
        # Check for mol_gfn2_coords (numpy array)
        mol_gfn2_coords = entry.get("mol_gfn2_coords")
        if mol_gfn2_coords is None or (
            hasattr(mol_gfn2_coords, "size") and mol_gfn2_coords.size == 0
        ):
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Missing or empty mol_gfn2_coords"
            )
            return False
        # Check for mol_gfn2_atype (list)
        mol_gfn2_atype = entry.get("mol_gfn2_atype")
        if not mol_gfn2_atype:
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Missing or empty mol_gfn2_atype"
            )
            return False

        # Only check for ligand coordinates if pocket extraction is enabled
        # (since sdf data is only extracted during pocket processing)
        if include_pocket:
            sdf_coords = entry.get("sdf_coords")
            if sdf_coords is None or (
                hasattr(sdf_coords, "size") and sdf_coords.size == 0
            ):
                logger.debug(
                    f"VALIDATION FAILED for {entry_name}: Missing or empty sdf_coords"
                )
                return False
            sdf_atype = entry.get("sdf_atype")
            if sdf_atype is None or (
                hasattr(sdf_atype, "size") and sdf_atype.size == 0
            ):
                logger.debug(
                    f"VALIDATION FAILED for {entry_name}: Missing or empty sdf_atype"
                )
                return False
            sdf_bonds = entry.get("sdf_bonds")
            if sdf_bonds is None or (
                hasattr(sdf_bonds, "size") and sdf_bonds.size == 0
            ):
                logger.debug(
                    f"VALIDATION FAILED for {entry_name}: Missing or empty sdf_bonds"
                )
                return False

    if include_pocket:
        # Check for pocket residues
        pocket_residues = entry.get("pocket_residues")
        if not pocket_residues:
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Missing or empty pocket_residues"
            )
            return False
        # Check for pocket embeddings
        pocket_embeddings = entry.get("pocket_embeddings")
        if pocket_embeddings is None:
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Missing pocket_embeddings"
            )
            return False
        # Check for pocket edges
        pocket_edges = entry.get("pocket_edges")
        if pocket_edges is None or (
            hasattr(pocket_edges, "size") and pocket_edges.size == 0
        ):
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Missing or empty pocket_edges"
            )
            return False
        # Check for pocket mask
        pocket_mask = entry.get("pocket_mask")
        if pocket_mask is None or (
            hasattr(pocket_mask, "size") and pocket_mask.size == 0
        ):
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Missing or empty pocket_mask"
            )
            return False

    # Check molecular embeddings (only if GFN2 is also enabled)
    if include_mace and include_gfn2:
        # Check for MACE embedding (make it optional - don't fail if missing)
        mace_embedding = entry.get("mol_mace_embedding")
        if (
            mace_embedding is not None
            and hasattr(mace_embedding, "size")
            and mace_embedding.size == 0
        ):
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Empty mol_mace_embedding"
            )
            return False

    if include_mbtr and include_gfn2:
        # Check for MBTR embedding (make it optional - don't fail if missing)
        mbtr_embedding = entry.get("mol_mbtr_embedding")
        if (
            mbtr_embedding is not None
            and hasattr(mbtr_embedding, "size")
            and mbtr_embedding.size == 0
        ):
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Empty mol_mbtr_embedding"
            )
            return False

    if include_soap and include_gfn2:
        soap_embedding = entry.get("mol_soap_embedding")
        if (
            soap_embedding is not None
            and hasattr(soap_embedding, "size")
            and soap_embedding.size == 0
        ):
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Empty mol_soap_embedding"
            )
            return False

    if include_acsf and include_gfn2:
        acsf_embedding = entry.get("mol_acsf_embedding")
        if (
            acsf_embedding is not None
            and hasattr(acsf_embedding, "size")
            and acsf_embedding.size == 0
        ):
            logger.debug(
                f"VALIDATION FAILED for {entry_name}: Empty mol_acsf_embedding"
            )
            return False

    logger.debug(f"VALIDATION PASSED for {entry_name}")
    return True


def create_hdf5_dataset(
    data_dir: Path,
    csv_file: str = None,
    output_file: str = "dataset.h5",
    include_protein: bool = False,
    include_gfn2: bool = True,
    include_pocket: bool = True,
    include_sdf: bool = True,
    include_mace: bool = True,
    include_mbtr: bool = False,
    include_soap: bool = False,
    include_acsf: bool = False,
    mace_model_path: str = "../mace_model/MACE-MP-0b3.model",
    pocket_threshold: float = DEFAULT_POCKET_THRESHOLD,
):
    """
    Create HDF5 dataset with improved structure and error handling.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory
    csv_file : str, optional
        Path to the CSV file containing entries to process
    output_file : str
        Name of the output HDF5 file
    include_protein : bool
        Whether to include protein PDB files
    include_gfn2 : bool
        Whether to include GFN2-xTB XYZ files
    include_pocket : bool
        Whether to extract pocket residues
    include_sdf : bool
        Whether to include SDF files from pocket geometry
    include_mace : bool
        Whether to include MACE embeddings
    mace_model_path : str
        Path to MACE model file
    pocket_threshold : float
        Distance threshold for pocket definition
    """
    # Clear failed entries list at start
    global failed_entries
    failed_entries = []

    if not include_protein and not include_pocket and not include_mace:
        logger.error(
            "At least one of --protein, --gfn2, or --pocket must be specified!"
        )
        return

    # Load valid names from CSV if provided
    name_to_target = None
    if csv_file and os.path.exists(csv_file):
        logger.info(f"Reading CSV file: {csv_file}")
        valid_names, tgt_vals = extract_organic_csv(csv_file)
        logger.info(f"Found {len(valid_names)} entries in CSV file")

        # Create mapping from entry names to target values
        name_to_target = dict(zip(valid_names, tgt_vals))
        logger.info(f"Created target value mapping for {len(name_to_target)} entries")
    else:
        valid_names = None
        logger.info(
            "No CSV file provided or file doesn't exist, processing all directories"
        )

    # Check if data directory exists
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return

    # Collect all subdirectories to process
    all_subdirs = []

    if valid_names is not None:
        # Optimized: construct paths directly from CSV entries
        logger.info("Constructing paths directly from CSV entries...")
        for entry_name in valid_names:
            # Extract PDB ID from entry name (first 4 characters)
            pdb_id = entry_name[:4]
            subdir_path = data_dir / pdb_id / entry_name

            # Check if the directory exists
            if subdir_path.exists() and subdir_path.is_dir():
                all_subdirs.append(subdir_path)
            else:
                logger.debug(f"Directory not found: {subdir_path}")

        logger.info(
            f"Found {len(all_subdirs)} valid directories from {len(valid_names)} CSV entries"
        )
    else:
        # Fallback: scan all directories (original behavior)
        logger.info("Scanning all directories...")
        pdb_dirs = [d for d in data_dir.iterdir() if d.is_dir() and len(d.name) == 4]
        logger.info(f"Found {len(pdb_dirs)} PDB directories")

        for pdb_dir in pdb_dirs:
            subdirs = [d for d in pdb_dir.iterdir() if d.is_dir()]
            all_subdirs.extend(subdirs)

        logger.info(f"Found {len(all_subdirs)} total subdirectories")

    all_subdirs = all_subdirs

    logger.info(f"Processing {len(all_subdirs)} subdirectories")
    logger.info(f"Including protein files: {include_protein}")
    logger.info(f"Including GFN2-xTB files: {include_gfn2}")
    logger.info(f"Including pocket extraction: {include_pocket}")
    logger.info(f"Including SDF files: {include_sdf}")
    logger.info(f"Including MACE embeddings: {include_mace}")
    logger.info(f"Including MBTR embeddings: {include_mbtr}")
    logger.info(f"Including SOAP embeddings: {include_soap}")
    logger.info(f"Including ACSF embeddings: {include_acsf}")
    if include_pocket:
        logger.info(f"Pocket threshold: {pocket_threshold} Å")

    if not all_subdirs:
        logger.error("No subdirectories to process!")
        return

    # Initialize model once at the beginning if needed
    if include_protein or include_pocket:
        logger.info("Initializing ESM-C model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ESMC.from_pretrained("esmc_600m").to(device)
        model.eval()
        # from transformers import AutoModelForMaskedLM

        # model = AutoModelForMaskedLM.from_pretrained(
        #     "Synthyra/ESMplusplus_large", trust_remote_code=True
        # )
        logger.info(f"Model initialized on device: {device}")
    else:
        model = None
        device = None

    # Initialize MACE calculator if needed
    mace_calculator = None
    if include_mace:
        logger.info("Initializing MACE calculator...")
        mace_calculator = create_mace_calculator(mace_model_path)
        if mace_calculator is None:
            logger.warning(
                "Failed to initialize MACE calculator, proceeding without MACE embeddings"
            )
            include_mace = False

    # Process entries sequentially with tqdm progress bar
    start_time = time.time()
    logger.info("Processing entries sequentially...")

    data_entries = []
    for subdir_path in tqdm(all_subdirs, desc="Processing entries"):
        result = process_single_entry(
            subdir_path,
            model=model,
            device=device,
            include_protein=include_protein,
            include_gfn2=include_gfn2,
            include_pocket=include_pocket,
            include_sdf=include_sdf,
            include_mace=include_mace,
            include_mbtr=include_mbtr,
            include_soap=include_soap,
            include_acsf=include_acsf,
            pocket_threshold=pocket_threshold,
            mace_calculator=mace_calculator,
        )
        if result is not None:
            data_entries.append(result)

        # clear the memory
        del result
        gc.collect()
        torch.cuda.empty_cache()

    processing_time = time.time() - start_time
    success_rate = len(data_entries) / len(all_subdirs) * 100

    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    logger.info(
        f"Successfully processed: {len(data_entries)}/{len(all_subdirs)} entries ({success_rate:.1f}%)"
    )

    # Report failed entries in detail
    if failed_entries:
        logger.info("=" * 60)
        logger.info(f"FAILED ENTRIES SUMMARY: {len(failed_entries)} entries failed")
        logger.info("=" * 60)

        # Group by error type
        error_types = {}
        for error in failed_entries:
            error_type = error["error_type"]
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)

        for error_type, errors in error_types.items():
            logger.info(f"\n{error_type} ERRORS ({len(errors)} entries):")
            logger.info("-" * 40)
            for error in errors:
                logger.info(f"  - {error['entry_name']}: {error['error_details']}")
                logger.info(f"    PDB file: {error['pdb_file']}")

        # Save failed entries to a text file for easy review
        failed_file = output_file.replace(".h5", "_failed_entries.txt")
        with open(failed_file, "w") as f:
            f.write(f"Failed Entries Report\n")
            f.write(f"Generated: {time.ctime()}\n")
            f.write(f"Total failed: {len(failed_entries)}\n\n")

            for error_type, errors in error_types.items():
                f.write(f"\n{error_type} ERRORS ({len(errors)} entries):\n")
                f.write("-" * 40 + "\n")
                for error in errors:
                    f.write(f"Entry: {error['entry_name']}\n")
                    f.write(f"PDB file: {error['pdb_file']}\n")
                    f.write(f"Error: {error['error_details']}\n")
                    f.write(f"Message: {error['error_message']}\n\n")

        logger.info(f"\nDetailed error report saved to: {failed_file}")
        logger.info("=" * 60)

    if not data_entries:
        logger.error("No entries were successfully processed!")
        return

    # Add target values to entries if available
    if name_to_target:
        logger.info("Adding target values to processed entries...")
        for entry in data_entries:
            entry_name = entry["name"]
            if entry_name in name_to_target:
                entry["target_value"] = name_to_target[entry_name]

        entries_with_targets = sum(
            1 for entry in data_entries if "target_value" in entry
        )
        logger.info(
            f"Added target values to {entries_with_targets}/{len(data_entries)} entries"
        )

    # Save to HDF5 (update the function call to remove num_processes)
    save_to_hdf5(
        data_entries,
        output_file,
        include_protein,
        include_gfn2,
        include_pocket,
        include_sdf,
        include_mace,
        include_mbtr,
        include_soap,
        include_acsf,
        pocket_threshold,
        processing_time,
    )


def save_to_hdf5(
    data_entries: List[Dict],
    output_file: str,
    include_protein: bool,
    include_gfn2: bool,
    include_pocket: bool,
    include_sdf: bool,
    include_mace: bool,
    include_mbtr: bool,
    include_soap: bool,
    include_acsf: bool,
    pocket_threshold: float,
    processing_time: float,
):
    """Save data entries to HDF5 file."""
    logger.info("Creating HDF5 file...")

    with h5py.File(output_file, "w") as h5f:
        dt_string = h5py.string_dtype(encoding="utf-8")

        for entry in data_entries:
            entry_name = entry["name"]
            entry_group = h5f.create_group(entry_name)

            # Store entry name
            entry_group.create_dataset("name", data=entry_name, dtype=dt_string)

            # Store target value if available
            if "target_value" in entry:
                entry_group.create_dataset("target_value", data=entry["target_value"])

            # Store protein data
            if include_protein:
                if "protein_sequence" in entry:
                    entry_group.create_dataset(
                        "protein_sequence",
                        data=entry["protein_sequence"],
                        dtype=dt_string,
                    )

                # Store protein embeddings
                if "protein_embeddings" in entry:
                    embedding_tensor = entry["protein_embeddings"]
                    embedding_array = embedding_tensor.detach().cpu().float().numpy()
                    entry_group.create_dataset(
                        "protein_embeddings", data=embedding_array
                    )
                if "protein_attentions" in entry:
                    attention_tensor = entry["protein_attentions"]
                    # convert the tensors in the tuple to numpy arrays
                    attention_array = [
                        np.array(tensor.detach().cpu().float().numpy())
                        for tensor in attention_tensor
                    ]
                    entry_group.create_dataset(
                        "protein_attentions", data=attention_array
                    )

            # Store pocket data
            if include_pocket:
                if "pocket_residues" in entry:
                    pocket_strings = [
                        str(residue) for residue in entry["pocket_residues"]
                    ]
                    if pocket_strings:
                        entry_group.create_dataset(
                            "pocket_residues", data=pocket_strings, dtype=dt_string
                        )
                    else:
                        entry_group.create_dataset(
                            "pocket_residues", data=[], dtype=dt_string
                        )

                # Store pocket embeddings
                if "pocket_embeddings" in entry:
                    pocket_embeddings = entry["pocket_embeddings"]
                    # Convert tensor to numpy array for storage - fix BFloat16 issue
                    pocket_emb_array = pocket_embeddings.detach().cpu().float().numpy()
                    entry_group.create_dataset(
                        "pocket_embeddings", data=pocket_emb_array
                    )
                if "pocket_edges" in entry:
                    entry_group.create_dataset(
                        "pocket_edges", data=entry["pocket_edges"]
                    )
                # Store pocket mask
                if "pocket_mask" in entry:
                    entry_group.create_dataset(
                        "pocket_mask", data=entry["pocket_mask"]
                    )
            if include_sdf:
                # Store ligand coordinates
                if "sdf_coords" in entry:
                    entry_group.create_dataset("sdf_coords", data=entry["sdf_coords"])
                if "sdf_atype" in entry:
                    entry_group.create_dataset("sdf_atype", data=entry["sdf_atype"])
                if "sdf_bonds" in entry:
                    entry_group.create_dataset("sdf_bonds", data=entry["sdf_bonds"])

            # Store GFN2-xTB data
            if include_gfn2:
                if "mol_gfn2_coords" in entry:
                    entry_group.create_dataset(
                        "mol_gfn2_coords", data=entry["mol_gfn2_coords"]
                    )

                if "mol_gfn2_atype" in entry:
                    atype_data = [
                        atype.encode("utf-8") for atype in entry["mol_gfn2_atype"]
                    ]
                    entry_group.create_dataset("mol_gfn2_atype", data=atype_data)

            # Store MACE embeddings
            if include_mace and "mol_mace_embedding" in entry:
                entry_group.create_dataset(
                    "mol_mace_embedding", data=entry["mol_mace_embedding"]
                )

            if include_mbtr and "mol_mbtr_embedding" in entry:
                entry_group.create_dataset(
                    "mol_mbtr_embedding", data=entry["mol_mbtr_embedding"]
                )

            if include_soap and "mol_soap_embedding" in entry:
                entry_group.create_dataset(
                    "mol_soap_embedding", data=entry["mol_soap_embedding"]
                )
            if include_acsf and "mol_acsf_embedding" in entry:
                entry_group.create_dataset(
                    "mol_acsf_embedding", data=entry["mol_acsf_embedding"]
                )

        # Save metadata
        all_names = [entry["name"] for entry in data_entries]
        h5f.create_dataset("entry_names", data=all_names, dtype=dt_string)

        # Add attributes
        description_parts = []
        if include_protein:
            description_parts.append("protein sequences and embeddings")
        if include_pocket:
            description_parts.append("pocket residues and embeddings")
        if include_gfn2:
            description_parts.append("GFN2-xTB XYZ files")
        if include_mace:
            description_parts.append("MACE embeddings")
        if include_mbtr:
            description_parts.append("MBTR embeddings")
        if include_soap:
            description_parts.append("SOAP embeddings")
        if include_acsf:
            description_parts.append("ACSF embeddings")

        h5f.attrs["description"] = (
            f"HiQBind dataset with {' and '.join(description_parts)}"
        )
        h5f.attrs["total_entries"] = len(data_entries)
        h5f.attrs["created_by"] = "create_organic_h5_improved.py"
        h5f.attrs["includes_protein"] = include_protein
        h5f.attrs["includes_gfn2"] = include_gfn2
        h5f.attrs["includes_pocket"] = include_pocket
        h5f.attrs["includes_mace"] = include_mace
        h5f.attrs["includes_mbtr"] = include_mbtr
        h5f.attrs["includes_soap"] = include_soap
        h5f.attrs["includes_acsf"] = include_acsf
        if include_pocket:
            h5f.attrs["pocket_threshold"] = pocket_threshold
        h5f.attrs["processing_time_seconds"] = processing_time

    logger.info(
        f"HDF5 file '{output_file}' created successfully with {len(data_entries)} entries"
    )


def main(args: argparse.Namespace):
    # Set up file paths
    csv_file = str(
        Path(__file__).resolve().parent.parent.parent.parent
        / "data" / "excel" / "organic" / args.affinity_type
        / f"organic_{args.affinity_type}.csv"
    )

    if args.output == ".":
        output_file = f"./organic_{args.affinity_type}_dataset.h5"
    else:
        output_file = args.output

    create_hdf5_dataset(
        Path(args.data_dir),
        csv_file,
        output_file,
        args.protein,
        args.gfn2,
        args.pocket,
        args.sdf,
        args.mace,
        args.mbtr,
        args.soap,
        args.acsf,
        args.mace_model_path,
        args.pocket_threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create HDF5 dataset from organic data (improved version)"
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        default=None,
        help="Path to the raw data directory (required)",
    )
    parser.add_argument(
        "--affinity_type",
        "-a",
        default="pic50",
        help="Type of affinity to process (pic50 or pk)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=".",
        help="Output HDF5 filename (default: organic_{affinity_type}_dataset.h5)",
    )
    parser.add_argument(
        "--protein",
        action="store_true",
        help="Include protein PDB files in the dataset",
    )
    parser.add_argument(
        "--gfn2", action="store_true", help="Include GFN2-xTB XYZ files in the dataset"
    )
    parser.add_argument(
        "--sdf",
        action="store_true",
        help="Include SDF files from pocket geometry in the dataset",
    )
    parser.add_argument(
        "--pocket",
        action="store_true",
        help="Extract pocket residues from protein-ligand complexes",
    )
    parser.add_argument(
        "--mace",
        action="store_true",
        help="Include MACE embeddings in the dataset",
    )
    parser.add_argument(
        "--mace-model-path",
        default=str(Path(__file__).resolve().parent.parent / "mace_model" / "MACE-MP-0b3.model"),
        help="Path to MACE model file",
    )
    parser.add_argument(
        "--mbtr",
        action="store_true",
        help="Include MBTR embeddings in the dataset",
    )
    parser.add_argument(
        "--soap",
        action="store_true",
        help="Include SOAP embeddings in the dataset",
    )
    parser.add_argument(
        "--acsf",
        action="store_true",
        help="Include ACSF embeddings in the dataset",
    )
    parser.add_argument(
        "--pocket_threshold",
        type=float,
        default=6.0,
        help=f"Distance threshold for pocket definition in Angstroms (default: 6.0)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Log the start of the program
    logger.info("=" * 50)
    logger.info("ORGANIC HDF5 DATASET CREATION STARTED")
    logger.info("=" * 50)

    try:
        main(args)
    except Exception as e:
        logger.exception(f"Fatal error in main execution: {str(e)}")
        raise
    finally:
        logger.info("=" * 50)
        logger.info("ORGANIC HDF5 DATASET CREATION FINISHED")
        logger.info("=" * 50)
