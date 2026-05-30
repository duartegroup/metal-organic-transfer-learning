import h5py
import os
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time
import numpy as np
from typing import Tuple, List, Dict, Optional
import re
import pandas as pd
import logging
import sys
from tqdm import tqdm
import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, ProteinComplex, LogitsConfig

# add relative path of src/data_src to path
path = str(Path(__file__).parent.parent.parent.parent / "src" / "data_src")
sys.path.append(path)

from create_hf5_common import (
    generate_embeddings_esmc,
    create_chain_position_mapping,
    extract_pocket_embeddings,
    extract_pocket_edges,
    # extract_sdf,
    create_mace_calculator,
    add_mace_embeddings_to_entry,
    compute_mbtr_embedding,
    compute_soap_embedding,
    compute_acsf_embedding,
    create_pocket_mask,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_caged_csv(
    csv_file: str,
    affinity_type: str,
) -> Tuple[List[str], List[str], List[str], List[str], List[int], np.ndarray]:
    """
    Extract data from caged CSV file with improved error handling.

    Parameters
    ----------
    csv_file : str
        Path to the caged CSV file.
    affinity_type : str
        Type of affinity ('pic50' or 'pk').

    Returns
    -------
    Tuple[List[str], List[str], List[str], List[str], List[int], np.ndarray]
        A tuple containing protein names, representative PDBs, uniprot codes,
        protein types, compound numbers, and target values.
    """
    caged_df = pd.read_csv(csv_file)
    caged_df.columns = caged_df.columns.str.lower()

    if affinity_type == "pic50":
        tgt_col = "pic50"
    elif affinity_type == "pk":
        tgt_col = "pki_"
    else:
        raise ValueError(f"Invalid affinity type: {affinity_type}")

    target_vals = caged_df[tgt_col].values
    caged_df = caged_df.dropna(subset=["protein_seq"])

    # Extract and clean data
    protein_names = [
        f"{name}-{idx}" for idx, name in enumerate(caged_df["protein"].values)
    ]
    protein_names = [re.sub(r"\xa0", "", name) for name in protein_names]

    representative_pdbs = caged_df["representative_pdb"].values.tolist()
    uniprot_codes = [
        re.sub(r"\xa0", "", code) for code in caged_df["uniprot_id"].values
    ]
    protein_types = caged_df["protein_type"].values.tolist()
    compound_numbers = caged_df["compound_n"].values.tolist()

    return (
        protein_names,
        representative_pdbs,
        uniprot_codes,
        protein_types,
        compound_numbers,
        target_vals,
    )


def load_p2rank_predictions(p2rank_path: str) -> List[str]:
    """
    Load pocket residue IDs from p2rank predictions.

    Parameters
    ----------
    p2rank_path : str
        Path to the p2rank predictions CSV file.

    Returns
    -------
    List[str]
        List of pocket residue identifiers from the best pocket.
    """
    p2rank_df = pd.read_csv(p2rank_path)
    p2rank_df.columns = p2rank_df.columns.str.replace(" ", "")

    if p2rank_df.empty or "residue_ids" not in p2rank_df.columns:
        return []

    # Get residues from the first (best) pocket
    residues_string = p2rank_df["residue_ids"].iloc[0].strip()
    return residues_string.split()


def extract_xyz(xyz_file: str) -> Tuple[List[List[float]], List[str]]:
    """
    Extract coordinates and atom types from XYZ file.

    Parameters
    ----------
    xyz_file : str
        Path to the XYZ file.

    Returns
    -------
    Tuple[List[List[float]], List[str]]
        A tuple containing the coordinates and atom types.
    """
    coords, atom_types = [], []
    try:
        with open(xyz_file, "r") as f:
            lines = f.readlines()[2:]  # Skip first two lines
            for line in lines:
                parts = line.split()
                atom_types.append(parts[0])
                coords.append([float(x) for x in parts[1:]])
    except Exception as e:
        logger.error(f"Error reading XYZ file {xyz_file}: {e}")

    return coords, atom_types


def process_single_entry(
    entry_data: Tuple,
    model: ESMC,
    device: str,
    base_compound_dir: str,
    base_protein_dir: str,
    include_protein: bool = True,
    include_gfn2: bool = True,
    include_pocket: bool = True,
    include_mace: bool = False,
    include_mbtr: bool = False,
    include_soap: bool = False,
    include_acsf: bool = False,
    mace_calculator: Optional = None,
) -> Optional[Dict]:
    """
    Process a single entry with improved structure and error handling.

    Parameters
    ----------
    entry_data : Tuple
        Tuple containing entry information (protein_name, representative_pdb, uniprot_code, protein_type, compound_n, target_value).
    model : ESMC
        ESM-C model for generating protein embeddings.
    device : str
        Device to use for model inference.
    base_compound_dir : str
        Base directory for compound files.
    base_protein_dir : str
        Base directory for protein files.
    include_protein : bool, optional
        Whether to include protein data (default: True).
    include_gfn2 : bool, optional
        Whether to include GFN2-xTB data (default: True).
    include_pocket : bool, optional
        Whether to include pocket data (default: True).
    include_mace : bool, optional
        Whether to include MACE embeddings (default: False).
    include_mbtr : bool, optional
        Whether to include MBTR embeddings (default: False).
    include_soap : bool, optional
        Whether to include SOAP embeddings (default: False).
    include_acsf : bool, optional
        Whether to include ACSF embeddings (default: False).
    mace_calculator : Optional, optional
        MACE calculator instance.

    Returns
    -------
    Optional[Dict]
        Entry data dictionary if successful, None if failed.
    """
    (
        protein_name,
        representative_pdb,
        uniprot_code,
        protein_type,
        compound_n,
        target_value,
    ) = entry_data

    entry_data_dict = {"name": f"compound_{compound_n}_{protein_name}"}

    try:
        # Process protein data
        if include_protein or include_pocket:
            protein_dir = _get_protein_directory(
                base_protein_dir, protein_type, uniprot_code, representative_pdb
            )

            if protein_type == "AlphaFold":
                pdb_files = list(protein_dir.glob("AF-*.pdb"))
            else:
                pdb_files = list(protein_dir.glob("*fixed.pdb"))

            if not pdb_files:
                logger.info(f"No PDB file found for {protein_name} in {protein_dir}")
                return None

            pdb_file = pdb_files[0]

            # Process protein with ESM-C
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

                    if include_protein:
                        entry_data_dict["protein_sequence"] = protein_multi.sequence
                        entry_data_dict["protein_embeddings"] = protein_embeddings
                except Exception as e:
                    logger.error(f"Error reading PDB file for {protein_name}: {e}")
                    return None

            # Process pocket extraction if requested
            if include_pocket:
                try:
                    # Get P2Rank file path
                    if protein_type == "AlphaFold":
                        p2rank_path = f"{base_protein_dir}/AlphaFoldDB/{uniprot_code}/output_dir/AF-{uniprot_code}-F1-model_v4.pdb_predictions.csv"
                    else:
                        p2rank_path = f"{base_protein_dir}/PDB/{representative_pdb}/output_dir/pdb{representative_pdb}_fixed.pdb_predictions.csv"

                    pocket_residues = load_p2rank_predictions(p2rank_path)

                    if pocket_residues:
                        pocket_embeddings = extract_pocket_embeddings(
                            protein_embeddings, pocket_residues, chain_mapping
                        )
                        pocket_edges = extract_pocket_edges(pocket_residues, pdb_file)

                        # Create binary pocket mask
                        sequence_length = len(protein_multi.sequence)
                        pocket_mask = create_pocket_mask(
                            pocket_residues, chain_mapping, sequence_length
                        )

                        entry_data_dict["pocket_residues"] = pocket_residues
                        entry_data_dict["pocket_embeddings"] = pocket_embeddings
                        entry_data_dict["pocket_edges"] = pocket_edges
                        entry_data_dict["pocket_mask"] = pocket_mask  # Add this line
                    else:
                        logger.debug(f"No pocket residues found for {protein_name}")
                        return None

                except Exception as e:
                    logger.error(f"Error extracting pocket for {protein_name}: {e}")
                    return None

        # Process compound data
        if include_gfn2 or include_mace or include_mbtr:
            compound_dir = Path(base_compound_dir) / f"compound_{compound_n}"
            xyz_files = compound_dir / f"compound_{compound_n}.xyz"
            # sdf_files = compound_dir / f"compound_{compound_n}.sdf"

            if not xyz_files:
                logger.debug(f"No XYZ file found for compound_{compound_n}")
                return None

            xyz_coords, xyz_atom_types = extract_xyz(str(xyz_files))
            # sdf_coords, sdf_atom_types, sdf_bonds = extract_sdf(str(sdf_files))

            if not xyz_coords or not xyz_atom_types:
                logger.debug(f"Empty XYZ data for compound_{compound_n}")
                return None

            if include_gfn2:
                entry_data_dict["mol_gfn2_coords"] = np.array(xyz_coords)
                entry_data_dict["mol_gfn2_atype"] = xyz_atom_types

                # entry_data_dict["sdf_coords"] = np.array(sdf_coords)
                # entry_data_dict["sdf_atype"] = sdf_atom_types
                # entry_data_dict["sdf_bonds"] = sdf_bonds

            if include_mbtr:
                entry_data_dict["mol_mbtr_embedding"] = compute_mbtr_embedding(
                    xyz_coords, xyz_atom_types
                )

            if include_soap:
                entry_data_dict["mol_soap_embedding"] = compute_soap_embedding(
                    xyz_coords, xyz_atom_types
                )

            if include_acsf:
                entry_data_dict["mol_acsf_embedding"] = compute_acsf_embedding(
                    xyz_coords, xyz_atom_types
                )

        # add the target value to the entry_data_dict
        entry_data_dict["target_value"] = target_value

        # Add MACE embeddings if requested
        if include_mace:
            entry_data_dict = add_mace_embeddings_to_entry(
                entry_data_dict,
                xyz_coords,
                xyz_atom_types,
                mace_calculator,
                include_mace,
            )

        # Validate entry
        if _validate_entry_data(
            entry_data_dict,
            include_protein,
            include_pocket,
            include_gfn2,
            include_mace,
            include_mbtr,
            include_soap,
            include_acsf,
        ):
            return entry_data_dict
        else:
            logger.info(f"Invalid data for {protein_name}_{compound_n}")
            return None

    except Exception as e:
        logger.error(f"Error processing {protein_name}_{compound_n}: {e}")
        return None


def _get_protein_directory(
    base_dir: str, protein_type: str, uniprot_code: str, representative_pdb: str
) -> Path:
    """
    Get protein directory based on type.

    Parameters
    ----------
    base_dir : str
        Base directory for protein files.
    protein_type : str
        Type of protein ('AlphaFold' or other).
    uniprot_code : str
        UniProt identifier.
    representative_pdb : str
        Representative PDB identifier.

    Returns
    -------
    Path
        Path to the protein directory.
    """
    if protein_type == "AlphaFold":
        return Path(base_dir) / "AlphaFoldDB" / uniprot_code
    else:
        return Path(base_dir) / "PDB" / representative_pdb


def _validate_entry_data(
    entry: Dict,
    include_protein: bool,
    include_pocket: bool,
    include_gfn2: bool,
    include_mace: bool,
    include_mbtr: bool,
    include_soap: bool,
    include_acsf: bool,
) -> bool:
    """
    Validate that entry contains required data.

    Parameters
    ----------
    entry : Dict
        Entry data dictionary to validate.
    include_protein : bool
        Whether protein data should be included.
    include_pocket : bool
        Whether pocket data should be included.
    include_gfn2 : bool
        Whether GFN2-xTB data should be included.
    include_mace : bool
        Whether MACE embeddings should be included.
    include_mbtr : bool
        Whether MBTR embeddings should be included.
    include_soap : bool
        Whether SOAP embeddings should be included.
    include_acsf : bool
        Whether ACSF embeddings should be included.

    Returns
    -------
    bool
        True if entry contains all required data, False otherwise.
    """
    if include_protein:
        # Check for protein sequence
        if not entry.get("protein_sequence") or not entry["protein_sequence"].strip():
            return logger.info(f"Invalid protein sequence for {entry['name']}")
        # Check for protein embeddings
        protein_embeddings = entry.get("protein_embeddings")
        if protein_embeddings is None or (
            hasattr(protein_embeddings, "numel") and protein_embeddings.numel() == 0
        ):
            return logger.info(f"Invalid protein embeddings for {entry['name']}")

    if include_pocket:
        # Check for pocket residues
        if not entry.get("pocket_residues"):
            return logger.info(f"Invalid pocket residues for {entry['name']}")
        # Check for pocket embeddings
        pocket_embeddings = entry.get("pocket_embeddings")
        if pocket_embeddings is None or (
            hasattr(pocket_embeddings, "numel") and pocket_embeddings.numel() == 0
        ):
            return logger.info(f"Invalid pocket embeddings for {entry['name']}")
        # Check for pocket edges
        pocket_edges = entry.get("pocket_edges")
        if pocket_edges is None or (
            hasattr(pocket_edges, "size") and pocket_edges.size == 0
        ):
            return logger.info(f"Invalid pocket edges for {entry['name']}")
        # Check for pocket mask
        pocket_mask = entry.get("pocket_mask")
        if pocket_mask is None or (
            hasattr(pocket_mask, "size") and pocket_mask.size == 0
        ):
            return logger.info(f"Invalid pocket mask for {entry['name']}")

    if include_gfn2:
        # Check for mol_gfn2_xyz (numpy array)
        xyz_data = entry.get("mol_gfn2_coords")
        if xyz_data is None or (hasattr(xyz_data, "size") and xyz_data.size == 0):
            return logger.info(f"Invalid GFN2 data for {entry['name']}")
        # Check for mol_gfn2_atype (list)
        atype_data = entry.get("mol_gfn2_atype")
        if not atype_data:
            return logger.info(f"Invalid GFN2 data for {entry['name']}")
        # Check for mol_gfn2_sdf_coords (numpy array)
        # sdf_coords = entry.get("sdf_coords")
        # if sdf_coords is None or (hasattr(sdf_coords, "size") and sdf_coords.size == 0):
        #     return False
        # # Check for mol_gfn2_sdf_atype (list)
        # sdf_atype = entry.get("sdf_atype")
        # if not sdf_atype:
        #     return False
        # # Check for mol_gfn2_sdf_bonds (list)
        # sdf_bonds = entry.get("sdf_bonds")
        # if not sdf_bonds:
        #     return False

    if include_mace:
        # Check for MACE embedding (make it optional - don't fail if missing)
        mace_embedding = entry.get("mol_mace_embedding")
        if (
            mace_embedding is not None
            and hasattr(mace_embedding, "size")
            and mace_embedding.size == 0
        ):
            return logger.info(f"Invalid MACE data for {entry['name']}")

    if include_mbtr:
        # Check for MBTR embedding (make it optional - don't fail if missing)
        mbtr_embedding = entry.get("mol_mbtr_embedding")
        if (
            mbtr_embedding is not None
            and hasattr(mbtr_embedding, "size")
            and mbtr_embedding.size == 0
        ):
            return logger.info(f"Invalid MBTR data for {entry['name']}")

    if include_soap:
        soap_embedding = entry.get("mol_soap_embedding")
        if (
            soap_embedding is not None
            and hasattr(soap_embedding, "size")
            and soap_embedding.size == 0
        ):
            return logger.info(f"Invalid SOAP data for {entry['name']}")

    if include_acsf:
        acsf_embedding = entry.get("mol_acsf_embedding")
        if (
            acsf_embedding is not None
            and hasattr(acsf_embedding, "size")
            and acsf_embedding.size == 0
        ):
            return logger.info(f"Invalid ACSF data for {entry['name']}")

    return True


def create_hdf5_dataset(
    csv_file: str,
    base_compound_dir: str,
    base_protein_dir: str,
    output_file: str = "dataset.h5",
    include_protein: bool = False,
    include_pocket: bool = True,
    include_gfn2: bool = False,
    include_mace: bool = True,
    mace_model_path: str = "../mace_model/MACE-MP-0b3.model",
    include_mbtr: bool = False,
    include_soap: bool = False,
    include_acsf: bool = False,
    affinity_type: str = "pic50",
) -> None:
    """
    Create HDF5 dataset with improved structure and error handling.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing entries to process.
    base_compound_dir : str
        Base directory for compound files.
    base_protein_dir : str
        Base directory for protein files.
    output_file : str, optional
        Name of the output HDF5 file (default: 'dataset.h5').
    include_protein : bool, optional
        Whether to include protein data (default: False).
    include_pocket : bool, optional
        Whether to include pocket data (default: True).
    include_gfn2 : bool, optional
        Whether to include GFN2-xTB data (default: False).
    include_mace : bool, optional
        Whether to include MACE embeddings (default: True).
    mace_model_path : str, optional
        Path to MACE model file (default: '../mace_model/MACE-MP-0b3.model').
    include_mbtr : bool, optional
        Whether to include MBTR embeddings (default: False).
    include_soap : bool, optional
        Whether to include SOAP embeddings (default: False).
    include_acsf : bool, optional
        Whether to include ACSF embeddings (default: False).
    affinity_type : str, optional
        Type of affinity ('pic50' or 'pk', default: 'pic50').

    Returns
    -------
    None
    """

    if not include_protein and not include_pocket and not include_mace:
        logger.error(
            "At least one of --protein, --pocket, or --mace must be specified!"
        )
        return

    # Extract data from CSV
    logger.info("Reading CSV file...")
    (
        protein_names,
        representative_pdbs,
        uniprot_codes,
        protein_types,
        compound_ns,
        target_vals,
    ) = extract_caged_csv(csv_file, affinity_type)

    if not protein_names:
        logger.error("No data extracted from CSV file")
        return

    entries = list(
        zip(
            protein_names,
            representative_pdbs,
            uniprot_codes,
            protein_types,
            compound_ns,
            target_vals,
        )
    )

    logger.info(f"Found {len(entries)} entries in CSV")
    logger.info(f"Including protein files: {include_protein}")
    logger.info(f"Including GFN2-xTB files: {include_gfn2}")
    logger.info(f"Including pocket residues: {include_pocket}")
    logger.info(f"Including MACE embeddings: {include_mace}")
    logger.info(f"Including MBTR embeddings: {include_mbtr}")
    logger.info(f"Including SOAP embeddings: {include_soap}")
    logger.info(f"Including ACSF embeddings: {include_acsf}")

    # Initialize model once at the beginning if needed
    if include_protein or include_pocket:
        logger.info("Initializing ESM-C model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ESMC.from_pretrained("esmc_600m").to(device)
        model.eval()
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
    for entry_data in tqdm(entries, desc="Processing entries"):
        result = process_single_entry(
            entry_data,
            model=model,
            device=device,
            base_compound_dir=base_compound_dir,
            base_protein_dir=base_protein_dir,
            include_protein=include_protein,
            include_gfn2=include_gfn2,
            include_pocket=include_pocket,
            include_mace=include_mace,
            include_mbtr=include_mbtr,
            include_soap=include_soap,
            include_acsf=include_acsf,
            mace_calculator=mace_calculator,
        )
        if result is not None:
            data_entries.append(result)

    processing_time = time.time() - start_time
    success_rate = len(data_entries) / len(entries) * 100

    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    logger.info(
        f"Successfully processed: {len(data_entries)}/{len(entries)} entries ({success_rate:.1f}%)"
    )

    if not data_entries:
        logger.error("No entries were successfully processed!")
        return

    # Save to HDF5
    _save_to_hdf5(
        data_entries,
        output_file,
        include_protein,
        include_pocket,
        include_gfn2,
        include_mace,
        include_mbtr,
        include_soap,
        include_acsf,
        processing_time,
    )


def _save_to_hdf5(
    data_entries: List[Dict],
    output_file: str,
    include_protein: bool,
    include_pocket: bool,
    include_gfn2: bool,
    include_mace: bool,
    include_mbtr: bool,
    include_soap: bool,
    include_acsf: bool,
    processing_time: float,
) -> None:
    """
    Save data entries to HDF5 file.

    Parameters
    ----------
    data_entries : List[Dict]
        List of entry data dictionaries to save.
    output_file : str
        Path to the output HDF5 file.
    include_protein : bool
        Whether protein data is included.
    include_pocket : bool
        Whether pocket data is included.
    include_gfn2 : bool
        Whether GFN2-xTB data is included.
    include_mace : bool
        Whether MACE embeddings are included.
    include_mbtr : bool
        Whether MBTR embeddings are included.
    include_soap : bool
        Whether SOAP embeddings are included.
    include_acsf : bool
        Whether ACSF embeddings are included.
    processing_time : float
        Time taken to process all entries in seconds.

    Returns
    -------
    None
    """
    logger.info("Creating HDF5 file...")

    with h5py.File(output_file, "w") as h5f:
        dt_string = h5py.string_dtype(encoding="utf-8")

        for entry in data_entries:
            entry_name = entry["name"]
            entry_group = h5f.create_group(entry_name)

            # Store target value
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
                    # Convert tensor to numpy array for storage
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
                    entry_group.create_dataset("pocket_mask", data=entry["pocket_mask"])

            # Store molecular data
            if include_gfn2:
                if "mol_gfn2_xyz" in entry:
                    entry_group.create_dataset(
                        "mol_gfn2_xyz", data=entry["mol_gfn2_xyz"]
                    )
                if "mol_gfn2_atype" in entry:
                    atype_data = [
                        atype.encode("utf-8") for atype in entry["mol_gfn2_atype"]
                    ]
                    entry_group.create_dataset("mol_gfn2_atype", data=atype_data)

                # if "sdf_coords" in entry:
                #     entry_group.create_dataset("sdf_coords", data=entry["sdf_coords"])
                # if "sdf_atype" in entry:
                #     atype_data = [atype.encode("utf-8") for atype in entry["sdf_atype"]]
                #     entry_group.create_dataset("sdf_atype", data=atype_data)
                # if "sdf_bonds" in entry:
                #     entry_group.create_dataset("sdf_bonds", data=entry["sdf_bonds"])

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
            description_parts.append("GFN2-xTB optimized XYZ files")
        if include_mace:
            description_parts.append("MACE embeddings")
        if include_mbtr:
            description_parts.append("MBTR embeddings")
        if include_soap:
            description_parts.append("SOAP embeddings")
        if include_acsf:
            description_parts.append("ACSF embeddings")

        h5f.attrs["description"] = (
            f"Caged dataset with {' and '.join(description_parts)}"
        )
        h5f.attrs["total_entries"] = len(data_entries)
        h5f.attrs["created_by"] = "create_caged_h5_improved.py"
        h5f.attrs["includes_protein"] = include_protein
        h5f.attrs["includes_gfn2"] = include_gfn2
        h5f.attrs["includes_pocket"] = include_pocket
        h5f.attrs["includes_mace"] = include_mace
        h5f.attrs["includes_mbtr"] = include_mbtr
        h5f.attrs["includes_soap"] = include_soap
        h5f.attrs["includes_acsf"] = include_acsf
        h5f.attrs["processing_time_seconds"] = processing_time

    logger.info(
        f"HDF5 file '{output_file}' created successfully with {len(data_entries)} entries"
    )


def main(args: argparse.Namespace) -> None:
    """
    Main function to create HDF5 dataset from command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing dataset configuration.

    Returns
    -------
    None
    """
    csv_file = (
        f"./../../excel/caged/{args.affinity_type}/caged_{args.affinity_type}.csv"
    )
    output_file = (
        args.output if args.output else f"caged_{args.affinity_type}_dataset.h5"
    )

    create_hdf5_dataset(
        csv_file,
        args.compound_dir,
        args.protein_dir,
        output_file,
        args.protein,
        args.pocket,
        args.gfn2,
        args.mace,
        args.mace_model_path,
        args.mbtr,
        args.soap,
        args.acsf,
        args.affinity_type,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create HDF5 dataset from Caged data (improved version)"
    )
    parser.add_argument(
        "--affinity_type",
        "-a",
        default="pic50",
        help="Type of affinity to process (pic50 or pk)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file name",
    )
    parser.add_argument(
        "--compound-dir",
        default=None,
        help="Base directory for compound files",
    )
    parser.add_argument(
        "--protein-dir",
        default=None,
        help="Base directory for protein files",
    )
    parser.add_argument(
        "--protein",
        action="store_true",
        help="Include protein PDB files in the dataset",
    )
    parser.add_argument(
        "--pocket", action="store_true", help="Include pocket residues in the dataset"
    )
    parser.add_argument(
        "--gfn2", action="store_true", help="Include GFN2-xTB XYZ files in the dataset"
    )
    parser.add_argument(
        "--mace", action="store_true", help="Include MACE embeddings in the dataset"
    )
    parser.add_argument(
        "--mace-model-path",
        default="../mace_model/MACE-MP-0b3.model",
        help="Path to MACE model file",
    )
    parser.add_argument(
        "--mbtr",
        default=False,
        action="store_true",
        help="Include MBTR embeddings in the dataset",
    )
    parser.add_argument(
        "--soap",
        default=False,
        action="store_true",
        help="Include SOAP embeddings in the dataset",
    )
    parser.add_argument(
        "--acsf",
        default=False,
        action="store_true",
        help="Include ACSF embeddings in the dataset",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=None,
        help="Number of processes to use (default: auto-detect)",
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
    logger.info("CAGED HDF5 DATASET CREATION STARTED")
    logger.info("=" * 50)

    try:
        main(args)
    except Exception as e:
        logger.exception(f"Fatal error in main execution: {str(e)}")
        raise
    finally:
        logger.info("=" * 50)
        logger.info("CAGED HDF5 DATASET CREATION FINISHED")
        logger.info("=" * 50)
