import torch
from typing import Dict, List, Tuple, Optional, Any
from esm.sdk.api import ProteinComplex, ESMProtein, LogitsConfig
import logging
from Bio.PDB import PDBParser
import numpy as np
import warnings

# Try to import MACE components (make it optional)
try:
    from ase import Atoms
    from mace.calculators import MACECalculator

    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ATOM_TYPES = set(["Se", "I", "P", "Cl", "C", "Ru", "Br", "F", "H", "N", "O", "S"])

from dscribe.descriptors import MBTR, SOAP, ACSF
from ase.io import read
from ase.atoms import Atoms
from scipy.linalg import orth

from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter("ignore", PDBConstructionWarning)


# === INHIBITOR FUNCTIONS === #
def extract_xyz(xyz_file: str) -> Tuple[np.ndarray, List[str]]:
    """
    Extract coordinates and atom types from XYZ file.

    Parameters
    ----------
    xyz_file : str
        Path to the XYZ file.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A tuple containing:
        - coords: Array of atomic coordinates with shape (n_atoms, 3)
        - atom_types: List of atom type strings
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

    return np.array(coords), atom_types


def extract_sdf(sdf_file: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]]]:
    """
    Extract atom types, coordinates, and bonds from an SDF file.

    Parameters
    ----------
    sdf_file : str
        Path to the SDF file.

    Returns
    -------
    tuple[np.ndarray, list[str], list[tuple[int, int]]]
        A tuple containing:
        - coords: Array of atomic coordinates with shape (n_atoms, 3)
        - atype: List of atom type strings
        - bonds: List of bond tuples (atom1_idx, atom2_idx) with 0-based indexing
    """
    coords, atype, bonds = [], [], []

    try:
        with open(sdf_file, "r") as fin:
            lines = fin.readlines()

            # Parse the counts line (4th line in SDF format)
            count_line = lines[3]
            num_atoms = int(count_line[0:3])
            num_bonds = int(count_line[3:6])

            # Extract atom information (starts at line 5)
            for i in range(4, 4 + num_atoms):
                line = lines[i].split()
                coords.append([float(line[0]), float(line[1]), float(line[2])])
                atype.append(line[3])

            # Extract bond information
            for i in range(4 + num_atoms, 4 + num_atoms + num_bonds):
                # line = lines[i].split()
                # SDF format uses 1-based indexing, convert to 0-based
                atom1 = int(lines[i][0:3]) - 1
                atom2 = int(lines[i][3:6]) - 1
                bonds.append((atom1, atom2))

        return np.array(coords), atype, bonds
    except Exception as e:
        logger.error(f"Error processing SDF file {sdf_file}: {e}")
        return [], [], []


def gram_schmidt_orthogonalization(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply Gram-Schmidt process using SciPy's efficient orthogonalization.

    Parameters
    ----------
    vectors : list[np.ndarray]
        List of vectors to orthogonalize.

    Returns
    -------
    list[np.ndarray]
        List of orthogonal unit vectors.
    """
    A = np.column_stack(vectors)
    Q = orth(A)
    return [Q[:, i] for i in range(Q.shape[1])]


def create_local_coordinate_system(
    positions: np.ndarray, atom_idx: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create local coordinate system for atom i based on neighboring atoms.

    Parameters
    ----------
    positions : np.ndarray
        Array of 3D coordinates for all atoms
    atom_idx : int
        Index of the current atom (0-based)

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        Tuple of (basis_vectors, orthogonal_vectors)
    """
    n_atoms = len(positions)

    # Handle edge cases for boundary atoms
    if atom_idx == 0:
        # First atom: use next two atoms
        if n_atoms >= 3:
            v1 = positions[1] - positions[0]  # x1 - x0
            v2 = positions[2] - positions[0]  # x2 - x0
            v3 = positions[2] - positions[1]  # x2 - x1
        else:
            raise ValueError("Need at least 3 atoms for local coordinate system")

    elif atom_idx == n_atoms - 1:
        # Last atom: use previous two atoms
        v1 = positions[atom_idx] - positions[atom_idx - 1]  # xi - xi-1
        v2 = positions[atom_idx - 1] - positions[atom_idx]  # xi-1 - xi
        v3 = positions[atom_idx - 1] - positions[atom_idx - 2]  # xi-1 - xi-2

    else:
        # Middle atoms: use neighbors as defined in equations (2)
        i = atom_idx
        v1 = positions[i] - positions[i - 1]  # xi - xi-1
        v2 = positions[i + 1] - positions[i]  # xi+1 - xi
        v3 = positions[i + 1] - positions[i - 1]  # xi+1 - xi-1

    # Apply Gram-Schmidt orthogonalization (equations 3)
    basis_vectors = [v1, v2, v3]
    orthogonal_vectors = gram_schmidt_orthogonalization(basis_vectors)

    return basis_vectors, orthogonal_vectors


def generate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Generate pairwise distance matrix for molecular coordinates using vectorized operations.

    Parameters
    ----------
    coords : np.ndarray
        Array of 3D coordinates for all atoms with shape (n_atoms, 3).

    Returns
    -------
    np.ndarray
        Pairwise distance matrix with shape (n_atoms, n_atoms) where element [i, j]
        represents the Euclidean distance between atoms i and j.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def generate_spatial_position_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Generate a spatial position matrix for a molecule using local coordinate systems.

    This function creates a spatial position matrix P where each row represents
    the local coordinate system for an atom, capturing the relative spatial
    positioning of neighboring atoms.

    Parameters
    ----------
    coords : np.ndarray
        Array of 3D coordinates for all atoms

    Returns
    -------
    np.ndarray
        Spatial position matrix where P[i] contains the flattened local coordinate
        system for atom i (9-dimensional vector: 3 vectors x 3 dimensions each)
    """
    n_atoms = len(coords)
    P = []

    for i in range(n_atoms):
        try:
            basis_vectors, orthogonal_vectors = create_local_coordinate_system(
                coords, i
            )

            # The spatial position element Pij captures relative spatial positioning
            # Here we use the orthogonal vectors as the local coordinate system
            local_coords = np.array(orthogonal_vectors)
            P.append(local_coords.flatten())  # Flatten to create feature vector

        except Exception as e:
            logger.debug(f"Error processing atom {i}: {e}")
            # Use zero vector as fallback
            P.append(np.zeros(9))  # 3 vectors × 3 dimensions each

    return np.array(P)


def generate_enhanced_molecular_features(coords: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate enhanced molecular features including distance matrix, spatial position matrix,
    and initial pair representation.

    Parameters
    ----------
    coords : np.ndarray
        Array of 3D coordinates for all atoms

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'distance_matrix': Standard distance matrix
        - 'spatial_position_matrix': Enhanced spatial position matrix
        - 'initial_pair_representation': Initial pair representation z^(0)
    """
    # Generate distance matrix
    D = generate_distance_matrix(coords)

    # Generate spatial position matrix
    P = generate_spatial_position_matrix(coords)

    return {
        "distance_matrix": D,
        "spatial_position_matrix": P,
    }


def compute_mbtr_embedding(coords: np.ndarray, atom_types: List[str]) -> np.ndarray:
    """
    Compute Many-Body Tensor Representation (MBTR) embedding for a molecule.

    MBTR creates a descriptor based on the distribution of geometric properties
    (angles in this case) weighted by distance, providing a rotationally and
    translationally invariant representation of the molecular structure.

    Parameters
    ----------
    coords : np.ndarray
        Array of 3D coordinates for all atoms with shape (n_atoms, 3).
    atom_types : list[str]
        List of atom type strings (element symbols) for each atom.

    Returns
    -------
    np.ndarray
        MBTR descriptor array with L2 normalization applied.
    """
    geometry = {
        "function": "angle",
    }
    grid = {
        "min": 0,
        "max": 1,
        "sigma": 0.1,
        "n": 100,
    }
    weighting = {"function": "exp", "r_cut": 10, "threshold": 1e-3}

    normalization = "l2"

    mbtr = MBTR(
        species=ATOM_TYPES,
        geometry=geometry,
        grid=grid,
        weighting=weighting,
        periodic=False,
        normalization=normalization,
    )

    ase_atoms = Atoms(positions=coords, symbols=atom_types)
    mbtr_desc = mbtr.create(ase_atoms, n_jobs=-1)
    return mbtr_desc


def compute_soap_embedding(coords: np.ndarray, atom_types: List[str]) -> np.ndarray:
    """
    Compute Smooth Overlap of Atomic Positions (SOAP) embedding for a molecule.

    SOAP creates a descriptor based on the local atomic environment by expanding
    the atomic density using spherical harmonics and radial basis functions,
    providing a rotationally and translationally invariant representation.

    Parameters
    ----------
    coords : np.ndarray
        Array of 3D coordinates for all atoms with shape (n_atoms, 3).
    atom_types : list[str]
        List of atom type strings (element symbols) for each atom.

    Returns
    -------
    np.ndarray
        SOAP descriptor array for each atom in the molecule.
    """
    # SOAP settings
    r_cutoff = 6
    n_max = 8
    l_max = 6
    average = "off"
    compression = "mu1nu1"

    soap = SOAP(
        species=ATOM_TYPES,
        periodic=False,
        r_cut=r_cutoff,
        n_max=n_max,
        l_max=l_max,
        average=average,
        compression={"mode": compression, "species_weighting": None},
    )
    ase_atoms = Atoms(positions=coords, symbols=atom_types)
    soap_desc = soap.create(ase_atoms, n_jobs=-1)
    return soap_desc


def compute_acsf_embedding(coords: np.ndarray, atom_types: List[str]) -> np.ndarray:
    """
    Compute Atom-Centered Symmetry Functions (ACSF) embedding for a molecule.

    ACSF creates a descriptor using radial and angular symmetry functions that
    characterize the local atomic environment in a rotationally and translationally
    invariant manner. Uses both G2 (radial) and G4 (angular) symmetry functions.

    Parameters
    ----------
    coords : np.ndarray
        Array of 3D coordinates for all atoms with shape (n_atoms, 3).
    atom_types : list[str]
        List of atom type strings (element symbols) for each atom.

    Returns
    -------
    np.ndarray
        ACSF descriptor array for each atom in the molecule.
    """
    r_cutoff = 6
    g2_params = [[1, 1], [1, 2], [1, 3]]
    g4_params = [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]

    acsf = ACSF(
        species=ATOM_TYPES,
        r_cut=r_cutoff,
        g2_params=g2_params,
        g4_params=g4_params,
    )

    ase_atoms = Atoms(positions=coords, symbols=atom_types)
    acsf_desc = acsf.create(ase_atoms, n_jobs=-1)
    return acsf_desc


# === PROTEIN FUNCTIONS === #
def generate_embeddings_esmc(
    protein_multi: Any,
    model: Any,
    device: torch.device,
    batch_size: int = 1,
    average: bool = False,
    layer_idx: int = 35,
) -> torch.Tensor:
    """
    Generate embeddings for protein sequences using ESM-C model.

    The ESM-C model produces embeddings with the following token structure:
    [CLS] residue_1 residue_2 ... residue_n [EOS] [PAD] [PAD] ...

    Parameters
    ----------
    protein_multi : Any
        Protein complex or sequence data compatible with ESM-C model encoding.
        Should have a 'sequence' attribute containing the protein sequence.
    model : Any
        The ESM-C model client with encode() and logits() methods.
    device : torch.device
        The device to use for the model (CPU or CUDA).
    batch_size : int, optional
        The batch size for processing. Default is 1 (ESM-C processes one by one).
    average : bool, optional
        Whether to average the embeddings over the sequence length. Default is False.
    layer_idx : int, optional
        The hidden layer index to extract embeddings from. Default is 35.

    Returns
    -------
    torch.Tensor
        Protein embeddings tensor. If average=True, shape is (1, hidden_dim).
        If average=False, shape is (1, seq_len, hidden_dim) including special tokens.
    """
    logger.debug(
        f"Starting ESM-C embedding generation for {len(protein_multi)} sequences"
    )
    logger.debug(f"Using layer {layer_idx}, average={average}")

    # Debug protein sequence information
    if hasattr(protein_multi, "sequence"):
        protein_seq = protein_multi.sequence
        logger.debug(f"Input protein sequence length: {len(protein_seq)}")
        logger.debug(
            f"First 50 chars: {protein_seq[:50]}{'...' if len(protein_seq) > 50 else ''}"
        )
        logger.debug(
            f"Last 50 chars: {'...' if len(protein_seq) > 50 else ''}{protein_seq[-50:]}"
        )
    else:
        logger.debug(f"Protein multi type: {type(protein_multi)}")

    protein_tensor = model.encode(protein_multi)

    # Debug protein tensor information
    logger.debug(f"Encoded protein tensor type: {type(protein_tensor)}")
    if hasattr(protein_tensor, "shape"):
        logger.debug(f"Protein tensor shape: {protein_tensor.shape}")

    logits_output = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True),
    )

    # Extract embeddings
    embeddings = logits_output.hidden_states[layer_idx]

    # Debug embeddings information
    logger.debug(f"Final embeddings shape: {embeddings.shape}")

    # Log token structure analysis
    if hasattr(protein_multi, "sequence"):
        protein_seq_len = len(protein_multi.sequence)
        embedding_seq_len = embeddings.shape[1]
        special_tokens_count = embedding_seq_len - protein_seq_len
        logger.debug(f"Token structure analysis:")
        logger.debug(f"  - Protein sequence length: {protein_seq_len}")
        logger.debug(f"  - Embedding sequence length: {embedding_seq_len}")
        logger.debug(f"  - Special tokens count: {special_tokens_count}")
        logger.debug(
            f"  - Expected structure: [CLS](pos 0) + {protein_seq_len} residues + [EOS](pos {protein_seq_len+1}) + padding"
        )

    if average:
        # Average over sequence length
        embedding = embeddings.mean(1).cpu()
    else:
        # Keep per-token representations
        embedding = embeddings.cpu()

    return embedding


def generate_embeddings_contact_esm_plus_plus(
    protein_multi: ProteinComplex,
    model: Any,
    device: torch.device,
    batch_size: int = 1,
    average: bool = False,
) -> Tuple[torch.Tensor, None]:
    """
    Generate embeddings for protein sequences using ESM++ (ESM2) model with contact prediction.

    Parameters
    ----------
    protein_multi : ProteinComplex
        Protein complex containing the sequence to encode.
    model : Any
        The ESM++ model with tokenizer and forward pass methods.
    device : torch.device
        The device to use for the model (CPU or CUDA).
    batch_size : int, optional
        The batch size for processing. Default is 1.
    average : bool, optional
        Whether to average the embeddings over the sequence length. Currently unused.

    Returns
    -------
    tuple[torch.Tensor, None]
        A tuple containing:
        - embeddings: Last hidden state from the model with shape (batch_size, seq_len, hidden_dim)
        - None: Placeholder for potential attention maps (currently not returned)
    """
    tokenizer = model.tokenizer
    tokenized = tokenizer(protein_multi.sequence, return_tensors="pt")

    output = model(**tokenized, output_attentions=True)
    embeddings = output.last_hidden_state
    # attentions = output.attentions
    # # also get the last attention map
    # attention_map = attentions[-1]
    return embeddings, None


# === POCKET FUNCTIONS === #
def create_chain_position_mapping(
    protein_complex: ProteinComplex, esm_sequence: str
) -> Dict[str, Dict[int, int]]:
    """
    Create mapping from PDB chain+residue number to ESM sequence position.

    Token Structure in ESM embeddings:
    [CLS] residue_1 residue_2 ... residue_n [EOS] [PAD] [PAD] ...
      0      1         2           n        n+1    n+2   n+3

    This function maps PDB residue numbers to their corresponding positions
    in the ESM embedding tensor (accounting for the CLS token offset).

    Parameters
    ----------
    protein_complex : ProteinComplex
        The protein complex from PDB file
    esm_sequence : str
        The sequence string from ESMProtein (with '|' chain breaks)

    Returns
    -------
    mapping : Dict[str, Dict[int, int]]
        Nested dict: chain_id -> {pdb_residue_num: esm_position}
        ESM positions include the CLS token offset (start from position 1)
    """
    mapping = {}
    current_esm_pos = 1  # Start after the initial CLS token (position 0)

    # Get chain information from metadata
    chain_lookup = (
        protein_complex.metadata.chain_lookup
    )  # {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    chain_boundaries = (
        protein_complex.metadata.chain_boundaries
    )  # [(0, 251), (252, 503), ...]

    # Get residue indices and chain IDs
    residue_indices = protein_complex.residue_index
    chain_ids = protein_complex.chain_id

    logger.debug(f"Chain lookup: {chain_lookup}")
    logger.debug(f"Chain boundaries: {chain_boundaries}")
    logger.debug(
        f"Starting ESM position mapping from position {current_esm_pos} (after CLS token)"
    )

    # Process each chain
    for chain_idx, (start_pos, end_pos) in enumerate(chain_boundaries):
        chain_letter = chain_lookup[chain_idx]  # Get chain letter (A, B, C, D)
        mapping[chain_letter] = {}

        # Get residue numbers for this chain
        chain_residue_indices = residue_indices[start_pos : end_pos + 1]
        chain_chain_ids = chain_ids[start_pos : end_pos + 1]

        # Filter out any chain break markers (-1 values)
        valid_positions = chain_residue_indices != -1
        valid_residue_indices = chain_residue_indices[valid_positions]

        logger.debug(f"Chain {chain_letter}: {len(valid_residue_indices)} residues")
        logger.debug(
            f"  Residue range: {valid_residue_indices.min()} to {valid_residue_indices.max()}"
        )
        logger.debug(
            f"  ESM positions: {current_esm_pos} to {current_esm_pos + len(valid_residue_indices) - 1}"
        )

        # Map each PDB residue number to ESM sequence position
        for i, pdb_res_num in enumerate(valid_residue_indices):
            esm_pos = current_esm_pos + i
            mapping[chain_letter][int(pdb_res_num)] = esm_pos
            logger.debug(
                f"  Mapped residue {chain_letter}_{pdb_res_num} -> ESM position {esm_pos}"
            )

        # Move to next chain (add chain length + 1 for chain break token '|')
        current_esm_pos += len(valid_residue_indices) + 1

    logger.debug(f"Completed chain position mapping for {len(mapping)} chains")
    return mapping


def extract_pocket_embeddings(
    embeddings: torch.Tensor,
    p2rank_residues: List[str],
    chain_mapping: Dict[str, Dict[int, int]],
) -> torch.Tensor:
    """
    Extract pocket embeddings based on P2rank predictions.

    Parameters
    ----------
    embeddings : torch.Tensor
        Full protein embeddings [batch_size, seq_len, hidden_dim]
        Expected structure: [CLS] residue_1 residue_2 ... residue_n [EOS] [PAD] ...
                             0    1        2              n        n+1   n+2
    p2rank_residues : List[str]
        List of residue IDs like ['A_116', 'B_127', ...]
    chain_mapping : Dict[str, Dict[int, int]]
        Mapping from chain+residue to ESM sequence position (includes CLS offset)

    Returns
    -------
    pocket_embeddings : torch.Tensor
        Embeddings for pocket residues [num_pocket_residues, hidden_dim]
    """
    pocket_indices = []
    valid_residues = []

    logger.debug(f"Extracting pocket embeddings from tensor shape: {embeddings.shape}")
    logger.debug(f"Processing {len(p2rank_residues)} pocket residues")

    for residue_id in p2rank_residues:
        try:
            # Parse residue ID (e.g., 'A_116' -> chain='A', res_num=116)
            chain_id, res_num_str = residue_id.split("_")

            # Handle cases like 'C_50B' where there's a letter suffix
            res_num = int("".join(filter(str.isdigit, res_num_str)))

            # Get ESM sequence position (this already includes the CLS token offset)
            if chain_id in chain_mapping and res_num in chain_mapping[chain_id]:
                esm_pos = chain_mapping[chain_id][res_num]

                # Validate that the position is within the embedding tensor bounds
                # Position should be between 1 (first actual residue) and seq_len-2 (last actual residue before EOS)
                if (
                    1 <= esm_pos < embeddings.shape[1] - 1
                ):  # -1 to account for EOS token
                    pocket_indices.append(esm_pos)
                    valid_residues.append(residue_id)
                    logger.debug(
                        f"Mapped residue {residue_id} to ESM position {esm_pos}"
                    )
                else:
                    logger.debug(
                        f"Warning: ESM position {esm_pos} out of bounds for residue {residue_id} (tensor shape: {embeddings.shape})"
                    )
            else:
                logger.debug(f"Warning: Could not map residue {residue_id}")

        except (ValueError, IndexError) as e:
            logger.debug(f"Warning: Could not parse residue ID {residue_id}: {e}")
            continue

    if not pocket_indices:
        logger.debug("Warning: No valid pocket residues found!")
        return torch.empty(0, embeddings.shape[-1])

    logger.debug(f"Extracting embeddings for {len(pocket_indices)} pocket residues")
    logger.debug(f"Valid residues: {valid_residues}")
    logger.debug(f"ESM positions: {pocket_indices}")

    # Extract embeddings for pocket positions
    # The chain_mapping positions already account for the CLS token offset,
    # so we can use them directly to index into the embedding tensor
    pocket_embeddings = embeddings[
        0, pocket_indices, :
    ]  # [num_pocket_residues, hidden_dim]

    logger.debug(f"Extracted pocket embeddings shape: {pocket_embeddings.shape}")
    return pocket_embeddings


def extract_pocket_edges(
    pocket_residues: List[str],
    pdb_file: str,
    cutoff_distance: float = 6.0,
    atom_type: str = "CA",
) -> List[Tuple[int, int]]:
    """
    Create edges between pocket residues based on spatial proximity.

    This function parses a PDB file and creates a graph edge list for binding pocket
    residues, where edges connect residues within a specified distance cutoff.

    Parameters
    ----------
    pocket_residues : list[str]
        List of residue identifiers in format 'CHAIN_ID_RESIDUE_NUMBER'
        (e.g., 'A_10', 'B_234').
    pdb_file : str
        Path to the PDB file containing the protein structure.
    cutoff_distance : float, optional
        Maximum distance (in Angstroms) between two residues for an edge to be created.
        Default is 6.0.
    atom_type : str, optional
        Type of atom to use for distance calculation:
        - 'CA': C-alpha distance (faster, standard for residue-residue distance)
        - 'heavy': Closest heavy atom distance (more accurate, computationally expensive)
        Default is 'CA'.

    Returns
    -------
    list[tuple[int, int]]
        List of edge tuples (idx1, idx2) representing undirected edges between residues.
        Indices correspond to positions in the pocket_residues list.
        Each edge appears only once (no duplicates like (i,j) and (j,i)).
    """
    edges = []
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    # Dictionary to store residue objects with their corresponding index in pocket_residues
    # and their chain for efficient lookup
    parsed_pocket_residues = {}

    # Map residue string 'CHAIN_RESID' to a tuple (Chain, Residue_Obj)
    for model in structure:
        for chain in model:
            for residue in chain:
                # Bio.PDB residue ID is a tuple: (hetero_flag, residue_number, insertion_code)
                # We need to match 'CHAIN_ID_RESIDUE_NUMBER'
                res_id_str = f"{chain.id}_{residue.id[1]}"  # e.g., 'A_10'

                if res_id_str in pocket_residues:
                    # Store a tuple of (Bio.PDB.Chain object, Bio.PDB.Residue object)
                    # and its original index in the input pocket_residues list
                    original_index = pocket_residues.index(res_id_str)
                    parsed_pocket_residues[original_index] = (chain.id, residue)

    # Ensure all specified pocket residues were found
    if len(parsed_pocket_residues) != len(pocket_residues):
        found_residues = set(
            f"{c_id}_{r_obj.id[1]}" for c_id, r_obj in parsed_pocket_residues.values()
        )
        missing_residues = set(pocket_residues) - found_residues
        logger.debug(
            f"Warning: Could not find all specified pocket residues in PDB file. Missing: {missing_residues}"
        )

    # Iterate through all unique pairs of residues to calculate distances and create edges
    # We iterate using their original indices from `pocket_residues` list
    for i in range(len(pocket_residues)):
        for j in range(
            i + 1, len(pocket_residues)
        ):  # Start from i+1 to avoid self-loops and duplicate pairs

            # Ensure both residues exist in the parsed_pocket_residues (i.e., were found in the PDB)
            if i not in parsed_pocket_residues or j not in parsed_pocket_residues:
                continue

            # Get Bio.PDB Residue objects
            _, res_i = parsed_pocket_residues[i]
            _, res_j = parsed_pocket_residues[j]

            distance = float("inf")  # Initialize with infinity

            if atom_type == "CA":
                # Check if C-alpha atoms exist
                if "CA" in res_i and "CA" in res_j:
                    distance = res_i["CA"] - res_j["CA"]
            elif atom_type == "heavy":
                # Calculate closest heavy atom distance
                min_dist_sq = float("inf")  # Use squared distance for comparison
                for atom_i in res_i.get_atoms():
                    if atom_i.name.startswith("H"):  # Skip hydrogen atoms
                        continue
                    for atom_j in res_j.get_atoms():
                        if atom_j.name.startswith("H"):  # Skip hydrogen atoms
                            continue
                        dist_sq = (atom_i.get_coord() - atom_j.get_coord()) ** 2
                        dist_sq = np.sum(
                            dist_sq
                        )  # Sum squared differences to get squared Euclidean distance
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                distance = np.sqrt(min_dist_sq)  # Take square root at the end

            # Add edge if within cutoff
            if distance <= cutoff_distance:
                edges.append((i, j))
    logger.debug(f"Extracted {len(edges)} edges")
    return edges


# === MACE FUNCTIONS === #
def create_mace_calculator(
    model_path: str = "mace_model/MACE-MP-0b3.model", device: str = "cuda"
) -> Optional[Any]:
    """
    Create a MACE calculator for generating molecular embeddings.

    Parameters
    ----------
    model_path : str
        Path to the MACE model file
    device : str
        Device to use for calculations ('cuda' or 'cpu')

    Returns
    -------
    calculator : Any or None
        MACE calculator instance, or None if MACE is not available
    """
    logger.info("Creating MACE calculator")
    logger.debug(f"Using model: {model_path} on {device} device")

    calculator = MACECalculator(
        model_paths=model_path,
        device=device,
    )

    logger.info("MACE calculator created successfully")
    return calculator


def generate_mace_embedding(
    coords: np.ndarray, atom_types: List[str], mace_calculator: Any
) -> Optional[np.ndarray]:
    """
    Generate MACE embedding for a single molecule.

    Parameters
    ----------
    coords : np.ndarray
        Molecular coordinates
    atom_types : List[str]
        List of atom type symbols
    mace_calculator : Any
        The MACE calculator instance

    Returns
    -------
    mace_embedding : np.ndarray or None
        The MACE embedding, or None if generation failed
    """
    # Create ASE Atoms object
    ase_atoms = Atoms(positions=coords, symbols=atom_types, pbc=False)

    # Generate MACE embedding
    mace_embedding = mace_calculator.get_descriptors(ase_atoms, num_layers=-1)

    return mace_embedding


def add_mace_embeddings_to_entry(
    entry_data: Dict,
    coords: np.ndarray,
    atom_types: List[str],
    mace_calculator: Any,
    include_mace: bool = True,
) -> Dict:
    """
    Add MACE embeddings to an entry data dictionary.

    Parameters
    ----------
    entry_data : dict
        Entry data dictionary to be updated with MACE embeddings.
    coords : np.ndarray
        Molecular coordinates with shape (n_atoms, 3).
    atom_types : list[str]
        List of atom type strings (element symbols) for each atom.
    mace_calculator : Any
        The MACE calculator instance for generating embeddings.
    include_mace : bool, optional
        Whether to include MACE embeddings. Default is True.

    Returns
    -------
    dict
        Updated entry data dictionary with 'mol_mace_embedding' key added (if successful).
    """
    # if not include_mace or not MACE_AVAILABLE or mace_calculator is None:
    #     return entry_data

    # # Check if we have the required molecular data
    # if "mol_gfn2_xyz" not in entry_data or "mol_gfn2_atype" not in entry_data:
    #     logger.debug("No molecular coordinate data found, skipping MACE embedding")
    #     return entry_data

    try:
        mace_embedding = generate_mace_embedding(coords, atom_types, mace_calculator)

        if mace_embedding is not None:
            entry_data["mol_mace_embedding"] = mace_embedding
            logger.debug(f"Added MACE embedding with shape: {mace_embedding.shape}")
        else:
            logger.debug("Failed to generate MACE embedding")

    except Exception as e:
        logger.error(f"Error adding MACE embedding to entry: {e}")

    return entry_data


def create_pocket_mask(
    pocket_residues: List[str],
    chain_mapping: Dict[str, Dict[int, int]],
    sequence_length: int,
) -> np.ndarray:
    """
    Create a binary mask indicating which residues are part of the binding pocket.

    The mask corresponds to the actual protein sequence positions (excluding special tokens).
    The embedding tensor structure is: [CLS] residue_1 residue_2 ... residue_n [EOS] [PAD] ...
                                        0    1        2              n        n+1   n+2

    But the pocket mask corresponds to:    residue_1 residue_2 ... residue_n
                                           0        1              n-1

    Parameters
    ----------
    pocket_residues : List[str]
        List of residue IDs like ['A_116', 'B_127', ...]
    chain_mapping : Dict[str, Dict[int, int]]
        Mapping from chain+residue to ESM sequence position (includes CLS offset)
    sequence_length : int
        Total length of the protein sequence (excluding special tokens)

    Returns
    -------
    pocket_mask : np.ndarray
        Binary array of shape (sequence_length,) where 1 indicates pocket residue
    """
    # Initialize mask with zeros
    pocket_mask = np.zeros(sequence_length, dtype=np.int32)

    logger.debug(f"Creating pocket mask for sequence length: {sequence_length}")
    logger.debug(f"Processing {len(pocket_residues)} pocket residues")

    # Set pocket positions to 1
    for residue_id in pocket_residues:
        try:
            # Parse residue ID (e.g., 'A_116' -> chain='A', res_num=116)
            chain_id, res_num_str = residue_id.split("_")

            # Handle cases like 'C_50B' where there's a letter suffix
            res_num = int("".join(filter(str.isdigit, res_num_str)))

            # Get ESM sequence position (includes CLS offset)
            if chain_id in chain_mapping and res_num in chain_mapping[chain_id]:
                esm_pos = chain_mapping[chain_id][res_num]

                # Convert ESM position (which includes CLS offset) to mask position
                # ESM position 1 -> mask position 0, ESM position 2 -> mask position 1, etc.
                mask_pos = esm_pos - 1  # Subtract 1 to account for CLS token

                # Check bounds to avoid index errors
                if 0 <= mask_pos < sequence_length:
                    pocket_mask[mask_pos] = 1
                    logger.debug(
                        f"Mapped residue {residue_id}: ESM pos {esm_pos} -> mask pos {mask_pos}"
                    )
                else:
                    logger.debug(
                        f"Warning: Mask position {mask_pos} (ESM pos {esm_pos}) out of bounds for residue {residue_id}"
                    )
            else:
                logger.debug(f"Warning: Could not map residue {residue_id}")

        except (ValueError, IndexError) as e:
            logger.debug(f"Warning: Could not parse residue ID {residue_id}: {e}")
            continue

    logger.debug(
        f"Created pocket mask: {np.sum(pocket_mask)} pocket residues out of {sequence_length} total"
    )
    return pocket_mask
