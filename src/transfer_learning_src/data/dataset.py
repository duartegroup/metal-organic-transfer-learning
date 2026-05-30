import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BindingAffinitiesDataset(Dataset):
    """
    PyTorch Dataset for binding affinity prediction tasks.

    This dataset handles molecular, protein, and pocket features along with binding affinities.
    It supports data preprocessing including SMOTE for balancing and feature scaling.

    Parameters
    ----------
    inhibitor_features : dict
        Dictionary mapping compound keys to inhibitor feature arrays.
    protein_embeddings : dict
        Dictionary mapping compound keys to protein embedding arrays.
    pocket_embeddings : dict
        Dictionary mapping compound keys to pocket embedding arrays.
    binding_affinities : dict
        Dictionary mapping compound keys to binding affinity values.
    modality : str
        Modality string indicating which features to use (e.g., 'mol', 'protein', 'pocket', or combinations).
    apply_smote : bool, optional
        Whether to apply SMOTE for balancing the dataset. Default is False.
    smote_k_neighbors : int, optional
        Number of neighbors for SMOTE algorithm. Default is 5.
    binding_threshold : float, optional
        Threshold for binary classification of binding affinity. Default is 7.0.
    scaler_type : str or None, optional
        Type of scaler to use ('standard', 'minmax', or None). Default is None.
    scaler_stats : dict or None, optional
        Precomputed scaler statistics for consistent scaling across datasets. Default is None.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    """

    def __init__(
        self,
        inhibitor_features: dict,
        protein_embeddings: dict,
        pocket_embeddings: dict,
        binding_affinities: dict,
        modality: str,
        apply_smote: bool = False,
        smote_k_neighbors: int = 5,
        binding_threshold: float = 7.0,
        scaler_type: Optional[str] = None,
        scaler_stats: Optional[dict] = None,
        random_state: int = 42,
    ) -> None:

        self.keys = list(binding_affinities.keys())
        assert (
            list(inhibitor_features.keys())
            == list(protein_embeddings.keys())
            == list(pocket_embeddings.keys())
            == self.keys
        ), "Keys are not the same across all inputs."

        # Convert raw data to tensors
        self.inhibitor_tensors = [
            _to_tensor(inhibitor_features[k]).squeeze() for k in self.keys
        ]
        self.protein_embeddings_tensors = [
            _to_tensor(protein_embeddings[k]).squeeze() for k in self.keys
        ]
        self.pocket_embeddings_tensors = [
            _to_tensor(pocket_embeddings[k]).squeeze() for k in self.keys
        ]
        # Continuous affinities as floats, keep as tensor
        self.binding_affinities_tensor = torch.tensor(
            [float(binding_affinities[k]) for k in self.keys], dtype=torch.float32
        )

        self.binding_threshold = binding_threshold
        self.key_to_index = {k: i for i, k in enumerate(self.keys)}
        self.scaler_type = scaler_type
        self.scaler_stats = scaler_stats
        self.random_state = random_state
        self.modality = modality

        # Compute fixed-length vectors first (no scaling yet)
        self._compute_fixed_vectors()

        # Apply SMOTE first (following supervised learning workflow)
        if apply_smote:
            self._apply_smote(smote_k_neighbors)

        # Apply scaling after SMOTE (following supervised learning workflow)
        if scaler_type is not None:
            self._apply_scaling()

        # Update dataset length and pos/neg stats based on continuous affinity threshold
        self.length = len(self.affinities)
        self.pos_count = int((self.affinities > binding_threshold).sum().item())
        self.neg_count = self.length - self.pos_count
        self.pos_weight = torch.tensor(
            self.neg_count / max(1, self.pos_count), dtype=torch.float32
        )

    def _compute_fixed_vectors(self) -> None:
        """
        Compute fixed-length vectors from variable-length embeddings without scaling.

        Molecular features are averaged over atoms, protein features use the CLS token (first row),
        and pocket features are averaged over residues.
        """
        # Mol: mean over atom dim; Protein: CLS token (0th row); Pocket: mean over residues
        mol_vecs = torch.stack([mol.mean(dim=0) for mol in self.inhibitor_tensors])
        protein_vecs = torch.stack(
            [prot[0] for prot in self.protein_embeddings_tensors]
        )
        pocket_vecs = torch.stack(
            [pocket.mean(dim=0) for pocket in self.pocket_embeddings_tensors]
        )

        self.mol_vecs = mol_vecs
        self.protein_vecs = protein_vecs
        self.pocket_vecs = pocket_vecs

        # Continuous affinities, unchanged
        self.affinities = self.binding_affinities_tensor.clone()

    def _apply_smote(self, k_neighbors: int) -> None:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

        Creates synthetic samples to balance positive and negative classes based on the
        binding threshold. Assigns synthetic affinities based on the class label.

        Parameters
        ----------
        k_neighbors : int
            Number of nearest neighbors to use for synthetic sample generation.
        """
        from imblearn.over_sampling import SMOTE

        # Create binary labels for SMOTE based on threshold
        binary_labels = (self.affinities > self.binding_threshold).int().numpy()

        # Concatenate features based on modality
        X_combined = self._get_combined_features()
        X_combined_np = X_combined.numpy()

        smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_combined_np, binary_labels)

        # Split back into components based on modality
        self._split_combined_features(torch.tensor(X_res, dtype=torch.float32))

        # Synthetic continuous affinities for SMOTE samples:
        # For synthetic positives assign a high affinity (e.g., 8.0),
        # for synthetic negatives assign a lower affinity (e.g., 6.0).
        synthetic_affinities = [
            self.binding_threshold + 1.0 if lbl == 1 else self.binding_threshold - 1.0
            for lbl in y_res
        ]
        self.affinities = torch.tensor(synthetic_affinities, dtype=torch.float32)

    def _get_combined_features(self) -> torch.Tensor:
        """
        Get combined feature tensor based on the active modality.

        Concatenates molecular, protein, and/or pocket features according to the
        modality specification.

        Returns
        -------
        torch.Tensor
            Combined feature tensor with shape (n_samples, total_feature_dim).

        Raises
        ------
        ValueError
            If the modality string is invalid or unrecognized.
        """
        if (
            "mol" in self.modality
            and "protein" in self.modality
            and "pocket" in self.modality
        ):
            return torch.cat(
                [self.mol_vecs, self.protein_vecs, self.pocket_vecs], dim=1
            )
        elif (
            "mol" in self.modality
            and "protein" in self.modality
            and "pocket" not in self.modality
        ):
            return torch.cat([self.mol_vecs, self.protein_vecs], dim=1)
        elif (
            "mol" in self.modality
            and "pocket" in self.modality
            and "protein" not in self.modality
        ):
            return torch.cat([self.mol_vecs, self.pocket_vecs], dim=1)
        elif self.modality in ("mol", "molecule"):  # Only molecular features
            return self.mol_vecs
        elif self.modality == "protein":
            return self.protein_vecs
        elif self.modality == "pocket":
            return self.pocket_vecs
        else:
            raise ValueError(
                f"Invalid modality: {self.modality}. Expected 'mol', 'protein', 'pocket', or combinations"
            )

    def _split_combined_features(self, X_combined_tensor: torch.Tensor) -> None:
        """
        Split combined feature tensor back into component features based on modality.

        Parameters
        ----------
        X_combined_tensor : torch.Tensor
            Combined feature tensor to be split into molecular, protein, and/or pocket components.

        Raises
        ------
        ValueError
            If the modality string is invalid or unrecognized.
        """
        if (
            "mol" in self.modality
            and "protein" in self.modality
            and "pocket" in self.modality
        ):
            mol_dim = self.mol_vecs.shape[1]
            protein_dim = self.protein_vecs.shape[1]
            pocket_dim = self.pocket_vecs.shape[1]

            self.mol_vecs = X_combined_tensor[:, :mol_dim]
            self.protein_vecs = X_combined_tensor[:, mol_dim : mol_dim + protein_dim]
            self.pocket_vecs = X_combined_tensor[:, mol_dim + protein_dim :]

        elif (
            "mol" in self.modality
            and "protein" in self.modality
            and "pocket" not in self.modality
        ):
            mol_dim = self.mol_vecs.shape[1]
            protein_dim = self.protein_vecs.shape[1]

            self.mol_vecs = X_combined_tensor[:, :mol_dim]
            self.protein_vecs = X_combined_tensor[:, mol_dim : mol_dim + protein_dim]

        elif (
            "mol" in self.modality
            and "pocket" in self.modality
            and "protein" not in self.modality
        ):
            mol_dim = self.mol_vecs.shape[1]
            pocket_dim = self.pocket_vecs.shape[1]

            self.mol_vecs = X_combined_tensor[:, :mol_dim]
            self.pocket_vecs = X_combined_tensor[:, mol_dim:]

        elif self.modality in ("mol", "molecule"):  # Only molecular features
            self.mol_vecs = X_combined_tensor
        elif self.modality == "protein":
            self.protein_vecs = X_combined_tensor
        elif self.modality == "pocket":
            self.pocket_vecs = X_combined_tensor
        else:
            raise ValueError(f"Invalid modality: {self.modality}")

    def _apply_scaling(self) -> None:
        """
        Apply feature scaling using scikit-learn scalers with optional precomputed statistics.

        Uses either StandardScaler or MinMaxScaler based on scaler_type. If scaler_stats
        are provided, applies precomputed scaling statistics for consistency across datasets.

        Raises
        ------
        ValueError
            If the scaler_type is unknown or unsupported.
        """
        if self.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif self.scaler_type == "none":
            return
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        # Get combined features based on modality
        X_combined = self._get_combined_features()
        X_combined_np = X_combined.numpy()

        if self.scaler_stats is not None:
            # Use precomputed statistics
            X_scaled = self._apply_precomputed_scaling(X_combined_np, scaler)
        else:
            # Compute statistics from current data
            X_scaled = scaler.fit_transform(X_combined_np)

        # Split back into components
        X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self._split_combined_features(X_scaled_tensor)

    def _apply_precomputed_scaling(
        self, x: np.ndarray, scaler: Union[StandardScaler, MinMaxScaler]
    ) -> np.ndarray:
        """
        Apply scaling using precomputed statistics for the combined feature vector.

        Parameters
        ----------
        x : np.ndarray
            Feature array to scale, shape (n_samples, n_features).
        scaler : StandardScaler or MinMaxScaler
            Scaler instance to configure with precomputed statistics.

        Returns
        -------
        np.ndarray
            Scaled feature array with the same shape as input.

        Raises
        ------
        KeyError
            If required scaling statistics are missing from scaler_stats.
        ValueError
            If the scaler_type is unknown or unsupported.
        """
        scaler_copy = type(scaler)()  # Create a new instance of the same scaler type

        if self.scaler_type == "standard":
            if (
                "combined_mean" in self.scaler_stats
                and "combined_std" in self.scaler_stats
            ):
                # Set StandardScaler internal attributes
                scaler_copy.mean_ = self.scaler_stats["combined_mean"]
                combined_std = self.scaler_stats["combined_std"]

                # StandardScaler uses scale_ = 1/std and var_ = std^2
                scaler_copy.var_ = combined_std**2
                scaler_copy.scale_ = 1.0 / combined_std
                scaler_copy.n_features_in_ = len(scaler_copy.mean_)

                return scaler_copy.transform(x)
            else:
                raise KeyError(
                    "Missing scaling statistics: combined_mean, combined_std"
                )

        elif self.scaler_type == "minmax":
            if (
                "combined_min" in self.scaler_stats
                and "combined_max" in self.scaler_stats
            ):
                data_min = self.scaler_stats["combined_min"]
                data_max = self.scaler_stats["combined_max"]

                # Set MinMaxScaler internal attributes
                scaler_copy.data_min_ = data_min
                scaler_copy.data_max_ = data_max
                scaler_copy.data_range_ = data_max - data_min

                # Handle zero variance features (where min == max)
                safe_range = np.where(
                    scaler_copy.data_range_ == 0, 1.0, scaler_copy.data_range_
                )
                scaler_copy.scale_ = 1.0 / safe_range
                scaler_copy.min_ = -data_min * scaler_copy.scale_

                # For zero variance features, keep original values (scale=1, min=0)
                zero_var_mask = scaler_copy.data_range_ == 0
                if np.any(zero_var_mask):
                    scaler_copy.scale_[zero_var_mask] = 1.0
                    scaler_copy.min_[zero_var_mask] = 0.0

                scaler_copy.n_features_in_ = len(data_min)
                scaler_copy.feature_range = (0, 1)  # Default range

                return scaler_copy.transform(x)
            else:
                raise KeyError("Missing scaling statistics: combined_min, combined_max")

        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.affinities)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset by index.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - mol_vec: Molecular feature vector (empty tensor if not in modality)
            - protein_vec: Protein feature vector (empty tensor if not in modality)
            - pocket_vec: Pocket feature vector (empty tensor if not in modality)
            - affinity: Binding affinity value
        """
        mol_vec = self.mol_vecs[index] if "mol" in self.modality else torch.tensor([])
        protein_vec = (
            self.protein_vecs[index] if "protein" in self.modality else torch.tensor([])
        )
        pocket_vec = (
            self.pocket_vecs[index] if "pocket" in self.modality else torch.tensor([])
        )
        return (
            mol_vec,
            protein_vec,
            pocket_vec,
            self.affinities[index],
        )

    def get_by_key(
        self, key: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset by its key identifier.

        Parameters
        ----------
        key : str
            The compound identifier key.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing molecular, protein, pocket feature vectors and binding affinity.

        Raises
        ------
        KeyError
            If the key is not found in the dataset.
        """
        idx = self.key_to_index.get(key)
        if idx is None:
            raise KeyError(f"Key {key} not found in dataset")
        return self.__getitem__(idx)


class BindingAffinitiesDatasetWithKeys(BindingAffinitiesDataset):
    """
    Subclass of BindingAffinitiesDataset that returns keys along with data.

    This allows tracking which specific samples are being processed during
    visualization without modifying the original training-compatible dataset.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the dataset with keys included in output.

        Parameters
        ----------
        *args
            Variable length argument list passed to parent class.
        **kwargs
            Arbitrary keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        # Create reverse mapping from index to key for efficient lookup
        self.index_to_key = {v: k for k, v in self.key_to_index.items()}

    def __getitem__(
        self, index: int
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Override __getitem__ to return (key, data) tuple.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - key: Compound identifier string
            - mol_vec: Molecular feature vector
            - protein_vec: Protein feature vector
            - pocket_vec: Pocket feature vector
            - affinity: Binding affinity value
        """
        # Get the original data from parent class
        data = super().__getitem__(index)

        # Get the corresponding key for this index
        key = self.index_to_key[index]

        # Return key along with the data
        return (key,) + data

    def get_by_key(
        self, key: str
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get data by key, returning key along with data.

        Parameters
        ----------
        key : str
            The compound identifier key to get the data for.

        Returns
        -------
        tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing key, molecular, protein, pocket feature vectors and binding affinity.

        Raises
        ------
        KeyError
            If the key is not found in the dataset.
        """
        idx = self.key_to_index.get(key)
        if idx is None:
            raise KeyError(f"Key {key} not found in dataset")
        return self.__getitem__(idx)


class WrappedDataLoader:
    """
    Wrapper for DataLoader that yields batches as dictionaries.

    This wrapper converts standard DataLoader batches into dictionary format
    for compatibility with multi-task learning frameworks.
    """

    def __init__(self, dataloader, key: str) -> None:
        """
        Initialize the wrapped dataloader.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The DataLoader to wrap.
        key : str
            The key to use for the batch dictionary.
        """
        self.dataloader = dataloader
        self.key = key

    def __iter__(self):
        """
        Iterate over batches yielding dictionaries.

        Yields
        ------
        dict
            Dictionary with self.key mapping to the batch data.
        """
        for batch in self.dataloader:
            yield {self.key: batch}

    def __len__(self) -> int:
        """
        Get the number of batches in the dataloader.

        Returns
        -------
        int
            Number of batches.
        """
        return len(self.dataloader)


def _to_tensor(
    data: Union[torch.Tensor, np.ndarray, list], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert data to tensor, handling both tensor and non-tensor inputs properly.

    Parameters
    ----------
    data : torch.Tensor, np.ndarray, or list
        The data to convert to tensor.
    dtype : torch.dtype, optional
        The desired tensor dtype. Default is torch.float32.

    Returns
    -------
    torch.Tensor
        The converted tensor.
    """
    if isinstance(data, torch.Tensor):
        return data.clone().detach().to(dtype)
    else:
        return torch.tensor(data, dtype=dtype)


def apply_scaling_like_training(
    mol_vec: Optional[np.ndarray],
    protein_vec: Optional[np.ndarray],
    pocket_vec: Optional[np.ndarray],
    modality: str,
    stats: dict,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply scaling exactly like in training using scikit-learn scalers.

    Processes and scales molecular, protein, and pocket features according to the
    modality specification and precomputed scaling statistics from training.

    Parameters
    ----------
    mol_vec : np.ndarray or None
        Molecular feature vector (may be per-atom or already averaged).
    protein_vec : np.ndarray or None
        Protein embedding vector (may include CLS token).
    pocket_vec : np.ndarray or None
        Pocket embedding vector (may be per-residue or already averaged).
    modality : str
        Modality string indicating which features to use and combine.
    stats : dict
        Dictionary containing scaling statistics and feature dimensions from training.

    Returns
    -------
    tuple[np.ndarray or None, np.ndarray or None, np.ndarray or None]
        A tuple containing scaled molecular, protein, and pocket feature vectors.

    Raises
    ------
    ValueError
        If no features are available to combine based on the modality.
    """
    # Process features exactly like training dataset._compute_fixed_vectors()
    # Mol: mean over atom dim; Protein: CLS token (0th row); Pocket: mean over residues

    processed_mol = None
    processed_protein = None
    processed_pocket = None

    # Process molecular features: average over atoms if it's per-atom
    if ("molecule" in modality or "mol" in modality) and mol_vec is not None:
        if len(mol_vec.shape) > 1:
            # Average over the first dimension (atoms) - same as training
            processed_mol = np.mean(mol_vec, axis=0).astype(np.float32)
        else:
            processed_mol = mol_vec.astype(np.float32)

    # Process protein features: take the first token (CLS token) - same as training
    if "protein" in modality and protein_vec is not None:
        if len(protein_vec.shape) > 1:
            # Take the first token (CLS token) - same as training
            processed_protein = protein_vec[0].astype(np.float32)
        else:
            processed_protein = protein_vec.astype(np.float32)

    # Process pocket features: average over residues - same as training
    if "pocket" in modality and pocket_vec is not None:
        if len(pocket_vec.shape) > 1:
            # Average over the first dimension (residues) - same as training
            processed_pocket = np.mean(pocket_vec, axis=0).astype(np.float32)
        else:
            processed_pocket = pocket_vec.astype(np.float32)

    # Combine features according to modality (same as training)
    parts = []
    if ("molecule" in modality or "mol" in modality) and processed_mol is not None:
        parts.append(processed_mol)
    if "protein" in modality and processed_protein is not None:
        parts.append(processed_protein)
    if "pocket" in modality and processed_pocket is not None:
        parts.append(processed_pocket)

    if not parts:
        raise ValueError("No features available to combine")

    combined = np.concatenate(parts, axis=0)

    # Apply scaling using scikit-learn scalers (same as training)
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
