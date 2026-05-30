import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryMatthewsCorrCoef,
    BinaryConfusionMatrix,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from typing import Optional, Dict, Any, Tuple, List, Union
from pytorch_optimizer import ADOPT
import os
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MolecularEncoder(nn.Module):
    """
    Encoder for molecular representations with mask-aware mean pooling.

    This encoder processes molecular embeddings and applies either mean pooling
    with masking or direct processing depending on configuration.
    """

    def __init__(
        self,
        input_dim: int = 256,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        mean_pooling: bool = True,
        activation: str = "PReLU",
    ) -> None:
        """
        Initialize the molecular encoder.

        Parameters
        ----------
        input_dim : int, optional
            Dimensionality of input molecular features, by default 256.
        embed_dim : int, optional
            Dimensionality of output embedding, by default 256.
        hidden_dim : int, optional
            Dimensionality of hidden layer, by default 512.
        dropout : float, optional
            Dropout rate for regularization, by default 0.1.
        mean_pooling : bool, optional
            Whether to apply mean pooling to 3D inputs, by default True.
        activation : str, optional
            Activation function type ('ReLU' or 'PReLU'), by default "PReLU".

        Returns
        -------
        None
        """
        super().__init__()
        self.mean_pooling = mean_pooling
        self.activation = activation
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() if self.activation == "ReLU" else nn.PReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU() if self.activation == "ReLU" else nn.PReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the molecular encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input molecular features. Can be 2D (batch_size, input_dim) or
            3D (batch_size, seq_len, input_dim) if mean_pooling is enabled.
        mask : torch.Tensor, optional
            Binary mask indicating valid tokens for mean pooling (required for 3D inputs).

        Returns
        -------
        torch.Tensor
            Encoded molecular embeddings of shape (batch_size, embed_dim).

        Raises
        ------
        ValueError
            If mean_pooling is enabled for 3D input but mask is not provided.
        """
        if x.dim() == 3 and self.mean_pooling:
            if mask is None:
                raise ValueError("mask is required for mean pooling")
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            x = (x * mask.unsqueeze(-1).float()).sum(dim=1) / denom

        return self.layers(x)


class BiologicalEncoder(nn.Module):
    """
    Encoder for protein/biological representations with flexible pooling strategies.

    Supports both CLS token extraction and mean pooling with masking for processing
    protein embeddings from transformer-based models.
    """

    def __init__(
        self,
        input_dim: int = 1152,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_cls: bool = False,
        mean_pooling: bool = True,
        activation: str = "PReLU",
    ) -> None:
        """
        Initialize the biological encoder.

        Parameters
        ----------
        input_dim : int, optional
            Dimensionality of input protein features, by default 1152.
        embed_dim : int, optional
            Dimensionality of output embedding, by default 256.
        hidden_dim : int, optional
            Dimensionality of hidden layer, by default 512.
        dropout : float, optional
            Dropout rate for regularization, by default 0.1.
        use_cls : bool, optional
            Whether to use CLS token (first token) instead of mean pooling, by default False.
        mean_pooling : bool, optional
            Whether to apply mean pooling to 3D inputs, by default True.
        activation : str, optional
            Activation function type ('ReLU' or 'PReLU'), by default "PReLU".

        Returns
        -------
        None
        """
        super().__init__()
        self.use_cls = use_cls
        self.mean_pooling = mean_pooling
        self.activation = activation
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() if self.activation == "ReLU" else nn.PReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU() if self.activation == "ReLU" else nn.PReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the biological encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input protein features. Can be 2D (batch_size, input_dim) or
            3D (batch_size, seq_len, input_dim) if mean_pooling is enabled.
        mask : torch.Tensor, optional
            Binary mask indicating valid tokens for mean pooling (required for 3D inputs
            when use_cls is False).

        Returns
        -------
        torch.Tensor
            Encoded protein embeddings of shape (batch_size, embed_dim).

        Raises
        ------
        ValueError
            If mean_pooling is enabled for 3D input, use_cls is False, but mask is not provided.
        """
        if x.dim() == 3 and self.mean_pooling:
            if self.use_cls:
                # take CLS token directly, ignore mask
                x = x[:, 0, :]
            else:
                if mask is None:
                    raise ValueError("mask is required for mean pooling")
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
                x = (x * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return self.layers(x)


class FeatureExtractor(nn.Module):
    """
    Feature extraction network for multi-modal molecular representations.

    Combines different modality embeddings and projects them to a normalized
    latent space for downstream tasks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        dropout: float = 0.1,
        activation: str = "ReLU",
    ) -> None:
        """
        Initialize the feature extractor.

        Parameters
        ----------
        input_dim : int
            Dimensionality of concatenated input embeddings from multiple modalities.
        hidden_dim : int
            Dimensionality of hidden layer.
        embed_dim : int
            Dimensionality of output embedding (latent space).
        dropout : float, optional
            Dropout rate for regularization, by default 0.1.
        activation : str, optional
            Activation function type ('ReLU' or 'PReLU'), by default "ReLU".

        Returns
        -------
        None
        """
        super().__init__()
        self.activation = activation
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() if self.activation == "ReLU" else nn.PReLU(),
            nn.Dropout(dropout),  # Dropout after activation of first hidden layer
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU() if self.activation == "ReLU" else nn.PReLU(),
            nn.Dropout(
                dropout
            ),  # Dropout after second activation, before latent embedding output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Parameters
        ----------
        x : torch.Tensor
            Concatenated embeddings from multiple modalities, shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            L2-normalized latent embeddings of shape (batch_size, embed_dim).
        """
        x = self.layers(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class BindingAffinityHead(nn.Module):
    """
    Classification head for binding affinity prediction.

    Predicts binary binding affinity from latent molecular-protein embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "ReLU",
    ) -> None:
        """
        Initialize the binding affinity classification head.

        Parameters
        ----------
        input_dim : int
            Dimensionality of input latent embeddings.
        hidden_dim : int
            Dimensionality of hidden layers (will be progressively reduced).
        dropout : float, optional
            Dropout rate for regularization, by default 0.1.
        activation : str, optional
            Activation function type ('ReLU' or 'PReLU'), by default "ReLU".

        Returns
        -------
        None
        """
        super().__init__()
        self.activation = activation
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU() if self.activation == "ReLU" else nn.PReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU() if self.activation == "ReLU" else nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict binding affinity logits.

        Parameters
        ----------
        embedding : torch.Tensor
            Latent embeddings from the feature extractor, shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Binary classification logits, shape (batch_size,).
        """
        return self.classifier(embedding).squeeze(-1)


class ProjectionHead(nn.Module):
    """
    SimCLR-style projection head for contrastive self-supervised learning.

    Projects embeddings to a space optimized for contrastive loss computation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        Initialize the projection head for contrastive learning.

        Parameters
        ----------
        input_dim : int
            Dimensionality of input embeddings from feature extractor.
        hidden_dim : int
            Dimensionality of hidden layer.
        output_dim : int
            Dimensionality of output projection space.

        Returns
        -------
        None
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False),  # No affine -> just normalization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection head.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings from feature extractor, shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Projected embeddings for contrastive learning, shape (batch_size, output_dim).
        """
        return self.net(x)


class MultiTaskPocket(pl.LightningModule):
    """
    Multi-task model for binding affinity classification in latent space.

    Combines molecule, protein, and pocket representations for binding affinity
    classification. Supports both supervised learning and self-supervised learning
    via contrastive methods.

    This model can be trained in different modes:
    - Supervised: Direct classification of binding affinity
    - Self-supervised: Contrastive learning on molecular representations
    - Transfer learning: Domain adaptation between source and target datasets
    """

    def __init__(
        self,
        domain: str = "src",
        self_ssl: bool = False,
        temperature: float = 0.07,
        **params: Any,
    ) -> None:
        """
        Initialize the multi-task pocket binding affinity model.

        Parameters
        ----------
        domain : str, optional
            Training domain ('src' for source or 'tgt' for target), by default "src".
        self_ssl : bool, optional
            Whether to use self-supervised learning mode with contrastive loss, by default False.
        temperature : float, optional
            Temperature parameter for InfoNCE contrastive loss, by default 0.07.
        **params : Any
            Additional configuration parameters including:
            - embed_dim (int): Embedding dimension, default 128
            - dropout (float): Dropout rate, default 0.4
            - learning_rate (float): Learning rate, default 1e-4
            - batch_size (int): Batch size for logging, default 128
            - threshold (float): Affinity threshold for binary classification, default 7.0
            - use_mol (bool): Whether to use molecular encoder, default False
            - use_protein (bool): Whether to use protein encoder, default False
            - use_pocket (bool): Whether to use pocket encoder, default False
            - inhibitor_input_dim (int): Input dimension for molecular features, default 256
            - biological_input_dim (int): Input dimension for protein/pocket features, default 1152
            - activation (str): Activation function type, default "PReLU"
            - ssl_noise_scale (float): Noise scale for SSL augmentations, default 0.3
            - ssl_node_drop_prob (float): Node dropout probability for SSL, default 0.3
            - scaler: Optional scaler for feature normalization

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no modalities are enabled (use_mol, use_protein, use_pocket all False).
        """
        super().__init__()
        self.domain = domain
        self.self_ssl = self_ssl
        self.temperature = temperature
        self.noise_scale = params.get("ssl_noise_scale", 0.3)
        self.drop_prob = params.get("ssl_node_drop_prob", 0.3)
        self.scaler = params.get("scaler", None)

        # Core parameters
        self.domain = domain
        embed_dim = params.get("embed_dim", 128)
        hidden_dim = embed_dim * 2
        dropout = params.get("dropout", 0.4)
        self.learning_rate = params.get("learning_rate", 1e-4)
        self.batch_size = params.get("batch_size", 128)
        # Make threshold robust (can be float or list/tuple)
        raw_threshold = params.get("threshold", 7.0)
        if isinstance(raw_threshold, (list, tuple)):
            self.threshold = float(raw_threshold[0]) if len(raw_threshold) > 0 else 7.0
        else:
            self.threshold = float(raw_threshold)

        # Store encoder flags
        self.use_mol = params.get("use_mol", False)
        self.use_protein = params.get("use_protein", False)
        self.use_pocket = params.get("use_pocket", False)

        self.pos_weight = None

        # Generate combination name based on active modalities
        combo_parts = []
        if self.use_mol:
            combo_parts.append("mol")
        if self.use_protein:
            combo_parts.append("protein")
        if self.use_pocket:
            combo_parts.append("pocket")
        self.combination_name = "_".join(combo_parts) if combo_parts else "empty"

        # Initialize encoders
        self.mol_encoder = None
        self.protein_encoder = None
        self.pocket_encoder = None

        # Calculate total input dimension for feature extractor
        total_input_dim = 0

        # Molecule encoder: mean pooling
        if self.use_mol:
            self.mol_encoder = MolecularEncoder(
                input_dim=params.get("inhibitor_input_dim", 256),
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation=params.get("activation", "PReLU"),
            )
            total_input_dim += embed_dim

        # Pocket encoder: mean pooling
        if self.use_pocket:
            self.pocket_encoder = BiologicalEncoder(
                input_dim=params.get("biological_input_dim", 1152),
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_cls=False,
                activation=params.get("activation", "PReLU"),
            )
            total_input_dim += embed_dim

        # Protein encoder: CLS token
        if self.use_protein:
            self.protein_encoder = BiologicalEncoder(
                input_dim=params.get("biological_input_dim", 1152),
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_cls=True,
                activation=params.get("activation", "PReLU"),
            )
            total_input_dim += embed_dim

        # Validate at least one modality is enabled
        if total_input_dim == 0:
            raise ValueError(
                "At least one modality must be enabled (use_mol, use_protein, or use_pocket)"
            )

        # FFN with dynamic input dimension
        self.feature_extractor = FeatureExtractor(
            input_dim=total_input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            activation=params.get("activation", "PReLU"),
        )

        # Example prediction head
        if not self_ssl:
            self.classification_head = BindingAffinityHead(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation=params.get("activation", "PReLU"),
            )
        else:
            # projection head for contrastive SSL
            self.projection_head = ProjectionHead(
                input_dim=embed_dim, hidden_dim=hidden_dim, output_dim=embed_dim
            )

        # Setup metrics
        if not self_ssl:
            self.setup_metrics()

        # Initialize probe storage lists
        self._reset_probe_storage()

    def _reset_probe_storage(self) -> None:
        """
        Reset probe storage lists to avoid memory leaks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.train_embeddings = []
        self.train_labels = []
        self.val_embeddings = []
        self.val_labels = []

    def print_model(self) -> None:
        """
        Log model architecture and active modalities to the logger.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logger.info(f"Model Architecture:\n{self}")
        logger.info(
            f"Active modalities: mol={self.use_mol}, protein={self.use_protein}, pocket={self.use_pocket}"
        )

    def setup_metrics(self) -> None:
        """
        Initialize all binary classification metrics for training and validation.

        Sets up metrics including accuracy, precision, recall, F1, AUROC, AUPRC,
        MCC (Matthews Correlation Coefficient), and confusion matrix.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()
        self.auroc = BinaryAUROC()
        self.auprc = BinaryAveragePrecision()
        self.mcc = BinaryMatthewsCorrCoef()
        self.confmat = BinaryConfusionMatrix()

    def _evaluate_probe(
        self,
        dataloader: Any,
        probe_type: str = "knn",
        k_list: List[int] = [5, 10, 20],
        probe_split_ratio: float = 0.8,
    ) -> Dict[str, float]:
        """
        Evaluate learned embeddings using probe classifiers (k-NN or linear).

        This method evaluates the quality of learned embeddings in self-supervised
        learning by training a simple probe classifier and measuring performance.

        Parameters
        ----------
        dataloader : Any
            PyTorch DataLoader containing probe evaluation data.
        probe_type : str, optional
            Type of probe classifier to use ("knn" or "linear"), by default "knn".
        k_list : List[int], optional
            List of k values to evaluate for k-NN classifier, by default [5, 10, 20].
        probe_split_ratio : float, optional
            Ratio for train/val split of probe data, by default 0.8.

        Returns
        -------
        Dict[str, float]
            Dictionary with probe evaluation metrics:
            - "val/knn_auroc": AUROC score averaged across k values
            - "val/knn_auprc": AUPRC score averaged across k values

        Raises
        ------
        ValueError
            If probe_type is not "knn" or "linear".
        """
        if not self.self_ssl:
            return {"val/knn_auroc": 0.0, "val/knn_auprc": 0.0}

        # Store current training mode and set to eval
        was_training = self.training
        self.eval()

        embeddings = []
        labels = []

        try:
            # Extract embeddings and labels
            with torch.no_grad():
                for batch in dataloader:
                    try:
                        # Get the batch for the current domain
                        domain_batch = batch[self.domain]

                        # Encode to get embeddings (encode method now handles device placement)
                        emb = self.encode(domain_batch)

                        # L2 normalize embeddings
                        emb = F.normalize(emb, dim=1)

                        # Get labels (last element of domain_batch)
                        batch_labels = domain_batch[-1]
                        binary_labels = (batch_labels > self.threshold).float()

                        # Move to CPU for sklearn processing
                        embeddings.append(emb.detach().cpu().numpy())
                        labels.append(binary_labels.detach().cpu().numpy())

                    except Exception as e:
                        # Skip this batch if there's an error
                        logger.debug(f"Skipping batch in probe evaluation: {e}")
                        continue

            if len(embeddings) == 0:
                return {"val/knn_auroc": 0.0, "val/knn_auprc": 0.0}

            # Concatenate all embeddings and labels
            embeddings = np.vstack(embeddings)
            labels = np.concatenate(labels)

            # Ensure we have both classes for meaningful evaluation
            if len(np.unique(labels)) < 2:
                return {"val/knn_auroc": 0.0, "val/knn_auprc": 0.0}

            # Split into train/val for probe evaluation
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    embeddings,
                    labels,
                    train_size=probe_split_ratio,
                    stratify=labels,
                    random_state=42,
                )
            except ValueError:
                # If stratification fails (e.g., too few samples), use random split
                X_train, X_val, y_train, y_val = train_test_split(
                    embeddings, labels, train_size=probe_split_ratio, random_state=42
                )

            if probe_type == "knn":
                return self._evaluate_knn_probe_core(
                    X_train, X_val, y_train, y_val, k_list
                )
            # elif probe_type == "linear":
            # return self._evaluate_linear_probe_core(X_train, X_val, y_train, y_val)
            else:
                raise ValueError(f"Unknown probe type: {probe_type}")

        except Exception as e:
            logger.debug(f"Probe evaluation failed: {e}")
            return {"val/knn_auroc": 0.0, "val/knn_auprc": 0.0}
        finally:
            # Restore original training mode
            if was_training:
                self.train()
            else:
                self.eval()

    def _evaluate_knn_probe_core(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        k_list: List[int],
    ) -> Dict[str, float]:
        """
        Core k-NN probe evaluation logic with multiple k values.

        Parameters
        ----------
        X_train : np.ndarray
            Training embeddings of shape (n_train, embed_dim).
        X_val : np.ndarray
            Validation embeddings of shape (n_val, embed_dim).
        y_train : np.ndarray
            Training labels of shape (n_train,).
        y_val : np.ndarray
            Validation labels of shape (n_val,).
        k_list : List[int]
            List of k values to evaluate for k-NN classifier.

        Returns
        -------
        Dict[str, float]
            Dictionary containing averaged AUROC and AUPRC scores:
            - "val/knn_auroc": Average AUROC across all k values
            - "val/knn_auprc": Average AUPRC across all k values
        """
        auroc_scores = []
        auprc_scores = []

        # Evaluate for each k value
        for k in k_list:
            # Ensure k doesn't exceed training data size
            k_actual = min(k, len(X_train) - 1)
            if k_actual < 1:
                continue

            try:
                # Fit k-NN classifier with cosine distance
                knn = KNeighborsClassifier(n_neighbors=k_actual, metric="cosine")
                knn.fit(X_train, y_train)

                # Get prediction probabilities
                y_pred_proba = knn.predict_proba(X_val)

                # Handle case where only one class is predicted
                if y_pred_proba.shape[1] == 1:
                    continue

                # Extract positive class probabilities
                y_scores = y_pred_proba[:, 1]

                # Calculate metrics
                auroc = roc_auc_score(y_val, y_scores)
                auprc = average_precision_score(y_val, y_scores)

                auroc_scores.append(auroc)
                auprc_scores.append(auroc)

            except Exception as e:
                # Skip this k value if evaluation fails
                continue

        # Return averages across k values
        avg_auroc = np.mean(auroc_scores) if auroc_scores else 0.0
        avg_auprc = np.mean(auprc_scores) if auprc_scores else 0.0

        return {"val/knn_auroc": float(avg_auroc), "val/knn_auprc": float(avg_auprc)}

    def info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss for self-supervised learning.

        Parameters
        ----------
        z1 : torch.Tensor
            First set of projected embeddings, shape (batch_size, projection_dim).
        z2 : torch.Tensor
            Second set of projected embeddings from augmented views, shape (batch_size, projection_dim).

        Returns
        -------
        torch.Tensor
            Symmetric InfoNCE contrastive loss (scalar).
        """
        # normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=self.device)
        return (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        ) / 2

    def ssl_augment(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Create two augmented views of the same batch for self-supervised learning.

        Applies data augmentation techniques including Gaussian noise and random
        token dropout to create two independent views of the same data for
        contrastive learning.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, ...]
            Input batch tuple containing (mol, protein, pocket, labels).

        Returns
        -------
        Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]
            Two augmented views (view1, view2), each with the same structure as the input batch
            but with independent augmentations applied. Each view is a tuple of
            (mol_augmented, protein_augmented, pocket_augmented, labels).
        """
        noise_scale = self.noise_scale
        drop_prob = self.drop_prob

        def random_mask_tokens(tensor, mask, p):
            # clone so original is not modified
            t = tensor.clone() if tensor is not None else None
            m = mask.clone() if mask is not None else None
            if p > 0 and m is not None and t is not None and t.numel() > 0:
                drop = (torch.rand_like(m.float()) < p) & (m.bool())
                t[drop] = 0  # zero out tokens
                m[drop] = False
            return t, m

        def add_gaussian_noise(tensor, scale, relative=True):
            """
            Add Gaussian noise to a tensor.

            Args:
                tensor (torch.Tensor): Input tensor
                scale (float): Noise scale (interpreted as fraction of std if relative=True)
                relative (bool): If True, scale is relative to feature std

            Returns:
                torch.Tensor
            """
            if tensor is None or scale <= 0 or tensor.numel() == 0:
                return tensor

            noise = torch.randn_like(tensor)

            if relative:
                # Compute per-feature std; avoid div-by-zero
                std = tensor.std(dim=0, keepdim=True).clamp(min=1e-6)
                noise = noise * (scale * std)
            else:
                noise = noise * scale

            return tensor + noise

        mol, protein, pocket, labels = batch

        def make_view():
            mol_v, protein_v, pocket_v = mol, protein, pocket

            if self.use_mol and mol.numel() > 0:
                mol_mask = torch.ones_like(mol)
                mol_v, _ = random_mask_tokens(mol, mol_mask, drop_prob)
                mol_v = add_gaussian_noise(mol_v, noise_scale)

            if self.use_protein and protein.numel() > 0:
                protein_mask = torch.ones_like(protein)
                protein_v, _ = random_mask_tokens(protein, protein_mask, drop_prob)
                protein_v = add_gaussian_noise(protein_v, noise_scale)

            if self.use_pocket and pocket.numel() > 0:
                pocket_mask = torch.ones_like(pocket)
                pocket_v, _ = random_mask_tokens(pocket, pocket_mask, drop_prob)
                pocket_v = add_gaussian_noise(pocket_v, noise_scale)

            return (mol_v, protein_v, pocket_v, labels)

        return make_view(), make_view()

    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward pass through the full model for supervised classification.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, ...]
            Input batch tuple containing (mol, protein, pocket, labels).

        Returns
        -------
        torch.Tensor
            Classification logits for binding affinity prediction, shape (batch_size,).
        """
        return self.classification_head(self.encode(batch))

    def encode(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Encode multi-modal inputs into a unified latent embedding.

        This method processes molecule, protein, and pocket features through
        their respective encoders and combines them into a single normalized
        latent representation.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, ...]
            Input batch tuple containing (mol, protein, pocket, labels).

        Returns
        -------
        torch.Tensor
            Normalized latent embeddings of shape (batch_size, embed_dim).

        Raises
        ------
        RuntimeError
            If no encoders are available (all modalities disabled).
        """
        mol, protein, pocket, labels = batch

        # Ensure all inputs are on the same device as the model
        device = next(self.parameters()).device

        mol = mol.to(device) if torch.is_tensor(mol) else mol
        protein = protein.to(device) if torch.is_tensor(protein) else protein
        pocket = pocket.to(device) if torch.is_tensor(pocket) else pocket

        embeddings = []
        if self.use_mol:
            mol_mask = torch.ones_like(mol, device=device) if mol.numel() > 0 else None
            mol_emb = self.mol_encoder(mol, mol_mask)
            embeddings.append(mol_emb)

        if self.use_protein:
            protein_mask = (
                torch.ones_like(protein, device=device) if protein.numel() > 0 else None
            )
            protein_emb = self.protein_encoder(protein, protein_mask)
            embeddings.append(protein_emb)

        if self.use_pocket:
            pocket_mask = (
                torch.ones_like(pocket, device=device) if pocket.numel() > 0 else None
            )
            pocket_emb = self.pocket_encoder(pocket, pocket_mask)
            embeddings.append(pocket_emb)

        if len(embeddings) == 0:
            raise RuntimeError("No encoders are available - check model configuration")
        elif len(embeddings) == 1:
            x = embeddings[0]
        else:
            x = torch.cat(embeddings, dim=1)

        return self.feature_extractor(x)

    def compute_loss(
        self, affinity: torch.Tensor, forward_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute binary classification loss for binding affinity prediction.

        Parameters
        ----------
        affinity : torch.Tensor
            Continuous affinity values (e.g., pIC50 or pKd), shape (batch_size,).
        forward_output : torch.Tensor
            Model logits for classification, shape (batch_size,).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - loss : Binary cross-entropy loss with logits (scalar).
            - labels : Binary labels based on affinity threshold, shape (batch_size,).
        """
        labels = (affinity > self.threshold).float()

        loss = F.binary_cross_entropy_with_logits(
            forward_output, labels, pos_weight=self.pos_weight
        )

        return loss, labels

    def training_step(
        self, batch: Dict[str, Tuple[torch.Tensor, ...]], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Parameters
        ----------
        batch : Dict[str, Tuple[torch.Tensor, ...]]
            Dictionary containing domain-specific batches. Keys are domain names ('src', 'tgt'),
            values are tuples of (mol, protein, pocket, labels).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Training loss for the current batch (scalar).
        """
        if self.self_ssl:
            view1, view2 = self.ssl_augment(batch[self.domain])
            z1 = self.projection_head(self.encode(view1))
            z2 = self.projection_head(self.encode(view2))
            loss = self.info_nce_loss(z1, z2)
            self.log("train/ssl_loss", loss, prog_bar=True, batch_size=self.batch_size)

            # Store embeddings and labels for probe evaluation
            with torch.no_grad():
                # Use view1 for consistency (could also use view2)
                embeddings = self.encode(view1)
                # L2 normalize embeddings before storing
                embeddings = F.normalize(embeddings, dim=1)

                # Get labels (last element of view1)
                batch_labels = view1[-1]
                binary_labels = (batch_labels > self.threshold).float()

                # Store for probe evaluation
                self.train_embeddings.append(embeddings.detach())
                self.train_labels.append(binary_labels.detach())
        else:
            forward_output = self.forward(batch[self.domain])
            loss, labels = self.compute_loss(batch[self.domain][-1], forward_output)
            self.log("train/loss", loss, prog_bar=True, batch_size=self.batch_size)

        return loss

    def validation_step(
        self, batch: Dict[str, Tuple[torch.Tensor, ...]], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single validation step.

        Parameters
        ----------
        batch : Dict[str, Tuple[torch.Tensor, ...]]
            Dictionary containing domain-specific batches. Keys are domain names ('src', 'tgt'),
            values are tuples of (mol, protein, pocket, labels).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Validation loss for the current batch (scalar).
        """
        if self.self_ssl:
            # Expect two augmented views for validation too
            view1, view2 = self.ssl_augment(batch[self.domain])
            z1 = self.projection_head(self.encode(view1))
            z2 = self.projection_head(self.encode(view2))
            loss = self.info_nce_loss(z1, z2)

            # Store for debugging purposes (not used for early stopping)
            self._last_ssl_loss = loss.detach()
            self.log(
                "val/ssl_loss_debug", loss, prog_bar=False, batch_size=self.batch_size
            )

            # Store embeddings and labels for probe evaluation
            with torch.no_grad():
                # Use view1 for consistency
                embeddings = self.encode(view1)
                # L2 normalize embeddings before storing
                embeddings = F.normalize(embeddings, dim=1)

                # Get labels (last element of view1)
                batch_labels = view1[-1]
                binary_labels = (batch_labels > self.threshold).float()

                # Store for probe evaluation
                self.val_embeddings.append(embeddings.detach())
                self.val_labels.append(binary_labels.detach())
        else:
            forward_output = self.forward(batch[self.domain])
            loss, labels = self.compute_loss(batch[self.domain][-1], forward_output)

            self.log("val/loss", loss, prog_bar=True, batch_size=self.batch_size)

            labels_int = labels.int()
            probs = torch.sigmoid(forward_output.detach())

            # existing metrics (kept as-is)
            self.accuracy.update(forward_output, labels_int)
            self.precision.update(forward_output, labels_int)
            self.recall.update(forward_output, labels_int)
            self.f1.update(forward_output, labels_int)
            self.auroc.update(forward_output, labels_int)
            self.auprc.update(forward_output, labels_int)
            self.mcc.update(forward_output, labels_int)

            # new confusion matrix updates
            self.confmat.update(probs, labels_int)

        return loss

    def get_combination_save_dir(
        self, base_dir: str, fold: int, mode: str = "supervised"
    ) -> str:
        """
        Get the save directory path for this model combination.

        Parameters
        ----------
        base_dir : str
            Base directory for saving model checkpoints.
        fold : int
            Cross-validation fold number.
        mode : str, optional
            Training mode ('supervised', 'self_ssl', etc.), by default "supervised".

        Returns
        -------
        str
            Full path to the save directory for this model configuration.
        """
        # Always create a new fold-specific path from the base directory
        return os.path.join(base_dir, f"fold_{fold}", f"{mode}_{self.combination_name}")

    def on_validation_epoch_end(self) -> None:
        """
        Compute and log metrics at the end of each validation epoch.

        For self-supervised learning mode, evaluates embeddings using k-NN probe.
        For supervised mode, logs classification metrics (accuracy, F1, AUROC, MCC, confusion matrix).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.self_ssl:
            # For self-supervised learning, evaluate using k-NN probe with collected embeddings
            try:
                # Check if we have enough data for probe evaluation
                if (
                    len(self.train_embeddings) == 0
                    or len(self.val_embeddings) == 0
                    or len(self.train_labels) == 0
                    or len(self.val_labels) == 0
                ):
                    logger.warning("Insufficient data for k-NN probe evaluation")
                    self.log("val/knn_auroc", 0.0, prog_bar=True)
                    self.log("val/knn_auprc", 0.0, prog_bar=True)
                    return

                # Concatenate all collected embeddings and labels
                train_emb = torch.cat(self.train_embeddings, dim=0).cpu().numpy()
                train_lbl = torch.cat(self.train_labels, dim=0).cpu().numpy()
                val_emb = torch.cat(self.val_embeddings, dim=0).cpu().numpy()
                val_lbl = torch.cat(self.val_labels, dim=0).cpu().numpy()

                # Ensure we have both classes for meaningful evaluation
                if len(np.unique(train_lbl)) < 2 or len(np.unique(val_lbl)) < 2:
                    logger.warning(
                        "Insufficient class diversity for k-NN probe evaluation"
                    )
                    self.log("val/knn_auroc", 0.0, prog_bar=True)
                    self.log("val/knn_auprc", 0.0, prog_bar=True)
                    return

                # Fit k-NN classifier on training embeddings and evaluate on validation embeddings
                try:
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.metrics import roc_auc_score, average_precision_score

                    # Determine if this is binary or multiclass classification
                    n_classes = len(np.unique(np.concatenate([train_lbl, val_lbl])))
                    is_binary = n_classes == 2

                    # Fit k-NN classifier with cosine distance
                    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
                    knn.fit(train_emb, train_lbl)

                    # Get prediction probabilities
                    y_pred_proba = knn.predict_proba(val_emb)

                    # Handle case where only one class is predicted
                    if y_pred_proba.shape[1] == 1:
                        logger.warning("k-NN classifier only predicted one class")
                        self.log("val/knn_auroc", 0.0, prog_bar=True)
                        self.log("val/knn_auprc", 0.0, prog_bar=True)
                        return

                    # Calculate metrics based on classification type
                    if is_binary:
                        # Binary classification: use positive class probabilities
                        y_scores = y_pred_proba[:, 1]
                        auroc = roc_auc_score(val_lbl, y_scores)
                        auprc = average_precision_score(val_lbl, y_scores)
                    else:
                        # Multiclass classification: use macro averaging
                        auroc = roc_auc_score(
                            val_lbl, y_pred_proba, multi_class="ovr", average="macro"
                        )
                        auprc = average_precision_score(
                            val_lbl, y_pred_proba, average="macro"
                        )

                    # Log k-NN probe metrics (these will be used for early stopping)
                    self.log("val/knn_auroc", auroc, prog_bar=True)
                    self.log("val/knn_auprc", auprc, prog_bar=True)

                    # Also log the SSL loss for debugging (but not used for early stopping)
                    if hasattr(self, "_last_ssl_loss"):
                        self.log(
                            "val/ssl_loss_debug", self._last_ssl_loss, prog_bar=False
                        )

                except Exception as e:
                    logger.warning(f"k-NN probe evaluation failed: {e}")
                    self.log("val/knn_auroc", 0.0, prog_bar=True)
                    self.log("val/knn_auprc", 0.0, prog_bar=True)

            except Exception as e:
                # If k-NN evaluation fails, log zeros and continue
                logger.warning(f"k-NN probe evaluation failed: {e}")
                self.log("val/knn_auroc", 0.0, prog_bar=True)
                self.log("val/knn_auprc", 0.0, prog_bar=True)
            finally:
                # Clear stored embeddings and labels to avoid memory leaks
                self._reset_probe_storage()
        else:
            # Standard supervised learning metrics
            metrics = {
                "val/accuracy": self.accuracy.compute(),
                # "val/precision": self.precision.compute(),
                # "val/recall": self.recall.compute(),
                "val/f1": self.f1.compute(),
                "val/auroc": self.auroc.compute(),
                # "val/auprc": self.auprc.compute(),
                "val/mcc": self.mcc.compute(),
            }
            # confusion matrix
            cm = self.confmat.compute()  # [[tn, fp],[fn, tp]]
            tn, fp, fn, tp = cm.flatten().tolist()
            cm_metrics = {"val/tn": tn, "val/fp": fp, "val/fn": fn, "val/tp": tp}

            self.log_dict(
                metrics | cm_metrics, prog_bar=True, batch_size=self.batch_size
            )

            # Reset metrics
            for metric in [self.accuracy, self.f1, self.auroc, self.mcc]:
                metric.reset()
            self.confmat.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.

        Uses AdamW optimizer with warmup followed by cosine annealing schedule.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - "optimizer": AdamW optimizer instance
            - "lr_scheduler": Dictionary with scheduler configuration (scheduler, interval, frequency)
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )

        warmup_epochs = 10
        total_epochs = self.trainer.max_epochs

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs),
            ],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class CCSAFinetune(pl.LightningModule):
    """
    Contrastive Semantic Alignment (CCSA) fine-tuning wrapper.

    Wraps a base MultiTaskPocket model and adds CCSA loss for domain adaptation.
    CCSA minimizes intra-class distance while maximizing inter-class distance
    across domains, enabling effective transfer learning.
    """

    def __init__(
        self, base_model: MultiTaskPocket, gamma: float = 0.8, margin: float = 2.0
    ) -> None:
        """
        Initialize the CCSA fine-tuning wrapper.

        Parameters
        ----------
        base_model : MultiTaskPocket
            Pre-trained base model to wrap for domain adaptation.
        gamma : float, optional
            Weight balancing parameter between source classification loss and
            contrastive alignment loss, by default 0.8.
        margin : float, optional
            Margin for contrastive separation loss, by default 2.0.

        Returns
        -------
        None
        """
        super().__init__()
        self.base_model = base_model
        self.gamma = gamma
        self.margin = margin

        # Mirror key properties for logging/optim
        self.learning_rate = self.base_model.learning_rate
        self.batch_size = self.base_model.batch_size
        self.threshold = self.base_model.threshold
        # Forward pos_weight from base model
        self.pos_weight = getattr(self.base_model, "pos_weight", None)

        self.use_mol = self.base_model.use_mol
        self.use_protein = self.base_model.use_protein
        self.use_pocket = self.base_model.use_pocket

        # Generate combination name based on active modalities
        combo_parts = []
        if self.use_mol:
            combo_parts.append("mol")
        if self.use_protein:
            combo_parts.append("protein")
        if self.use_pocket:
            combo_parts.append("pocket")
        self.combination_name = "_".join(combo_parts) if combo_parts else "empty"

        # Metrics (like MultiTaskPocket)
        self.accuracy = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.auroc = BinaryAUROC()
        self.mcc = BinaryMatthewsCorrCoef()
        self.confmat = BinaryConfusionMatrix()

    def get_combination_save_dir(
        self, base_dir: str, fold: int, mode: str = "ccsa"
    ) -> str:
        """Get save directory path for this combination"""
        # Always create a new fold-specific path from the base directory
        return os.path.join(
            base_dir, f"fold_{fold}", f"{mode}_{self.base_model.combination_name}"
        )

    def pairwise_X(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise squared Euclidean distances between two sets of embeddings.

        Parameters
        ----------
        X : torch.Tensor
            First set of embeddings, shape (n, embed_dim).
        Y : torch.Tensor
            Second set of embeddings, shape (m, embed_dim).

        Returns
        -------
        torch.Tensor
            Pairwise squared Euclidean distances, shape (n, m).
        """
        X2 = torch.sum(X.pow(2), dim=1, keepdim=True).expand(-1, Y.shape[0])
        Y2 = torch.sum(Y.pow(2), dim=1, keepdim=True).expand(-1, X.shape[0])
        XY = torch.matmul(X, Y.t())
        return X2 + Y2.t() - 2 * XY

    def pairwise_cosine_distance(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise cosine distances between two sets of normalized embeddings.

        Parameters
        ----------
        X : torch.Tensor
            First set of normalized embeddings, shape (n, embed_dim).
        Y : torch.Tensor
            Second set of normalized embeddings, shape (m, embed_dim).

        Returns
        -------
        torch.Tensor
            Pairwise cosine distances in [0, 2], shape (n, m).
        """
        # assumes X and Y are normalized embeddings (FeatureExtractor does F.normalize)
        sim = torch.matmul(X, Y.t())  # in [-1,1]
        return 1.0 - sim  # distance in [0,2]

    def pairwise_y_similarity(
        self, src_y: torch.Tensor, tgt_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise label similarity matrix between source and target labels.

        Parameters
        ----------
        src_y : torch.Tensor
            Source domain labels, shape (n,).
        tgt_y : torch.Tensor
            Target domain labels, shape (m,).

        Returns
        -------
        torch.Tensor
            Binary similarity matrix (1.0 if labels match, 0.0 otherwise), shape (n, m).
        """
        src_y = src_y.flatten()
        tgt_y = tgt_y.flatten()
        return (src_y.unsqueeze(1) == tgt_y.unsqueeze(0)).float()

    def semantic_alignment_loss(
        self, sim: torch.Tensor, dist: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute semantic alignment loss to minimize intra-class distance.

        Parameters
        ----------
        sim : torch.Tensor
            Pairwise label similarity matrix, shape (n, m).
        dist : torch.Tensor
            Pairwise distance matrix, shape (n, m).

        Returns
        -------
        torch.Tensor
            Per-sample semantic alignment loss, shape (n,).
        """
        aligned = sim * dist
        num_similar = torch.sum(sim, dim=1)
        return torch.sum(aligned, dim=1) / (num_similar + 1e-8)

    def separation_loss(self, sim: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """
        Compute separation loss to maximize inter-class distance.

        Parameters
        ----------
        sim : torch.Tensor
            Pairwise label similarity matrix, shape (n, m).
        dist : torch.Tensor
            Pairwise distance matrix, shape (n, m).

        Returns
        -------
        torch.Tensor
            Per-sample separation loss with margin, shape (n,).
        """
        sep = (1 - sim) * torch.clamp(self.margin - dist, min=0.0)
        num_dissim = torch.sum(1 - sim, dim=1)
        return torch.sum(sep, dim=1) / (num_dissim + 1e-8)

    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward pass through the base model.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, ...]
            Input batch tuple containing (mol, protein, pocket, labels).

        Returns
        -------
        torch.Tensor
            Classification logits from the base model, shape (batch_size,).
        """
        # Reuse base for logits
        return self.base_model.forward(batch)

    def get_combination_save_dir(
        self, base_dir: str, fold: int, mode: str = "supervised"
    ) -> str:
        """
        Get the save directory path for this model combination.

        Parameters
        ----------
        base_dir : str
            Base directory for saving model checkpoints.
        fold : int
            Cross-validation fold number.
        mode : str, optional
            Training mode (typically 'ccsa'), by default "supervised".

        Returns
        -------
        str
            Full path to the save directory for this model configuration.
        """
        # Always create a new fold-specific path from the base directory
        return os.path.join(
            base_dir, f"fold_{fold}", f"{mode}_{self.base_model.combination_name}"
        )

    def compute_src_loss(
        self, src_batch: Tuple[torch.Tensor, ...], src_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute classification loss on source domain data.

        Parameters
        ----------
        src_batch : Tuple[torch.Tensor, ...]
            Source batch tuple containing (mol, protein, pocket, labels).
        src_logits : torch.Tensor
            Classification logits from the model, shape (batch_size,).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - loss : Binary cross-entropy loss (scalar).
            - labels : Binary labels based on affinity threshold, shape (batch_size,).
        """
        affinity = src_batch[-1]
        labels = (affinity > self.threshold).float()
        loss = F.binary_cross_entropy_with_logits(
            src_logits, labels, pos_weight=self.pos_weight
        )
        return loss, labels

    def training_step(
        self, batch: Dict[str, Tuple[torch.Tensor, ...]], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single CCSA training step with both source and target data.

        Parameters
        ----------
        batch : Dict[str, Tuple[torch.Tensor, ...]]
            Dictionary containing 'src' and 'tgt' domain batches. Each batch is a tuple
            of (mol, protein, pocket, labels).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Combined training loss (source classification + contrastive alignment), scalar.
        """
        # batch is dict: {'source': src_batch, 'target': tgt_batch}
        src_batch = batch["src"]
        tgt_batch = batch["tgt"]

        # Classification on source
        src_emb = self.base_model.encode(src_batch)
        src_logits = self.base_model.classification_head(src_emb)
        src_loss, src_labels = self.compute_src_loss(src_batch, src_logits)

        # Embeddings and labels (binary via threshold) for contrastive term
        tgt_emb = self.base_model.encode(tgt_batch)
        tgt_affinity = tgt_batch[-1]
        tgt_labels = (tgt_affinity > self.threshold).float()

        # Contrastive semantic alignment
        dist = self.pairwise_X(src_emb, tgt_emb)
        sim = self.pairwise_y_similarity(src_labels.long(), tgt_labels.long())
        sa = self.semantic_alignment_loss(sim, dist)
        sep = self.separation_loss(sim, dist)
        contrastive = (sa + sep).mean()

        total = self.gamma * src_loss + (1 - self.gamma) * contrastive

        self.log("train/loss", total, prog_bar=True, batch_size=self.batch_size)
        self.log("train/src_loss", src_loss, prog_bar=True, batch_size=self.batch_size)
        self.log("train/SA_loss", sa.mean(), prog_bar=True, batch_size=self.batch_size)
        self.log("train/S_loss", sep.mean(), prog_bar=True, batch_size=self.batch_size)
        return total

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single validation step on target domain data.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, ...]
            Target domain batch tuple containing (mol, protein, pocket, labels).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Validation loss (binary cross-entropy), scalar.
        """
        # Validation on target only (standard classification metrics)
        logits = self.base_model.forward(batch)
        if isinstance(logits, tuple):
            logits = logits  # if base.forward was altered; but ours returns logits
        affinity = batch[-1]
        labels = (affinity > self.threshold).float()

        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight
        )
        labels_int = labels.int()

        probs = torch.sigmoid(logits.detach())
        labels_int = labels.int()  # or labels.long()
        self.accuracy.update(probs, labels_int)
        self.f1.update(probs, labels_int)
        self.auroc.update(probs, labels_int)
        self.mcc.update(probs, labels_int)

        # new confusion matrix update
        self.confmat.update(probs, labels_int)

        self.log("val/loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Compute and log validation metrics at the end of each epoch.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        metrics = {
            "val/accuracy": self.accuracy.compute(),
            "val/f1": self.f1.compute(),
            "val/auroc": self.auroc.compute(),
            "val/mcc": self.mcc.compute(),
        }
        cm = self.confmat.compute()  # [[tn, fp],[fn, tp]]
        tn, fp, fn, tp = cm.flatten().tolist()
        cm_metrics = {"val/tn": tn, "val/fp": fp, "val/fn": fn, "val/tp": tp}
        self.log_dict(metrics | cm_metrics, prog_bar=True, batch_size=self.batch_size)
        for m in [self.accuracy, self.f1, self.auroc, self.mcc]:
            m.reset()
        self.confmat.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler for CCSA training.

        Uses AdamW optimizer with ReduceLROnPlateau scheduler that reduces learning
        rate when validation loss plateaus.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - "optimizer": AdamW optimizer instance
            - "lr_scheduler": ReduceLROnPlateau scheduler instance
            - "monitor": Metric to monitor for learning rate reduction
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }
