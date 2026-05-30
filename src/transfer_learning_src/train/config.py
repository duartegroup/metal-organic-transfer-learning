from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Centralized configuration for all training phases.

    This dataclass stores hyperparameters and settings for pretraining, CCSA
    (Contrastive Semantic Alignment), and fine-tuning phases of the multi-task
    binding affinity classification model.

    Attributes
    ----------
    data_scaler : str
        Type of data scaling to apply ('none', 'standard', 'minmax').
    apply_smote : bool
        Whether to apply SMOTE oversampling to balance classes.
    source_smote : bool
        Whether to apply SMOTE to source domain data.
    cross_domain_scaling : bool
        Whether to use consistent scaling across source and target domains.
    pretrain_batch_size : int
        Batch size for pretraining phase.
    learning_rate_pretrain : float
        Learning rate for pretraining phase.
    dropout_pretrain : float
        Dropout rate for pretraining phase.
    ssl_temperature : float
        Temperature parameter for contrastive SSL loss.
    ssl_noise_scale : float
        Noise scale for SSL augmentation.
    ssl_node_drop_prob : float
        Probability of dropping nodes during SSL.
    monitor_metric_pretrain : str
        Metric to monitor during pretraining.
    monitor_mode_pretrain : str
        Mode for monitoring ('min' or 'max').
    learning_rate_ccsa : float
        Learning rate for CCSA phase.
    dropout_ccsa : float
        Dropout rate for CCSA phase.
    gamma : float
        Weight for CCSA loss component.
    margin : float
        Margin for CCSA contrastive loss.
    monitor_metric_ccsa : str
        Metric to monitor during CCSA.
    monitor_mode_ccsa : str
        Mode for monitoring CCSA.
    learning_rate_finetune : float
        Learning rate for fine-tuning phase.
    dropout_finetune : float
        Dropout rate for fine-tuning phase.
    monitor_metric_finetune : str
        Metric to monitor during fine-tuning.
    monitor_mode_finetune : str
        Mode for monitoring fine-tuning.
    max_epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience (epochs without improvement).
    enable_progress_bar : bool
        Whether to show progress bar during training.
    """

    # data config
    data_scaler: str = "none"
    apply_smote: bool = True
    source_smote: bool = False
    cross_domain_scaling: bool = False

    # Pretraining config
    pretrain_batch_size: int = 512
    learning_rate_pretrain: float = 1e-4
    dropout_pretrain: float = 0.05
    ssl_temperature: float = 0.07
    ssl_noise_scale: float = 0.03
    ssl_node_drop_prob: float = 0.3
    monitor_metric_pretrain: str = "val/knn_auroc"
    monitor_mode_pretrain: str = "max"

    # CCSA config
    learning_rate_ccsa: float = 1e-4
    dropout_ccsa: float = 0.4
    gamma: float = 0.5
    margin: float = 1.0
    monitor_metric_ccsa: str = "val/loss"
    monitor_mode_ccsa: str = "min"

    # Finetune config
    learning_rate_finetune: float = 1e-4
    dropout_finetune: float = 0.4
    monitor_metric_finetune: str = "val/loss"
    monitor_mode_finetune: str = "min"

    # Training config
    max_epochs: int = 500
    patience: int = 35
    enable_progress_bar: bool = True
