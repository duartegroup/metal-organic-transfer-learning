import argparse
from typing import Optional

features_dim_dict = {"MACE": 256, "SOAP": 5376, "ACSF": 360, "ESM-C": 1152}


def normalize_data_scaler(data_scaler: str) -> Optional[str]:
    """
    Normalize data_scaler parameter to handle various forms of 'none'.

    Args:
        data_scaler: The data_scaler value (str, None, or other)

    Returns:
        None if data_scaler represents 'none', otherwise returns the original value
    """
    if data_scaler is None:
        return None

    if isinstance(data_scaler, str):
        # Handle case-insensitive variations of "none"
        normalized = data_scaler.strip().lower()
        if normalized in ("none", ""):
            return None

    return data_scaler


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training configuration.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing all training configuration parameters.
    """
    parser = argparse.ArgumentParser()

    # === DATA PARAMS ===#
    parser.add_argument(
        "--base_path", type=str, default="../data", help="Path to the data"
    )
    parser.add_argument(
        "--affinity_type",
        type=str,
        default="pic50",
        choices=["pic50", "pk", "pk+pic50"],
    )
    # === MODEL PARAMS ===#
    parser.add_argument(
        "--inhibitor_descriptor",
        type=str,
        default="MACE",
        choices=["MACE", "SOAP", "ACSF"],
    )
    parser.add_argument(
        "--biological_descriptor", type=str, default="ESM-C", choices=["ESM-C"]
    )
    parser.add_argument("--sweep", action="store_true", default=False)
    parser.add_argument(
        "--mode", type=str, default="metal", choices=["metal", "pdbbind", "ccsa"]
    )
    parser.add_argument("--smote", action="store_true", default=False)
    parser.add_argument(
        "--data_scaler", type=str, default=None, choices=["none", "standard", "minmax"]
    )
    parser.add_argument("--phase", type=int, default=None, choices=[None, 1, 2])
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--split_layer", type=int, default=2, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--pipeline", type=str, default="ssl,ccsa,task")
    parser.add_argument(
        "--train_on",
        type=str,
        default="src+tgt",
        choices=["src", "tgt", "src+tgt"],
    )
    parser.add_argument(
        "--val_dataset", type=str, default="tgt", choices=["tgt", "src"]
    )

    # === MODALITY SELECTION ===#
    parser.add_argument(
        "--use_mol",
        action="store_true",
        default=False,
        help="Use molecular encoder in the model",
    )
    parser.add_argument(
        "--use_protein",
        action="store_true",
        default=False,
        help="Use protein encoder in the model",
    )
    parser.add_argument(
        "--use_pocket",
        action="store_true",
        default=False,
        help="Use pocket encoder in the model",
    )

    parser.add_argument(
        "--criterion", type=str, default="bce", choices=["bce", "mse", "focal"]
    )
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_source", type=int, default=64)

    # === PARAMS PRETRAIN ===#
    parser.add_argument("--learning_rate_pretrain", type=float, default=1e-4)
    parser.add_argument("--dropout_pretrain", type=float, default=0.4)
    parser.add_argument("--ssl_temperature", type=float, default=0.07)
    parser.add_argument("--ssl_noise_scale", type=float, default=0.02)
    parser.add_argument("--ssl_node_drop_prob", type=float, default=0.2)

    # === PARAMS CCSA ===#
    parser.add_argument("--learning_rate_ccsa", type=float, default=1e-4)
    parser.add_argument("--dropout_ccsa", type=float, default=0.4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--ccsa_followup_modes", type=list, default=["task", "encoder_task"]
    )

    # === PARAMS FINETUNE ===#
    parser.add_argument("--learning_rate_finetune", type=float, default=1e-4)
    parser.add_argument("--dropout_finetune", type=float, default=0.4)

    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=0.001)

    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Local rank for distributed training"
    )

    # === THRESHOLD GROUP ===#
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=None,
        help="Threshold(s) to binarize or split classes",
    )

    # === VISUALIZATION ===#
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--visualize_method", type=str, default="tsne", choices=["tsne", "pca", "umap"]
    )

    # === OTHER ===#
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--enable_progress_bar", default=False, action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")

    parser.add_argument(
        "--source_smote",
        action="store_true",
        default=False,
        help="Apply SMOTE to source domain data",
    )
    parser.add_argument(
        "--target_smote",
        action="store_true",
        default=False,
        help="Apply SMOTE to target domain data",
    )
    parser.add_argument(
        "--cross_domain_scaling",
        action="store_true",
        default=False,
        help="Use source statistics for scaling both domains. If False, compute separate statistics for each domain.",
    )

    return parser.parse_args()


def build_params_from_args(args: argparse.Namespace) -> dict:
    """
    Convert parsed arguments into a parameters dictionary for model training.

    Performs validation and normalization of arguments, calculates derived parameters,
    and constructs modality strings.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments from parse_args().

    Returns
    -------
    dict
        Dictionary containing all training parameters.

    Raises
    ------
    ValueError
        If modality selection is invalid (no modalities enabled or conflicting settings).
    """
    # Ensure thresholds are converted to list
    if args.threshold is not None and not isinstance(args.threshold, list):
        args.threshold = [args.threshold]

    if args.threshold is not None:
        args.classes = len(args.threshold) + 1

    # Validate modality selection - at least one must be enabled
    if not any([args.use_mol, args.use_protein, args.use_pocket]):
        raise ValueError(
            "At least one modality (mol, protein, or pocket) must be enabled"
        )

    # Normalize data_scaler parameter
    args.data_scaler = normalize_data_scaler(args.data_scaler)

    # get the input dimensions
    args.inhibitor_input_dim = features_dim_dict[args.inhibitor_descriptor]
    args.biological_input_dim = features_dim_dict[args.biological_descriptor]

    # create a modality string
    if args.use_mol and args.use_protein and args.use_pocket:
        args.modality = "mol_protein_pocket"
    elif args.use_mol and args.use_protein:
        args.modality = "mol_protein"
    elif args.use_mol and args.use_pocket:
        args.modality = "mol_pocket"
    elif args.use_protein and args.use_pocket:
        args.modality = "protein_pocket"
    elif args.use_mol:
        args.modality = "mol"
    elif args.use_protein:
        args.modality = "protein"
    elif args.use_pocket:
        args.modality = "pocket"
    else:
        raise ValueError(f"Invalid modality: all of modalities are set to False")

    return {
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "batch_size_source": args.batch_size_source,
        "learning_rate_pretrain": args.learning_rate_pretrain,
        "learning_rate_ccsa": args.learning_rate_ccsa,
        "learning_rate_finetune": args.learning_rate_finetune,
        "dropout_pretrain": args.dropout_pretrain,
        "dropout_ccsa": args.dropout_ccsa,
        "dropout_finetune": args.dropout_finetune,
        "gradient_clip_val": args.gradient_clip_val,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "margin": args.margin,
        "gamma": args.gamma,
        "fold": args.fold,
        "n_folds": args.n_folds,
        "split_layer": args.split_layer,
        "base_path": args.base_path,
        "criterion": args.criterion,
        "threshold": args.threshold,
        "affinity_type": args.affinity_type,
        "inhibitor_input_dim": args.inhibitor_input_dim,
        "biological_input_dim": args.biological_input_dim,
        "inhibitor_descriptor": args.inhibitor_descriptor,
        "biological_descriptor": args.biological_descriptor,
        "mode": args.mode,
        "phase": args.phase,
        "smote": args.smote,
        "data_scaler": args.data_scaler,
        "visualize": args.visualize,
        "visualize_method": args.visualize_method,
        "classes": args.classes,
        "enable_progress_bar": args.enable_progress_bar,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "pipeline": args.pipeline,
        "train_on": args.train_on,
        "val_dataset": args.val_dataset,
        # Add modality selection parameters
        "use_mol": args.use_mol,
        "use_protein": args.use_protein,
        "use_pocket": args.use_pocket,
        "modality": args.modality,
        "ccsa_followup_modes": args.ccsa_followup_modes,
        "source_smote": args.source_smote,
        "target_smote": args.target_smote,
        "cross_domain_scaling": args.cross_domain_scaling,
    }
