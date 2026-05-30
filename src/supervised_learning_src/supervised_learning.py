import os
import re
import math
import pickle
import torch
import sys
import argparse
from pathlib import Path
from typing import Dict, Callable, Tuple, List, Optional
import warnings
from sklearn.exceptions import ConvergenceWarning

import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    matthews_corrcoef,
    accuracy_score,
    f1_score,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
from matplotlib.transforms import ScaledTranslation

# Import transfer learning modules with safe path handling
try:
    from load_data import load_data_hdf5
    from split_data import split_data_by_group
    from dataset import BindingAffinitiesDataset
except ModuleNotFoundError:
    # A fallback path for when the script is run from a different location
    # _root_dir = Path(os.getcwd())
    _transfer_learning_src_path = Path("./../../src/transfer_learning_src/data")
    if _transfer_learning_src_path.exists():
        sys.path.append(str(_transfer_learning_src_path))
        from load_data import load_data_hdf5
        from split_data import split_data_by_group
        from dataset import BindingAffinitiesDataset
    else:
        print("Warning: Could not find the data utility modules.")


import logging

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==== MODELS AND PARAMETERS ==== #
# import all the models
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier

# Define models
random_seed = 1996
models = {
    "regression": {
        "RandomForest": RandomForestRegressor(random_state=random_seed),
        "Ridge": Ridge(),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_seed),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "MLP": MLPRegressor(random_state=random_seed, max_iter=1500),
        "XGBoost": XGBRegressor(random_state=random_seed),
    },
    "classification": {
        "RandomForest": RandomForestClassifier(random_state=random_seed),
        "LogisticRegression": LogisticRegression(
            random_state=random_seed, max_iter=1000
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_seed),
        # "SVC": SVC(probability=True, random_state=random_seed),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(random_state=random_seed, max_iter=1500),
        "XGBoost": XGBClassifier(random_state=random_seed),
    },
}

# Define parameter grids for hyperparameter tuning
param_grids = {
    "regression": {
        "RandomForest": {
            "regressor__n_estimators": [50, 100, 200],
            "regressor__max_depth": [None, 10, 20],
            "regressor__min_samples_split": [2, 5, 10],
        },
        "Ridge": {
            "regressor__alpha": [0.1, 1.0, 10.0, 100.0],
            "regressor__solver": ["auto", "svd", "cholesky"],
        },
        "GradientBoosting": {
            "regressor__n_estimators": [50, 100, 200],
            "regressor__learning_rate": [0.01, 0.1, 0.2],
            "regressor__max_depth": [3, 5, 7],
        },
        "SVR": {
            "regressor__C": [0.1, 1, 10, 100],
            "regressor__kernel": ["linear", "rbf"],
            "regressor__gamma": ["scale", "auto", 0.1, 0.01],
        },
        "KNN": {
            "regressor__n_neighbors": [3, 5, 7, 9],
            "regressor__weights": ["uniform", "distance"],
            "regressor__p": [1, 2],  # Manhattan or Euclidean distance
        },
        "MLP": {
            "regressor__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "regressor__activation": ["relu", "tanh"],
            "regressor__alpha": [0.0001, 0.001, 0.01],
            "regressor__learning_rate": ["constant", "adaptive"],
        },
        "XGBoost": {
            "regressor__n_estimators": [50, 100, 200],
            "regressor__learning_rate": [0.01, 0.1, 0.2],
            "regressor__max_depth": [3, 5, 7],
            "regressor__subsample": [0.8, 1.0],
        },
    },
    "classification": {
        "RandomForest": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__class_weight": ["balanced", "balanced_subsample", None],
        },
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10.0, 100.0],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear", "saga"],
            "classifier__class_weight": ["balanced", None],
        },
        "GradientBoosting": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__max_depth": [3, 5, 7],
            "classifier__subsample": [0.8, 1.0],
        },
        "SVC": {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto", 0.1, 0.01],
            "classifier__class_weight": ["balanced", None],
        },
        "KNN": {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],  # Manhattan or Euclidean distance
        },
        "MLP": {
            "classifier__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "classifier__activation": ["relu", "tanh"],
            "classifier__alpha": [0.0001, 0.001, 0.01],
            "classifier__learning_rate": ["constant", "adaptive"],
        },
        "XGBoost": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__max_depth": [3, 5, 7],
            "classifier__subsample": [0.8, 1.0],
            "classifier__scale_pos_weight": [1, 3, 5],  # For imbalanced datasets
        },
    },
}


def safe_scorer(scorer_func: Callable, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Wrapper for a scorer function that handles cases where y_true has only one class.

    Parameters
    ----------
    scorer_func : Callable
        The scoring function to wrap (e.g., matthews_corrcoef).
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The score, or 0.0 if only one class is present in y_true.
    """
    if len(np.unique(y_true)) < 2:
        logger.warning(
            "Skipping metric calculation for a fold because it contains only one class."
        )
        return 0.0  # MCC is 0 for random predictions, a safe neutral value
    return scorer_func(y_true, y_pred)


def load_data(params: Dict) -> Tuple[Dict, Dict]:
    """
    Load and split the caged dataset for training and testing.

    Parameters
    ----------
    params : Dict
        Dictionary containing configuration parameters including 'inhibitor_descriptor',
        'affinity_type', 'fold', and 'n_folds'.

    Returns
    -------
    Tuple[Dict, Dict]
        Tuple containing (train_dataset, test_dataset) dictionaries with features and targets.
    """
    tgt_dataset = load_data_hdf5(
        dataset_name="caged",
        inhibitor_type=params["inhibitor_descriptor"],
        target_type=params["affinity_type"],
    )

    # split the target data into train and test
    tgt_train_dataset, tgt_test_dataset = split_data_by_group(
        **tgt_dataset,
        dataset="caged",
        affinity_type=params["affinity_type"],
        fold=params["fold"],
        n_folds=params["n_folds"],
    )

    return tgt_train_dataset, tgt_test_dataset


def dict_to_array(
    feature_dict: Dict, use_cls_token: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert dictionary of embeddings to numpy array, maintaining order.

    Parameters
    ----------
    feature_dict : Dict
        Dictionary of embeddings with compound names as keys.
    use_cls_token : bool
        If True, extract first token (CLS) instead of averaging (default: False).

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Tuple containing (feature_array, compound_names).
    """
    compound_names = sorted(feature_dict.keys())  # Sort for consistency
    arrays = []
    for compound_name in compound_names:
        if use_cls_token:
            # Extract first token (CLS token) for biological descriptors
            arrays.append(feature_dict[compound_name][0, 0, :])  # First token only
        else:
            # Average over atoms/residues for molecular descriptors
            arrays.append(feature_dict[compound_name].mean(axis=0))
    return np.vstack(arrays), compound_names


def prepare_features_and_targets(
    dataset: Dict, biological_type: str = "inhibitor_pocket"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and combine features and get target values for a given dataset.

    Parameters
    ----------
    dataset : Dict
        Dictionary containing 'inhibitor_feats', 'protein_feats', 'pocket_feats', and 'binding_affs'.
    biological_type : str
        Type of features to use: 'inhibitor_pocket', 'inhibitor_protein', 'inhibitor',
        'protein', 'pocket', or 'inhibitor_protein_pocket' (default: 'inhibitor_pocket').

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (feature_matrix, target_values).
    """
    inhibitor_array, _ = dict_to_array(dataset["inhibitor_feats"], use_cls_token=False)
    protein_array, _ = dict_to_array(dataset["protein_feats"], use_cls_token=True)
    pocket_array, _ = dict_to_array(dataset["pocket_feats"], use_cls_token=False)

    if biological_type == "inhibitor_pocket":
        X = np.concatenate([inhibitor_array, pocket_array], axis=1)
    elif biological_type == "inhibitor_protein":
        X = np.concatenate([inhibitor_array, protein_array], axis=1)
    elif biological_type == "inhibitor":
        X = inhibitor_array
    elif biological_type == "protein":
        X = protein_array
    elif biological_type == "pocket":
        X = pocket_array
    elif biological_type == "inhibitor_protein_pocket":
        X = np.concatenate([inhibitor_array, protein_array, pocket_array], axis=1)
    else:
        raise ValueError(f"Invalid biological type: {biological_type}")

    y, _ = extract_target_values(dataset["binding_affs"])
    return X, y


def extract_target_values(
    binding_affinities_dict: Dict[str, float],
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract target values from dictionary while maintaining compound order.

    Parameters
    ----------
    binding_affinities_dict : Dict[str, float]
        Dictionary mapping compound names to binding affinity values.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Tuple containing (target_values_array, compound_names).
    """
    compound_names = sorted(binding_affinities_dict.keys())
    target_values = []
    for compound_name in compound_names:
        target_values.append(binding_affinities_dict[compound_name])
    return np.array(target_values), compound_names


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for supervised learning experiments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing configuration for the experiment.
    """
    parser = argparse.ArgumentParser(
        description="Run supervised learning experiments with configurable options"
    )

    # Task selection
    parser.add_argument(
        "--regression", action="store_true", help="Run regression models"
    )
    parser.add_argument(
        "--classification", action="store_true", help="Run classification models"
    )

    # Data parameters
    parser.add_argument(
        "--inhibitor-descriptor",
        type=str,
        default="MACE",
        choices=["MACE", "ACSF", "SOAP"],
        help="Inhibitor descriptor type (default: MACE)",
    )
    parser.add_argument(
        "--biological-type",
        type=str,
        default="inhibitor_pocket",
        choices=[
            "inhibitor",
            "protein",
            "pocket",
            "inhibitor_protein",
            "inhibitor_pocket",
            "inhibitor_protein_pocket",
        ],
        help="Biological descriptor type (default: inhibitor_pocket)",
    )
    parser.add_argument(
        "--affinity-type",
        type=str,
        default="pic50",
        choices=["pic50", "pk"],
        help="Affinity type (default: pic50)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help="Number of folds for cross-validation (default: 4)",
    )

    # Feature processing parameters
    parser.add_argument(
        "--use-pca", action="store_true", help="Apply PCA to features (default: False)"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=100,
        help="Number of PCA components (default: 200)",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="standard",
        choices=["standard", "minmax", "none"],
        help="Scaler type to use (default: standard)",
    )

    # Model parameters
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Apply SMOTE for imbalanced classification data (default: False)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of random hyperparameter trials per model (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=7.0,
        help="Classification threshold for binary classification (default: 7.0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1996,
        help="Random seed for reproducibility (default: 1996)",
    )

    # Output parameters
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix to add to output filename (default: empty)",
    )

    parser.add_argument(
        "--evaluate-from-file",
        type=str,
        default=None,
        help="Path to a .pkl results file. If provided, skips hyperparameter search and evaluates models using the best parameters from this file.",
    )
    parser.add_argument(
        "--evaluation-fold",
        type=int,
        default=1,
        help="Which fold to use as the test set for the final evaluation (1-based index). Default is 1.",
    )

    args = parser.parse_args()

    # If neither regression nor classification is specified, run both
    if not args.regression and not args.classification:
        args.regression = True
        args.classification = True
        logger.info("No task specified, running both regression and classification")

    return args


def build_pipeline(
    base_model: BaseEstimator,
    task_type: str,
    scaler: Optional[TransformerMixin],
    use_pca: bool,
    pca_components: int,
    use_smote: bool,
    random_seed: int,
) -> imbpipeline:
    """
    Build an sklearn pipeline based on the provided configuration.

    Parameters
    ----------
    base_model : BaseEstimator
        The base estimator (classifier or regressor) to use.
    task_type : str
        Type of task: 'classification' or 'regression'.
    scaler : Optional[TransformerMixin]
        Scaler to use (StandardScaler, MinMaxScaler, or None).
    use_pca : bool
        Whether to include PCA in the pipeline.
    pca_components : int
        Number of PCA components if use_pca is True.
    use_smote : bool
        Whether to include SMOTE (only for classification).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    imbpipeline
        Configured sklearn/imblearn pipeline.
    """
    pipeline_steps = []

    # Classification-specific step (SMOTE)
    if task_type == "classification" and use_smote:
        pipeline_steps.append(("smote", SMOTE(random_state=random_seed)))

    # Common steps
    if scaler is not None:
        pipeline_steps.append(("scaler", scaler))

    if use_pca:
        pipeline_steps.append(("pca", PCA(random_state=random_seed)))

    # Final estimator step
    estimator_name = "classifier" if task_type == "classification" else "regressor"
    pipeline_steps.append((estimator_name, base_model))

    return imbpipeline(pipeline_steps)


def run_hyperparameter_search(args: argparse.Namespace) -> Dict:
    """
    Run the full nested cross-validation to find the best hyperparameters.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing experiment configuration.

    Returns
    -------
    Dict
        Dictionary containing results for regression and/or classification tasks.
    """
    logger.info("--- Starting Hyperparameter Search with Nested Cross-Validation ---")

    # Map the string 'none' (case-insensitive) to Python None
    if isinstance(args.biological_type, str) and args.biological_type.lower() == "none":
        args.biological_type = None

    # Map scaler argument to scaler instance (only when not "none")
    scaler_mapping = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
    }
    scaler = scaler_mapping.get(args.scaler) if args.scaler != "none" else None

    # Log configuration
    logger.info("Configuration:")
    logger.info(
        f"  Tasks: {'Regression' if args.regression else ''} {'Classification' if args.classification else ''}"
    )
    logger.info(f"  Inhibitor descriptor: {args.inhibitor_descriptor}")
    logger.info(f"  Biological type: {args.biological_type}")
    logger.info(f"  Affinity type: {args.affinity_type}")
    logger.info(f"  Number of folds: {args.n_folds}")
    logger.info(f"  Use PCA: {args.use_pca}")
    if args.use_pca:
        logger.info(f"  PCA components: {args.pca_components}")
    logger.info(f"  Scaler: {args.scaler}")
    if args.classification:
        logger.info(f"  Use SMOTE: {args.use_smote}")
    logger.info(f"  Trials per model: {args.n_trials}")
    logger.info(f"  Threshold: {args.threshold}")
    logger.info(f"  Random seed: {args.random_seed}")

    params = {
        "n_folds": args.n_folds,
        "inhibitor_descriptor": args.inhibitor_descriptor,
        "biological_type": args.biological_type,
        "affinity_type": args.affinity_type,
    }

    robust_mcc_scorer = make_scorer(
        lambda y_true, y_pred: safe_scorer(matthews_corrcoef, y_true, y_pred)
    )

    # Load data for all folds first
    logger.info("Loading data for all folds...")
    all_folds_data = []
    for fold in range(1, params["n_folds"] + 1):
        params["fold"] = fold
        _, test_dataset = load_data(
            params
        )  # We only need the test sets from each split
        all_folds_data.append(test_dataset)
    logger.info("Data loaded.")

    # Initialize results storage
    all_results = {"regression": {}, "classification": {}}

    # Outer cross-validation loop
    for outer_fold_idx in range(args.n_folds):
        outer_fold_num = outer_fold_idx + 1
        logger.info(f"{'='*80}")
        logger.info(f"--- Outer Fold: {outer_fold_num}/{args.n_folds} ---")
        logger.info(f"{'='*80}")

        # Split data into outer train and test sets
        outer_test_data = all_folds_data[outer_fold_idx]
        outer_train_data_list = (
            all_folds_data[:outer_fold_idx] + all_folds_data[outer_fold_idx + 1 :]
        )

        # Prepare data for inner CV loop
        inner_X_parts, inner_y_parts, test_fold_assignments = [], [], []
        for i, fold_dataset in enumerate(outer_train_data_list):
            X_fold, y_fold = prepare_features_and_targets(
                fold_dataset, args.biological_type
            )
            inner_X_parts.append(X_fold)
            inner_y_parts.append(y_fold)
            test_fold_assignments.extend([i] * len(y_fold))

        X_inner = np.concatenate(inner_X_parts)
        y_inner = np.concatenate(inner_y_parts)
        inner_cv_split = PredefinedSplit(test_fold_assignments)

        # Dynamically adjust PCA components if necessary for inner CV
        pca_components_to_use = args.pca_components
        if args.use_pca:
            n_features = X_inner.shape[1]
            min_train_samples_inner_cv = min(
                [
                    len(train_idx)
                    for train_idx, _ in inner_cv_split.split(X_inner, y_inner)
                ]
            )
            max_possible_components = min(n_features, min_train_samples_inner_cv)

            if pca_components_to_use > max_possible_components:
                logger.warning(
                    f"Outer Fold {outer_fold_num}: Requested PCA components ({pca_components_to_use}) is larger than "
                    f"the max possible for the smallest inner fold ({max_possible_components}). Adjusting PCA components."
                )
                pca_components_to_use = max_possible_components

        # Prepare outer test data
        X_test, y_test = prepare_features_and_targets(
            outer_test_data, args.biological_type
        )
        logger.info(
            f"Outer Fold {outer_fold_num}: Train samples: {len(y_inner)}, Test samples: {len(y_test)}"
        )
        logger.info(
            f"Feature matrix shapes - Train: {X_inner.shape}, Test: {X_test.shape}"
        )

        # Classification tasks
        if args.classification:
            y_inner_class = (y_inner >= args.threshold).astype(int)
            y_test_class = (y_test >= args.threshold).astype(int)

            if len(np.unique(y_inner_class)) < 2:
                logger.error(
                    f"Outer Fold {outer_fold_num}: Inner training data has only one class. "
                    f"Cannot perform classification. Skipping classification for this outer fold."
                )
                continue

            for model_name, base_model in models["classification"].items():
                logger.info(f"Running classification search for {model_name}...")
                pipeline = build_pipeline(
                    base_model,
                    "classification",
                    scaler,
                    args.use_pca,
                    args.pca_components,
                    args.use_smote,
                    args.random_seed,
                )

                param_dist = param_grids["classification"][model_name].copy()
                if args.use_pca:
                    param_dist["pca__n_components"] = [pca_components_to_use]

                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_dist,
                    n_iter=args.n_trials,
                    cv=inner_cv_split,
                    scoring=robust_mcc_scorer,
                    random_state=args.random_seed,
                    n_jobs=-1,
                )
                search.fit(X_inner, y_inner_class)
                best_model = search.best_estimator_

                logger.info(f"  Inner CV Results for {model_name}:")
                logger.info(f"    Best Avg. MCC score: {search.best_score_:.4f}")
                logger.info(f"    Best parameters: {search.best_params_}")

                y_pred = best_model.predict(X_test)

                ## NEW/MODIFIED ## - Calculate accuracy and F1 score
                mcc = matthews_corrcoef(y_test_class, y_pred)
                accuracy = accuracy_score(y_test_class, y_pred)
                f1 = f1_score(y_test_class, y_pred, zero_division=0)

                logger.info(f"  Performance on Outer Fold {outer_fold_num} (Test Set):")
                logger.info(
                    f"    MCC: {mcc:.4f}, Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}"
                )

                ## NEW/MODIFIED ## - Store new metrics
                if model_name not in all_results["classification"]:
                    all_results["classification"][model_name] = []
                all_results["classification"][model_name].append(
                    {
                        "fold": outer_fold_num,
                        "model": model_name,
                        "mcc": mcc,
                        "accuracy": accuracy,
                        "f1_score": f1,
                        "threshold": args.threshold,
                        "best_params": search.best_params_,
                    }
                )

        # Regression tasks
        if args.regression:
            for model_name, base_model in models["regression"].items():
                logger.info(f"Running regression search for {model_name}...")
                pipeline = build_pipeline(
                    base_model,
                    "regression",
                    scaler,
                    args.use_pca,
                    args.pca_components,
                    False,  # SMOTE is not used for regression
                    args.random_seed,
                )

                param_dist = param_grids["regression"][model_name].copy()
                if args.use_pca:
                    param_dist["pca__n_components"] = [pca_components_to_use]

                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_dist,
                    n_iter=args.n_trials,
                    cv=inner_cv_split,
                    scoring="r2",
                    random_state=args.random_seed,
                    n_jobs=-1,
                )
                search.fit(X_inner, y_inner)
                best_model = search.best_estimator_

                logger.info(f"  Inner CV Results for {model_name}:")
                logger.info(f"    Best Avg. R2 score: {search.best_score_:.4f}")
                logger.info(f"    Best parameters: {search.best_params_}")

                y_pred = best_model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = math.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                pearson_r, _ = pearsonr(y_test, y_pred)
                spearman_r, _ = spearmanr(y_test, y_pred)

                logger.info(f"  Performance on Outer Fold {outer_fold_num} (Test Set):")
                logger.info(f"    R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                logger.info(
                    f"    Pearson's r: {pearson_r:.4f}, Spearman's r: {spearman_r:.4f}"
                )

                if model_name not in all_results["regression"]:
                    all_results["regression"][model_name] = []
                all_results["regression"][model_name].append(
                    {
                        "fold": outer_fold_num,
                        "model": model_name,
                        "mae": mae,
                        "mse": mse,
                        "rmse": rmse,
                        "r2": r2,
                        "pearsonr": pearson_r,
                        "spearman_r": spearman_r,
                        "best_params": search.best_params_,
                    }
                )

    # Convert to DataFrames
    final_results = {"regression": {}, "classification": {}}
    for task_type in ["regression", "classification"]:
        if task_type in all_results and all_results[task_type]:
            for model_name, results_list in all_results[task_type].items():
                if results_list:
                    df = pd.DataFrame(results_list)
                    final_results[task_type][model_name] = df

    # Save results
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    task_suffix = (
        "_reg"
        if args.regression and not args.classification
        else "_class" if args.classification and not args.regression else ""
    )
    biological_type_str = args.biological_type
    output_dir = f"results/threshold_{args.threshold}/{args.inhibitor_descriptor}/{biological_type_str}/{args.scaler}_scaler/{params['affinity_type']}"
    os.makedirs(output_dir, exist_ok=True)
    pca_suffix = f"_with_pca_{args.pca_components}c" if args.use_pca else "_no_pca"
    smote_suffix = "_with_smote" if args.classification and args.use_smote else ""
    output_file = f"{output_dir}/baseline_results{task_suffix}{pca_suffix}{smote_suffix}{suffix}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(final_results, f)
    logger.info(f"Results saved to {output_file}")

    # Print summary statistics
    print_summary(final_results)

    return final_results


def evaluate_with_best_params(args: argparse.Namespace) -> None:
    """
    Load best parameters from a results file and run a single evaluation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments including path to results file and evaluation fold.

    Returns
    -------
    None
    """
    logger.info(
        f"--- Running Final Evaluation from Results File: {args.evaluate_from_file} ---"
    )

    # 1. Load results file
    results_path = Path(args.evaluate_from_file)
    if not results_path.is_file():
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)
    with open(results_path, "rb") as f:
        previous_results = pickle.load(f)
    logger.info("Successfully loaded previous results.")

    # 2. Infer parameters from the file path to ensure consistency
    try:
        parts = results_path.parts
        args.threshold = float(next(p for p in parts if "threshold" in p).split("_")[1])
        args.inhibitor_descriptor = parts[-5]
        biological_type_str = parts[-4]
        args.biological_type = biological_type_str
        args.scaler = parts[-3].replace("_scaler", "")
        args.affinity_type = parts[-2]

        filename = results_path.name
        args.use_pca = "_with_pca" in filename
        if args.use_pca:
            match = re.search(r"_(\d+)c", filename)
            args.pca_components = int(match.group(1)) if match else 100

        # Infer SMOTE setting
        args.use_smote = "_with_smote" in filename

    except (StopIteration, IndexError, ValueError) as e:
        logger.error(
            f"Could not auto-infer parameters from path: {e}. Please ensure path structure is correct."
        )
        sys.exit(1)

    logger.info("Inferred Configuration:")
    logger.info(
        f"  Inhibitor Descriptor: {args.inhibitor_descriptor}, Biological Type: {args.biological_type}"
    )
    logger.info(f"  Affinity Type: {args.affinity_type}, Scaler: {args.scaler}")
    logger.info(
        f"  PCA: {args.use_pca}, Components: {args.pca_components if args.use_pca else 'N/A'}"
    )
    if previous_results.get("classification"):
        logger.info(f"  Use SMOTE: {args.use_smote}")
    logger.info(f"  Classification Threshold: {args.threshold}")
    logger.info(f"  Test fold for evaluation: {args.evaluation_fold}")

    # 3. Prepare Data for a single train/test split
    params = {
        "n_folds": args.n_folds,
        "inhibitor_descriptor": args.inhibitor_descriptor,
        "biological_type": args.biological_type,
        "affinity_type": args.affinity_type,
    }
    logger.info("Loading and splitting data for final evaluation...")
    all_folds_data = [
        load_data({**params, "fold": f})[1] for f in range(1, params["n_folds"] + 1)
    ]

    test_fold_idx = args.evaluation_fold - 1
    if not (0 <= test_fold_idx < args.n_folds):
        logger.error(
            f"Invalid evaluation-fold: {args.evaluation_fold}. Must be between 1 and {args.n_folds}."
        )
        sys.exit(1)

    test_data = all_folds_data[test_fold_idx]
    train_data_list = (
        all_folds_data[:test_fold_idx] + all_folds_data[test_fold_idx + 1 :]
    )

    # Concatenate training data from multiple folds
    train_X_parts, train_y_parts = [], []
    for dataset in train_data_list:
        X_part, y_part = prepare_features_and_targets(dataset, args.biological_type)
        train_X_parts.append(X_part)
        train_y_parts.append(y_part)
    X_train, y_train = np.concatenate(train_X_parts), np.concatenate(train_y_parts)
    X_test, y_test = prepare_features_and_targets(test_data, args.biological_type)

    logger.info(
        f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples."
    )

    # 4. Evaluate each model with its best parameters
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler()}.get(args.scaler)
    evaluation_summary = {"regression": [], "classification": []}

    for task_type, models_results in previous_results.items():
        if not models_results:
            continue

        y_train_task = (
            (y_train >= args.threshold).astype(int)
            if task_type == "classification"
            else y_train
        )
        y_test_task = (
            (y_test >= args.threshold).astype(int)
            if task_type == "classification"
            else y_test
        )

        for model_name, df_results in models_results.items():
            logger.info(f"--- Evaluating {model_name} for {task_type} ---")

            # Find best params from the loaded results dataframe
            metric_col = "mcc" if task_type == "classification" else "r2"
            best_fold_idx = df_results[metric_col].idxmax()
            best_params = df_results.loc[best_fold_idx]["best_params"]
            logger.info(
                f"  Using best parameters found in fold {df_results.loc[best_fold_idx]['fold']}: {best_params}"
            )

            # Re-create pipeline and set parameters
            base_model = models[task_type][model_name]
            pipeline = build_pipeline(
                base_model,
                task_type,
                scaler,
                args.use_pca,
                args.pca_components,
                args.use_smote,
                args.random_seed,
            )
            pipeline.set_params(**best_params)

            # Fit and predict
            logger.info("  Training final model...")
            pipeline.fit(X_train, y_train_task)
            y_pred = pipeline.predict(X_test)

            # Calculate and store metrics
            result_row = {"Model": model_name}
            if task_type == "classification":
                result_row["MCC"] = matthews_corrcoef(y_test_task, y_pred)
                result_row["Accuracy"] = accuracy_score(y_test_task, y_pred)
                result_row["F1-score"] = f1_score(y_test_task, y_pred, zero_division=0)
                logger.info(
                    f"  Test Performance -> MCC: {result_row['MCC']:.4f}, Accuracy: {result_row['Accuracy']:.4f}, F1: {result_row['F1-score']:.4f}"
                )
            else:
                result_row["R²"] = r2_score(y_test_task, y_pred)
                result_row["MAE"] = mean_absolute_error(y_test_task, y_pred)
                result_row["RMSE"] = math.sqrt(mean_squared_error(y_test_task, y_pred))
                result_row["Pearson R"], _ = pearsonr(y_test_task, y_pred)
                logger.info(
                    f"  Test Performance -> R²: {result_row['R²']:.4f}, MAE: {result_row['MAE']:.4f}, RMSE: {result_row['RMSE']:.4f}"
                )
            evaluation_summary[task_type].append(result_row)

    # 5. Print final summary tables
    logger.info(f"\n{'#'*80}\n" + "FINAL EVALUATION SUMMARY".center(80) + f"\n{'#'*80}")
    for task_type, results in evaluation_summary.items():
        if results:
            logger.info(f"\n--- {task_type.title()} Final Performance ---")
            df = pd.DataFrame(results)
            logger.info(
                "\n"
                + tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False)
            )


def print_summary(final_results: Dict) -> None:
    """
    Print a formatted summary of the cross-validation results.

    Parameters
    ----------
    final_results : Dict
        Dictionary containing results for regression and/or classification tasks.

    Returns
    -------
    None
    """
    logger.info(
        f"\n{'#'*80}\n" + "CROSS-VALIDATION RESULTS SUMMARY".center(80) + f"\n{'#'*80}"
    )

    for task_type in ["regression", "classification"]:
        if final_results.get(task_type):
            logger.info(
                f"\n{'='*60}\n"
                + f"{task_type.title()} Task Summary".center(60)
                + f"\n{'='*60}"
            )
            summary_data = []

            if task_type == "regression":
                headers = ["Model", "Mean R²", "Mean MAE", "Mean Pearson"]
                for model_name, df in final_results[task_type].items():
                    summary_data.append(
                        [
                            model_name,
                            f"{df['r2'].mean():.3f} ± {df['r2'].std():.3f}",
                            f"{df['mae'].mean():.3f} ± {df['mae'].std():.3f}",
                            f"{df['pearsonr'].mean():.3f} ± {df['pearsonr'].std():.3f}",
                        ]
                    )
            else:  # classification
                headers = ["Model", "Mean MCC", "Mean Accuracy", "Mean F1-score"]
                for model_name, df in final_results[task_type].items():
                    summary_data.append(
                        [
                            model_name,
                            f"{df['mcc'].mean():.3f} ± {df['mcc'].std():.3f}",
                            f"{df['accuracy'].mean():.3f} ± {df['accuracy'].std():.3f}",
                            f"{df['f1_score'].mean():.3f} ± {df['f1_score'].std():.3f}",
                        ]
                    )
            logger.info(
                "\n" + tabulate(summary_data, headers=headers, tablefmt="fancy_grid")
            )


def main() -> None:
    """
    Main function to parse arguments and run the appropriate mode.

    Dispatches to either hyperparameter search or evaluation mode based on arguments.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    args = parse_arguments()

    if args.evaluate_from_file:
        # If a results file is provided, run the evaluation mode
        evaluate_with_best_params(args)
    else:
        # Otherwise, run the full hyperparameter search
        run_hyperparameter_search(args)


if __name__ == "__main__":
    main()
