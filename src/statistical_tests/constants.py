# constants.py
"""
Shared constants used across reporting, plotting, and analysis modules.

Centralising these here avoids the same dict/list being defined in multiple
files, which previously caused silent drift when values needed to be updated.
"""

STRATEGY_MAP = {
    "no_pretrain_encoder_task": "S1",
    "pretrain_encoder_task": "S2A",
    "pretrain_task": "S2B",
    "no_pretrain_ccsa": "S3A",
    "no_pretrain_ccsa_then_task": "S3B",
    "no_pretrain_ccsa_then_encoder_task": "S3C",
    "pretrain_ccsa": "S4A",
    "pretrain_ccsa_then_task": "S4B",
    "pretrain_ccsa_then_encoder_task": "S4C",
}

# The transfer-learning baseline strategy: plain fine-tuning, no pretraining and
# no domain adaptation. Every other strategy is measured against it, so the label
# lives here and the method key is derived from STRATEGY_MAP rather than repeated.
BASELINE_STRATEGY = "S1"

BASELINE_METHOD = next(
    method for method, label in STRATEGY_MAP.items() if label == BASELINE_STRATEGY
)


MODALITY_MAP = {
    # Transfer learning modality names
    "molecule": "Molecule",
    "molecule_protein": "Molecule+Protein",
    "molecule_pocket": "Molecule+Pocket",
    "molecule_protein_pocket": "Molecule+Protein+Pocket",
    # Supervised learning modality names (mapped to same display names)
    "inhibitor": "Molecule",
    "protein": "Protein",
    "pocket": "Pocket",
    "inhibitor_protein": "Molecule+Protein",
    "inhibitor_pocket": "Molecule+Pocket",
    "inhibitor_protein_pocket": "Molecule+Protein+Pocket",
}

SCALER_MAP = {
    "none_scaler": "None",
    "minmax_scaler": "MinMax",
    "standard_scaler": "Standard",
}

SORT_ORDER_MAP = {
    "method": list(STRATEGY_MAP.keys()),
    "modality": [
        "inhibitor",
        "inhibitor_protein",
        "inhibitor_pocket",
        "inhibitor_protein_pocket",
        "protein",
        "pocket",
    ],
    "scaler": list(SCALER_MAP.keys()),
    "inhibitor": ["ACSF", "MACE", "SOAP"],
    "pca_status": ["with_pca_50components", "no_pca"],
    "ccsa_status": ["True", "False"],
    "pretrain_status": ["True", "False"],
    "cross_domain_scaling": ["True", "False"],
    "source_smote": ["True", "False"],
    "target_smote": ["True", "False"],
}

# Noise-ceiling fallback values used when pkl files are unavailable.
# Keyed by affinity type, then by metric name.
FALLBACK_MAX_VALUES = {
    "pic50": {
        "mcc":      [0.619462, 0.73992,  0.855959, 0.72795],
        "accuracy": [0.869262, 0.876362, 0.941841, 0.872405],
        "f1":       [0.697286, 0.834561, 0.891203, 0.825929],
    },
    "pk": {
        "mcc":      [0.903711, 0.695456, 0.64705,  0.719323],
        "accuracy": [0.958435, 0.86128,  0.849796, 0.910104],
        "f1":       [0.932307, 0.789192, 0.750986, 0.765908],
    },
}
