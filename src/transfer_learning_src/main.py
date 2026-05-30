import sys
import os
import logging
from pathlib import Path

import numpy as np
import torch

# Import from the train_pipline module
from utils.config_loader import (
    parse_args, 
    build_params_from_args
)

from train.pipeline import MultitaskTrainingPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def init_seed(seed: int) -> None:
    """Initialize random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

def run_training_pipeline(params: dict) -> dict:
    """Run the training pipeline with the given parameters."""
    pipeline = MultitaskTrainingPipeline(params)
    return pipeline.run()

def run_all_folds_training_pipeline(params: dict) -> list:
    """Run training pipeline for all folds."""
    results = []
    for fold in range(1, params["n_folds"] + 1):
        params["fold"] = fold
        result = run_training_pipeline(params)
        results.append(result)
    return results

def main() -> None:
    """Main entry point for the transfer learning pipeline."""
    logging.info("=" * 55)
    logging.info("      STARTING TRANSFER LEARNING PIPELINE")
    logging.info("=" * 55)
    
    args = parse_args()
    
    # Determine number of classes
    if args.threshold is not None:
        if not isinstance(args.threshold, list):
            args.threshold = [args.threshold]
        args.classes = len(args.threshold) + 1
        logger.info(f"Using thresholds {args.threshold}, setting classes to {args.classes}")
    elif args.top is None:
        args.threshold = [7.0]
        args.classes = 2
        logger.info("Using default threshold 7.0 and classes=2")

    params = build_params_from_args(args)
    
    init_seed(params["seed"])

    logger.info("Configuration:")
    for k, v in params.items():
        logger.info(f"{k}: {v}")
    
    # Run training pipeline
    if args.all_folds:
        run_all_folds_training_pipeline(params)
    else:
        run_training_pipeline(params)

if __name__ == "__main__":
    main()
