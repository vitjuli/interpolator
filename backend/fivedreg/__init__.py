"""
FiveDReg: 5D Regression Interpolation System

This package provides:
- FiveDRegressor: Neural network for 5D regression
- Utility functions for dataset and model management
"""

from .model import FiveDRegressor
from .utils import (
    # Dataset functions
    create_synthetic_dataset,
    save_dataset,
    load_dataset,
    list_datasets,
    create_train_val_test_datasets,
    ground_truth_function,
    # Model functions
    save_model,
    load_model,
    list_models,
    # Path functions
    get_data_dir,
    get_models_dir,
)

__version__ = "0.1.0"

__all__ = [
    # Model
    "FiveDRegressor",
    # Dataset functions
    "create_synthetic_dataset",
    "save_dataset",
    "load_dataset",
    "list_datasets",
    "create_train_val_test_datasets",
    "ground_truth_function",
    # Model functions
    "save_model",
    "load_model",
    "list_models",
    # Path functions
    "get_data_dir",
    "get_models_dir",
]
