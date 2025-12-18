"""
Utility functions for dataset and model management.

This module provides backend functions for:
- Creating and saving synthetic datasets as .pkl files
- Loading datasets from .pkl files
- Saving trained models to backend/models/
- Loading models from backend/models/
"""

import pickle
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from .model import FiveDRegressor


# Directory paths
BACKEND_DIR = Path(__file__).parent.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def ground_truth_function(X: np.ndarray) -> np.ndarray:
    """
    Ground truth function for synthetic data generation.

    Formula: y = 2.0·x₁ + (-1.5)·x₂² + 3.0·sin(x₃) + 0.5·x₄·x₅

    Args:
        X: Input array of shape (n_samples, 5)

    Returns:
        Output array of shape (n_samples,)
    """
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    return (
        2.0 * x1 +
        (-1.5) * (x2 ** 2) +
        3.0 * np.sin(x3) +
        0.5 * x4 * x5
    )


def create_synthetic_dataset(
    n_samples: int,
    seed: int = 42,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic 5D dataset using ground truth function.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        save_path: Optional path to save dataset as .pkl file
                  (relative to backend/data/ or absolute path)

    Returns:
        X, y: Feature matrix and target vector

    Example:
        >>> X, y = create_synthetic_dataset(1000, seed=42, save_path="train_data.pkl")
        >>> print(X.shape, y.shape)
        (1000, 5) (1000,)
    """
    np.random.seed(seed)

    # Generate random features from standard normal distribution
    X = np.random.randn(n_samples, 5).astype(np.float32)

    # Compute targets using ground truth function
    y = ground_truth_function(X)

    # Save if path provided
    if save_path is not None:
        save_dataset(X, y, save_path)

    return X, y


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    filename: str
) -> Path:
    """
    Save dataset to backend/data/ directory as .pkl file.

    Args:
        X: Feature matrix of shape (n_samples, 5)
        y: Target vector of shape (n_samples,)
        filename: Filename for the dataset (e.g., "my_data.pkl")
                 Will be saved to backend/data/

    Returns:
        Path to saved file

    Example:
        >>> save_dataset(X_train, y_train, "train_data.pkl")
        PosixPath('.../backend/data/train_data.pkl')
    """
    # Ensure .pkl extension
    if not filename.endswith('.pkl'):
        filename = f"{filename}.pkl"

    filepath = DATA_DIR / filename

    # Save as pickle
    data = {
        'X': X,
        'y': y,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    return filepath


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from backend/data/ directory.

    Args:
        filename: Filename of the dataset (e.g., "my_data.pkl")
                 Will be loaded from backend/data/

    Returns:
        X, y: Feature matrix and target vector

    Example:
        >>> X, y = load_dataset("train_data.pkl")
        >>> print(X.shape, y.shape)
        (1000, 5) (1000,)
    """
    # Ensure .pkl extension
    if not filename.endswith('.pkl'):
        filename = f"{filename}.pkl"

    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            f"Available datasets: {list_datasets()}"
        )

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data['X'], data['y']


def list_datasets() -> list:
    """
    List all available datasets in backend/data/ directory.

    Returns:
        List of dataset filenames

    Example:
        >>> datasets = list_datasets()
        >>> print(datasets)
        ['train_data.pkl', 'test_data.pkl']
    """
    return [f.name for f in DATA_DIR.glob("*.pkl")]


def save_model(
    model: FiveDRegressor,
    model_name: str
) -> Path:
    """
    Save trained model to backend/models/ directory.

    Args:
        model: Trained FiveDRegressor instance
        model_name: Name for the model (e.g., "my_model.pt")
                   Will be saved to backend/models/

    Returns:
        Path to saved model

    Example:
        >>> save_model(model, "trained_interpolator.pt")
        PosixPath('.../backend/models/trained_interpolator.pt')
    """
    # Ensure .pt extension
    if not model_name.endswith('.pt'):
        model_name = f"{model_name}.pt"

    filepath = MODELS_DIR / model_name

    # Use the model's save method
    model.save_model(str(filepath))

    return filepath


def load_model(model_name: str) -> FiveDRegressor:
    """
    Load trained model from backend/models/ directory.

    Args:
        model_name: Name of the model (e.g., "my_model.pt")
                   Will be loaded from backend/models/

    Returns:
        Loaded FiveDRegressor instance

    Example:
        >>> model = load_model("trained_interpolator.pt")
        >>> predictions = model.predict(X_test)
    """
    # Ensure .pt extension
    if not model_name.endswith('.pt'):
        model_name = f"{model_name}.pt"

    filepath = MODELS_DIR / model_name

    if not filepath.exists():
        raise FileNotFoundError(
            f"Model not found: {filepath}\n"
            f"Available models: {list_models()}"
        )

    # Use the model's class method to load
    return FiveDRegressor.load_model(str(filepath))


def list_models() -> list:
    """
    List all available models in backend/models/ directory.

    Returns:
        List of model filenames

    Example:
        >>> models = list_models()
        >>> print(models)
        ['trained_interpolator.pt', 'best_model.pt']
    """
    return [f.name for f in MODELS_DIR.glob("*.pt")]


def get_data_dir() -> Path:
    """Get the backend/data/ directory path."""
    return DATA_DIR


def get_models_dir() -> Path:
    """Get the backend/models/ directory path."""
    return MODELS_DIR


# Convenience function for creating train/val/test splits
def create_train_val_test_datasets(
    n_train: int = 3000,
    n_val: int = 500,
    n_test: int = 1000,
    seed: int = 42,
    save: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create and optionally save train/val/test datasets.

    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        seed: Random seed for reproducibility
        save: If True, save datasets to backend/data/

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test

    Example:
        >>> X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_datasets()
        Dataset saved: backend/data/train_data.pkl (3000 samples)
        Dataset saved: backend/data/val_data.pkl (500 samples)
        Dataset saved: backend/data/test_data.pkl (1000 samples)
    """
    # Generate datasets with different seeds
    X_train, y_train = create_synthetic_dataset(n_train, seed=seed)
    X_val, y_val = create_synthetic_dataset(n_val, seed=seed + 1)
    X_test, y_test = create_synthetic_dataset(n_test, seed=seed + 2)

    if save:
        train_path = save_dataset(X_train, y_train, "train_data.pkl")
        val_path = save_dataset(X_val, y_val, "val_data.pkl")
        test_path = save_dataset(X_test, y_test, "test_data.pkl")

        print(f"Dataset saved: {train_path} ({n_train} samples)")
        print(f"Dataset saved: {val_path} ({n_val} samples)")
        print(f"Dataset saved: {test_path} ({n_test} samples)")

    return X_train, y_train, X_val, y_val, X_test, y_test
