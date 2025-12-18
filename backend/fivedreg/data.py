import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


Array = np.ndarray


@dataclass
class PreparedData:
    """
    Container for preprocessed dataset splits and fitted preprocessing objects.

    Attributes:
        X_train: Standardized training features of shape (n_train, 5).
        X_val: Standardized validation features of shape (n_val, 5).
        X_test: Standardized test features of shape (n_test, 5).
        y_train: Training targets of shape (n_train,) or None.
        y_val: Validation targets of shape (n_val,) or None.
        y_test: Test targets of shape (n_test,) or None.
        imputer: Fitted imputer used to handle missing values.
        scaler: Fitted scaler used to standardize features.
    """
    X_train: Array
    X_val: Array
    X_test: Array
    y_train: Optional[Array]
    y_val: Optional[Array]
    y_test: Optional[Array]
    imputer: SimpleImputer
    scaler: StandardScaler


def load_dataset(filepath: Union[str, Path]) -> Tuple[Array, Optional[Array]]:
    """
    Load and validate a 5D regression dataset from a `.pkl` file.

    The dataset is expected to be stored as a dictionary with key ``"X"``
    containing the feature matrix and an optional key ``"y"`` containing
    the target values. This allows the function to handle both training
    datasets (with targets) and test datasets (without targets).

    Args:
        filepath: Path to the `.pkl` file containing the dataset.

    Returns:
        X: Feature array of shape ``(n_samples, 5)``.
        y: Target array of shape ``(n_samples,)`` if present, otherwise ``None``.

    Raises:
        ValueError: If the file extension is not `.pkl`.
        ValueError: If the loaded object is not a dictionary with key ``"X"``.
        ValueError: If ``X`` is not two-dimensional or does not have exactly
            five feature columns.
        ValueError: If ``y`` has incompatible dimensions or does not match
            the number of samples in ``X``.
        ValueError: If ``X`` or ``y`` contains infinite values.

    Note:
        Missing values (NaN) are allowed and will be imputed during preprocessing.
        Only infinite values are rejected.
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() != ".pkl":
        raise ValueError(f"Expected .pkl file, got {filepath.name}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "X" not in data:
        raise ValueError("Expected a dict with key 'X' (and optionally 'y').")

    X = np.asarray(data["X"], dtype=float)

    y = None
    if "y" in data and data["y"] is not None:
        y = np.asarray(data["y"], dtype=float)

    # Validate X
    if X.ndim != 2:
        raise ValueError(f"X must be 2D with shape (n_samples, 5). Got {X.shape}")
    if X.shape[1] != 5:
        raise ValueError(f"Expected 5 features, got {X.shape[1]} (shape={X.shape})")
    if np.isinf(X).any():
        raise ValueError("X contains infinite values.")

    # Validate y (if present)
    if y is not None:
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (n_samples,) or 2D (n_samples, 1). Got {y.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y must contain the same number of samples. "
                f"Got {X.shape[0]} and {y.shape[0]}."
            )
        if np.isinf(y).any():
            raise ValueError("y contains infinite values.")

    return X, y


def split_data(
    X: Array,
    y: Optional[Array],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split a dataset into training, validation, and test sets.

    The test set is first separated using ``test_size``. The validation
    set is then taken as a fraction of the remaining data.

    Args:
        X: Feature array of shape ``(n_samples, 5)``.
        y: Target array of shape ``(n_samples,)`` or ``None`` for test datasets.
        test_size: Fraction of the dataset reserved for testing.
        val_size: Fraction of the remaining data reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        X_test: Test feature matrix.
        y_train: Training targets or ``None``.
        y_val: Validation targets or ``None``.
        y_test: Test targets or ``None``.
    """
    if y is None:
        X_trainval, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state, shuffle=True
        )
        X_train, X_val = train_test_split(
            X_trainval, test_size=val_size, random_state=random_state, shuffle=True
        )
        return X_train, X_val, X_test, None, None, None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=random_state, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_features(
    X_train: Array,
    X_val: Array,
    X_test: Array,
    impute_strategy: str = "mean",
):
    """
    Handle missing values and standardize feature distributions.

    Missing values are imputed using statistics computed from the training
    set only. Feature standardization is also fitted exclusively on the
    training data to avoid information leakage.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        X_test: Test feature matrix.
        impute_strategy: Strategy used by the imputer (default: ``"mean"``).

    Returns:
        X_train: Standardized training features.
        X_val: Standardized validation features.
        X_test: Standardized test features.
        imputer: Fitted imputer instance.
        scaler: Fitted scaler instance.
    """
    imputer = SimpleImputer(strategy=impute_strategy)
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, imputer, scaler


def prepare_data(
    filepath: Union[str, Path],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    impute_strategy: str = "mean",
) -> PreparedData:
    """
    End-to-end data preparation pipeline.

    This function loads a dataset from disk, validates its structure,
    splits it into training, validation, and test sets, handles missing
    values, and standardizes all features.

    Args:
        filepath: Path to the `.pkl` dataset file.
        test_size: Fraction of samples reserved for the test set.
        val_size: Fraction of remaining samples reserved for validation.
        random_state: Random seed for reproducibility.
        impute_strategy: Imputation strategy for missing values.

    Returns:
        PreparedData: Object containing all dataset splits and fitted
        preprocessing components.
    """
    X, y = load_dataset(filepath)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )
    X_train, X_val, X_test, imputer, scaler = standardize_features(
        X_train, X_val, X_test, impute_strategy=impute_strategy
    )
    return PreparedData(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        imputer, scaler
    )