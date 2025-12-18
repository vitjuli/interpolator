"""
Core experimental logic for Task 8.

Consolidates all benchmark experiments into a clean, reusable module.
"""

import sys
from pathlib import Path
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fivedreg.model import FiveDRegressor
from task8.config import (
    DATASET_SIZES,
    DEFAULT_HYPERPARAMS,
    N_RUNS,
    RANDOM_SEED,
    USE_WANDB,
    WANDB_PROJECT,
    GROUND_TRUTH_COEFFS,
    ISO_PARAM_ARCHITECTURES,
)


def ground_truth_function(X: np.ndarray) -> np.ndarray:
    """
    Ground truth function for data generation.

    y = 2.0·x₁ + (-1.5)·x₂² + 3.0·sin(x₃) + 0.5·x₄·x₅

    Args:
        X: Input array of shape (n_samples, 5)

    Returns:
        Output array of shape (n_samples,)
    """
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    return (
        GROUND_TRUTH_COEFFS["x1"] * x1 +
        GROUND_TRUTH_COEFFS["x2"] * (x2 ** 2) +
        GROUND_TRUTH_COEFFS["x3"] * np.sin(x3) +
        GROUND_TRUTH_COEFFS["x4_x5"] * x4 * x5
    )


def generate_dataset(n_samples: int, seed: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset.

    Args:
        n_samples: Number of samples
        seed: Random seed

    Returns:
        X, y arrays
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, 5).astype(np.float32)
    y = ground_truth_function(X)
    return X, y


def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    hyperparams: Optional[Dict] = None,
    use_wandb: bool = False,
    wandb_run_name: Optional[str] = None,
) -> Tuple[FiveDRegressor, Dict]:
    """
    Train a single model and return results.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Optional validation data
        hyperparams: Model hyperparameters
        use_wandb: Enable wandb logging
        wandb_run_name: Optional wandb run name

    Returns:
        Trained model and results dict
    """
    if hyperparams is None:
        hyperparams = DEFAULT_HYPERPARAMS.copy()

    # Add wandb params if enabled
    if use_wandb:
        hyperparams["use_wandb"] = True
        hyperparams["wandb_project"] = WANDB_PROJECT
        if wandb_run_name:
            hyperparams["wandb_run_name"] = wandb_run_name

    # Create and train model
    model = FiveDRegressor(**hyperparams)

    start_time = time.perf_counter()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.perf_counter() - start_time

    # Compute metrics
    results = {
        "train_time_sec": train_time,
        "epochs_trained": len(model.history["epoch"]),
        "final_train_loss": model.history["train_loss"][-1] if model.history["train_loss"] else None,
        "final_val_loss": model.history["val_loss"][-1] if model.history["val_loss"] else None,
        "final_train_r2": model.history["train_r2"][-1] if model.history["train_r2"] else None,
        "final_val_r2": model.history["val_r2"][-1] if model.history["val_r2"] else None,
    }

    return model, results


def run_training_time_scaling(
    n_runs: int = N_RUNS,
    use_wandb: bool = USE_WANDB,
    verbose: bool = True,
) -> Dict:
    """
    Experiment 1: Training time scaling with dataset size.

    Tests how training time scales with dataset size (1K, 5K, 10K).

    Args:
        n_runs: Number of runs per dataset size
        use_wandb: Enable wandb logging
        verbose: Print progress

    Returns:
        Dictionary with results for each dataset size
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 1: Training Time Scaling")
        print("="*70)

    results = {}

    for dataset_name, n_samples in DATASET_SIZES.items():
        if verbose:
            print(f"\n{dataset_name} samples ({n_samples:,})...")

        dataset_results = {
            "n_samples": n_samples,
            "n_runs": n_runs,
            "train_times": [],
            "epochs": [],
        }

        for run_idx in range(n_runs):
            if verbose:
                print(f"  Run {run_idx + 1}/{n_runs}...", end=" ")

            # Generate data
            X_train, y_train = generate_dataset(n_samples, seed=RANDOM_SEED + run_idx)
            n_val = n_samples // 5
            X_val, y_val = generate_dataset(n_val, seed=RANDOM_SEED + 1000 + run_idx)

            # Train model
            wandb_run_name = f"training-time-{dataset_name}-run{run_idx+1}" if use_wandb else None
            model, run_results = train_single_model(
                X_train, y_train, X_val, y_val,
                use_wandb=use_wandb,
                wandb_run_name=wandb_run_name,
            )

            dataset_results["train_times"].append(run_results["train_time_sec"])
            dataset_results["epochs"].append(run_results["epochs_trained"])

            if verbose:
                print(f"{run_results['train_time_sec']:.2f}s")

        # Compute statistics
        times = dataset_results["train_times"]
        dataset_results["mean_time"] = float(np.mean(times))
        dataset_results["std_time"] = float(np.std(times))
        dataset_results["min_time"] = float(np.min(times))
        dataset_results["max_time"] = float(np.max(times))

        results[dataset_name] = dataset_results

        if verbose:
            print(f"  Mean: {dataset_results['mean_time']:.2f}s ± {dataset_results['std_time']:.2f}s")

    return results


def run_accuracy_analysis(
    n_runs: int = N_RUNS,
    use_wandb: bool = USE_WANDB,
    verbose: bool = True,
) -> Dict:
    """
    Experiment 2: Accuracy and generalization analysis.

    Measures train/val/test performance across dataset sizes.

    Args:
        n_runs: Number of runs per dataset size
        use_wandb: Enable wandb logging
        verbose: Print progress

    Returns:
        Dictionary with accuracy results
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 2: Accuracy Analysis")
        print("="*70)

    results = {}

    for dataset_name, n_samples in DATASET_SIZES.items():
        if verbose:
            print(f"\n{dataset_name} samples ({n_samples:,})...")

        dataset_results = {
            "n_samples": n_samples,
            "train_mse": [],
            "val_mse": [],
            "test_mse": [],
            "train_r2": [],
            "val_r2": [],
            "test_r2": [],
        }

        for run_idx in range(n_runs):
            # Generate data with proper splits
            n_train = int(n_samples * 0.64)
            n_val = int(n_samples * 0.16)
            n_test = int(n_samples * 0.20)

            X_train, y_train = generate_dataset(n_train, seed=RANDOM_SEED + run_idx)
            X_val, y_val = generate_dataset(n_val, seed=RANDOM_SEED + 1000 + run_idx)
            X_test, y_test = generate_dataset(n_test, seed=RANDOM_SEED + 2000 + run_idx)

            # Train model
            wandb_run_name = f"accuracy-{dataset_name}-run{run_idx+1}" if use_wandb else None
            model, run_results = train_single_model(
                X_train, y_train, X_val, y_val,
                use_wandb=use_wandb,
                wandb_run_name=wandb_run_name,
            )

            # Test set evaluation
            y_pred = model.predict(X_test)
            test_mse = float(np.mean((y_pred - y_test) ** 2))
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            test_r2 = float(1 - ss_res / ss_tot)

            dataset_results["train_mse"].append(run_results["final_train_loss"])
            dataset_results["val_mse"].append(run_results["final_val_loss"])
            dataset_results["test_mse"].append(test_mse)
            dataset_results["train_r2"].append(run_results["final_train_r2"])
            dataset_results["val_r2"].append(run_results["final_val_r2"])
            dataset_results["test_r2"].append(test_r2)

        # Compute statistics
        for metric in ["train_mse", "val_mse", "test_mse", "train_r2", "val_r2", "test_r2"]:
            values = dataset_results[metric]
            dataset_results[f"{metric}_mean"] = float(np.mean(values))
            dataset_results[f"{metric}_std"] = float(np.std(values))

        results[dataset_name] = dataset_results

        if verbose:
            print(f"  Test R²: {dataset_results['test_r2_mean']:.4f} ± {dataset_results['test_r2_std']:.4f}")

    return results


def run_error_evolution_analysis(
    n_samples: int = 5000,
    use_wandb: bool = USE_WANDB,
    verbose: bool = True,
) -> Dict:
    """
    Experiment 3: Error evolution over training time.

    Tracks approximation error on fixed validation slice during training.

    Args:
        n_samples: Dataset size
        use_wandb: Enable wandb logging
        verbose: Print progress

    Returns:
        Dictionary with error evolution data
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 3: Error Evolution Analysis")
        print("="*70)

    # Create fixed validation slice (1D slice varying x₁)
    n_points = 100
    x1_values = np.linspace(-3, 3, n_points)
    slice_X = np.zeros((n_points, 5), dtype=np.float32)
    slice_X[:, 0] = x1_values
    slice_X[:, 1] = 0.5  # Fixed x₂
    slice_X[:, 2] = 1.0  # Fixed x₃
    slice_X[:, 3] = -0.5  # Fixed x₄
    slice_X[:, 4] = 0.3  # Fixed x₅
    slice_y_true = ground_truth_function(slice_X)

    # Generate training data
    X_train, y_train = generate_dataset(n_samples)
    X_val, y_val = generate_dataset(n_samples // 5)

    # Train with error tracking
    hyperparams = DEFAULT_HYPERPARAMS.copy()
    hyperparams["verbose"] = verbose
    if use_wandb:
        hyperparams["use_wandb"] = True
        hyperparams["wandb_project"] = WANDB_PROJECT
        hyperparams["wandb_run_name"] = "error-evolution"

    # We need a custom training loop to track error per epoch
    # For simplicity, just train and compute error at checkpoints
    model = FiveDRegressor(**hyperparams)
    model.fit(X_train, y_train, X_val, y_val)

    # Compute error evolution (retrospectively from history)
    error_history = {
        "epochs": list(range(1, len(model.history["epoch"]) + 1)),
        "train_loss": model.history["train_loss"],
        "val_loss": model.history["val_loss"],
        "train_r2": model.history["train_r2"],
        "val_r2": model.history["val_r2"],
    }

    # Final slice error
    y_pred_slice = model.predict(slice_X)
    slice_mae = float(np.mean(np.abs(y_pred_slice - slice_y_true)))

    results = {
        "n_samples": n_samples,
        "final_slice_mae": slice_mae,
        "error_history": error_history,
        "slice_X": slice_X.tolist(),
        "slice_y_true": slice_y_true.tolist(),
        "slice_y_pred": y_pred_slice.tolist(),
    }

    if verbose:
        print(f"  Final slice MAE: {slice_mae:.4f}")
        print(f"  Trained for {len(error_history['epochs'])} epochs")

    return results


def run_iso_parameter_comparison(
    n_samples: int = 5000,
    use_wandb: bool = USE_WANDB,
    verbose: bool = True,
) -> Dict:
    """
    Experiment 4: Iso-parameter architecture comparison.

    Compares architectures with similar parameter counts but different structures.

    Args:
        n_samples: Dataset size
        use_wandb: Enable wandb logging
        verbose: Print progress

    Returns:
        Dictionary with comparison results
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 4: Iso-Parameter Architecture Comparison")
        print("="*70)

    # Generate data once
    X_train, y_train = generate_dataset(n_samples)
    X_val, y_val = generate_dataset(n_samples // 5)
    X_test, y_test = generate_dataset(n_samples // 5)

    results = {}

    for arch_name, hidden_layers in ISO_PARAM_ARCHITECTURES.items():
        if verbose:
            print(f"\n{arch_name}: {hidden_layers}")

        hyperparams = DEFAULT_HYPERPARAMS.copy()
        hyperparams["hidden_layers"] = hidden_layers

        wandb_run_name = f"iso-param-{arch_name.lower().replace(' ', '-')}" if use_wandb else None

        model, train_results = train_single_model(
            X_train, y_train, X_val, y_val,
            hyperparams=hyperparams,
            use_wandb=use_wandb,
            wandb_run_name=wandb_run_name,
        )

        # Test evaluation
        y_pred = model.predict(X_test)
        test_mse = float(np.mean((y_pred - y_test) ** 2))
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        test_r2 = float(1 - ss_res / ss_tot)

        # Model complexity
        summary = model.get_model_summary()

        results[arch_name] = {
            "hidden_layers": hidden_layers,
            "params": summary["parameters"]["total"],
            "flops": summary["flops"]["total_flops"],
            "train_time": train_results["train_time_sec"],
            "test_r2": test_r2,
            "test_mse": test_mse,
        }

        if verbose:
            print(f"  Params: {results[arch_name]['params']:,}")
            print(f"  FLOPs: {results[arch_name]['flops']:,}")
            print(f"  Test R²: {test_r2:.6f}")
            print(f"  Time: {train_results['train_time_sec']:.2f}s")

    return results


def run_approximation_analysis(
    n_samples: int = 5000,
    use_wandb: bool = USE_WANDB,
    verbose: bool = True,
) -> Dict:
    """
    Experiment 5: Ground truth vs model approximation.

    Analyzes how well the model approximates the ground truth function
    using 1D and 2D slices.

    Args:
        n_samples: Dataset size
        use_wandb: Enable wandb logging
        verbose: Print progress

    Returns:
        Dictionary with approximation analysis results
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 5: Approximation Analysis")
        print("="*70)

    # Train model
    X_train, y_train = generate_dataset(n_samples)
    X_val, y_val = generate_dataset(n_samples // 5)

    wandb_run_name = "approximation-analysis" if use_wandb else None
    model, _ = train_single_model(
        X_train, y_train, X_val, y_val,
        use_wandb=use_wandb,
        wandb_run_name=wandb_run_name,
    )

    # 1D Slice (varying x₁, others fixed)
    n_points = 100
    x1_values = np.linspace(-3, 3, n_points)
    slice_1d_X = np.zeros((n_points, 5), dtype=np.float32)
    slice_1d_X[:, 0] = x1_values
    slice_1d_X[:, 1] = 0.0
    slice_1d_X[:, 2] = 0.0
    slice_1d_X[:, 3] = 0.0
    slice_1d_X[:, 4] = 0.0

    slice_1d_y_true = ground_truth_function(slice_1d_X)
    slice_1d_y_pred = model.predict(slice_1d_X)
    slice_1d_error = slice_1d_y_pred - slice_1d_y_true
    slice_1d_mae = float(np.mean(np.abs(slice_1d_error)))

    # 2D Slice (varying x₁ and x₂, others fixed)
    n_points_2d = 50
    x1_2d = np.linspace(-2, 2, n_points_2d)
    x2_2d = np.linspace(-2, 2, n_points_2d)
    X1, X2 = np.meshgrid(x1_2d, x2_2d)

    slice_2d_X = np.zeros((n_points_2d * n_points_2d, 5), dtype=np.float32)
    slice_2d_X[:, 0] = X1.ravel()
    slice_2d_X[:, 1] = X2.ravel()

    slice_2d_y_true = ground_truth_function(slice_2d_X).reshape(n_points_2d, n_points_2d)
    slice_2d_y_pred = model.predict(slice_2d_X).reshape(n_points_2d, n_points_2d)
    slice_2d_error = slice_2d_y_pred - slice_2d_y_true
    slice_2d_mae = float(np.mean(np.abs(slice_2d_error)))

    results = {
        "1d_slice": {
            "x1_values": x1_values.tolist(),
            "y_true": slice_1d_y_true.tolist(),
            "y_pred": slice_1d_y_pred.tolist(),
            "error": slice_1d_error.tolist(),
            "mae": slice_1d_mae,
        },
        "2d_slice": {
            "x1_grid": X1.tolist(),
            "x2_grid": X2.tolist(),
            "y_true": slice_2d_y_true.tolist(),
            "y_pred": slice_2d_y_pred.tolist(),
            "error": slice_2d_error.tolist(),
            "mae": slice_2d_mae,
        },
    }

    if verbose:
        print(f"  1D Slice MAE: {slice_1d_mae:.4f}")
        print(f"  2D Slice MAE: {slice_2d_mae:.4f}")

    return results
