"""
Memory and time profiling for Task 8.

Consolidates all profiling logic from benchmark_memory.py and analyze_memory_growth.py
"""

import sys
from pathlib import Path
import numpy as np
import time
import psutil
import os
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from fivedreg.model import FiveDRegressor
from task8.config import DATASET_SIZES, DEFAULT_HYPERPARAMS, RANDOM_SEED
from task8.experiments import generate_dataset


def get_memory_usage_mb() -> float:
    """
    Get current process memory usage in MB.

    Returns:
        Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_memory_profiling(
    n_samples: int = 5000,
    verbose: bool = True,
) -> Dict:
    """
    Profile memory usage during training and inference.

    Args:
        n_samples: Dataset size
        verbose: Print progress

    Returns:
        Dictionary with memory profiling results
    """
    if verbose:
        print("\n" + "="*70)
        print("MEMORY PROFILING")
        print("="*70)

    # Baseline memory
    baseline_memory = get_memory_usage_mb()
    if verbose:
        print(f"Baseline memory: {baseline_memory:.2f} MB")

    # Generate data
    X_train, y_train = generate_dataset(n_samples)
    X_val, y_val = generate_dataset(n_samples // 5)

    memory_after_data = get_memory_usage_mb()
    data_memory = memory_after_data - baseline_memory

    if verbose:
        print(f"Memory after data generation: {memory_after_data:.2f} MB (+{data_memory:.2f} MB)")

    # Create model
    model = FiveDRegressor(**DEFAULT_HYPERPARAMS)
    memory_after_model = get_memory_usage_mb()
    model_memory = memory_after_model - memory_after_data

    if verbose:
        print(f"Memory after model creation: {memory_after_model:.2f} MB (+{model_memory:.2f} MB)")

    # Train model
    model.fit(X_train, y_train, X_val, y_val)
    peak_memory = get_memory_usage_mb()
    training_memory = peak_memory - memory_after_model

    if verbose:
        print(f"Peak memory after training: {peak_memory:.2f} MB (+{training_memory:.2f} MB)")

    # Inference memory (single prediction)
    memory_before_inference = get_memory_usage_mb()
    _ = model.predict(X_val)
    memory_after_inference = get_memory_usage_mb()
    inference_memory = memory_after_inference - memory_before_inference

    if verbose:
        print(f"Memory for inference: {inference_memory:.2f} MB")

    # Model complexity
    summary = model.get_model_summary()
    total_params = summary["parameters"]["total"]
    total_flops = summary["flops"]["total_flops"]

    # Theoretical memory
    theoretical_model_memory = (total_params * 4) / 1024 / 1024  # float32
    theoretical_optimizer_memory = theoretical_model_memory * 2  # Adam
    max_layer_width = max(DEFAULT_HYPERPARAMS["hidden_layers"])
    theoretical_batch_memory = (DEFAULT_HYPERPARAMS["batch_size"] * max_layer_width * 4) / 1024 / 1024

    results = {
        "n_samples": n_samples,
        "baseline_memory_mb": baseline_memory,
        "data_memory_mb": data_memory,
        "model_memory_mb": model_memory,
        "training_memory_mb": training_memory,
        "peak_memory_mb": peak_memory,
        "inference_memory_mb": inference_memory,
        "total_params": total_params,
        "total_flops": total_flops,
        "theoretical_model_memory_mb": theoretical_model_memory,
        "theoretical_optimizer_memory_mb": theoretical_optimizer_memory,
        "theoretical_batch_memory_mb": theoretical_batch_memory,
    }

    if verbose:
        print(f"\nTheoretical breakdown:")
        print(f"  Model params: {theoretical_model_memory:.3f} MB")
        print(f"  Optimizer: {theoretical_optimizer_memory:.3f} MB")
        print(f"  Batch: {theoretical_batch_memory:.3f} MB")

    return results


def run_memory_scaling_analysis(
    dataset_sizes: List[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Analyze memory scaling with dataset size.

    Tests how memory usage scales with dataset size to confirm O(1) behavior.

    Args:
        dataset_sizes: List of dataset sizes to test
        verbose: Print progress

    Returns:
        Dictionary with scaling analysis results
    """
    if dataset_sizes is None:
        dataset_sizes = [1000, 5000, 10000, 20000]

    if verbose:
        print("\n" + "="*70)
        print("MEMORY SCALING ANALYSIS")
        print("="*70)
        print(f"Testing {len(dataset_sizes)} dataset sizes: {', '.join(f'{n//1000}K' for n in dataset_sizes)}")

    results = []

    for n_samples in dataset_sizes:
        if verbose:
            print(f"\n{n_samples:,} samples...")

        # Baseline
        baseline_memory = get_memory_usage_mb()

        # Generate data
        X_train, y_train = generate_dataset(n_samples)
        X_val, y_val = generate_dataset(min(1000, n_samples // 5))

        # Train
        model = FiveDRegressor(
            **{**DEFAULT_HYPERPARAMS, "max_epochs": 20, "verbose": False}
        )
        model.fit(X_train, y_train, X_val, y_val)

        # Peak memory
        peak_memory = get_memory_usage_mb()
        memory_growth = peak_memory - baseline_memory

        result = {
            "n_samples": n_samples,
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "memory_growth_mb": memory_growth,
        }

        results.append(result)

        if verbose:
            print(f"  Peak: {peak_memory:.1f} MB, Growth: {memory_growth:.1f} MB")

    # Analyze scaling
    peak_memories = [r["peak_memory_mb"] for r in results]
    memory_range = max(peak_memories) - min(peak_memories)
    mean_memory = np.mean(peak_memories)

    scaling_summary = {
        "mean_memory_mb": mean_memory,
        "memory_range_mb": memory_range,
        "std_memory_mb": float(np.std(peak_memories)),
        "is_constant": memory_range < 20,  # Less than 20 MB variation
    }

    if verbose:
        print(f"\nScaling Analysis:")
        print(f"  Mean memory: {mean_memory:.1f} MB")
        print(f"  Range: {memory_range:.1f} MB")
        print(f"  Std: {scaling_summary['std_memory_mb']:.1f} MB")
        if scaling_summary["is_constant"]:
            print(f"  ✓ Memory is O(1) w.r.t. dataset size")
        else:
            print(f"  ⚠ Memory shows significant variation")

    return {
        "results": results,
        "summary": scaling_summary,
    }
