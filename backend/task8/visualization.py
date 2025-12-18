"""
Visualization functions for Task 8.

Consolidates all plotting logic into clean, reusable functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from task8.config import FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT


def plot_training_time_scaling(results: Dict, save_path: Path = None):
    """
    Plot training time scaling with dataset size.

    Args:
        results: Dictionary from run_training_time_scaling()
        save_path: Optional path to save figure
    """
    if save_path is None:
        save_path = FIGURES_DIR / f"training_time_scaling.{FIGURE_FORMAT}"

    dataset_names = list(results.keys())
    n_samples = [results[k]["n_samples"] for k in dataset_names]
    mean_times = [results[k]["mean_time"] for k in dataset_names]
    std_times = [results[k]["std_time"] for k in dataset_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax1.errorbar(n_samples, mean_times, yerr=std_times, marker='o', markersize=8,
                capsize=5, linewidth=2, label='Observed')
    ax1.set_xlabel('Dataset Size (samples)', fontsize=12)
    ax1.set_ylabel('Training Time (seconds)', fontsize=12)
    ax1.set_title('Training Time vs Dataset Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Log-log scale for scaling analysis
    ax2.loglog(n_samples, mean_times, 'o-', markersize=8, linewidth=2, label='Observed')

    # Fit power law: t = a * n^α
    log_n = np.log(n_samples)
    log_t = np.log(mean_times)
    coeffs = np.polyfit(log_n, log_t, 1)
    alpha = coeffs[0]
    a = np.exp(coeffs[1])

    # Plot fitted line
    n_fit = np.linspace(min(n_samples), max(n_samples), 100)
    t_fit = a * (n_fit ** alpha)
    ax2.loglog(n_fit, t_fit, '--', linewidth=2, label=f'Fitted: t ∝ n^{alpha:.2f}')

    ax2.set_xlabel('Dataset Size (samples)', fontsize=12)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Scaling Analysis (log-log)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_memory_usage(profiling_results: Dict, save_path: Path = None):
    """
    Plot memory usage breakdown.

    Args:
        profiling_results: Dictionary from run_memory_profiling()
        save_path: Optional path to save figure
    """
    if save_path is None:
        save_path = FIGURES_DIR / f"memory_usage.{FIGURE_FORMAT}"

    components = ['Model', 'Data', 'Training\nOverhead']
    memory_values = [
        profiling_results['model_memory_mb'],
        profiling_results['data_memory_mb'],
        profiling_results['training_memory_mb'],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax1.bar(components, memory_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax1.set_title('Memory Usage Breakdown', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, v in enumerate(memory_values):
        ax1.text(i, v + 0.5, f'{v:.1f} MB', ha='center', va='bottom', fontweight='bold')

    # Theoretical vs Observed
    theoretical_total = (
        profiling_results['theoretical_model_memory_mb'] +
        profiling_results['theoretical_optimizer_memory_mb'] +
        profiling_results['theoretical_batch_memory_mb']
    )
    observed_total = profiling_results['peak_memory_mb'] - profiling_results['baseline_memory_mb']

    categories = ['Theoretical', 'Observed']
    values = [theoretical_total, observed_total]
    colors = ['#9b59b6', '#e67e22']

    ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Total Memory (MB)', fontsize=12)
    ax2.set_title('Theoretical vs Observed', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(values):
        ax2.text(i, v + 5, f'{v:.1f} MB', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_memory_scaling(scaling_results: Dict, save_path: Path = None):
    """
    Plot memory scaling with dataset size.

    Args:
        scaling_results: Dictionary from run_memory_scaling_analysis()
        save_path: Optional path to save figure
    """
    if save_path is None:
        save_path = FIGURES_DIR / f"memory_scaling.{FIGURE_FORMAT}"

    results = scaling_results["results"]
    n_samples = [r["n_samples"] for r in results]
    peak_memory = [r["peak_memory_mb"] for r in results]

    mean_memory = np.mean(peak_memory)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_samples, peak_memory, 'o-', markersize=10, linewidth=2, label='Peak Memory')
    ax.axhline(y=mean_memory, color='r', linestyle='--', linewidth=2,
               label=f'Mean: {mean_memory:.1f} MB')

    ax.set_xlabel('Dataset Size (samples)', fontsize=12)
    ax.set_ylabel('Peak Memory Usage (MB)', fontsize=12)
    ax.set_title('Memory Usage vs Dataset Size (O(1) Scaling)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add variation annotation
    memory_range = max(peak_memory) - min(peak_memory)
    ax.text(0.95, 0.05, f'Variation: {memory_range:.1f} MB',
           transform=ax.transAxes, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_accuracy_comparison(accuracy_results: Dict, save_path: Path = None):
    """
    Plot accuracy across dataset sizes.

    Args:
        accuracy_results: Dictionary from run_accuracy_analysis()
        save_path: Optional path to save figure
    """
    if save_path is None:
        save_path = FIGURES_DIR / f"accuracy_comparison.{FIGURE_FORMAT}"

    dataset_names = list(accuracy_results.keys())
    n_samples = [accuracy_results[k]["n_samples"] for k in dataset_names]

    train_r2 = [accuracy_results[k]["train_r2_mean"] for k in dataset_names]
    val_r2 = [accuracy_results[k]["val_r2_mean"] for k in dataset_names]
    test_r2 = [accuracy_results[k]["test_r2_mean"] for k in dataset_names]

    train_r2_std = [accuracy_results[k]["train_r2_std"] for k in dataset_names]
    val_r2_std = [accuracy_results[k]["val_r2_std"] for k in dataset_names]
    test_r2_std = [accuracy_results[k]["test_r2_std"] for k in dataset_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # R² scores
    x = np.arange(len(dataset_names))
    width = 0.25

    ax1.bar(x - width, train_r2, width, yerr=train_r2_std, label='Train',
           capsize=3, alpha=0.8, color='#3498db')
    ax1.bar(x, val_r2, width, yerr=val_r2_std, label='Validation',
           capsize=3, alpha=0.8, color='#2ecc71')
    ax1.bar(x + width, test_r2, width, yerr=test_r2_std, label='Test',
           capsize=3, alpha=0.8, color='#e74c3c')

    ax1.set_xlabel('Dataset Size', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Accuracy vs Dataset Size', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.96, 1.0])

    # MSE
    train_mse = [accuracy_results[k]["train_mse_mean"] for k in dataset_names]
    val_mse = [accuracy_results[k]["val_mse_mean"] for k in dataset_names]
    test_mse = [accuracy_results[k]["test_mse_mean"] for k in dataset_names]

    ax2.plot(n_samples, train_mse, 'o-', label='Train', markersize=8, linewidth=2)
    ax2.plot(n_samples, val_mse, 's-', label='Validation', markersize=8, linewidth=2)
    ax2.plot(n_samples, test_mse, '^-', label='Test', markersize=8, linewidth=2)

    ax2.set_xlabel('Dataset Size (samples)', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('Mean Squared Error vs Dataset Size', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_error_evolution(error_results: Dict, save_path: Path = None):
    """
    Plot error evolution over training.

    Args:
        error_results: Dictionary from run_error_evolution_analysis()
        save_path: Optional path to save figure
    """
    if save_path is None:
        save_path = FIGURES_DIR / f"error_evolution.{FIGURE_FORMAT}"

    history = error_results["error_history"]
    epochs = history["epochs"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Loss Evolution During Training', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # R² curves
    ax2.plot(epochs, history["train_r2"], label='Train R²', linewidth=2)
    ax2.plot(epochs, history["val_r2"], label='Val R²', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² Evolution During Training', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_iso_parameter_comparison(iso_results: Dict, save_path: Path = None):
    """
    Plot iso-parameter architecture comparison.

    Args:
        iso_results: Dictionary from run_iso_parameter_comparison()
        save_path: Optional path to save figure
    """
    if save_path is None:
        save_path = FIGURES_DIR / f"iso_parameter_comparison.{FIGURE_FORMAT}"

    arch_names = list(iso_results.keys())
    params = [iso_results[k]["params"] for k in arch_names]
    flops = [iso_results[k]["flops"] for k in arch_names]
    test_r2 = [iso_results[k]["test_r2"] for k in arch_names]
    train_time = [iso_results[k]["train_time"] for k in arch_names]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Params vs FLOPs
    scatter = ax1.scatter(params, flops, c=test_r2, s=200, cmap='RdYlGn',
                         vmin=min(test_r2), vmax=max(test_r2), alpha=0.7)
    for i, name in enumerate(arch_names):
        ax1.annotate(name, (params[i], flops[i]), fontsize=9, ha='center', va='bottom')
    ax1.set_xlabel('Parameters', fontsize=12)
    ax1.set_ylabel('FLOPs', fontsize=12)
    ax1.set_title('Parameters vs FLOPs (color = R²)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Test R²')

    # Test R²
    x = np.arange(len(arch_names))
    bars = ax2.bar(x, test_r2, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(arch_names, rotation=15, ha='right')
    ax2.set_ylabel('Test R²', fontsize=12)
    ax2.set_title('Test Accuracy by Architecture', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(test_r2):
        ax2.text(i, v + 0.0001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    # Training time
    ax3.bar(x, train_time, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(arch_names, rotation=15, ha='right')
    ax3.set_ylabel('Training Time (seconds)', fontsize=12)
    ax3.set_title('Training Time by Architecture', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Efficiency (R² per second)
    efficiency = [r2 / t for r2, t in zip(test_r2, train_time)]
    ax4.bar(x, efficiency, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax4.set_xticks(x)
    ax4.set_xticklabels(arch_names, rotation=15, ha='right')
    ax4.set_ylabel('R² / Training Time', fontsize=12)
    ax4.set_title('Training Efficiency', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_approximation_analysis(approx_results: Dict, save_path: Path = None):
    """
    Plot ground truth vs model approximation.

    Args:
        approx_results: Dictionary from run_approximation_analysis()
        save_path: Optional path to save figure
    """
    if save_path is None:
        save_path = FIGURES_DIR / f"approximation_analysis.{FIGURE_FORMAT}"

    # 1D slice
    slice_1d = approx_results["1d_slice"]
    x1 = np.array(slice_1d["x1_values"])
    y_true_1d = np.array(slice_1d["y_true"])
    y_pred_1d = np.array(slice_1d["y_pred"])
    error_1d = np.array(slice_1d["error"])

    # 2D slice
    slice_2d = approx_results["2d_slice"]
    X1 = np.array(slice_2d["x1_grid"])
    X2 = np.array(slice_2d["x2_grid"])
    y_true_2d = np.array(slice_2d["y_true"])
    y_pred_2d = np.array(slice_2d["y_pred"])
    error_2d = np.array(slice_2d["error"])

    fig = plt.figure(figsize=(16, 10))

    # 1D Slice - Ground Truth vs Prediction
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x1, y_true_1d, 'b-', linewidth=2, label='Ground Truth')
    ax1.plot(x1, y_pred_1d, 'r--', linewidth=2, label='Model Prediction')
    ax1.set_xlabel('x₁', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('1D Slice: Prediction vs Ground Truth', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1D Slice - Error
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(x1, error_1d, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('x₁', fontsize=11)
    ax2.set_ylabel('Error', fontsize=11)
    ax2.set_title(f'1D Slice Error (MAE={slice_1d["mae"]:.4f})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 1D Slice - Absolute Error
    ax3 = plt.subplot(2, 3, 3)
    ax3.fill_between(x1, 0, np.abs(error_1d), alpha=0.5, color='red')
    ax3.set_xlabel('x₁', fontsize=11)
    ax3.set_ylabel('|Error|', fontsize=11)
    ax3.set_title('1D Slice: Absolute Error', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 2D Slices
    vmin = min(y_true_2d.min(), y_pred_2d.min())
    vmax = max(y_true_2d.max(), y_pred_2d.max())

    ax4 = plt.subplot(2, 3, 4)
    im1 = ax4.contourf(X1, X2, y_true_2d, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    ax4.set_xlabel('x₁', fontsize=11)
    ax4.set_ylabel('x₂', fontsize=11)
    ax4.set_title('2D Slice: Ground Truth', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax4)

    ax5 = plt.subplot(2, 3, 5)
    im2 = ax5.contourf(X1, X2, y_pred_2d, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    ax5.set_xlabel('x₁', fontsize=11)
    ax5.set_ylabel('x₂', fontsize=11)
    ax5.set_title('2D Slice: Model Prediction', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax5)

    ax6 = plt.subplot(2, 3, 6)
    im3 = ax6.contourf(X1, X2, error_2d, levels=20, cmap='RdBu_r', center=0)
    ax6.set_xlabel('x₁', fontsize=11)
    ax6.set_ylabel('x₂', fontsize=11)
    ax6.set_title(f'2D Slice: Error (MAE={slice_2d["mae"]:.4f})', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
