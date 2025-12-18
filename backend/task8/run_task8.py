"""
Task 8: Main Orchestrator

Runs all experiments, generates all plots, and saves results.
This is the single entry point for reproducing all Task 8 analysis.

Usage:
    python -m task8.run_task8
    python -m task8.run_task8 --use-wandb
    python -m task8.run_task8 --quick  # Run with fewer iterations
"""

import sys
import argparse
import json
from pathlib import Path
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task8 import (
    # Experiments
    run_training_time_scaling,
    run_accuracy_analysis,
    run_error_evolution_analysis,
    run_iso_parameter_comparison,
    run_approximation_analysis,
    # Profiling
    run_memory_profiling,
    run_memory_scaling_analysis,
    # Visualization
    plot_training_time_scaling,
    plot_memory_usage,
    plot_memory_scaling,
    plot_accuracy_comparison,
    plot_error_evolution,
    plot_iso_parameter_comparison,
    plot_approximation_analysis,
)

from task8.config import FIGURES_DIR, N_RUNS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Task 8: Performance, Profiling and Experimental Analysis"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging (optional)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick mode with fewer iterations (for testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: figures/)"
    )
    return parser.parse_args()


def main():
    """Main orchestrator for Task 8."""
    args = parse_args()

    # Configuration
    use_wandb = args.use_wandb
    n_runs = 2 if args.quick else N_RUNS  # Use 2 runs in quick mode

    print("="*80)
    print("TASK 8: PERFORMANCE, PROFILING AND EXPERIMENTAL ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Number of runs: {n_runs}")
    print(f"  - Weights & Biases: {'Enabled' if use_wandb else 'Disabled'}")
    print(f"  - Output directory: {FIGURES_DIR}")
    print(f"  - Mode: {'Quick' if args.quick else 'Full'}")

    if use_wandb:
        print("\n  wandb is enabled. Make sure you're logged in (wandb login)")

    print("\n" + "="*80)

    start_time = time.time()

    # ===================================================================
    # EXPERIMENT 1: Training Time Scaling
    # ===================================================================
    print("\n  Running Experiment 1/7: Training Time Scaling...")
    training_time_results = run_training_time_scaling(
        n_runs=n_runs,
        use_wandb=use_wandb,
        verbose=True
    )

    # Save results
    with open(FIGURES_DIR / "training_time_results.json", "w") as f:
        json.dump(training_time_results, f, indent=2)

    # Plot
    plot_training_time_scaling(training_time_results)

    # ===================================================================
    # EXPERIMENT 2: Accuracy Analysis
    # ===================================================================
    print("\n  Running Experiment 2/7: Accuracy Analysis...")
    accuracy_results = run_accuracy_analysis(
        n_runs=n_runs,
        use_wandb=use_wandb,
        verbose=True
    )

    # Save results
    with open(FIGURES_DIR / "accuracy_results.json", "w") as f:
        json.dump(accuracy_results, f, indent=2)

    # Plot
    plot_accuracy_comparison(accuracy_results)

    # ===================================================================
    # EXPERIMENT 3: Error Evolution
    # ===================================================================
    print("\n  Running Experiment 3/7: Error Evolution Analysis...")
    error_evolution_results = run_error_evolution_analysis(
        use_wandb=use_wandb,
        verbose=True
    )

    # Save results
    with open(FIGURES_DIR / "error_evolution_results.json", "w") as f:
        json.dump(error_evolution_results, f, indent=2)

    # Plot
    plot_error_evolution(error_evolution_results)

    # ===================================================================
    # EXPERIMENT 4: Iso-Parameter Comparison
    # ===================================================================
    print("\n  Running Experiment 4/7: Iso-Parameter Architecture Comparison...")
    iso_param_results = run_iso_parameter_comparison(
        use_wandb=use_wandb,
        verbose=True
    )

    # Save results
    with open(FIGURES_DIR / "iso_parameter_results.json", "w") as f:
        json.dump(iso_param_results, f, indent=2)

    # Plot
    plot_iso_parameter_comparison(iso_param_results)

    # ===================================================================
    # EXPERIMENT 5: Approximation Analysis
    # ===================================================================
    print("\n  Running Experiment 5/7: Ground Truth Approximation Analysis...")
    approximation_results = run_approximation_analysis(
        use_wandb=use_wandb,
        verbose=True
    )

    # Save results
    with open(FIGURES_DIR / "approximation_results.json", "w") as f:
        json.dump(approximation_results, f, indent=2)

    # Plot
    plot_approximation_analysis(approximation_results)

    # ===================================================================
    # EXPERIMENT 6: Memory Profiling
    # ===================================================================
    print("\n  Running Experiment 6/7: Memory Profiling...")
    memory_profiling_results = run_memory_profiling(verbose=True)

    # Save results
    with open(FIGURES_DIR / "memory_profiling_results.json", "w") as f:
        json.dump(memory_profiling_results, f, indent=2)

    # Plot
    plot_memory_usage(memory_profiling_results)

    # ===================================================================
    # EXPERIMENT 7: Memory Scaling
    # ===================================================================
    print("\n  Running Experiment 7/7: Memory Scaling Analysis...")
    memory_scaling_results = run_memory_scaling_analysis(verbose=True)

    # Save results
    with open(FIGURES_DIR / "memory_scaling_results.json", "w") as f:
        json.dump(memory_scaling_results, f, indent=2)

    # Plot
    plot_memory_scaling(memory_scaling_results)

    # ===================================================================
    # Summary
    # ===================================================================
    elapsed_time = time.time() - start_time

    print("\n" + "="*80)
    print("TASK 8 COMPLETE!")
    print("="*80)
    print(f"\n All experiments completed in {elapsed_time/60:.1f} minutes")
    print(f"\n Generated figures:")

    figure_files = list(FIGURES_DIR.glob(f"*.png"))
    for fig_file in sorted(figure_files):
        print(f"  - {fig_file.name}")

    print(f"\n Results saved to: {FIGURES_DIR}")
    print(f"\n Next step: Review Performance_analysis.md for detailed analysis")

    # Print key findings
    print("\n" + "="*80)
    print("Obsevations")
    print("="*80)

    # Training time scaling
    sizes = list(training_time_results.keys())
    times = [training_time_results[k]["mean_time"] for k in sizes]
    print(f"\n1. Training Time Scaling:")
    for i, size in enumerate(sizes):
        print(f"   {size}: {times[i]:.2f}s")

    # Best accuracy
    best_dataset = max(sizes, key=lambda k: accuracy_results[k]["test_r2_mean"])
    best_r2 = accuracy_results[best_dataset]["test_r2_mean"]
    print(f"\n2. Best Test Accuracy: {best_r2:.4f} R² ({best_dataset} samples)")

    # Memory usage
    peak_mem = memory_profiling_results["peak_memory_mb"]
    print(f"\n3. Peak Memory: {peak_mem:.1f} MB")

    # Memory scaling
    is_constant = memory_scaling_results["summary"]["is_constant"]
    memory_range = memory_scaling_results["summary"]["memory_range_mb"]
    print(f"\n4. Memory Scaling: {'O(1) - Constant' if is_constant else 'Not constant'}")
    print(f"   Range: {memory_range:.1f} MB across 20× dataset size increase")

    # Best architecture
    best_arch = max(iso_param_results.keys(), key=lambda k: iso_param_results[k]["test_r2"])
    best_arch_r2 = iso_param_results[best_arch]["test_r2"]
    best_arch_params = iso_param_results[best_arch]["params"]
    print(f"\n5. Best Architecture: {best_arch}")
    print(f"   Test R²: {best_arch_r2:.6f}, Params: {best_arch_params:,}")

    # Approximation quality
    approx_mae_1d = approximation_results["1d_slice"]["mae"]
    approx_mae_2d = approximation_results["2d_slice"]["mae"]
    print(f"\n6. Approximation Quality:")
    print(f"   1D Slice MAE: {approx_mae_1d:.4f}")
    print(f"   2D Slice MAE: {approx_mae_2d:.4f}")


if __name__ == "__main__":
    main()
