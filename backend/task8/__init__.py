"""
Task 8: Performance, Profiling and Experimental Analysis

A clean, consolidated module for all Task 8 experiments.
"""

__version__ = "1.0.0"
__author__ = "Task 8 Implementation"

from .experiments import (
    run_training_time_scaling,
    run_accuracy_analysis,
    run_error_evolution_analysis,
    run_iso_parameter_comparison,
    run_approximation_analysis,
)

from .profiling import (
    run_memory_profiling,
    run_memory_scaling_analysis,
)

from .visualization import (
    plot_training_time_scaling,
    plot_memory_usage,
    plot_memory_scaling,
    plot_accuracy_comparison,
    plot_error_evolution,
    plot_iso_parameter_comparison,
    plot_approximation_analysis,
)

__all__ = [
    # Experiments
    "run_training_time_scaling",
    "run_accuracy_analysis",
    "run_error_evolution_analysis",
    "run_iso_parameter_comparison",
    "run_approximation_analysis",
    # Profiling
    "run_memory_profiling",
    "run_memory_scaling_analysis",
    # Visualization
    "plot_training_time_scaling",
    "plot_memory_usage",
    "plot_memory_scaling",
    "plot_accuracy_comparison",
    "plot_error_evolution",
    "plot_iso_parameter_comparison",
    "plot_approximation_analysis",
]
