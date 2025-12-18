"""
Configuration for Task 8 experiments.
"""

from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
FIGURES_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Dataset configurations
DATASET_SIZES = {
    "1K": 1000,
    "5K": 5000,
    "10K": 10000,
}

# Training hyperparameters
DEFAULT_HYPERPARAMS = {
    "hidden_layers": (64, 32, 16),
    "learning_rate": 1e-3,
    "max_epochs": 100,
    "batch_size": 256,
    "patience": 20,
    "weight_decay": 0.0,
    "random_state": 42,
    "verbose": False,
}

# Experimental settings
N_RUNS = 5  # Number of runs for statistical significance
RANDOM_SEED = 42

# Weights & Biases (optional)
USE_WANDB = False  # Set to True to enable wandb logging
WANDB_PROJECT = "fivedreg-task8"

# Ground truth function coefficients
GROUND_TRUTH_COEFFS = {
    "x1": 2.0,
    "x2": -1.5,
    "x3": 3.0,
    "x4_x5": 0.5,
}

# Iso-parameter architectures
ISO_PARAM_ARCHITECTURES = {
    "Balanced": (64, 32, 16),
    "Wide-Shallow": (52, 52),
    "Deep-Narrow": (24, 24, 24, 24),
}

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
