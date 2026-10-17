"""Shared default input paths for training, evaluation, and experiments.

Edit the five roots below for this machine. Use an absolute Path for external
storage, or PROJECT_ROOT / "..." for a repository-relative location.
Command-line path arguments override these defaults.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset roots, not individual splits or task directories.
GAP_DATA_ROOT = PROJECT_ROOT / "datasets" / "GAP_fast"
JPWLEG_DATA_ROOT = PROJECT_ROOT / "datasets" / "MET_Dataset"
LSEJ_DATA_ROOT = PROJECT_ROOT / "datasets" / "ImageNet_LSEJ"

# Input weights; keep the released filenames inside these directories.
COMPATIBILITY_ROOT = PROJECT_ROOT / "checkpoints" / "compatibility"
PRETRAINED_ROOT = PROJECT_ROOT / "checkpoints" / "pretrained"
