# Source components

| Component | Location |
| :-- | :-- |
| ViT architecture, compatibility encoder, scoring, and training | `compatibility/` |
| GAP, JPwLEG, and ImageNet-LSEJ adapters | `data_loaders/` |
| Gallagher, Pomeranz, LP, and partial-layout completion | `initial_solvers/` |
| Reliable-region selection and layout search | `rg_lns/` |
| Complete Baseline-to-RG-LNS evaluation commands | `evaluation/commands/` |
| Shared evaluation arguments, checkpoint lookup, and utilities | `evaluation/common/` |
| Optional standalone initial-solver baselines | `evaluation/baselines/` |
| Dataset metrics and reporting | `evaluation/` |
| Ablation, sensitivity, runtime, and failure analysis | `experiments/` |
| Optional standalone greedy comparisons | `experiments/comparisons/` |
| Compact reference results | `results/reference/` |

The repository includes its own ViT implementation and Gallagher assembly
code. JPwLEG loading and label conversion are implemented in
`data_loaders/jpwleg_common.py`; ImageNet-LSEJ uses the official adapter from
the selected dataset root.

Datasets, checkpoints, generated results, and analysis caches remain local and
are ignored by Git. Reference records retain their original field names and
artifact identifiers so they remain usable for regression comparisons.
