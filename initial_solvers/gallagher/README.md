# Gallagher initial solver

This module provides the Python assembly implementation of Gallagher's
CVPR 2012 method. RG-LNS supplies learned ViT-T directional compatibility
scores to this solver to obtain a baseline layout.

Run a baseline-to-final comparison from the repository root:

```bash
python evaluate.py gap gallagher rg-lns --dataset GAP-5 --checkpoint checkpoints/compatibility/gap5_best.pth
```

The run reports **Baseline (ViT-T + Gallagher)** and **Baseline + RG-LNS**.
Use `jpwleg` or `lsej` with the matching dataset/task and checkpoint to evaluate
the other benchmarks; see the [main README](../../README.md#quick-start).
The standalone `initial` mode evaluates only the baseline when needed.

`mgc.py` also retains the solver's original MGC scoring routines. They are
separate from the learned compatibility used in the RG-LNS evaluations.
Internal piece IDs are 1-based, with `0` denoting an empty cell; the evaluation
adapters convert these to the repository's piece-to-position representation.

The evaluation scripts report PA, AA, and SRA, with HA, VA, and NA where
supported. See the main README and [reproducibility guide](../../docs/reproducibility.md)
for metrics and dataset paths.
