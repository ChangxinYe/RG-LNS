# Experiment artifacts

The Python modules in this directory are the runnable experiment entry points:

- `ablation.py`: RG-LNS ablations;
- `sensitivity.py`: hyperparameter sensitivity;
- `runtime.py`: runtime summaries.

These drivers share the dataset and input-weight locations in
[`paths.py`](../paths.py), which defaults to released weights under
`checkpoints/compatibility/`, using the layout in the root README. Custom
weights can still be passed through the checkpoint arguments.

## Standalone comparisons

Normal `evaluate.py ... rg-lns` runs already report both **Baseline** and
**Baseline + RG-LNS**. No standalone comparison is needed to reproduce those
two results. The nine complete evaluation entry points remain under
`evaluation/commands/`.

For a targeted baseline-only check, use the optional modules under
`evaluation/baselines/`, for example:

```bash
python -m evaluation.baselines.evaluate_gap_gallagher --dataset GAP-5 --checkpoint checkpoints/compatibility/gap5_best.pth
```

Single-path greedy comparisons are isolated under `experiments/comparisons/`:

```bash
python -m experiments.comparisons gap gallagher --dataset GAP-5 --checkpoint checkpoints/compatibility/gap5_best.pth
```

The comparison dispatcher accepts the same three dataset families and three
initial solvers as normal evaluation. Append `--help` after dataset and solver
to inspect a comparison's options. These are experimental variants, not extra
stages that users must run before or after RG-LNS. Ablation, sensitivity, and
runtime entry points remain unchanged.

## Failure analysis

`failure_analysis.ipynb` is a runnable companion for the LSEJ failure analysis.
It is distributed without saved outputs. Install its optional dependencies
from the repository root:

```bash
python -m pip install -r requirements-notebook.txt
```

After preparing ImageNet-LSEJ and its released erode2 checkpoint, generate
paired results for all three initial solvers from the repository root:

```bash
python evaluate.py ImageNet-LSEJ-10x10-2px gallagher rg-lns --split test --checkpoint checkpoints/compatibility/lsej_grid10_erode2_best.pth --save-predictions true --save-diagnostics true --output-dir results/runs/failure_analysis/gallagher/lsej_grid10_erode2
python evaluate.py ImageNet-LSEJ-10x10-2px pomeranz rg-lns --split test --checkpoint checkpoints/compatibility/lsej_grid10_erode2_best.pth --save-predictions true --save-diagnostics true --output-dir results/runs/failure_analysis/pomeranz/lsej_grid10_erode2
python evaluate.py ImageNet-LSEJ-10x10-2px linear-programming rg-lns --split test --checkpoint checkpoints/compatibility/lsej_grid10_erode2_best.pth --save-predictions true --save-diagnostics true --output-dir results/runs/failure_analysis/lp/lsej_grid10_erode2
python -m jupyterlab experiments/failure_analysis.ipynb
```

Run all notebook cells in order. The setup cell exposes `TASKS`, `DATA_ROOT`,
`CHECKPOINTS`, and `RESULT_ROOT`; update them together when analyzing other
tasks or local paths. All solvers must use the same test samples, checkpoint,
and RG-LNS structural parameters. Each RG-LNS archive already includes its
paired baseline predictions, so separate initial-only runs are unnecessary.

The notebook compares **Baseline** with **Baseline + RG-LNS**. Optional
environment settings use the `RG_LNS_ANALYSIS_` prefix, including
`RG_LNS_ANALYSIS_DATA_ROOT`, `RG_LNS_ANALYSIS_RESULT_ROOT`, and
`RG_LNS_ANALYSIS_MAX_FAILED`; earlier environment names are still accepted.
`--save-diagnostics true` exports search records for detailed inspection.
It does not change the predictions used by this analysis.

Full-population statistics require complete test splits. For a workflow
check, add `--max-samples N` to each evaluation command and set
`MAX_FAILED_SAMPLES_PER_TASK` to a positive integer in the notebook. This
mode skips population-level conclusions and must not be used for paper
tables. To isolate such runs, set `RG_LNS_ANALYSIS_RESULT_ROOT` to another
local directory before launching Jupyter and use that directory for the
evaluation outputs as well.

Prediction archives, completion details, and analysis caches stay under
`results/runs/`, which is ignored by Git. They are not included in the compact
records under `results/reference/`.
