# Experiment artifacts

The Python modules in this directory are the runnable experiment entry points:

- `ablation.py`: RG-LNS ablations;
- `sensitivity.py`: hyperparameter sensitivity;
- `runtime.py`: runtime summaries.

`failure_analysis.ipynb` is the executed notebook used during the paper study.
It is kept for inspection and provenance, including its original outputs. The
notebook itself still records private-workspace paths; these are not imported
or used by the public training and evaluation commands.

Rerunning that analysis requires full per-sample LSEJ predictions and
completion details. Generate those locally into `results/runs/` and update a
working copy of the notebook to point to that run. These bulky artifacts are
not included in `results/reference/` or tracked by Git.
