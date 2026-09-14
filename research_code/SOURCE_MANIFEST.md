# Source snapshot manifest

Snapshot date: 2026-09-14

Source project: `Jigsaw_Puzzles_V2/s1_based_on_metric_learning`

Destination: `research_code/s1_based_on_metric_learning`

## Code selection

The copied Python files are the recursive local-import closure of the following
paper-facing entry points:

- `s1a_train_{gap,jpleg,lsej}.py`;
- the Gallagher, Pomeranz, and LP variants of `s1a_eval_*`;
- the Gallagher, Pomeranz, and LP variants of the final `s1a3_eval_*`
  component-reassembly programs;
- the nine `s1a4_eval_*_profiled_large_neighborhood_reassembly.py` programs;
- `s1a4_ablation_experiment.py`;
- `s1a4_hyperparameter_sensitivity_experiment.py`;
- `s1a4_runtime_experiment.py`.

The executed failure-analysis notebook
`s1a_s4a_eval_lsej_data_analysis.ipynb` is also retained for provenance. Its
outputs and machine-specific paths must be cleaned before it is presented as a
public reproduction notebook.

## Deliberately excluded

- ImageNet-LSEJ experiments with 8-pixel erosion;
- JPwLEG-3 experiments;
- exploratory `s2`--`s10` methods not reported in the submission;
- `logs/`, `prepared_data/`, `mining_result/`, caches, and temporary figures;
- datasets, pretrained checkpoints, and precomputed compatibility matrices;
- the locally stored PDF copy of the Pomeranz paper.

No source file in the original research project was modified or removed while
creating this snapshot.
