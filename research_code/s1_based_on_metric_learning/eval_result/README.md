# Staged evaluation records

This directory preserves compact records used to verify the submitted paper.

## Included main experiments

- GAP-3 and GAP-5 with Gallagher, Pomeranz, and LP;
- JPwLEG-5 with Gallagher, Pomeranz, and LP;
- ImageNet-LSEJ with 2- and 5-pixel erosion and all three layout optimizers.

The submitted main runs are organized as
`rg_lns_<dataset>_<optimizer>/`. GAP and JPwLEG records use a
`submitted/` run directory. ImageNet-LSEJ records use `erode2/` and
`erode5/` to distinguish the two erosion settings. The complete checkpoint,
task, solver, and RG-LNS configuration remain recorded in each run's
`metrics.json` and `summary.txt`.

## Included analyses

- cumulative ablation for paper Table 4: `table4_ablation/submitted/`;
- final ImageNet-LSEJ hyperparameter study for paper Fig. 3:
  `fig3_sensitivity/submitted/`;
- runtime study for paper Table 7: `table7_runtime/submitted/`.

The compact copy contains 411 record files and occupies approximately 34.7 MB.
Detailed `completion_results.csv`, `pose_results.csv`, `candidates.csv`, and
`visuals/` directories are omitted; their original copies remain in the
research workspace. ImageNet-LSEJ 8-pixel and JPwLEG-3 results are also omitted
because they are not reported in the submitted paper.
