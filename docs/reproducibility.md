# Reproducibility notes

Normal evaluation compares two results on the same samples and frozen ViT-T
compatibility model:

- **Baseline (ViT-T + solver)**: the original layout from Gallagher, Pomeranz,
  or LP.
- **Baseline + RG-LNS**: the final layout after applying the complete RG-LNS
  method to that baseline.

RG-LNS automatically completes partial initial layouts before reliable-region
selection, translation, and beam completion. Candidate selection uses the
compatibility objective, never ground-truth positions. These operations are
internal parts of one method; they do not require separate evaluation runs.

The implementation is organized into `compatibility/` for visual compatibility,
`initial_solvers/` for initial layouts and partial-layout completion, and
`rg_lns/` for layout refinement.

## Local directory contract

Datasets are never committed. The default layout is:

```text
datasets/
├── GAP_fast/
│   ├── GAP-3/{train,val,test}/
│   └── GAP-5/{train,val,test}/
├── MET_Dataset/
│   ├── JPwLEG-3/                  # optional; not reported in the paper
│   └── JPwLEG-5/
└── ImageNet_LSEJ/
    ├── configs/
    ├── images/
    ├── permutations/
    ├── splits/
    └── lsej_dataloader.py
```

Training and evaluation accept `--data-root` for data stored elsewhere.
Point it to `GAP_fast/`, `MET_Dataset/`, or `ImageNet_LSEJ/`, respectively,
while preserving the subdirectory layout above. The ImageNet-LSEJ adapter is
imported from `lsej_dataloader.py` in the selected root.

For paper-level experiments, ablation and runtime accept `--gap-data-root`,
`--jpwleg-data-root`, and `--lsej-data-root`. Sensitivity accepts repeatable
`--data-root FAMILY=PATH` arguments, where `FAMILY` is `gap`, `jpwleg`, or `lsej`.

Learned weights are also local:

```text
checkpoints/
├── pretrained/
│   └── vit_tiny_patch16_224_augreg_in21k_ft_in1k_handwritten.pth
├── compatibility/
│   ├── gap3_best.pth
│   ├── gap5_best.pth
│   ├── jpwleg5_best.pth
│   ├── lsej_grid10_erode2_best.pth
│   └── lsej_grid10_erode5_best.pth
├── train_gap/<dataset>/<run>/checkpoints/       # generated training runs
├── train_jpwleg/<dataset>/<run>/checkpoints/    # generated training runs
└── train_lsej/<task>/<run>/checkpoints/        # generated training runs
```

Every evaluator accepts `--checkpoint`, so the directory convention is not a
hard dependency when reproducing from an archived checkpoint.

## GAP data preparation

Download `GAP_fast.zip` from
[GAP-fast on Hugging Face](https://huggingface.co/datasets/changxinye/GAP-fast/tree/main)
and extract it into `datasets/` under the repository root. The archive contains
a `GAP_fast/` directory; avoid creating an extra nested `GAP_fast/GAP_fast/`.
The resulting paths must be `datasets/GAP_fast/GAP-3/{train,val,test}/` and
`datasets/GAP_fast/GAP-5/{train,val,test}/`, with `puzzles.h5` and
`labels_indices.h5` in each split. No conversion is needed after extraction.

Alternatively, download the original dataset from
[Ofirish/GAP](https://huggingface.co/datasets/Ofirish/GAP) and place its
`GAP-3/` and `GAP-5/` directories under `datasets/GAP_Download/GAP/`. After
installing the requirements, run the following from the repository root:

```bash
python scripts/repack_gap_hdf5.py --source-root datasets/GAP_Download/GAP --output-root datasets/GAP_fast
```

Always specify both root arguments: the unchanged script defaults are based
on its own location, not the repository root. It preserves puzzle arrays and
index labels while rewriting their HDF5 storage into a contiguous,
uncompressed layout. The complete output is approximately 44.6 GB; retain
the original data and allow additional free space for conversion. The script
checks available space and refuses existing output files unless `--overwrite`
is explicitly supplied.

The GAP loader requires this repacked layout and does not fall back to the
original compressed files. GAP remains subject to the original dataset's
CC BY-NC 4.0 license and usage terms; please cite the original GAP paper.
The repository's MIT license does not relicense the dataset.

## Exact evaluation paths

The root dispatcher is a thin wrapper. For example:

```bash
python evaluate.py gap gallagher rg-lns --checkpoint CHECKPOINT
```

dispatches to:

```bash
python -m evaluation.commands.evaluate_gap_rg_lns --checkpoint CHECKPOINT
```

The same pattern covers three datasets (`gap`, `jpwleg`, `lsej`) and three
initial solvers (`gallagher`, `pomeranz`, `linear-programming`). The `rg-lns`
mode already reports both the baseline and the refined result. Standalone
`initial` and `greedy-rg-lns` modes remain available for targeted comparisons.

## Evaluation outputs

The terminal shows the run configuration followed by **Baseline (ViT-T + solver)**
and **Baseline + RG-LNS**. Progress metrics are cumulative over the samples
evaluated so far. `summary.txt` additionally records the exact sample range,
evaluation settings, baseline-to-final improvements and degradations, runtime
statistics, saved output files, and **Runtime Environment**. GAP summaries also
retain true-neighbor MRR, mean rank, and Recall@K. Saved visualizations show
`Input`, `Baseline`, `Baseline + RG-LNS`, and `Ground truth`; colored boxes
indicate correct (green) or incorrect (red) piece positions.

Normal runs save `metrics.json`, `sample_results.csv`, optional
`predictions.npz` and `visuals/`, and baseline-to-final repair/degradation lists.
`metrics.json` retains intermediate metrics, search diagnostics, stage timings,
and the runtime environment for reproducibility.

The summary's runtime section reports compatibility scoring, the initial solver,
RG-LNS refinement (including partial-layout completion), and total inference.
Its per-puzzle statistics exclude warmup samples; evaluation elapsed time also
includes loop overhead such as metrics and visualization. Detailed internal
stage timings remain available in `metrics.json`.

For detailed candidate and round records, add `--save-diagnostics true`.
This also writes `round_results.csv`, `candidates.csv`, `pose_results.csv`,
`completion_results.csv`, and single-path comparison lists. The existing
`--save-completion-details` argument is an alias for the same option.
Diagnostic export does not change the selected layout or evaluation metrics.

Existing JSON/NPZ keys are retained so analysis scripts can read earlier runs.
In `metrics.json`, `baseline` is the original baseline, `s1a4` is the final
RG-LNS result, and the other stage sections are diagnostic records.

## Verification policy

Refactoring beyond imports and paths should only be accepted after comparing:

- per-sample compatibility tensors or their hashes;
- initial piece-to-position predictions;
- RG-LNS selected components, translations, completion candidates, and final
  predictions;
- aggregate PA, AA, HA, VA, SRA, and NA.

The compact records under `results/reference/` provide aggregate reference
results. Generated runs go to `results/runs/`, which is ignored by Git.

## Failure-analysis notebook

`experiments/failure_analysis.ipynb` reads paired LSEJ results generated by the
public evaluation commands. See the [failure-analysis instructions](../experiments/README.md#failure-analysis)
for optional dependencies, input-generation commands, and notebook settings.
The notebook is distributed without saved outputs; population-level results
require complete test-split archives for all three solvers. Predictions,
completion details, and analysis caches belong under `results/runs/`, not in
the compact records under `results/reference/`.
