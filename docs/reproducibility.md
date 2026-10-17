# Reproducibility notes

Supplementary data-preparation and output details for the
[Quick Start](../README.md#quick-start). The
[Code Layout](../README.md#code-layout) and
[Checkpoints](../README.md#checkpoints) sections describe the default directories
and released weights.

## Custom data and checkpoint paths

Set machine-wide defaults in [`paths.py`](../paths.py). Its three dataset
roots and two input-weight directories are shared by training, evaluation,
the experiment drivers, and the failure-analysis notebook. For example:

```python
GAP_DATA_ROOT = Path("E:/datasets/GAP_fast")
COMPATIBILITY_ROOT = Path("E:/weights/rg_lns")
```

Use absolute paths for external storage or `PROJECT_ROOT / "..."` for paths
relative to the repository. Keep the released weight filenames unchanged.
These settings select inputs; training outputs and evaluation results retain
their existing locations unless `--output-dir` is supplied.
Restart the notebook kernel after changing `paths.py`.

Command-line path arguments override the shared defaults. For training and
evaluation, point `--data-root` to the appropriate family root:

| Dataset family | Point `--data-root` to the directory containing |
| :-- | :-- |
| GAP | `GAP-3/` and/or `GAP-5/` |
| JPwLEG | `JPwLEG-5/` (or `JPwLEG-3/` for optional experiments) |
| ImageNet-LSEJ | `lsej_dataloader.py`, `configs/`, `images/`, `permutations/`, and `splits/` |

The ImageNet-LSEJ adapter is imported from `lsej_dataloader.py` in the selected
root. Use `--checkpoint PATH` to select weights stored anywhere locally.

For paper-level experiments, ablation and runtime accept `--gap-data-root`,
`--jpwleg-data-root`, and `--lsej-data-root`. Sensitivity accepts repeatable
`--data-root FAMILY=PATH` arguments, where `FAMILY` is `gap`, `jpwleg`, or `lsej`.

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

Always specify both root arguments: the script defaults are based
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

## Evaluation outputs

Each normal evaluation reports **Baseline (ViT-T + solver)** and
**Baseline + RG-LNS**. Progress metrics are cumulative over the samples evaluated
so far, not the metrics of the most recent sample.

| Output | Contents |
| :-- | :-- |
| `summary.txt` | Run inputs, both result groups, improvement/degradation counts, runtime statistics, output locations, evaluation settings, and Runtime Environment; GAP also includes neighbor MRR, mean rank, and Recall@K |
| `metrics.json` | Machine-readable results, full configuration, diagnostic statistics, internal stage timings, and runtime environment |
| `sample_results.csv` | Per-sample metrics and diagnostic fields |
| `predictions.npz` | Saved layouts and targets; controlled by `--save-predictions` |
| `visuals/` | Input, Baseline, Baseline + RG-LNS, and ground truth; green/red boxes mark correct/incorrect absolute piece positions; controlled by `--save-visuals` |
| `pa_repaired_images.txt` / `pa_broken_images.txt` | Samples changed from failed to perfect layouts, or from perfect to failed layouts, relative to the baseline |

Use `--output-dir PATH` to choose a destination. New runs otherwise go under
`results/runs/`, separately from the archived records in `results/reference/`.

For programmatic access in `metrics.json`, `baseline` is the original baseline
and `s1a4` is the final RG-LNS result; other stage sections are diagnostic records.

### Runtime

The summary's runtime section reports compatibility scoring, the initial solver,
RG-LNS refinement (including partial-layout completion), and total inference.
Its per-puzzle statistics exclude warmup samples; evaluation elapsed time also
includes loop overhead such as metrics and visualization. Detailed internal
stage timings remain available in `metrics.json`.

### Detailed diagnostics

For detailed candidate and round records, add `--save-diagnostics true`.
This also writes `round_results.csv`, `candidates.csv`, `pose_results.csv`,
`completion_results.csv`, and single-path comparison lists. The existing
`--save-completion-details` argument is an alias for the same option.
Diagnostic export does not change the selected layout or evaluation metrics.

## Optional experiments

See the [standalone comparisons](../experiments/README.md#standalone-comparisons)
and [failure-analysis notebook](../experiments/README.md#failure-analysis)
instructions for optional workflows. The notebook's population-level statistics
require complete test-split archives for all three initial solvers; bounded
workflow checks must not be reported as full-population results.
