# Reproducibility notes

The public layout separates the two stages described in the paper without
rewriting their numerical implementation:

1. `compatibility/` trains or loads the frozen directional compatibility
   model and produces an `N x N x 4` distance tensor.
2. `initial_solvers/` produces the initial piece-to-position layout. If it is
   incomplete, `initial_solvers/partial_completion.py` completes it before
   RG-LNS begins.
3. `rg_lns/` applies reliability-guided component selection, legal rigid
   translations, profiled beam completion, the fixed E1 objective, and the
   strict-improvement acceptance rule.

The move from the submitted workspace was intentionally mechanical. Function
bodies for the paper-facing training, solver, and evaluation paths were kept;
module imports and repository-relative paths were changed. The former
`s1a*`, `s1a3*`, and `s1a4*` entry-point names were replaced by descriptive
dataset/solver names under `evaluation/commands/`.

## Local directory contract

Datasets are never committed. The loaders expect:

```text
datasets/
├── GAP_fast/
│   ├── GAP-3/{train,val,test}/
│   └── GAP-5/{train,val,test}/
├── MET_Dataset/
│   ├── JPLEG-3/                  # optional; not reported in the paper
│   └── JPLEG-5/
└── ImageNet_LSEJ/
    ├── configs/
    ├── images/
    ├── permutations/
    ├── splits/
    └── lsej_dataloader.py
```

Learned weights are also local:

```text
checkpoints/
├── pretrained/
│   └── vit_tiny_patch16_224_augreg_in21k_ft_in1k_handwritten.pth
├── compatibility/
│   ├── gap3_best.pth
│   ├── gap5_best.pth
│   ├── jpleg5_best.pth
│   ├── lsej_grid10_erode2_best.pth
│   └── lsej_grid10_erode5_best.pth
├── train_gap/<dataset>/<run>/checkpoints/       # generated training runs
├── train_jpleg/<dataset>/<run>/checkpoints/    # generated training runs
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

If `GAP_fast.zip` is not yet listed under **Files and versions**, use the
conversion method below or check back after the archive is uploaded.

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

The same pattern covers three datasets (`gap`, `jpleg`, `lsej`), three initial
solvers (`gallagher`, `pomeranz`, `linear-programming`), and three evaluation
modes (`initial`, `greedy-rg-lns`, `rg-lns`). Keeping these commands explicit
avoids silently sharing dataset-specific label conversion or solver wrappers.

## Verification policy

Refactoring beyond imports and paths should only be accepted after comparing:

- per-sample compatibility tensors or their hashes;
- initial piece-to-position predictions;
- RG-LNS selected components, translations, completion candidates, and final
  predictions;
- aggregate PA, AA, HA, VA, SRA, and NA.

The compact submitted records under `results/reference/` are the regression
oracle for aggregate results. Generated runs go to `results/runs/`, which is
ignored by Git.

## Failure-analysis notebook

`experiments/failure_analysis.ipynb` is retained as an executed provenance
artifact, so its figures and conclusions remain inspectable without shipping
the large per-sample outputs. Its embedded code reflects the original private
workspace and is not part of the public execution path. To rerun the analysis,
first generate a full LSEJ run with prediction and completion-detail saving
enabled, then point a working copy of the notebook at that run under
`results/runs/`. The compact records in `results/reference/` intentionally do
not contain those large prediction artifacts.
