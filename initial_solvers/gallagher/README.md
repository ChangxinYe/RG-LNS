# PuzzleDemoMGC CVPR 2012 Python Port

This is a Python reimplementation of the MATLAB demo in
`PuzzleDemoMGC_CVPR2012`.

The port keeps the original algorithm structure:

- split an image into square pieces
- optionally scramble positions and rotations
- optionally damage piece borders
- compute MGC pairwise compatibility scores
- greedily assemble pieces
- trim and fill holes
- render and evaluate the result

Example:

```bash
python demo.py --image ..\PuzzleDemoMGC_CVPR2012\DSC_1062.JPG --pp 56 --nc 12 --nr 9 --damage-pixels 4 --scramble-positions True --scramble-rotations False --seed 1
```

With unknown rotations enabled:

```bash
python demo.py --image ..\PuzzleDemoMGC_CVPR2012\DSC_1062.JPG --pp 56 --nc 12 --nr 9 --damage-pixels 4 --scramble-positions True --scramble-rotations True --seed 1
```

To reuse MATLAB's exact `ppp` and `randscram`, first run this in MATLAB:

```matlab
ExportShuffleForPython
```

Then run:

```bash
python demo_with_matlab_shuffle.py --shuffle-mat ..\PuzzleDemoMGC_CVPR2012\matlab_shuffle.mat
```

Outputs are written under `outputs/` by default.

## Single-image timing

Use `s1_mgc_single_image_infer.py` to solve one synthetically shuffled source
image and measure preprocessing, compatibility calculation, assembly, and total
inference time:

```bash
python s1_mgc_single_image_infer.py --image ..\PuzzleDemoMGC_CVPR2012\DSC_1062.JPG --pp 50 --nc 10 --nr 10 --damage-pixels 0 --scramble-rotations False
```

The script saves the shuffled, damaged, and solved images together with
`timing.json` and `prediction.npz` under `single_image_result/` by default.

To compare different puzzle sizes directly from the PNG visualizations under
`LSEJ_result`, edit `DEFAULT_GRID_SIZES` and `DEFAULT_NUM_IMAGES` at the top of
`s1_mgc_lsej_result_infer.py`, then run the file directly. No command-line
arguments are needed. The script extracts each visualization's Ground truth
panel and re-cuts it into every requested grid. The default grids are 3, 5, 10,
and 20 (9, 25, 100, and 400 pieces), with aggregate JSON/CSV timings written
under `LSEJ_result_grid_infer/`.

Notes:

- Piece IDs are kept 1-based internally to match the MATLAB code.
- Empty cells are represented by `0`, also matching the MATLAB code.
- The first version favors behavior parity and readability over speed.

## Evaluation metrics

The LSEJ and JPLEG evaluation scripts report the same six assembly metrics used
by the local TEN and Edge2Vec LSEJ evaluators:

- `PA`: fraction of puzzles whose every piece is in the correct position.
- `AA`: fraction of individual pieces in the correct absolute position.
- `HA`: direction-sensitive left-to-right adjacency accuracy.
- `VA`: direction-sensitive top-to-bottom adjacency accuracy.
- `SRA`: aggregate direction-sensitive spatial-relationship accuracy over the
  horizontal and vertical relationships.
- `NA`: direction-agnostic four-neighbor adjacency accuracy.

The JSON outputs include both the metric values and their correct/total counts.

## GAP evaluation

`s1_mgc_gap_test.py` evaluates MGC on the official shuffled GAP-3 and GAP-5
RGBA pieces stored in `datasets/GAP_fast`. Its top-level `DEFAULT_*` block is
ready for direct IDE execution; command-line arguments are optional overrides.
Transparent pixels are alpha-composited onto a white RGB background because
the original MGC implementation accepts three-channel pieces only.

The evaluator reports PA, AA, HA, VA, SRA, and NA; saves `metrics.json`,
`summary.txt`, optional `predictions.npz`, and up to 10 visualizations per
dataset under `GAP_result/` by default. It deliberately requires contiguous,
uncompressed `GAP_fast` files and does not silently fall back to the older
compressed dataset. The text and JSON summaries also record the host OS, CPU,
RAM, resolved execution device, GPU usage, Python/PyTorch versions, CUDA/cuDNN,
NVIDIA driver, and `CUDA_VISIBLE_DEVICES`, following the S1 evaluation format.
The LSEJ and JPLEG test summaries and JSON metrics record the same runtime
environment fields and identify their execution backend as NumPy CPU.
