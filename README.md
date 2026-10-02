# RG-LNS

**Reliability-Guided Large Neighborhood Search for Eroded Jigsaw Puzzle Reassembly**

**Paper status:** Under review at ICASSP 2027.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RG-LNS formulates the correction of assembled eroded-jigsaw layouts as a
separate combinatorial optimization problem under fixed directional
compatibility. Our framework decouples visual perception from spatial
reasoning: a ViT-Tiny (ViT-T) model learns directional compatibility between
pieces, while RG-LNS corrects the resulting layouts without retraining the
visual model.

We construct three two-stage baselines by pairing the same frozen ViT-T
compatibility with Gallagher, Pomeranz, and linear programming (LP) as layout
optimizers. Experiments are conducted on GAP, JPwLEG-5, and our ImageNet
Large-Scale Eroded Jigsaw (ImageNet-LSEJ) dataset.

> **Release status.** The paper-facing implementation is organized by method
> stage. The numerical training, initial-solver, and RG-LNS routines retain the
> submitted implementation; only paths, module names, and public dispatchers
> were changed. See the [reproducibility notes](docs/reproducibility.md) and
> [source manifest](docs/source_manifest.md).

## Code layout

```text
compatibility/          Stage I: ViT compatibility model, scorer, and training
data_loaders/           GAP, JPwLEG, and ImageNet-LSEJ adapters
initial_solvers/        Gallagher, Pomeranz, LP, and partial-layout completion
rg_lns/                 Stage II: the proposed destroy/translate/repair search
evaluation/             Metrics plus explicit dataset/solver evaluation commands
experiments/            Ablation, sensitivity, runtime, and failure analysis
datasets/               Local data only; ignored by Git
checkpoints/            Local pretrained/trained weights; ignored by Git
results/reference/      Compact records from the submitted experiments
results/runs/           Newly generated runs; ignored by Git
```

Stage-I perception is deliberately outside `rg_lns/`. The latter contains
only the proposed layout-correction algorithm and consumes a fixed directional
compatibility tensor plus an initial layout.

## Installation

The verified development environment uses Python 3.10 and the package versions
listed in `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

For CUDA, install the PyTorch build matching the local driver before installing
the remaining requirements.

## Data and checkpoints

Keep uncompressed datasets at these exact local paths:

```text
datasets/GAP_fast/GAP-3
datasets/GAP_fast/GAP-5
datasets/MET_Dataset/JPLEG-5
datasets/ImageNet_LSEJ/configs
datasets/ImageNet_LSEJ/images
```

Training writes complete run directories below `checkpoints/train_<dataset>/`.
Copy the selected paper checkpoints to stable names under
`checkpoints/compatibility/`:

```text
checkpoints/
├── pretrained/
│   └── vit_tiny_patch16_224_augreg_in21k_ft_in1k_handwritten.pth
└── compatibility/
    ├── gap3_best.pth
    ├── gap5_best.pth
    ├── jpleg5_best.pth
    ├── lsej_grid10_erode2_best.pth
    └── lsej_grid10_erode5_best.pth
```

The pretrained file initializes Stage-I training. The five compatibility
checkpoints are the task-specific models used for paper evaluation and can be
loaded directly with `--checkpoint`; evaluation does not reload the pretrained
initialization.

## Main commands

```bash
# Stage-I compatibility training
python train_compatibility.py gap --dataset GAP-5
python train_compatibility.py jpleg --dataset jpleg5
python train_compatibility.py lsej --task grid10_erode2

# Initial layout only, or the final RG-LNS correction
python evaluate.py gap gallagher initial --checkpoint checkpoints/compatibility/gap5_best.pth
python evaluate.py gap gallagher rg-lns --checkpoint checkpoints/compatibility/gap5_best.pth
python evaluate.py lsej linear-programming rg-lns --checkpoint checkpoints/compatibility/lsej_grid10_erode2_best.pth

# Paper-level experiments
python reproduce.py ablation
python reproduce.py sensitivity
python reproduce.py runtime
```

Run the lightweight synthetic regression tests with:

```bash
python -m unittest discover -s tests -v
```

All original dataset/solver combinations remain available as explicit modules
under `evaluation/commands/`, which keeps each reported path independently
auditable. Compact submitted records are inventoried in
[`results/reference/README.md`](results/reference/README.md).

## Motivation

Even when a complete puzzle layout is incorrect, it can retain many correct
local relationships. Our empirical analysis identifies two dominant failure
modes.

### 1. Region shift

Some regions preserve the correct relative positions among their pieces but
are displaced as a whole. The cyan outline below marks the same region in the
initial layout and its target location.

<p align="center">
  <img src="assets/motivation_1.png" width="100%" alt="Example of region shift">
</p>

### 2. Local ambiguity

Multiple neighboring pieces can exhibit high visual similarity, making their
relative spatial positions difficult to infer. The reliability map below
highlights where the available neighboring relationships provide insufficient
support for an unambiguous arrangement.

<p align="center">
  <img src="assets/motivation_2.png" width="100%" alt="Example of local ambiguity">
</p>

## Method Overview

RG-LNS addresses these two failure modes through reliability-guided destroy
and repair.

1. **Directional compatibility.** The four edges of each piece are mapped to a
   shared canonical orientation and embedded by a frozen ViT-T model.
2. **Reliability-guided destroy.** Mutually strong piece matches supported by
   surrounding pieces are grouped into reliable regions. The internal
   arrangement of the largest reliable region is preserved, while its absolute
   location and the positions of the remaining pieces are released. This
   defines a structured large neighborhood.
3. **Region translation.** The repair operator explores every feasible
   translation of the preserved region, directly addressing region shifts.
4. **Beam-search completion.** For each translation, the remaining pieces are
   placed while multiple layout hypotheses are retained. Newly formed
   adjacencies provide richer compatibility information for distinguishing
   locally ambiguous arrangements.
5. **Evaluation and acceptance.** Completed layouts are evaluated by a fixed
   compatibility objective, and a candidate is accepted only when it strictly
   improves the current layout.

## Main Results

RG-LNS consistently improves all three ViT-T-based baselines across the three
datasets. Its best perfect accuracy (PA) exceeds the strongest competing result
by:

- **33.4 percentage points** on GAP-5 (33.7% vs. 0.3%);
- **28.2 percentage points** on JPwLEG-5 (60.7% vs. 32.5%); and
- **37.7 percentage points** on ImageNet-LSEJ (55.9% vs. 18.2%).

On the ImageNet-LSEJ failures used for our analysis, RG-LNS repairs
19.3–42.1% of failures attributed to region shift and 13.9–26.8% of those
attributed to local ambiguity across the three baselines.

## Failure Taxonomy for Table 1

This section documents the deterministic sample-level classification used to
produce Table 1 of the paper. The three labels are applied to all failed
predictions from each baseline and form a disjoint partition.

### Step 1: Absolute-position errors

For a complete failed sample $s$, let $\mathbf{x}_i$ and
$\mathbf{x}_i^{*}$ be the predicted and ground-truth grid coordinates of piece
$i$. The set of pieces at incorrect absolute positions is

```math
\mathcal{W}_s=\{i\in\mathcal{P}:\mathbf{x}_i\neq\mathbf{x}_i^{*}\}.
```

Only failed samples are classified, so $|\mathcal{W}_s|>0$ for every complete
prediction considered below.

### Step 2: Reliable components

For a currently adjacent pair $(i,j)$ in direction $d$, let
$r_{i\rightarrow j}^{d}$ be the rank of $j$ among the candidate neighbors of
$i$, where rank 1 is best. The adjacency is mutual Top-$K$ when

```math
\max(r_{i\rightarrow j}^{d},\;r_{j\rightarrow i}^{\bar d})\le K,
```

where $\bar d$ denotes the opposite direction. A current $2\times2$ block
provides cycle support only when all four perimeter adjacencies are mutual
Top-$K$. The reliability graph $G_s^{\mathrm{rel}}$ contains the perimeter
edges supported by at least one such complete cycle. Its connected components
with at least $M$ pieces form

```math
\mathcal{C}_s=
\{Q\in\mathrm{CC}(G_s^{\mathrm{rel}}):|Q|\ge M\}.
```

The experiments use $K=3$ and $M=4$.

### Step 3: Evidence for the two failure modes

For every piece $i$, define the translation needed to move it from its
predicted position to its target position as

```math
\boldsymbol{\Delta}_i=\mathbf{x}_i^{*}-\mathbf{x}_i.
```

A reliable component provides evidence of region shift when all of its pieces
share the same nonzero translation:

```math
\mathcal{C}_s^{\mathrm{shift}}=
\{Q\in\mathcal{C}_s:\exists\,\boldsymbol{\delta}\neq\mathbf{0},\;
\forall i\in Q,\;\boldsymbol{\Delta}_i=\boldsymbol{\delta}\}.
```

Let $\mathcal{R}_s$ contain the pieces covered by shifted reliable components,
and let $\mathcal{L}_s$ contain the pieces outside every reliable component:

```math
\mathcal{R}_s=\bigcup_{Q\in\mathcal{C}_s^{\mathrm{shift}}}Q,
\qquad
\mathcal{L}_s=\mathcal{P}\setminus\bigcup_{Q\in\mathcal{C}_s}Q.
```

We measure how much of the absolute-position error is explained by each type
of evidence:

```math
q_s^{\mathrm{shift}}=
\frac{|\mathcal{W}_s\cap\mathcal{R}_s|}{|\mathcal{W}_s|},
\qquad
q_s^{\mathrm{local}}=
\frac{|\mathcal{W}_s\cap\mathcal{L}_s|}{|\mathcal{W}_s|}.
```

### Step 4: Ordered sample-level classification

With the dominance threshold $\tau=0.4$, the failure label is assigned in the
following order:

```math
y_s=
\begin{cases}
\text{Region shift},
& q_s^{\mathrm{shift}}\ge\tau,\\[2pt]
\text{Local ambiguity},
& q_s^{\mathrm{shift}}<\tau\ \text{and}\ q_s^{\mathrm{local}}\ge\tau,\\[2pt]
\text{Other failure},
& \text{otherwise}.
\end{cases}
```

The ordered rule makes the two dominant categories mutually exclusive.
Predictions that do not define a complete grid permutation are assigned
directly to **Other failure**.

### Table 1 results

The following results are obtained on ImageNet-LSEJ ($10\times10$) with
2-pixel erosion. All three baselines use the same frozen ViT-T directional
compatibility.

| Failure type | Gallagher | Pomeranz | LP |
|:--|--:|--:|--:|
| Region shift | 202 (16.6%) | 107 (10.3%) | 135 (12.3%) |
| Local ambiguity | 731 (60.1%) | 742 (71.6%) | 823 (74.7%) |
| Other failures | 283 (23.3%) | 188 (18.1%) | 144 (13.1%) |
| **Total** | **1,216 (100%)** | **1,037 (100%)** | **1,102 (100%)** |

Displayed percentages are rounded to one decimal place.

## Benchmarks

- **[GAP-3 / GAP-5](https://github.com/OfirShahar/puzzle-flow-matching):**
  public $3\times3$ and $5\times5$ benchmarks with irregular eroded
  fragments.
- **[JPwLEG-5](https://drive.google.com/drive/folders/1MjPm7ar-u6H5WX6Bw2qshPiYPT_eQCZE):**
  a public $5\times5$ benchmark with large gaps, evaluated under the
  no-fixed-center protocol.
- **[ImageNet-LSEJ](https://github.com/ChangxinYe/ImageNet-LSEJ):** our
  $10\times10$ benchmark derived from ImageNet, with $50\times50$-pixel pieces
  under 2- and 5-pixel erosion.

## Acknowledgments

We thank the authors of the following projects for making their code publicly
available. Their implementations supported our ImageNet-LSEJ baseline
evaluation:

- [JigsawGAN](https://github.com/liru0126/JigsawGAN)
- [JPDVT](https://github.com/JinyangMarkLiu/JPDVT)
- [DiffAssemble](https://github.com/IIT-PAVIS/DiffAssemble)
- [FCViT](https://github.com/HiMyNameIsDavidKim/fcvit)
- [PuzzleFlow](https://github.com/OfirShahar/puzzle-flow-matching)

Other baselines were reproduced by following their original papers.

## Citation

Citation information will be added with the public paper release.

## License

This project is released under the [MIT License](LICENSE).
