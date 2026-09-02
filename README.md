# RG-LNS

**Reliability-Guided Large Neighborhood Search for Eroded Jigsaw Puzzle Reassembly**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RG-LNS is a plug-and-play layout refinement method for eroded square jigsaw puzzles. It starts from the output of an existing solver, preserves reliable internal structures, destroys unreliable placement decisions, and repairs the layout through feasible component translations and beam-search completion. A candidate is accepted only when it strictly improves the frozen full-layout compatibility objective.

The method is evaluated with Gallagher, Pomeranz, and LP initial solvers on ImageNet-LSEJ, GAP, and JPwLEG. We also provide a strong ViT-Tiny directional compatibility baseline trained with metric learning.

> **Release status.** The source code, configurations, pretrained checkpoints, and dataset instructions are being organized for public release in this repository.

## Highlights

- **Reliability-guided destroy:** reliable components are extracted from mutual Top-K adjacencies with complete $2\times2$ cycle support. Their internal relations are preserved, while their absolute positions and the remaining placements are released.
- **Structured large neighborhood:** each reliable component is evaluated at its feasible translated positions rather than being fixed at the position selected by the initial solver.
- **Beam-search repair:** the remaining pieces are completed along multiple promising repair paths.
- **Full-layout acceptance:** only candidates that strictly improve the same frozen compatibility objective are accepted.
- **Cross-solver refinement:** RG-LNS can refine different initial solvers without replacing them.

## Motivation

Existing solvers often recover many correct local adjacencies but still fail to organize them into the correct complete layout. Our failure analysis reveals two dominant patterns.

### 1. Rigid displacement of reliable components

The reliable component in the initial result preserves its internal arrangement but appears at an incorrect global position. The cyan outline marks the same component in the solver result and its target position.

<p align="center">
  <img src="assets/motivation_1.png" width="100%" alt="Rigidly displaced reliable component and its refinement">
</p>

### 2. Errors concentrated in low-reliability regions

In many failed reconstructions, most of the puzzle is already supported by reliable adjacencies, while the remaining errors are concentrated in regions with weak mutual rankings or without complete cycle support.

<p align="center">
  <img src="assets/motivation_2.png" width="100%" alt="Errors concentrated in low-reliability regions">
</p>

## Extended Failure Taxonomy for Table 1

This section gives the complete sample-level classification protocol used to produce Table 1 of the paper. The three categories are applied to **all failed predictions** of each initial solver and therefore form a disjoint partition whose counts sum to the total number of failures.

### Step 1: Absolute-position errors

For a complete failed sample $s$, let $\mathbf{x}_i$ and $\mathbf{x}_i^{*}$ be the predicted and ground-truth grid coordinates of piece $i$. The set of pieces at incorrect absolute positions is

```math
\mathcal{W}_s=\{i\in\mathcal{P}:\mathbf{x}_i\neq\mathbf{x}_i^{*}\}.
```

Only failed samples are classified, so $|\mathcal{W}_s|>0$ for every complete prediction considered below.

### Step 2: Reliable components

For a currently adjacent pair $(i,j)$ in direction $d$, let $r_{i\rightarrow j}^{d}$ be the rank of $j$ among the candidate neighbors of $i$, where rank 1 is best. The adjacency is mutual Top-K when

```math
\max(r_{i\rightarrow j}^{d},\;r_{j\rightarrow i}^{\bar d})\le K.
```

where $\bar d$ denotes the opposite direction. A current $2\times2$ block provides cycle support only when all four perimeter adjacencies are mutual Top-K. The reliable graph $G_s^{\mathrm{rel}}$ contains the perimeter edges supported by at least one such complete cycle. Its connected components with at least $M$ pieces form

```math
\mathcal{C}_s=
\{Q\in\mathrm{CC}(G_s^{\mathrm{rel}}):|Q|\ge M\}.
```

The experiments use $K=3$ and $M=4$.

### Step 3: Rigidly displaced and low-reliability pieces

For every piece $i$, define the translation needed to move it from its predicted position to its target position as

```math
\boldsymbol{\Delta}_i=\mathbf{x}_i^{*}-\mathbf{x}_i.
```

A reliable component is a rigidly displaced component when all of its pieces share the same nonzero translation:

```math
\mathcal{C}_s^{\mathrm{rig}}=
\{Q\in\mathcal{C}_s:\exists\,\boldsymbol{\delta}\neq\mathbf{0},\;
\forall i\in Q,\;\boldsymbol{\Delta}_i=\boldsymbol{\delta}\}.
```

The pieces covered by rigidly displaced components and the pieces outside all reliable components are respectively

```math
\mathcal{R}_s=\bigcup_{Q\in\mathcal{C}_s^{\mathrm{rig}}}Q,
\qquad
\mathcal{L}_s=\mathcal{P}\setminus\bigcup_{Q\in\mathcal{C}_s}Q.
```

We then measure how much of the absolute-position error is explained by each mechanism:

```math
q_s^{\mathrm{rig}}=
\frac{|\mathcal{W}_s\cap\mathcal{R}_s|}{|\mathcal{W}_s|},
\qquad
q_s^{\mathrm{low}}=
\frac{|\mathcal{W}_s\cap\mathcal{L}_s|}{|\mathcal{W}_s|}.
```

### Step 4: Ordered sample-level classification

With the dominance threshold $\tau=0.4$, the failure label is assigned in the following order:

```math
y_s=
\begin{cases}
\text{Rigid-displacement-dominant},
& q_s^{\mathrm{rig}}\ge\tau,\\[2pt]
\text{Low-reliability-error-dominant},
& q_s^{\mathrm{rig}}<\tau\ \text{and}\ q_s^{\mathrm{low}}\ge\tau,\\[2pt]
\text{Other/mixed failure},
& \text{otherwise}.
\end{cases}
```

The ordered rule makes the two dominant categories mutually exclusive. Predictions that do not define a complete grid permutation are assigned directly to **Other/mixed failure**.

### Table 1 results

The following results are obtained on ImageNet-LSEJ-10 with 2-pixel erosion. All three initial solvers use our frozen ViT-Tiny directional compatibility.

| Failure type | Gallagher | Pomeranz | LP |
|:--|--:|--:|--:|
| Rigid-displacement-dominant | 202 (16.6%) | 107 (10.3%) | 135 (12.3%) |
| Low-reliability-error-dominant | 731 (60.1%) | 742 (71.6%) | 823 (74.7%) |
| Other/mixed failure | 283 (23.3%) | 188 (18.1%) | 144 (13.1%) |
| **All failed predictions** | **1,216 (100%)** | **1,037 (100%)** | **1,102 (100%)** |

Displayed percentages are rounded to one decimal place.

## Method at a Glance

1. Transform the four edges of every piece into a shared canonical orientation.
2. Extract directional embeddings with the trained ViT-Tiny encoder and construct the compatibility tensor.
3. Obtain an initial layout from an existing solver.
4. Build reliable components and apply the reliability-guided destroy operator.
5. Repair each structured neighborhood with feasible component translations and beam-search completion.
6. Evaluate complete layouts and accept a candidate only if it strictly improves the frozen objective.

## Benchmarks

- **[ImageNet-LSEJ](https://github.com/ChangxinYe/ImageNet-LSEJ):** our large-scale eroded-puzzle benchmark with controlled grid sizes and erosion levels.
- **[GAP-3 / GAP-5](https://github.com/OfirShahar/puzzle-flow-matching):** public irregularly eroded square-puzzle benchmarks released with PuzzleFlow.
- **[JPwLEG-3 / JPwLEG-5](https://drive.google.com/drive/folders/1MjPm7ar-u6H5WX6Bw2qshPiYPT_eQCZE):** public large-gap square-puzzle benchmarks, evaluated under the unfixed-center protocol in our experiments.

## Acknowledgments

We sincerely thank the authors of the following projects for making their code publicly available. Their official repositories supported our ImageNet-LSEJ-10 baseline evaluation:

- [JigsawGAN](https://github.com/liru0126/JigsawGAN)
- [JPDVT](https://github.com/JinyangMarkLiu/JPDVT)
- [DiffAssemble](https://github.com/IIT-PAVIS/DiffAssemble)
- [FCViT](https://github.com/HiMyNameIsDavidKim/fcvit)
- [PuzzleFlow](https://github.com/OfirShahar/puzzle-flow-matching)

Baselines for which we could not verify an official public implementation were reproduced or adapted from their papers and are therefore not linked here.

## Citation

Citation information will be added with the public paper release.

## License

This project is released under the [MIT License](LICENSE).
