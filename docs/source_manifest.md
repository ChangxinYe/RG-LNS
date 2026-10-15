# Source migration manifest

Migration date: 2026-10-01

Source workspace: `Jigsaw_Puzzles_V2/s1_based_on_metric_learning`

The release contains the recursive local-import closure of all paper-facing
training, evaluation, RG-LNS, ablation, sensitivity, runtime, and failure
analysis entry points. The main mapping is:

| Submitted workspace | Release location |
|---|---|
| `metric_vit_model.py`, `metric_scorer.py`, transforms and geometry | `compatibility/` |
| legacy compatibility-training entry points (`s1a_train_{gap,jpleg,lsej}.py`) | `compatibility/training/train_*.py` |
| `metric_*_data.py` | `data_loaders/` |
| Gallagher, Pomeranz, and LP assembly code | `initial_solvers/` |
| `single_component_e1.py` | `rg_lns/component_search.py` |
| `profiled_completion_e1.py` | `rg_lns/beam_search.py` |
| dataset evaluators | `evaluation/{gap,jpleg,lsej}.py` |
| former stage-labeled (`s1a*`, `s1a3*`, `s1a4*`) entry points | `evaluation/commands/` |
| ablation, sensitivity, runtime, and failure notebook | `experiments/` |
| compact `eval_result/` records | `results/reference/` |

Two required dependencies that previously lived elsewhere in the private
workspace are now self-contained:

- the handwritten ViT definition is `compatibility/vit.py`;
- the Gallagher MGC implementation is `initial_solvers/gallagher/mgc.py`.

JPwLEG array/label conversion is vendored in `data_loaders/jpleg_common.py`, so
the release no longer imports the Edge2Vec baseline tree.

Unused reinforcement-learning prototype modules were excluded. The only part
of that package required by the submitted pipeline—the deterministic
incomplete-layout completion routine—was retained as
`initial_solvers/partial_completion.py`.

Datasets, training logs, model checkpoints, caches, detailed visualization
folders, and unreported exploratory methods are deliberately excluded from
Git. The source migration does not modify or remove files in the original
workspace.

The public command names describe the operation directly (for example,
`evaluate_gap_greedy_rg_lns.py` and `evaluate_lsej_rg_lns.py`). Historical
stage labels are retained only inside machine-readable result fields and old
artifact names where they are needed to read previously generated outputs.
