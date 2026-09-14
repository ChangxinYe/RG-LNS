# Research-code snapshot

This directory contains a minimally reduced snapshot of the research code used
for the ICASSP 2027 submission. The internal module names and relative imports
are intentionally preserved so that code cleanup does not silently change the
reported results.

The snapshot is currently a staging copy rather than the final public API. It
still requires environment documentation, path normalization, checkpoint and
dataset download instructions, and end-to-end reproduction tests.

## Included code

The snapshot currently contains:

- the ViT-T directional compatibility model, transforms, loss, scorer, dataset
  loaders, and evaluators;
- the Gallagher, Pomeranz, and LP layout-optimizer interfaces used by the
  paper;
- the original RG-LNS implementation in
  `assembly_solvers/our_iterative_component_reassembly/`;
- the Stage-I training and evaluation entry points for GAP, JPwLEG-5, and
  ImageNet-LSEJ;
- the nine final RG-LNS evaluation entry points covering three datasets and
  three layout optimizers;
- the ablation, hyperparameter, runtime, and failure-analysis programs.

Exploratory methods unrelated to the submitted paper, including the `s2`--`s10`
experiments, training logs, caches, prepared datasets, and mining outputs, are
not included.

## Evaluation records

Selected raw evaluation records are staged under
`s1_based_on_metric_learning/eval_result/`. They cover the settings reported in
the paper and currently occupy approximately 34.7 MB. Detailed completion,
pose, and candidate traces, together with per-sample visualization folders,
have been omitted from this repository copy. See `eval_result/README.md` for
the included runs.

## Reproducibility policy

The first public version will prioritize exact reproduction over refactoring.
Changes to paths, command-line interfaces, or internal module structure should
be made only after the original and cleaned versions produce matching
per-sample predictions and aggregate metrics.
