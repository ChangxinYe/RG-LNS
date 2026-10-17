"""
@file: evaluate_gap_gallagher.py
@description: Load the specified compatibility checkpoint, compute the RGBA piece compatibility matrix, and use Gallagher Solver
              to reassemble GAP-3/GAP-5. Output PA, AA, HA, VA, SRA, NA, Recall@K, predictions, and visualizations.
@author: Changxin Ye
@created: 2026-07-27
@version: 1.0
"""

from __future__ import annotations

from evaluation.common.gap_inputs import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_DATASET,
    DEFAULT_LOG_ROOT,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_RUN_DIR,
    DEFAULT_RUN_NAMES,
    DEFAULT_SAVE_VISUALS,
    DEFAULT_SPLIT,
    DEFAULT_VISUAL_LIMIT,
    SCRIPT_DIR,
    build_parser,
    checkpoint_from_run_dir,
    parse_bool,
    resolve_checkpoint,
    validate_args,
)

import argparse
from pathlib import Path

from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
from data_loaders.gap import DEFAULT_GAP_DATA_ROOT, GAP_CONFIGS, GAP_SPLITS
from evaluation.gap import GAPMetrics, evaluate_gap_split, solve_with_gallagher
from compatibility.scorer import MetricCompatibilityScorer, load_metric_checkpoint, resolve_device


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run_evaluation(
    args: argparse.Namespace,
    *,
    solver=solve_with_gallagher,
    solver_name: str = "Gallagher",
    method_name: str = "Baseline (ViT-T + Gallagher)",
    solver_configuration: dict | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output_suffix: str = "",
) -> tuple[GAPMetrics, Path]:
    """Run a standalone GAP baseline through the same scorer/solver path as RG-LNS."""

    validate_args(args)
    if args.run_name is None:
        args.run_name = DEFAULT_RUN_NAMES[args.dataset]
    device = resolve_device(args.device, args.gpu_id)
    checkpoint = resolve_checkpoint(args)
    model, model_config, checkpoint_payload = load_metric_checkpoint(checkpoint, device)
    checkpoint_dataset = checkpoint_payload.get("dataset") or checkpoint_payload.get("args", {}).get("dataset")
    if checkpoint_dataset not in {None, args.dataset}:
        raise ValueError(f"Checkpoint dataset {checkpoint_dataset} does not match requested {args.dataset}")
    if int(checkpoint_payload.get("model_config", {}).get("input_channels", 3)) != 4:
        raise ValueError("Selected checkpoint is not an RGBA GAP metric checkpoint")
    scorer = MetricCompatibilityScorer(
        model=model,
        device=device,
        input_size=model_config["input_size"],
        normalization=model_config["normalization"],
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
        score_metric=model_config["score_metric"],
        canonical_edge=model_config["canonical_edge"],
        geometry_mode=model_config["geometry_mode"],
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else Path(output_root)
        / args.dataset
        / (
            f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{args.split}"
            f"{('_' + output_suffix) if output_suffix else ''}"
        )
    )
    output_dir = prepare_unique_output_dir(output_dir)
    configuration = {
        "solver": solver_name,
        "postprocess": args.postprocess,
        "score_batch_size": args.score_batch_size,
        "score_metric": model_config["score_metric"],
        "normalization": model_config["normalization"],
        "geometry_mode": model_config["geometry_mode"],
        "input_channels": 4,
        "rgba_adapter": "learned_conv1x1_4_to_3",
    }
    if solver_configuration is not None:
        configuration["initial_solver"] = solver_configuration
    print(f"dataset/split: {args.dataset} / {args.split}")
    print(f"data_root: {Path(args.data_root).expanduser().resolve()}")
    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    print(f"output: {output_dir}")
    metrics = evaluate_gap_split(
        scorer=scorer,
        solver=solver,
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        output_dir=output_dir,
        checkpoint=checkpoint,
        max_samples=args.max_samples,
        start_index=args.start_index,
        progress_interval=args.progress_interval,
        verbose_solver=args.verbose_solver,
        save_visuals=args.save_visuals,
        visual_limit=args.visual_limit,
        visual_interval=args.visual_interval,
        save_predictions=args.save_predictions,
        method_name=method_name,
        configuration=configuration,
    )
    return metrics, output_dir


def print_metrics(metrics: GAPMetrics, output_dir: Path) -> None:
    print("\nGAP evaluation")
    print("=" * 72)
    print(f"PA:  {metrics.pa:.4%}")
    print(f"AA:  {metrics.aa:.4%}")
    print(f"HA:  {metrics.ha:.4%}")
    print(f"VA:  {metrics.va:.4%}")
    print(f"SRA: {metrics.sra:.4%}")
    print(f"NA:  {metrics.na:.4%} (PuzzleFlow neighbor_accuracy)")
    for k in metrics.neighbor_retrieval.recall_ks:
        print(f"Recall@{k}: {metrics.neighbor_retrieval.recall_at(k):.4%}")
    print(f"Summary: {output_dir / 'summary.txt'}")


def main() -> None:
    metrics, output_dir = run_evaluation(parse_args())
    print_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()
