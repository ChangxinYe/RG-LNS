"""
@file: evaluate_jpwleg_rg_lns.py
@description: Compare a ViT-T compatibility baseline with the complete RG-LNS method.
              Complete partial initial layouts without labels, then search reliable-component
              translations and beam completions using a fixed compatibility objective.
              Save detailed search records only when diagnostics are requested.
@author: Changxin Ye
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from evaluation.reporting import (
    format_evaluation_summary,
    format_metric_summary,
    format_run_configuration,
)

from rg_lns import (
    iterative_component_reassembly,
    profiled_large_neighborhood_reassembly,
)
from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
from evaluation.runtime_info import collect_runtime_environment
from evaluation.timing import (
    StageTimingCollector,
    add_stage_timing_arguments,
)
from data_loaders.jpwleg import (
    JPWLEG_CONFIGS,
    SPLIT_ALIASES,
    image_to_pieces,
    label_to_target_positions,
    load_jpwleg_arrays,
)
from evaluation.jpwleg import (
    aggregate_ground_truth_metrics,
    compose_puzzle,
    is_complete_prediction,
    sample_ground_truth_metrics,
    solve_with_gallagher,
)
from compatibility.scorer import MetricCompatibilityScorer, resolve_device
from initial_solvers.partial_completion import complete_partial_layout
from evaluation.common.jpwleg_search import (
    build_parser as build_search_parser,
    validate_args as validate_search_args,
)
from evaluation.common.jpwleg_inputs import parse_bool, resolve_checkpoint_arg


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")

# Defaults for direct runs; checkpoints are selected for the requested dataset.
DEFAULT_DATASET = "jpwleg5"
DEFAULT_SPLIT = "test"
DEFAULT_START_INDEX = 0
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_RUN_NAME = None

DEFAULT_COMPLETION_BEAM_WIDTH = 4
DEFAULT_PARTIAL_COMPLETION_BEAM_WIDTH = 2
DEFAULT_MUTUAL_TOP_K = 3
DEFAULT_TRANSLATION_MODE = "all"
DEFAULT_COMPONENT_INDEX = 0
DEFAULT_MIN_COMPONENT_SIZE = 4
DEFAULT_REFINEMENT_ROUNDS = 1
DEFAULT_SCORE_BATCH_SIZE = 128
DEFAULT_POSTPROCESS = True
DEFAULT_SAVE_VISUALS = True
DEFAULT_SAVE_COMPLETION_DETAILS = False
DEFAULT_VISUAL_LIMIT = 10
DEFAULT_SAVE_PREDICTIONS = True
DEFAULT_PROGRESS_INTERVAL = 25
METRIC_KEYS = ("PA", "AA", "HA", "VA", "SRA", "NA")


def build_parser() -> argparse.ArgumentParser:
    """Add RG-LNS multi-path parameters to the shared greedy JPwLEG parameters."""

    parser = build_search_parser()
    parser.description = (
        "Evaluate RG-LNS on JPwLEG with frozen visual compatibility"
    )
    parser.prog = "Evaluate RG-LNS on JPwLEG"
    parser.add_argument(
        "--completion-beam-width",
        default=DEFAULT_COMPLETION_BEAM_WIDTH,
        type=int,
        help="maximum complete fill candidates retained per component pose; 1 strictly reduces to greedy reassembly",
    )
    parser.add_argument(
        "--partial-completion-beam-width",
        default=DEFAULT_PARTIAL_COMPLETION_BEAM_WIDTH,
        type=int,
        help="ground-truth-free compatibility completion beam width for incomplete initial layouts",
    )
    parser.add_argument(
        "--translation-mode",
        default=DEFAULT_TRANSLATION_MODE,
        choices=("fixed", "all"),
        help="keep the current absolute position or enumerate all legal translations",
    )
    parser.add_argument(
        "--save-diagnostics", "--save-completion-details",
        dest="save_completion_details",
        default=DEFAULT_SAVE_COMPLETION_DETAILS,
        type=parse_bool,
        help="save detailed search CSVs and single-path comparison lists",
    )
    parser.set_defaults(
        dataset=DEFAULT_DATASET,
        split=DEFAULT_SPLIT,
        start_index=DEFAULT_START_INDEX,
        max_samples=DEFAULT_MAX_SAMPLES,
        run_name=DEFAULT_RUN_NAME,
        mutual_top_k=DEFAULT_MUTUAL_TOP_K,
        component_index=DEFAULT_COMPONENT_INDEX,
        min_component_size=DEFAULT_MIN_COMPONENT_SIZE,
        refinement_rounds=DEFAULT_REFINEMENT_ROUNDS,
        score_batch_size=DEFAULT_SCORE_BATCH_SIZE,
        postprocess=DEFAULT_POSTPROCESS,
        save_visuals=DEFAULT_SAVE_VISUALS,
        visual_limit=DEFAULT_VISUAL_LIMIT,
        save_predictions=DEFAULT_SAVE_PREDICTIONS,
        progress_interval=DEFAULT_PROGRESS_INTERVAL,
    )
    add_stage_timing_arguments(parser)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_search_args(args)
    if args.completion_beam_width <= 0:
        raise ValueError("completion-beam-width must be a positive integer")
    if args.partial_completion_beam_width <= 0:
        raise ValueError("partial-completion-beam-width must be a positive integer")
    if args.translation_mode not in {"fixed", "all"}:
        raise ValueError("translation-mode must be fixed or all")
    if args.runtime_warmup_samples < 0:
        raise ValueError("runtime-warmup-samples must be non-negative")


def _default_output_dir(
    args: argparse.Namespace,
    checkpoint: Path,
    *,
    output_root: Path,
) -> Path:
    """Keep only dataset and checkpoint source in the directory; write ablation parameters to the summary."""

    folder = f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}"
    return Path(output_root) / args.dataset / folder


def _prepare_output_dir(output_dir: Path, *, automatically_named: bool) -> Path:
    """Create an isolated run directory to prevent mixing old and new evaluation results."""

    del automatically_named
    return prepare_unique_output_dir(output_dir)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_pa_case_lists(
    output_dir: Path,
    rows: list[dict],
    *,
    reference_prefix: str,
    filename_suffix: str,
    write_files: bool = True,
) -> tuple[list[str], list[str]]:
    """Save PA repair and degradation samples relative to the specified reference."""

    repaired = [
        f"sample_{int(row['sample_index']):06d}.png"
        for row in rows
        if int(row[f"{reference_prefix}_PA"]) == 0
        and int(row["selected_PA"]) == 1
    ]
    broken = [
        f"sample_{int(row['sample_index']):06d}.png"
        for row in rows
        if int(row[f"{reference_prefix}_PA"]) == 1
        and int(row["selected_PA"]) == 0
    ]
    if not write_files:
        return repaired, broken
    for kind, values in (("repaired", repaired), ("broken", broken)):
        filename = f"pa_{kind}{filename_suffix}_images.txt"
        text = "\n".join(values)
        (output_dir / filename).write_text(
            text + ("\n" if text else ""),
            encoding="utf-8",
        )
    return repaired, broken


def _save_visual(
    path: Path,
    pieces: list[np.ndarray],
    baseline: np.ndarray,
    selected: np.ndarray,
    target: np.ndarray,
    grid: int,
    title: str,
    initial_solver_name: str = "Gallagher",
) -> None:
    """Save a comparison of the baseline and complete RG-LNS method."""
    from evaluation.visualization import save_comparison_visual

    save_comparison_visual(
        path, pieces, baseline, selected, target, grid,
        title, initial_solver_name, compose_puzzle=compose_puzzle,
    )


def _prefixed_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": metrics[key] for key in METRIC_KEYS}


def _append_diagnostic_rows(
    *,
    args: argparse.Namespace,
    sample_index: int,
    target: np.ndarray,
    grid: int,
    result: dict,
    round_rows: list[dict],
    candidate_rows: list[dict],
    pose_rows: list[dict],
    completion_rows: list[dict],
) -> None:
    """Expand RG-LNS rounds, poses, and fill candidates for later ablation analysis."""

    for round_record in result["rounds"]:
        graph = round_record["graph_statistics"]
        improved_poses = sum(
            int(pose["selected_completion_index"] != 0)
            for pose in round_record["poses"]
        )
        round_rows.append(
            {
                "sample_index": sample_index,
                "round": round_record["round"],
                "status": round_record["status"],
                "accepted": int(round_record["accepted"]),
                "component_size": len(round_record["component"]),
                "component_count": graph["components"],
                "complete_cycles": graph["complete_cycles"],
                "cycle_supported_edges": graph["cycle_supported_edges"],
                "global_candidate_count": len(round_record["candidates"]),
                "pose_count": round_record["pose_count"],
                "recorded_pose_count": round_record["recorded_pose_count"],
                "completion_candidate_count": round_record["completion_candidate_count"],
                "recorded_completion_count": round_record["recorded_completion_count"],
                "pose_diagnostics_complete": int(round_record["pose_diagnostics_complete"]),
                "completion_beam_width": args.completion_beam_width,
                "poses_improved_by_beam": improved_poses,
                "selected_label": round_record["selected_label"],
                "selected_row_shift": round_record["selected_row_shift"],
                "selected_column_shift": round_record["selected_column_shift"],
                "selected_completion_index": round_record["selected_completion_index"],
                "selected_completion_origin": round_record["selected_completion_origin"],
                "single_path_selected_label": round_record["single_path_selected_label"],
                "single_path_selected_row_shift": round_record[
                    "single_path_selected_row_shift"
                ],
                "single_path_selected_column_shift": round_record[
                    "single_path_selected_column_shift"
                ],
            }
        )

        for candidate in round_record["candidates"]:
            metric = sample_ground_truth_metrics(candidate["prediction"], target, grid)
            candidate_rows.append(
                {
                    "sample_index": sample_index,
                    "round": round_record["round"],
                    "round_status": round_record["status"],
                    "round_accepted": int(round_record["accepted"]),
                    "component_size": len(round_record["component"]),
                    "label": candidate["label"],
                    "row_shift": candidate["row_shift"],
                    "column_shift": candidate["column_shift"],
                    "completion_count": candidate["completion_count"],
                    "selected_completion_index": candidate["selected_completion_index"],
                    "selected_completion_origin": candidate["selected_completion_origin"],
                    "e1_mean_adjacency_score": candidate["mean_adjacency_score"],
                    "e1_mutual_top1_edges": candidate["mutual_top1_edges"],
                    "selected": int(candidate["selected"]),
                    **metric,
                }
            )

        for pose in round_record["poses"]:
            metric = sample_ground_truth_metrics(pose["prediction"], target, grid)
            pose_rows.append(
                {
                    "sample_index": sample_index,
                    "round": round_record["round"],
                    "round_status": round_record["status"],
                    "round_accepted": int(round_record["accepted"]),
                    "component_size": len(round_record["component"]),
                    "pose_index": pose["pose_index"],
                    "label": pose["label"],
                    "row_shift": pose["row_shift"],
                    "column_shift": pose["column_shift"],
                    "completion_count": pose["completion_count"],
                    "beam_final_states": pose["beam_final_states"],
                    "greedy_e1_mean_adjacency_score": pose[
                        "greedy_mean_adjacency_score"
                    ],
                    "profiled_e1_mean_adjacency_score": pose["mean_adjacency_score"],
                    "profile_mean_gain_vs_greedy": pose["profile_mean_gain_vs_greedy"],
                    "greedy_e1_mutual_top1_edges": pose["greedy_mutual_top1_edges"],
                    "profiled_e1_mutual_top1_edges": pose["mutual_top1_edges"],
                    "profile_mutual_top1_gain_vs_greedy": pose[
                        "profile_mutual_top1_gain_vs_greedy"
                    ],
                    "selected_completion_index": pose["selected_completion_index"],
                    "selected_completion_origin": pose["selected_completion_origin"],
                    "selected_pose": int(pose["selected_pose"]),
                    "accepted_pose": int(pose["accepted_pose"]),
                    "deduplicated_from_global_candidates": int(
                        pose["deduplicated_from_global_candidates"]
                    ),
                    **metric,
                }
            )
            if not args.save_completion_details:
                continue
            for completion in pose["completions"]:
                completion_metric = sample_ground_truth_metrics(
                    completion["prediction"],
                    target,
                    grid,
                )
                completion_rows.append(
                    {
                        "sample_index": sample_index,
                        "round": round_record["round"],
                        "pose_index": pose["pose_index"],
                        "pose_label": pose["label"],
                        "round_status": round_record["status"],
                        "round_accepted": int(round_record["accepted"]),
                        "row_shift": pose["row_shift"],
                        "column_shift": pose["column_shift"],
                        "completion_index": completion["completion_index"],
                        "profile_rank": completion["profile_rank"],
                        "origin": completion["origin"],
                        "beam_partial_mean_score": completion["beam_partial_mean_score"],
                        "beam_closed_edges": completion["beam_closed_edges"],
                        "e1_mean_adjacency_score": completion["mean_adjacency_score"],
                        "e1_mutual_top1_edges": completion["mutual_top1_edges"],
                        "selected_within_pose": int(completion["selected_within_pose"]),
                        "selected_in_round": int(completion["selected_in_round"]),
                        "accepted_global": int(completion["accepted_global"]),
                        **completion_metric,
                    }
                )


def run_evaluation(
    args: argparse.Namespace,
    *,
    initial_solver=solve_with_gallagher,
    initial_solver_name: str = "Gallagher",
    initial_solver_tag: str = "gallagher",
    initial_solver_config: dict | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    validate_args(args)
    args.checkpoint = resolve_checkpoint_arg(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    split = SPLIT_ALIASES[args.split]
    config = JPWLEG_CONFIGS[args.dataset]
    images, labels = load_jpwleg_arrays(args.data_root, args.dataset, split)
    if args.start_index >= len(images):
        raise ValueError(f"start-index {args.start_index} exceeds dataset length {len(images)}")
    end_index = (
        len(images)
        if args.max_samples == 0
        else min(len(images), args.start_index + args.max_samples)
    )
    sample_indices = list(range(args.start_index, end_index))
    grid = config.grid

    device = resolve_device(args.device, args.gpu_id)
    scorer = MetricCompatibilityScorer.from_checkpoint(
        args.checkpoint,
        device=device,
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
    )
    automatically_named_output = args.output_dir is None
    if automatically_named_output:
        args.output_dir = _default_output_dir(
            args,
            args.checkpoint,
            output_root=output_root,
        )
    args.output_dir = _prepare_output_dir(
        Path(args.output_dir),
        automatically_named=automatically_named_output,
    )
    visual_dir = args.output_dir / "visuals"

    baseline_metrics: list[dict] = []
    completed_baseline_metrics: list[dict] = []
    s1a3_metrics: list[dict] = []
    selected_metrics: list[dict] = []
    sample_rows: list[dict] = []
    round_rows: list[dict] = []
    candidate_rows: list[dict] = []
    pose_rows: list[dict] = []
    completion_rows: list[dict] = []
    baseline_predictions: list[np.ndarray] = []
    completed_baseline_predictions: list[np.ndarray] = []
    s1a3_predictions: list[np.ndarray] = []
    selected_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    component_masks: list[np.ndarray] = []
    row_shifts: list[int] = []
    column_shifts: list[int] = []
    completion_indices: list[int] = []
    completion_origins: list[str] = []
    status_counts: dict[str, int] = {}
    complete_initial_baselines = 0
    complete_after_partial_completion = 0
    runtime_collector = StageTimingCollector(
        device=device,
        warmup_samples=args.runtime_warmup_samples,
        total_samples=len(sample_indices),
    )
    started = time.perf_counter()

    print(
        "\n".join(format_run_configuration(
            dataset=config.folder,
            split=split,
            samples=len(sample_indices),
            initial_solver_name=initial_solver_name,
            device=str(device),
            data_root=args.data_root,
            checkpoint=args.checkpoint,
        )),
        flush=True,
    )
    for evaluation_number, sample_index in enumerate(sample_indices, start=1):
        image = np.asarray(images[sample_index])
        pieces = list(image_to_pieces(image, grid))
        target = label_to_target_positions(
            np.asarray(labels[sample_index]),
            grid,
        ).astype(np.int32)
        with runtime_collector.measure(synchronize_cuda=True) as compatibility_timing:
            scores = scorer.score_pieces(pieces, rot_flag=0)
        with runtime_collector.measure() as initial_solver_timing:
            baseline = np.asarray(
                initial_solver(pieces, scores, grid, verbose=args.verbose_solver),
                dtype=np.int32,
            )
        complete_initial_baselines += int(is_complete_prediction(baseline, grid))

        completed_baseline = baseline.copy()
        partial_completion = None
        partial_completion_error = None
        partial_completion_seconds = 0.0
        if not is_complete_prediction(baseline, grid):
            try:
                with runtime_collector.measure() as partial_completion_timing:
                    partial_completion = complete_partial_layout(
                        scores,
                        baseline,
                        grid,
                        completion_beam_width=args.partial_completion_beam_width,
                    )
                    completed_baseline = np.asarray(
                        partial_completion["prediction"], dtype=np.int32
                    ).copy()
            except (RuntimeError, ValueError) as error:
                partial_completion_error = str(error)
            partial_completion_seconds = partial_completion_timing.seconds
        complete_after_partial_completion += int(
            is_complete_prediction(completed_baseline, grid)
        )

        rgls_seconds = 0.0
        if not is_complete_prediction(completed_baseline, grid):
            status = f"incomplete_{initial_solver_tag}_completion_failed"
            selected = baseline.copy()
            s1a3_reference = baseline.copy()
            display_component = np.empty(0, dtype=np.int32)
            rounds_attempted = rounds_accepted = 0
            total_candidates = total_poses = total_recorded_poses = 0
            total_completions = total_recorded_completions = 0
            initial_e1_score = final_e1_score = final_mutual_top1 = None
            selected_row_shift = selected_column_shift = 0
            selected_completion_index = -1
            selected_completion_origin = "partial_completion_failed"
        else:
            with runtime_collector.measure() as rgls_timing:
                result = profiled_large_neighborhood_reassembly(
                    scores,
                    completed_baseline,
                    grid,
                    mutual_top_k=args.mutual_top_k,
                    translation_mode=args.translation_mode,
                    component_index=args.component_index,
                    min_component_size=args.min_component_size,
                    max_rounds=args.refinement_rounds,
                    completion_beam_width=args.completion_beam_width,
                )
            rgls_seconds = rgls_timing.seconds
            selected = result["prediction"]
            s1a3_reference = result["single_path_reference_prediction"]
            if s1a3_reference is None:
                s1a3_reference = iterative_component_reassembly(
                    scores,
                    completed_baseline,
                    grid,
                    mutual_top_k=args.mutual_top_k,
                    translation_mode=args.translation_mode,
                    component_index=args.component_index,
                    min_component_size=args.min_component_size,
                    max_rounds=args.refinement_rounds,
                )["prediction"]
            status = str(result["stop_reason"])
            if partial_completion is not None and bool(partial_completion["applied"]):
                status = f"partial_completed__{status}"
            rounds_attempted = int(result["rounds_attempted"])
            rounds_accepted = int(result["rounds_accepted"])
            total_candidates = sum(
                len(record["candidates"]) for record in result["rounds"]
            )
            total_poses = sum(int(record["pose_count"]) for record in result["rounds"])
            total_recorded_poses = sum(
                int(record["recorded_pose_count"]) for record in result["rounds"]
            )
            total_completions = sum(
                int(record["completion_candidate_count"]) for record in result["rounds"]
            )
            total_recorded_completions = sum(
                int(record["recorded_completion_count"]) for record in result["rounds"]
            )
            initial_e1_score = float(result["initial_score"]["mean_adjacency_score"])
            final_e1_score = float(result["final_score"]["mean_adjacency_score"])
            final_mutual_top1 = int(result["final_score"]["mutual_top1_edges"])
            display_component = result["last_accepted_component"]
            selected_row_shift = int(result["last_accepted_row_shift"])
            selected_column_shift = int(result["last_accepted_column_shift"])
            selected_completion_index = int(result["last_accepted_completion_index"])
            selected_completion_origin = str(result["last_accepted_completion_origin"])
            if len(display_component) == 0 and result["rounds"]:
                display_component = result["rounds"][-1]["component"]

            _append_diagnostic_rows(
                args=args,
                sample_index=sample_index,
                target=target,
                grid=grid,
                result=result,
                round_rows=round_rows,
                candidate_rows=candidate_rows,
                pose_rows=pose_rows,
                completion_rows=completion_rows,
            )

        refinement_seconds = partial_completion_seconds + rgls_seconds
        runtime_row = runtime_collector.add_sample(
            sample_position=evaluation_number - 1,
            sample_index=sample_index,
            stages={
                "compatibility": compatibility_timing.seconds,
                "initial_solver": initial_solver_timing.seconds,
                "partial_completion": partial_completion_seconds,
                "rgls": rgls_seconds,
                "refinement_total": refinement_seconds,
                "inference_total": (
                    compatibility_timing.seconds
                    + initial_solver_timing.seconds
                    + refinement_seconds
                ),
            },
        )

        baseline_metric = sample_ground_truth_metrics(baseline, target, grid)
        completed_baseline_metric = sample_ground_truth_metrics(
            completed_baseline, target, grid
        )
        s1a3_metric = sample_ground_truth_metrics(s1a3_reference, target, grid)
        selected_metric = sample_ground_truth_metrics(selected, target, grid)
        baseline_metrics.append(baseline_metric)
        completed_baseline_metrics.append(completed_baseline_metric)
        s1a3_metrics.append(s1a3_metric)
        selected_metrics.append(selected_metric)
        status_counts[status] = status_counts.get(status, 0) + 1
        sample_rows.append(
            {
                "sample_index": sample_index,
                "status": status,
                **runtime_row,
                "partial_completion_applied": int(
                    partial_completion is not None and bool(partial_completion["applied"])
                ),
                "partial_completion_error": partial_completion_error,
                "partial_completion_missing_pieces": (
                    None if partial_completion is None else partial_completion["missing_pieces"]
                ),
                "partial_completion_legal_translations": (
                    None
                    if partial_completion is None
                    else partial_completion["legal_translation_count"]
                ),
                "partial_completion_candidates": (
                    None if partial_completion is None else partial_completion["candidate_count"]
                ),
                "rounds_attempted": rounds_attempted,
                "rounds_accepted": rounds_accepted,
                "total_global_candidates": total_candidates,
                "total_poses": total_poses,
                "total_recorded_poses": total_recorded_poses,
                "total_completion_candidates": total_completions,
                "total_recorded_completion_candidates": total_recorded_completions,
                "completion_beam_width": args.completion_beam_width,
                "selected_row_shift": selected_row_shift,
                "selected_column_shift": selected_column_shift,
                "selected_completion_index": selected_completion_index,
                "selected_completion_origin": selected_completion_origin,
                "prediction_changed": int(not np.array_equal(baseline, selected)),
                "beam_prediction_changed_vs_s1a3": int(
                    not np.array_equal(s1a3_reference, selected)
                ),
                "initial_e1_mean_adjacency_score": initial_e1_score,
                "final_e1_mean_adjacency_score": final_e1_score,
                "final_e1_mutual_top1_edges": final_mutual_top1,
                "baseline_correct_pieces": baseline_metric["correct_pieces"],
                "completed_baseline_correct_pieces": completed_baseline_metric[
                    "correct_pieces"
                ],
                "s1a3_reference_correct_pieces": s1a3_metric["correct_pieces"],
                "selected_correct_pieces": selected_metric["correct_pieces"],
                "piece_change": selected_metric["correct_pieces"]
                - baseline_metric["correct_pieces"],
                "beam_piece_change_vs_s1a3": selected_metric["correct_pieces"]
                - s1a3_metric["correct_pieces"],
                **_prefixed_metrics("baseline", baseline_metric),
                **_prefixed_metrics("completed_baseline", completed_baseline_metric),
                **_prefixed_metrics("s1a3_reference", s1a3_metric),
                **_prefixed_metrics("selected", selected_metric),
            }
        )
        baseline_predictions.append(baseline.copy())
        completed_baseline_predictions.append(completed_baseline.copy())
        s1a3_predictions.append(s1a3_reference.copy())
        selected_predictions.append(selected.copy())
        targets.append(target.copy())
        mask = np.zeros(grid * grid, dtype=np.uint8)
        mask[np.asarray(display_component, dtype=np.int32)] = 1
        component_masks.append(mask)
        row_shifts.append(selected_row_shift)
        column_shifts.append(selected_column_shift)
        completion_indices.append(selected_completion_index)
        completion_origins.append(selected_completion_origin)

        if args.save_visuals and (
            args.visual_limit == 0 or evaluation_number <= args.visual_limit
        ):
            _save_visual(
                visual_dir / f"sample_{sample_index:06d}.png",
                pieces, baseline, selected, target, grid,
                f"{config.folder} | {split} | sample {sample_index}",
                initial_solver_name,
            )

        if args.progress_interval > 0 and (
            evaluation_number % args.progress_interval == 0
            or evaluation_number == len(sample_indices)
        ):
            running_baseline = aggregate_ground_truth_metrics(baseline_metrics)
            running_selected = aggregate_ground_truth_metrics(selected_metrics)
            print(
                f"[{evaluation_number}/{len(sample_indices)}] Cumulative: "
                f"Baseline (ViT-T + {initial_solver_name}) -> Baseline + RG-LNS | "
                f"PA {running_baseline['PA']:.2%}->{running_selected['PA']:.2%}, "
                f"AA {running_baseline['AA']:.2%}->{running_selected['AA']:.2%}, "
                f"SRA {running_baseline['SRA']:.2%}->{running_selected['SRA']:.2%}"
            )

    elapsed_seconds = time.perf_counter() - started
    runtime_summary = runtime_collector.summary()
    baseline_aggregate = aggregate_ground_truth_metrics(baseline_metrics)
    completed_baseline_aggregate = aggregate_ground_truth_metrics(
        completed_baseline_metrics
    )
    s1a3_aggregate = aggregate_ground_truth_metrics(s1a3_metrics)
    selected_aggregate = aggregate_ground_truth_metrics(selected_metrics)
    piece_changes = [int(row["piece_change"]) for row in sample_rows]
    beam_changes = [int(row["beam_piece_change_vs_s1a3"]) for row in sample_rows]
    comparison = {
        "improved_puzzles": sum(change > 0 for change in piece_changes),
        "damaged_puzzles": sum(change < 0 for change in piece_changes),
        "unchanged_puzzles": sum(change == 0 for change in piece_changes),
        "changed_predictions": sum(int(row["prediction_changed"]) for row in sample_rows),
        "repaired_to_perfect": sum(
            int(row["baseline_PA"]) == 0 and int(row["selected_PA"]) == 1
            for row in sample_rows
        ),
        "broken_perfect": sum(
            int(row["baseline_PA"]) == 1 and int(row["selected_PA"]) == 0
            for row in sample_rows
        ),
        "accepted_rounds": sum(int(row["rounds_accepted"]) for row in sample_rows),
        "accepted_beam_rounds": sum(
            int(row["accepted"])
            and row["selected_completion_origin"] == "profiled_beam"
            for row in round_rows
        ),
    }
    partial_completion_diagnostics = {
        "beam_width": args.partial_completion_beam_width,
        "applied_samples": sum(
            int(row["partial_completion_applied"]) for row in sample_rows
        ),
        "failed_samples": sum(
            row["partial_completion_error"] is not None for row in sample_rows
        ),
        "complete_after_completion": complete_after_partial_completion,
        "repaired_to_perfect": sum(
            int(row["baseline_PA"]) == 0
            and int(row["completed_baseline_PA"]) == 1
            for row in sample_rows
        ),
        "improved_puzzles": sum(
            int(row["completed_baseline_correct_pieces"])
            > int(row["baseline_correct_pieces"])
            for row in sample_rows
        ),
        "damaged_puzzles": sum(
            int(row["completed_baseline_correct_pieces"])
            < int(row["baseline_correct_pieces"])
            for row in sample_rows
        ),
    }
    incremental = {
        "improved_puzzles": sum(change > 0 for change in beam_changes),
        "damaged_puzzles": sum(change < 0 for change in beam_changes),
        "unchanged_puzzles": sum(change == 0 for change in beam_changes),
        "changed_predictions": sum(
            int(row["beam_prediction_changed_vs_s1a3"]) for row in sample_rows
        ),
        "repaired_to_perfect": sum(
            int(row["s1a3_reference_PA"]) == 0 and int(row["selected_PA"]) == 1
            for row in sample_rows
        ),
        "broken_perfect": sum(
            int(row["s1a3_reference_PA"]) == 1 and int(row["selected_PA"]) == 0
            for row in sample_rows
        ),
    }
    search_diagnostics = {
        "evaluated_poses": sum(int(row["total_poses"]) for row in sample_rows),
        "recorded_pose_rows": len(pose_rows),
        "evaluated_completion_hypotheses": sum(
            int(row["total_completion_candidates"]) for row in sample_rows
        ),
        "recorded_completion_rows": len(completion_rows),
        "pose_diagnostics_complete": all(
            int(row["total_poses"]) == int(row["total_recorded_poses"])
            for row in sample_rows
        ),
        "poses_improved_over_s1a3_greedy": sum(
            int(row["selected_completion_index"] != 0) for row in pose_rows
        ),
        "selected_non_greedy_poses": sum(
            int(row["selected_pose"])
            and row["selected_completion_origin"] == "profiled_beam"
            for row in pose_rows
        ),
    }
    repaired, broken = _write_pa_case_lists(
        args.output_dir,
        sample_rows,
        reference_prefix="baseline",
        filename_suffix="",
    )
    repaired_vs_s1a3, broken_vs_s1a3 = _write_pa_case_lists(
        args.output_dir,
        sample_rows,
        reference_prefix="s1a3_reference",
        filename_suffix="_vs_s1a3",
        write_files=args.save_completion_details,
    )

    payload = {
        "method": (
            f"s1a4_profiled_large_neighborhood_reassembly_"
            f"{initial_solver_tag}_initialization"
        ),
        "initial_solver": initial_solver_name,
        "uses_training": False,
        "uses_ground_truth_for_selection": False,
        "selection_rule": (
            "first complete an incomplete initial layout with label-free E1 profiled beam; "
            "then, for each component placement, select the retained completion minimizing "
            "full-layout E1 mean adjacency score (mutual-Top1 tie-break); then select "
            "across placements with the same E1 rule and accept only a strict improvement"
        ),
        "beam1_contract": (
            "after the same partial-layout completion, completion_beam_width=1 directly "
            "reuses the formal greedy path"
        ),
        "dataset": args.dataset,
        "split": split,
        "start_index": args.start_index,
        "end_index_exclusive": end_index,
        "evaluated_samples": len(sample_indices),
        "checkpoint": str(args.checkpoint),
        "configuration": {
            "completion_beam_width": args.completion_beam_width,
            "partial_completion_beam_width": args.partial_completion_beam_width,
            "mutual_top_k": args.mutual_top_k,
            "translation_mode": args.translation_mode,
            "component_index": args.component_index,
            "min_component_size": args.min_component_size,
            "refinement_rounds": args.refinement_rounds,
            "postprocess": args.postprocess,
            "score_batch_size": args.score_batch_size,
            "runtime_warmup_samples": args.runtime_warmup_samples,
            "device": str(device),
            "save_visuals": args.save_visuals,
            "visual_limit": args.visual_limit,
            "save_predictions": args.save_predictions,
            "save_completion_details": args.save_completion_details,
            "initial_solver": initial_solver_config or {},
        },
        "complete_initial_solver_baselines": complete_initial_baselines,
        "complete_after_partial_completion": complete_after_partial_completion,
        "status_counts": status_counts,
        "baseline": baseline_aggregate,
        "partial_completion": completed_baseline_aggregate,
        "partial_completion_diagnostics": partial_completion_diagnostics,
        "s1a3_single_path_reference": s1a3_aggregate,
        "s1a4": selected_aggregate,
        "comparison": comparison,
        "incremental_vs_s1a3": incremental,
        "search_diagnostics": search_diagnostics,
        "runtime": runtime_summary,
        "elapsed_seconds": elapsed_seconds,
        "pa_case_lists": {
            "repaired_images": len(repaired),
            "broken_images": len(broken),
            "repaired_vs_s1a3_images": len(repaired_vs_s1a3),
            "broken_vs_s1a3_images": len(broken_vs_s1a3),
        },
    }
    payload["runtime_environment"] = collect_runtime_environment(
        requested_device=args.device, resolved_device=device, gpu_id=args.gpu_id,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "sample_results.csv", sample_rows)
    if args.save_completion_details:
        _write_csv(args.output_dir / "round_results.csv", round_rows)
        _write_csv(args.output_dir / "candidates.csv", candidate_rows)
        _write_csv(args.output_dir / "pose_results.csv", pose_rows)
        _write_csv(args.output_dir / "completion_results.csv", completion_rows)

    if args.save_predictions:
        np.savez_compressed(
            args.output_dir / "predictions.npz",
            sample_indices=np.asarray(sample_indices, dtype=np.int32),
            baseline_predictions=np.stack(baseline_predictions),
            completed_baseline_predictions=np.stack(completed_baseline_predictions),
            s1a3_reference_predictions=np.stack(s1a3_predictions),
            selected_predictions=np.stack(selected_predictions),
            targets=np.stack(targets),
            selected_component_masks=np.stack(component_masks),
            selected_row_shifts=np.asarray(row_shifts, dtype=np.int32),
            selected_column_shifts=np.asarray(column_shifts, dtype=np.int32),
            selected_completion_indices=np.asarray(completion_indices, dtype=np.int32),
            selected_completion_origins=np.asarray(completion_origins),
            grid_size=np.asarray(grid, dtype=np.int32),
        )

    metric_summary = format_metric_summary(
        baseline_aggregate, selected_aggregate, initial_solver_name, METRIC_KEYS
    )
    summary_lines = format_run_configuration(
        dataset=config.folder, split=split, samples=len(sample_indices),
        initial_solver_name=initial_solver_name, device=str(device),
        data_root=args.data_root, checkpoint=args.checkpoint,
    )
    summary_lines = format_evaluation_summary(summary_lines, metric_summary, payload, args.output_dir)
    (args.output_dir / "summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8",
    )

    print("\n".join(metric_summary))
    print(f"Saved: {args.output_dir}")
    return payload


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
