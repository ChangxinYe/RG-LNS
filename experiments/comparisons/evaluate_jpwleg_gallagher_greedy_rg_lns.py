"""
@file: evaluate_jpwleg_gallagher_greedy_rg_lns.py
@description: Run the single-path component-reassembly comparison on a Gallagher initial layout for JPwLEG and compare metrics before and after reassembly.
@author: Changxin Ye
@created: 2026-07-26
@version: 1.1
"""

from __future__ import annotations

from evaluation.common.jpwleg_search import (
    DEFAULT_COMPONENT_INDEX,
    DEFAULT_DATASET,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_COMPONENT_SIZE,
    DEFAULT_MUTUAL_TOP_K,
    DEFAULT_POSTPROCESS,
    DEFAULT_PROGRESS_INTERVAL,
    DEFAULT_REFINEMENT_ROUNDS,
    DEFAULT_SAVE_PREDICTIONS,
    DEFAULT_SAVE_VISUALS,
    DEFAULT_SCORE_BATCH_SIZE,
    DEFAULT_SPLIT,
    DEFAULT_START_INDEX,
    DEFAULT_VISUAL_LIMIT,
    build_parser,
    validate_args,
)

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from rg_lns import iterative_component_reassembly
from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
from evaluation.runtime_info import append_runtime_environment
from data_loaders.jpwleg import (
    DEFAULT_DATA_ROOT,
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
from evaluation.common.jpwleg_inputs import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_RUN_DIR,
    parse_bool,
    resolve_checkpoint_arg,
)


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_RUN_NAMES = {
    "jpwleg3": "hard_triplet_d128_s224_2026-07-27-08-14-53",
    "jpwleg5": "jpwleg5_vit_tiny_patch16_224_pretrained_puzzle_hard_triplet_d128_s224_2026-07-17-09-52-33",
}
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]

# First keep the same parameters as LSEJ greedy reassembly; freeze formal values after selecting them on JPwLEG validation.


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _write_pa_case_lists(output_dir: Path, rows: list[dict]) -> tuple[list[str], list[str]]:
    repaired = [
        f"sample_{int(row['sample_index']):06d}.png"
        for row in rows
        if int(row["baseline_PA"]) == 0 and int(row["selected_PA"]) == 1
    ]
    broken = [
        f"sample_{int(row['sample_index']):06d}.png"
        for row in rows
        if int(row["baseline_PA"]) == 1 and int(row["selected_PA"]) == 0
    ]
    for filename, values in (("pa_repaired_images.txt", repaired), ("pa_broken_images.txt", broken)):
        text = "\n".join(values)
        (output_dir / filename).write_text(text + ("\n" if text else ""), encoding="utf-8")
    return repaired, broken


def _save_visual(
    path: Path,
    pieces: list[np.ndarray],
    baseline: np.ndarray,
    seed_prediction: np.ndarray,
    selected: np.ndarray,
    target: np.ndarray,
    component: np.ndarray,
    grid: int,
    title: str,
    initial_solver_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    images = [
        compose_puzzle(pieces, np.arange(grid * grid), grid),
        compose_puzzle(pieces, baseline, grid),
        compose_puzzle(pieces, seed_prediction, grid),
        compose_puzzle(pieces, selected, grid),
        compose_puzzle(pieces, target, grid),
    ]
    names = [
        "Shuffled input puzzle",
        f"Baseline (ViT-T + {initial_solver_name}) (AA {np.count_nonzero(baseline == target)}/{grid * grid})",
        "Last component seed (cyan box)",
        f"Greedy reassembly final solution (AA {np.count_nonzero(selected == target)}/{grid * grid})",
        "Ground-truth puzzle",
    ]
    component_set = {int(piece) for piece in component}
    piece_size = int(pieces[0].shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
    for panel, (axis, image, name) in enumerate(zip(axes, images, names)):
        axis.imshow(image)
        axis.set_title(name)
        axis.axis("off")
        if panel in {1, 3}:
            positions = baseline if panel == 1 else selected
            for piece, position in enumerate(positions):
                if not 0 <= int(position) < grid * grid:
                    continue
                row, column = divmod(int(position), grid)
                color = "#2fb344" if int(position) == int(target[piece]) else "#e03131"
                axis.add_patch(Rectangle(
                    (column * piece_size + 1.5, row * piece_size + 1.5),
                    piece_size - 4,
                    piece_size - 4,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.8,
                ))
        if panel == 2:
            for piece in component_set:
                position = int(seed_prediction[piece])
                if not 0 <= position < grid * grid:
                    continue
                row, column = divmod(position, grid)
                axis.add_patch(Rectangle(
                    (column * piece_size + 1.5, row * piece_size + 1.5),
                    piece_size - 4,
                    piece_size - 4,
                    fill=False,
                    edgecolor="#15aabf",
                    linewidth=2.0,
                ))
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _default_output_dir(
    args: argparse.Namespace,
    checkpoint: Path,
    split: str,
    end_index: int,
    dataset_length: int,
    *,
    output_root: Path,
    initial_solver_tag: str,
) -> Path:
    range_tag = "all" if args.start_index == 0 and end_index == dataset_length else f"start{args.start_index}_n{end_index - args.start_index}"
    solver_part = f"init{initial_solver_tag}_"
    folder = (
        f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{split}_{range_tag}_"
        f"{solver_part}k{args.mutual_top_k}_c{args.component_index}_"
        f"min{args.min_component_size}_r{args.refinement_rounds}_selecte1"
    )
    return Path(output_root) / args.dataset / folder


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
    if args.run_name is None:
        args.run_name = DEFAULT_RUN_NAMES[args.dataset]
    args.checkpoint = resolve_checkpoint_arg(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    split = SPLIT_ALIASES[args.split]
    config = JPWLEG_CONFIGS[args.dataset]
    images, labels = load_jpwleg_arrays(args.data_root, args.dataset, split)
    if args.start_index >= len(images):
        raise ValueError(f"start-index {args.start_index} exceeds dataset length {len(images)}")
    end_index = len(images) if args.max_samples == 0 else min(len(images), args.start_index + args.max_samples)
    sample_indices = list(range(args.start_index, end_index))
    grid = config.grid

    device = resolve_device(args.device, args.gpu_id)
    scorer = MetricCompatibilityScorer.from_checkpoint(
        args.checkpoint,
        device=device,
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
    )
    if args.output_dir is None:
        args.output_dir = _default_output_dir(
            args,
            args.checkpoint,
            split,
            end_index,
            len(images),
            output_root=output_root,
            initial_solver_tag=initial_solver_tag,
        )
    args.output_dir = prepare_unique_output_dir(args.output_dir)
    visual_dir = args.output_dir / "visuals"

    baseline_metrics: list[dict] = []
    selected_metrics: list[dict] = []
    sample_rows: list[dict] = []
    round_rows: list[dict] = []
    candidate_rows: list[dict] = []
    baseline_predictions: list[np.ndarray] = []
    selected_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    selected_component_masks: list[np.ndarray] = []
    status_counts: dict[str, int] = {}
    started = time.perf_counter()

    print(f"Evaluating greedy reassembly on {config.folder} {split}: samples={len(sample_indices)}, grid={grid}x{grid}")
    for evaluation_number, sample_index in enumerate(sample_indices, start=1):
        image = np.asarray(images[sample_index])
        pieces = list(image_to_pieces(image, grid))
        target = label_to_target_positions(np.asarray(labels[sample_index]), grid).astype(np.int32)
        scores = scorer.score_pieces(pieces, rot_flag=0)
        baseline = np.asarray(
            initial_solver(pieces, scores, grid, verbose=args.verbose_solver),
            dtype=np.int32,
        )

        if not is_complete_prediction(baseline, grid):
            status = f"incomplete_{initial_solver_tag}_baseline"
            selected = baseline.copy()
            display_component = np.empty(0, dtype=np.int32)
            display_seed = baseline.copy()
            rounds_attempted = rounds_accepted = total_candidates = 0
            initial_e1_score = final_e1_score = final_mutual_top1 = None
        else:
            result = iterative_component_reassembly(
                scores,
                baseline,
                grid,
                mutual_top_k=args.mutual_top_k,
                component_index=args.component_index,
                min_component_size=args.min_component_size,
                max_rounds=args.refinement_rounds,
            )
            selected = result["prediction"]
            status = str(result["stop_reason"])
            rounds_attempted = int(result["rounds_attempted"])
            rounds_accepted = int(result["rounds_accepted"])
            total_candidates = sum(len(record["candidates"]) for record in result["rounds"])
            initial_e1_score = float(result["initial_score"]["mean_adjacency_score"])
            final_e1_score = float(result["final_score"]["mean_adjacency_score"])
            final_mutual_top1 = int(result["final_score"]["mutual_top1_edges"])
            display_component = result["last_accepted_component"]
            display_seed = result["last_accepted_seed_prediction"]
            if len(display_component) == 0 and result["rounds"]:
                display_component = result["rounds"][-1]["component"]
                display_seed = result["rounds"][-1]["seed_prediction"]

            for round_record in result["rounds"]:
                graph = round_record["graph_statistics"]
                round_rows.append({
                    "sample_index": sample_index,
                    "round": round_record["round"],
                    "status": round_record["status"],
                    "accepted": int(round_record["accepted"]),
                    "component_size": len(round_record["component"]),
                    "component_count": graph["components"],
                    "complete_cycles": graph["complete_cycles"],
                    "cycle_supported_edges": graph["cycle_supported_edges"],
                    "candidate_count": len(round_record["candidates"]),
                    "selected_label": round_record["selected_label"],
                    "selected_row_shift": round_record["selected_row_shift"],
                    "selected_column_shift": round_record["selected_column_shift"],
                })
                for candidate in round_record["candidates"]:
                    candidate_rows.append({
                        "sample_index": sample_index,
                        "round": round_record["round"],
                        "round_status": round_record["status"],
                        "round_accepted": int(round_record["accepted"]),
                        "component_size": len(round_record["component"]),
                        "label": candidate["label"],
                        "row_shift": candidate["row_shift"],
                        "column_shift": candidate["column_shift"],
                        "e1_mean_adjacency_score": candidate["mean_adjacency_score"],
                        "e1_mutual_top1_edges": candidate["mutual_top1_edges"],
                        "selected": int(candidate["selected"]),
                        **sample_ground_truth_metrics(candidate["prediction"], target, grid),
                    })

        baseline_metric = sample_ground_truth_metrics(baseline, target, grid)
        selected_metric = sample_ground_truth_metrics(selected, target, grid)
        baseline_metrics.append(baseline_metric)
        selected_metrics.append(selected_metric)
        status_counts[status] = status_counts.get(status, 0) + 1
        sample_rows.append({
            "sample_index": sample_index,
            "status": status,
            "rounds_attempted": rounds_attempted,
            "rounds_accepted": rounds_accepted,
            "total_candidates": total_candidates,
            "prediction_changed": int(not np.array_equal(baseline, selected)),
            "initial_e1_mean_adjacency_score": initial_e1_score,
            "final_e1_mean_adjacency_score": final_e1_score,
            "final_e1_mutual_top1_edges": final_mutual_top1,
            "baseline_correct_pieces": baseline_metric["correct_pieces"],
            "selected_correct_pieces": selected_metric["correct_pieces"],
            "piece_change": selected_metric["correct_pieces"] - baseline_metric["correct_pieces"],
            **{f"baseline_{key}": baseline_metric[key] for key in ("PA", "AA", "HA", "VA", "SRA", "NA")},
            **{f"selected_{key}": selected_metric[key] for key in ("PA", "AA", "HA", "VA", "SRA", "NA")},
        })
        baseline_predictions.append(baseline.copy())
        selected_predictions.append(selected.copy())
        targets.append(target.copy())
        component_mask = np.zeros(grid * grid, dtype=np.uint8)
        component_mask[np.asarray(display_component, dtype=np.int32)] = 1
        selected_component_masks.append(component_mask)

        if args.save_visuals and (args.visual_limit == 0 or evaluation_number <= args.visual_limit):
            _save_visual(
                visual_dir / f"sample_{sample_index:06d}.png",
                pieces,
                baseline,
                display_seed,
                selected,
                target,
                display_component,
                grid,
                f"{config.folder} | sample {sample_index} | accepted {rounds_accepted}/{rounds_attempted} | {status}",
                initial_solver_name,
            )
        if args.progress_interval > 0 and (
            evaluation_number % args.progress_interval == 0 or evaluation_number == len(sample_indices)
        ):
            running_baseline = aggregate_ground_truth_metrics(baseline_metrics)
            running_selected = aggregate_ground_truth_metrics(selected_metrics)
            print(
                f"[{evaluation_number}/{len(sample_indices)}] sample {sample_index}: {status}, "
                f"rounds={rounds_accepted}/{rounds_attempted}, "
                f"PA {running_baseline['PA']:.2%}->{running_selected['PA']:.2%}, "
                f"AA {running_baseline['AA']:.2%}->{running_selected['AA']:.2%}, "
                f"SRA {running_baseline['SRA']:.2%}->{running_selected['SRA']:.2%}"
            )

    elapsed_seconds = time.perf_counter() - started
    baseline_aggregate = aggregate_ground_truth_metrics(baseline_metrics)
    selected_aggregate = aggregate_ground_truth_metrics(selected_metrics)
    piece_changes = [int(row["piece_change"]) for row in sample_rows]
    comparison = {
        "improved_puzzles": sum(change > 0 for change in piece_changes),
        "damaged_puzzles": sum(change < 0 for change in piece_changes),
        "unchanged_puzzles": sum(change == 0 for change in piece_changes),
        "changed_predictions": sum(int(row["prediction_changed"]) for row in sample_rows),
        "repaired_to_perfect": sum(int(row["baseline_PA"]) == 0 and int(row["selected_PA"]) == 1 for row in sample_rows),
        "broken_perfect": sum(int(row["baseline_PA"]) == 1 and int(row["selected_PA"]) == 0 for row in sample_rows),
        "accepted_rounds": sum(int(row["rounds_accepted"]) for row in sample_rows),
    }
    repaired_images, broken_images = _write_pa_case_lists(args.output_dir, sample_rows)
    payload = {
        "method": f"s1a3_iterative_component_reassembly_{initial_solver_tag}_initialization",
        "initial_solver": initial_solver_name,
        "uses_training": False,
        "uses_ground_truth_for_selection": False,
        "selection_rule": "strictly minimize E1 mean adjacency score; maximize mutual-Top1 edges only as tie-break",
        "dataset": args.dataset,
        "split": split,
        "start_index": args.start_index,
        "end_index_exclusive": end_index,
        "evaluated_samples": len(sample_indices),
        "checkpoint": str(args.checkpoint),
        "configuration": {
            "mutual_top_k": args.mutual_top_k,
            "component_index": args.component_index,
            "min_component_size": args.min_component_size,
            "refinement_rounds": args.refinement_rounds,
            "postprocess": args.postprocess,
            "initial_solver": initial_solver_config or {},
        },
        "status_counts": status_counts,
        "baseline": baseline_aggregate,
        "s1a3": selected_aggregate,
        "comparison": comparison,
        "elapsed_seconds": elapsed_seconds,
        "pa_case_lists": {
            "repaired": "pa_repaired_images.txt",
            "broken": "pa_broken_images.txt",
            "repaired_images": len(repaired_images),
            "broken_images": len(broken_images),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for filename, rows in (("sample_results.csv", sample_rows), ("round_results.csv", round_rows), ("candidates.csv", candidate_rows)):
        if not rows:
            continue
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if args.save_predictions:
        np.savez_compressed(
            args.output_dir / "predictions.npz",
            sample_indices=np.asarray(sample_indices, dtype=np.int32),
            baseline_predictions=np.stack(baseline_predictions),
            selected_predictions=np.stack(selected_predictions),
            targets=np.stack(targets),
            selected_component_masks=np.stack(selected_component_masks),
            grid_size=np.asarray(grid, dtype=np.int32),
        )

    summary_lines = [
        f"JPwLEG Greedy Component Reassembly ({initial_solver_name} initialization)",
        "=" * 80,
        f"checkpoint: {args.checkpoint}",
        f"dataset/split: {args.dataset} / {split}",
        f"samples: [{args.start_index}, {end_index}) = {len(sample_indices)}",
        f"configuration: k={args.mutual_top_k}, component={args.component_index}, min_size={args.min_component_size}, rounds={args.refinement_rounds}",
        f"selection: {payload['selection_rule']}",
        f"status: {status_counts}",
        "",
        f"Baseline (ViT-T + {initial_solver_name}):",
        f"  PA={baseline_aggregate['PA']:.4%}",
        f"  AA={baseline_aggregate['AA']:.4%}",
        f"  HA={baseline_aggregate['HA']:.4%}",
        f"  VA={baseline_aggregate['VA']:.4%}",
        f"  SRA={baseline_aggregate['SRA']:.4%}",
        f"  NA={baseline_aggregate['NA']:.4%}",
        "",
        "Greedy reassembly:",
        f"  PA={selected_aggregate['PA']:.4%}",
        f"  AA={selected_aggregate['AA']:.4%}",
        f"  HA={selected_aggregate['HA']:.4%}",
        f"  VA={selected_aggregate['VA']:.4%}",
        f"  SRA={selected_aggregate['SRA']:.4%}",
        f"  NA={selected_aggregate['NA']:.4%}",
        "",
        f"Comparison: {comparison}",
        f"PA repaired image list: pa_repaired_images.txt ({len(repaired_images)})",
        f"PA broken image list: pa_broken_images.txt ({len(broken_images)})",
        f"elapsed_seconds: {elapsed_seconds:.10g}",
    ]
    (args.output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    append_runtime_environment(
        args.output_dir / "summary.txt",
        requested_device=args.device,
        resolved_device=device,
        gpu_id=args.gpu_id,
    )

    print(f"device: {device}")
    print(
        f"Final AA: {baseline_aggregate['AA']:.4%} -> {selected_aggregate['AA']:.4%}; "
        f"SRA: {baseline_aggregate['SRA']:.4%} -> {selected_aggregate['SRA']:.4%}; "
        f"PA: {baseline_aggregate['PA']:.4%} -> {selected_aggregate['PA']:.4%}"
    )
    print(f"Saved: {args.output_dir}")
    return payload


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
