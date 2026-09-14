"""
@file: s1a3_eval_gap_iterative_component_reassembly.py
@description: 在 GAP-3/GAP-5 上加载对应的 S1A RGBA 度量模型，先以 Gallagher Solver
              生成初始布局，再执行 S1A3 单路径迭代组件重组。评估同时报告重组前后的
              PA、AA、HA、VA、SRA、NA，以及不受求解器影响的真实邻居 Recall@K；并保存
              逐样本、逐轮、候选、预测、PA 修复/破坏列表和带正确性框的可视化结果。
@author: Changxin Ye
@created: 2026-08-01
@version: 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

try:
    from .assembly_solvers.our_iterative_component_reassembly import iterative_component_reassembly
    from .experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from .experiment_runtime_info import append_runtime_environment
    from .metric_gap_data import DEFAULT_GAP_DATA_ROOT, GAP_CONFIGS, GAP_SPLITS, GAPPuzzleDataset
    from .metric_gap_evaluator import DEFAULT_NEIGHBOR_RECALL_KS, rgba_to_rgb
    from .metric_jpleg_evaluator import (
        aggregate_ground_truth_metrics,
        compose_puzzle,
        is_complete_prediction,
        sample_ground_truth_metrics,
        solve_with_gallagher,
    )
    from .metric_lsej_evaluator import NeighborRetrievalMetrics, compute_true_neighbor_ranks
    from .metric_scorer import (
        MetricCompatibilityScorer,
        load_metric_checkpoint,
        resolve_device,
    )
    from .s1a_eval_gap import (
        DEFAULT_LOG_ROOT,
        parse_bool,
        resolve_checkpoint,
    )
except ImportError:
    from assembly_solvers.our_iterative_component_reassembly import iterative_component_reassembly
    from experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from experiment_runtime_info import append_runtime_environment
    from metric_gap_data import DEFAULT_GAP_DATA_ROOT, GAP_CONFIGS, GAP_SPLITS, GAPPuzzleDataset
    from metric_gap_evaluator import DEFAULT_NEIGHBOR_RECALL_KS, rgba_to_rgb
    from metric_jpleg_evaluator import (
        aggregate_ground_truth_metrics,
        compose_puzzle,
        is_complete_prediction,
        sample_ground_truth_metrics,
        solve_with_gallagher,
    )
    from metric_lsej_evaluator import NeighborRetrievalMetrics, compute_true_neighbor_ranks
    from metric_scorer import MetricCompatibilityScorer, load_metric_checkpoint, resolve_device
    from s1a_eval_gap import (
        DEFAULT_LOG_ROOT,
        parse_bool,
        resolve_checkpoint,
    )


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "GAP-3"  # "GAP-3" or "GAP-5"
DEFAULT_SPLIT = "test"
DEFAULT_START_INDEX = 0
DEFAULT_MAX_SAMPLES = 0  # 0 表示评估完整 split
DEFAULT_RUN_NAMES = {
    "GAP-3": "hard_triplet_d128_s224_kall_2026-07-31-20-21-04",
    "GAP-5": "hard_triplet_d128_s224_k15_2026-07-31-20-43-26",
}
DEFAULT_RUN_DIR = None
DEFAULT_CHECKPOINT_NAME = "best.pth"

# 与 LSEJ / JPwLEG 的正式 S1A3 设置保持一致。
DEFAULT_MUTUAL_TOP_K = 3
DEFAULT_COMPONENT_INDEX = 0
DEFAULT_MIN_COMPONENT_SIZE = 4
DEFAULT_REFINEMENT_ROUNDS = 1

DEFAULT_SCORE_BATCH_SIZE = 128
DEFAULT_POSTPROCESS = True
DEFAULT_SAVE_VISUALS = True
DEFAULT_VISUAL_LIMIT = 10
DEFAULT_SAVE_PREDICTIONS = True
DEFAULT_PROGRESS_INTERVAL = 25
METRIC_KEYS = ("PA", "AA", "HA", "VA", "SRA", "NA")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Evaluate S1A3 iterative component reassembly on GAP-3/GAP-5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=GAP_CONFIGS.keys())
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=GAP_SPLITS)
    parser.add_argument("--data-root", default=DEFAULT_GAP_DATA_ROOT, type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path, help="精确 checkpoint 路径，优先级最高")
    parser.add_argument(
        "--run-name",
        default=None,
        help="logs/s1a_train_gap/<dataset> 下的实验名；留空时按 dataset 使用 DEFAULT_RUN_NAMES",
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--start-index", default=DEFAULT_START_INDEX, type=int)
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int)
    parser.add_argument("--mutual-top-k", default=DEFAULT_MUTUAL_TOP_K, type=int)
    parser.add_argument("--component-index", default=DEFAULT_COMPONENT_INDEX, type=int)
    parser.add_argument("--min-component-size", default=DEFAULT_MIN_COMPONENT_SIZE, type=int)
    parser.add_argument("--refinement-rounds", default=DEFAULT_REFINEMENT_ROUNDS, type=int)
    parser.add_argument("--score-batch-size", default=DEFAULT_SCORE_BATCH_SIZE, type=int)
    parser.add_argument("--postprocess", default=DEFAULT_POSTPROCESS, type=parse_bool)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--save-visuals", default=DEFAULT_SAVE_VISUALS, type=parse_bool)
    parser.add_argument("--visual-limit", default=DEFAULT_VISUAL_LIMIT, type=int)
    parser.add_argument("--save-predictions", default=DEFAULT_SAVE_PREDICTIONS, type=parse_bool)
    parser.add_argument("--progress-interval", default=DEFAULT_PROGRESS_INTERVAL, type=int)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0 or args.max_samples < 0:
        raise ValueError("start-index 和 max-samples 必须非负")
    if args.mutual_top_k <= 0:
        raise ValueError("mutual-top-k 必须为正整数")
    if args.component_index < 0:
        raise ValueError("component-index 不能为负数")
    if args.min_component_size <= 0 or args.refinement_rounds <= 0:
        raise ValueError("min-component-size 和 refinement-rounds 必须为正整数")
    if args.score_batch_size <= 0:
        raise ValueError("score-batch-size 必须为正整数")
    if args.visual_limit < 0 or args.progress_interval < 0:
        raise ValueError("visual-limit 和 progress-interval 必须非负")


def _prepare_output_dir(output_dir: Path, *, automatically_named: bool) -> Path:
    """避免不同参数的重复评估混写到同一个结果目录。"""

    del automatically_named
    return prepare_unique_output_dir(output_dir)


def _default_output_dir(checkpoint: Path, dataset: str, *, output_root: Path) -> Path:
    # 目录名只保留 checkpoint 身份；全部评估参数写入 summary.txt / metrics.json。
    folder = f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}"
    return Path(output_root) / dataset / folder


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    """保存输入、初始求解器布局、组件种子、S1A3 解和真实布局。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rgb_pieces = [rgba_to_rgb(piece) for piece in pieces]
    images = [
        compose_puzzle(rgb_pieces, np.arange(grid * grid), grid),
        compose_puzzle(rgb_pieces, baseline, grid),
        compose_puzzle(rgb_pieces, seed_prediction, grid),
        compose_puzzle(rgb_pieces, selected, grid),
        compose_puzzle(rgb_pieces, target, grid),
    ]
    names = [
        "Official shuffled input",
        f"S1A {initial_solver_name} baseline "
        f"(AA {np.count_nonzero(baseline == target)}/{grid * grid})",
        "Last component seed (cyan)",
        f"S1A3 final (AA {np.count_nonzero(selected == target)}/{grid * grid})",
        "Ground truth",
    ]
    component_set = {int(piece) for piece in component}
    piece_height, piece_width = rgb_pieces[0].shape[:2]
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
                axis.add_patch(
                    Rectangle(
                        (column * piece_width + 1.5, row * piece_height + 1.5),
                        piece_width - 4,
                        piece_height - 4,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.8,
                    )
                )
        if panel == 2:
            for piece in component_set:
                position = int(seed_prediction[piece])
                if not 0 <= position < grid * grid:
                    continue
                row, column = divmod(position, grid)
                axis.add_patch(
                    Rectangle(
                        (column * piece_width + 1.5, row * piece_height + 1.5),
                        piece_width - 4,
                        piece_height - 4,
                        fill=False,
                        edgecolor="#15aabf",
                        linewidth=2.0,
                    )
                )
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _build_scorer(args: argparse.Namespace, checkpoint: Path):
    device = resolve_device(args.device, args.gpu_id)
    model, model_config, payload = load_metric_checkpoint(checkpoint, device)
    checkpoint_dataset = payload.get("dataset") or payload.get("args", {}).get("dataset")
    if checkpoint_dataset not in {None, args.dataset}:
        raise ValueError(
            f"Checkpoint dataset {checkpoint_dataset} does not match requested {args.dataset}"
        )
    if int(payload.get("model_config", {}).get("input_channels", 3)) != 4:
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
    return scorer, device, model_config


def _retrieval_metrics(
    rank_histogram: np.ndarray,
    horizontal_rank_histogram: np.ndarray,
    vertical_rank_histogram: np.ndarray,
    num_pieces: int,
) -> NeighborRetrievalMetrics:
    recall_ks = tuple(k for k in DEFAULT_NEIGHBOR_RECALL_KS if k <= num_pieces - 1)
    return NeighborRetrievalMetrics(
        recall_ks=recall_ks,
        rank_histogram=rank_histogram,
        horizontal_rank_histogram=horizontal_rank_histogram,
        vertical_rank_histogram=vertical_rank_histogram,
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
    if args.run_name is None:
        args.run_name = DEFAULT_RUN_NAMES[args.dataset]
    checkpoint = resolve_checkpoint(args)
    scorer, device, model_config = _build_scorer(args, checkpoint)

    gap_data = GAPPuzzleDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        start_index=args.start_index,
        max_samples=args.max_samples or None,
    )
    grid = gap_data.config.grid
    num_pieces = gap_data.config.num_pieces
    automatically_named_output = args.output_dir is None
    output_dir = (
        _default_output_dir(checkpoint, args.dataset, output_root=output_root)
        if automatically_named_output
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir = _prepare_output_dir(output_dir, automatically_named=automatically_named_output)
    visual_dir = output_dir / "visuals"

    baseline_metrics: list[dict] = []
    selected_metrics: list[dict] = []
    sample_rows: list[dict] = []
    round_rows: list[dict] = []
    candidate_rows: list[dict] = []
    baseline_predictions: list[np.ndarray] = []
    selected_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    selected_component_masks: list[np.ndarray] = []
    sample_indices: list[int] = []
    status_counts: dict[str, int] = {}
    rank_histogram = np.zeros(num_pieces, dtype=np.int64)
    horizontal_rank_histogram = np.zeros(num_pieces, dtype=np.int64)
    vertical_rank_histogram = np.zeros(num_pieces, dtype=np.int64)
    complete_initial_baselines = 0
    started = time.perf_counter()

    print(
        f"Evaluating S1A3 on {args.dataset} {args.split}: "
        f"samples={len(gap_data)}, grid={grid}x{grid}"
    )
    for evaluation_number in range(1, len(gap_data) + 1):
        sample = gap_data[evaluation_number - 1]
        sample_index = int(sample["sample_index"])
        pieces = list(sample["pieces"])
        target = np.asarray(sample["target"], dtype=np.int32)
        scores = scorer.score_pieces(pieces, rot_flag=0)

        ranks = compute_true_neighbor_ranks(scores, target, grid)
        rank_histogram += np.bincount(ranks["all"], minlength=num_pieces)[:num_pieces]
        horizontal_rank_histogram += np.bincount(
            ranks["horizontal"], minlength=num_pieces
        )[:num_pieces]
        vertical_rank_histogram += np.bincount(
            ranks["vertical"], minlength=num_pieces
        )[:num_pieces]

        solver_pieces = [rgba_to_rgb(piece) for piece in pieces]
        baseline = np.asarray(
            initial_solver(
                solver_pieces,
                scores,
                grid,
                verbose=args.verbose_solver,
            ),
            dtype=np.int32,
        )
        complete_initial_baselines += int(is_complete_prediction(baseline, grid))

        if not is_complete_prediction(baseline, grid):
            status = f"incomplete_{initial_solver_tag}_baseline"
            selected = baseline.copy()
            display_component = np.empty(0, dtype=np.int32)
            display_seed = baseline.copy()
            rounds_attempted = rounds_accepted = total_candidates = 0
            initial_e1_score = final_e1_score = None
            final_mutual_top1 = None
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
            selected = np.asarray(result["prediction"], dtype=np.int32)
            status = str(result["stop_reason"])
            rounds_attempted = int(result["rounds_attempted"])
            rounds_accepted = int(result["rounds_accepted"])
            total_candidates = sum(len(record["candidates"]) for record in result["rounds"])
            initial_e1_score = float(result["initial_score"]["mean_adjacency_score"])
            final_e1_score = float(result["final_score"]["mean_adjacency_score"])
            final_mutual_top1 = int(result["final_score"]["mutual_top1_edges"])
            display_component = np.asarray(result["last_accepted_component"], dtype=np.int32)
            display_seed = np.asarray(result["last_accepted_seed_prediction"], dtype=np.int32)
            if len(display_component) == 0 and result["rounds"]:
                display_component = np.asarray(result["rounds"][-1]["component"], dtype=np.int32)
                display_seed = np.asarray(result["rounds"][-1]["seed_prediction"], dtype=np.int32)

            for round_record in result["rounds"]:
                graph = round_record["graph_statistics"]
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
                        "candidate_count": len(round_record["candidates"]),
                        "selected_label": round_record["selected_label"],
                        "selected_row_shift": round_record["selected_row_shift"],
                        "selected_column_shift": round_record["selected_column_shift"],
                    }
                )
                for candidate in round_record["candidates"]:
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
                            "e1_mean_adjacency_score": candidate["mean_adjacency_score"],
                            "e1_mutual_top1_edges": candidate["mutual_top1_edges"],
                            "selected": int(candidate["selected"]),
                            **sample_ground_truth_metrics(candidate["prediction"], target, grid),
                        }
                    )

        baseline_metric = sample_ground_truth_metrics(baseline, target, grid)
        selected_metric = sample_ground_truth_metrics(selected, target, grid)
        baseline_metrics.append(baseline_metric)
        selected_metrics.append(selected_metric)
        status_counts[status] = status_counts.get(status, 0) + 1
        sample_rows.append(
            {
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
                "piece_change": selected_metric["correct_pieces"]
                - baseline_metric["correct_pieces"],
                **{f"baseline_{key}": baseline_metric[key] for key in METRIC_KEYS},
                **{f"selected_{key}": selected_metric[key] for key in METRIC_KEYS},
            }
        )
        sample_indices.append(sample_index)
        baseline_predictions.append(baseline.copy())
        selected_predictions.append(selected.copy())
        targets.append(target.copy())
        component_mask = np.zeros(num_pieces, dtype=np.uint8)
        component_mask[display_component] = 1
        selected_component_masks.append(component_mask)

        if args.save_visuals and (
            args.visual_limit == 0 or evaluation_number <= args.visual_limit
        ):
            _save_visual(
                visual_dir / f"sample_{sample_index:06d}.png",
                pieces,
                baseline,
                display_seed,
                selected,
                target,
                display_component,
                grid,
                f"{args.dataset} {args.split} | sample {sample_index} | "
                f"accepted {rounds_accepted}/{rounds_attempted} | {status}",
                initial_solver_name,
            )
        if args.progress_interval > 0 and (
            evaluation_number % args.progress_interval == 0
            or evaluation_number == len(gap_data)
        ):
            running_baseline = aggregate_ground_truth_metrics(baseline_metrics)
            running_selected = aggregate_ground_truth_metrics(selected_metrics)
            print(
                f"[{evaluation_number}/{len(gap_data)}] sample {sample_index}: {status}, "
                f"rounds={rounds_accepted}/{rounds_attempted}, "
                f"PA {running_baseline['PA']:.2%}->{running_selected['PA']:.2%}, "
                f"AA {running_baseline['AA']:.2%}->{running_selected['AA']:.2%}, "
                f"SRA {running_baseline['SRA']:.2%}->{running_selected['SRA']:.2%}"
            )

    elapsed_seconds = time.perf_counter() - started
    gap_data.close()
    baseline_aggregate = aggregate_ground_truth_metrics(baseline_metrics)
    selected_aggregate = aggregate_ground_truth_metrics(selected_metrics)
    retrieval = _retrieval_metrics(
        rank_histogram,
        horizontal_rank_histogram,
        vertical_rank_histogram,
        num_pieces,
    )
    piece_changes = [int(row["piece_change"]) for row in sample_rows]
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
    }
    repaired_images, broken_images = _write_pa_case_lists(output_dir, sample_rows)
    payload = {
        "method": f"s1a3_iterative_component_reassembly_{initial_solver_tag}_initialization",
        "initial_solver": initial_solver_name,
        "uses_training": False,
        "uses_ground_truth_for_selection": False,
        "selection_rule": (
            "strictly minimize E1 mean adjacency score; maximize mutual-Top1 edges only as tie-break"
        ),
        "dataset": args.dataset,
        "split": args.split,
        "start_index": args.start_index,
        "evaluated_samples": len(sample_rows),
        "checkpoint": str(checkpoint),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "configuration": {
            "mutual_top_k": args.mutual_top_k,
            "component_index": args.component_index,
            "min_component_size": args.min_component_size,
            "refinement_rounds": args.refinement_rounds,
            "postprocess": args.postprocess,
            "score_batch_size": args.score_batch_size,
            "score_metric": model_config["score_metric"],
            "input_channels": 4,
            "rgba_adapter": "learned_conv1x1_4_to_3",
            "initial_solver": initial_solver_config or {},
        },
        "complete_initial_solver_baselines": complete_initial_baselines,
        "status_counts": status_counts,
        "baseline": baseline_aggregate,
        "s1a3": selected_aggregate,
        "neighbor_retrieval": retrieval.to_dict(),
        "comparison": comparison,
        "elapsed_seconds": elapsed_seconds,
        "pa_case_lists": {
            "repaired": "pa_repaired_images.txt",
            "broken": "pa_broken_images.txt",
            "repaired_images": len(repaired_images),
            "broken_images": len(broken_images),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(output_dir / "sample_results.csv", sample_rows)
    _write_csv(output_dir / "round_results.csv", round_rows)
    _write_csv(output_dir / "candidates.csv", candidate_rows)

    if args.save_predictions:
        np.savez_compressed(
            output_dir / "predictions.npz",
            sample_indices=np.asarray(sample_indices, dtype=np.int32),
            baseline_predictions=np.stack(baseline_predictions),
            selected_predictions=np.stack(selected_predictions),
            targets=np.stack(targets),
            selected_component_masks=np.stack(selected_component_masks),
            grid_size=np.asarray(grid, dtype=np.int32),
            dataset=np.asarray(args.dataset),
            split=np.asarray(args.split),
        )

    summary_lines = [
        f"GAP S1A3 Iterative Component Reassembly ({initial_solver_name} initialization)",
        "=" * 80,
        f"checkpoint: {checkpoint}",
        f"data_root: {Path(args.data_root).expanduser().resolve()}",
        f"dataset/split: {args.dataset} / {args.split}",
        f"samples: {len(sample_rows)} (start_index={args.start_index})",
        f"complete_initial_solver_baselines: {complete_initial_baselines}/{len(sample_rows)}",
        f"configuration: k={args.mutual_top_k}, component={args.component_index}, "
        f"min_size={args.min_component_size}, rounds={args.refinement_rounds}, "
        f"postprocess={args.postprocess}, score_batch_size={args.score_batch_size}",
        f"selection: {payload['selection_rule']}",
        f"initial_solver: {initial_solver_name}",
        f"initial_solver_config: {initial_solver_config or {}}",
        f"status: {status_counts}",
        "",
    ]
    for label, metrics in (
        (f"S1A {initial_solver_name} baseline", baseline_aggregate),
        ("S1A3", selected_aggregate),
    ):
        summary_lines.append(f"{label}:")
        summary_lines.extend(f"  {key}={metrics[key]:.4%}" for key in METRIC_KEYS)
        summary_lines.append("")
    summary_lines.extend(
        [
            f"True-Neighbor MRR: {retrieval.mrr:.6f}",
            f"Mean True-Neighbor Rank: {retrieval.mean_rank:.3f}",
        ]
    )
    for k in retrieval.recall_ks:
        summary_lines.append(f"True-Neighbor Recall@{k}: {retrieval.recall_at(k):.4%}")
    summary_lines.extend(
        [
            "",
            f"Comparison: {comparison}",
            f"PA repaired image list: pa_repaired_images.txt ({len(repaired_images)})",
            f"PA broken image list: pa_broken_images.txt ({len(broken_images)})",
            f"elapsed_seconds: {elapsed_seconds:.10g}",
        ]
    )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    append_runtime_environment(
        output_dir / "summary.txt",
        requested_device=args.device,
        resolved_device=device,
        gpu_id=args.gpu_id,
    )

    print(f"device: {device}")
    for key in ("PA", "AA", "SRA", "NA"):
        print(
            f"Final {key} ({initial_solver_name} -> S1A3): "
            f"{baseline_aggregate[key]:.4%} -> {selected_aggregate[key]:.4%}"
        )
    print(f"Saved: {output_dir}")
    return payload


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
