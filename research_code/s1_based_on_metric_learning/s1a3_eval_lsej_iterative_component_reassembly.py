"""
@file: s1a3_eval_lsej_iterative_component_reassembly.py
@description: 复用 S1A 已训练的 E1 piece-to-piece embedding 与兼容性矩阵，先通过
              Gallagher 求解器获得初始解，再根据 E1 互惠 Top-K 关系与局部四边环约束，
              迭代提取高置信连通组件。每轮枚举组件的合法平移，使用 E1 兼容性
              贪心重填剩余 piece，并仅在全局 E1 邻接目标严格改善时接受新布局。
@author: Changxin Ye
@created: 2026-07-24
@version: 1.2
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

try:
    from .experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from .experiment_runtime_info import append_runtime_environment
    from .metric_lsej_data import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS, chw_uint8_to_hwc_numpy, load_official_dataset_class
    from .metric_lsej_evaluator import (
        compose_puzzle,
        compute_relationship_counts,
        is_complete_prediction,
        solve_with_gallagher,
    )
    from .assembly_solvers.our_iterative_component_reassembly import iterative_component_reassembly
    from .metric_scorer import MetricCompatibilityScorer, resolve_device
    from .s1a_eval_lsej import (
        LOSS_TYPES,
        parse_bool,
        resolve_checkpoint_arg,
        validate_checkpoint,
    )
except ImportError:
    from experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from experiment_runtime_info import append_runtime_environment
    from metric_lsej_data import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS, chw_uint8_to_hwc_numpy, load_official_dataset_class
    from metric_lsej_evaluator import compose_puzzle, compute_relationship_counts, is_complete_prediction, solve_with_gallagher
    from assembly_solvers.our_iterative_component_reassembly import iterative_component_reassembly
    from metric_scorer import MetricCompatibilityScorer, resolve_device
    from s1a_eval_lsej import (
        LOSS_TYPES,
        parse_bool,
        resolve_checkpoint_arg,
        validate_checkpoint,
    )


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET_NAME = "ImageNet_LSEJ"
DEFAULT_TASK = "grid10_erode2"
DEFAULT_SPLIT = "test"
DEFAULT_START_INDEX = 0
DEFAULT_MAX_SAMPLES = 0
DEFAULT_LOSS_TYPE = "hard_triplet"
DEFAULT_RUN_NAME = None
if DEFAULT_LOSS_TYPE == "hard_triplet":
    if DEFAULT_TASK == "grid10_erode2":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-07-25-10-22-41"
    elif DEFAULT_TASK == "grid10_erode5":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-07-19-16-45-16"
    elif DEFAULT_TASK == "grid10_erode8":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-07-19-16-46-17"
DEFAULT_RUN_DIR = None
DEFAULT_CHECKPOINT_NAME = "best.pth"

# S1A3 核心参数。ROUNDS=1 用于复现单轮 E1-only 候选重排结果。
DEFAULT_MUTUAL_TOP_K = 3
DEFAULT_COMPONENT_INDEX = 0
DEFAULT_MIN_COMPONENT_SIZE = 4
DEFAULT_REFINEMENT_ROUNDS = 1

DEFAULT_SCORE_BATCH_SIZE = 128
DEFAULT_POSTPROCESS = True
DEFAULT_SAVE_VISUALS = True
DEFAULT_VISUAL_LIMIT = 10
DEFAULT_SAVE_PREDICTIONS = True
DEFAULT_PROGRESS_INTERVAL = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "使用冻结的 S1A E1 兼容性执行迭代式高置信组件引导重组",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK, choices=OFFICIAL_TASKS, help="要评估的官方 LSEJ 任务")
    parser.add_argument("--loss-type", default=DEFAULT_LOSS_TYPE, choices=LOSS_TYPES, help="checkpoint 对应的训练损失")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["train", "val", "test"], help="要评估的数据划分")
    parser.add_argument("--data-root", default=DEFAULT_LSEJ_DATA_ROOT, type=Path, help="ImageNet-LSEJ 数据集根目录")
    parser.add_argument("--checkpoint", default=None, type=Path, help="精确 checkpoint 路径；设置后优先级最高")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="当前 task 日志目录下的实验文件夹名")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="直接指定包含 checkpoints 的实验目录")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="要读取的 checkpoint 文件名")
    parser.add_argument("--start-index", default=DEFAULT_START_INDEX, type=int, help="从 split 中第几张拼图开始")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="最多评估多少张拼图；0 表示完整 split")
    parser.add_argument("--mutual-top-k", default=DEFAULT_MUTUAL_TOP_K, type=int, help="构造互惠邻接图时每个方向保留的候选数")
    parser.add_argument("--component-index", default=DEFAULT_COMPONENT_INDEX, type=int, help="按规模排序后选取第几个闭环组件，0 表示最大组件")
    parser.add_argument("--min-component-size", default=DEFAULT_MIN_COMPONENT_SIZE, type=int, help="允许执行重组的最小组件 piece 数")
    parser.add_argument("--refinement-rounds", default=DEFAULT_REFINEMENT_ROUNDS, type=int, help="最多执行多少轮组件提取和平移重组")
    parser.add_argument("--score-batch-size", default=DEFAULT_SCORE_BATCH_SIZE, type=int, help="提取边缘 embedding 时的前向批量")
    parser.add_argument("--postprocess", default=DEFAULT_POSTPROCESS, type=parse_bool, help="是否执行 S1A 兼容分数后处理")
    parser.add_argument("--device", default="auto", help="运行设备，例如 auto、cpu、cuda:0")
    parser.add_argument("--gpu-id", default=0, type=int, help="device=auto 时使用的 GPU 编号")
    parser.add_argument("--verbose-solver", action="store_true", help="显示初始求解器的内部信息")
    parser.add_argument("--output-dir", default=None, type=Path, help="结果保存目录；留空时自动生成")
    parser.add_argument("--save-visuals", default=DEFAULT_SAVE_VISUALS, type=parse_bool, help="是否保存重组前后可视化")
    parser.add_argument("--visual-limit", default=DEFAULT_VISUAL_LIMIT, type=int, help="最多保存多少张可视化图片")
    parser.add_argument("--save-predictions", default=DEFAULT_SAVE_PREDICTIONS, type=parse_bool, help="是否保存 predictions.npz")
    parser.add_argument("--progress-interval", default=DEFAULT_PROGRESS_INTERVAL, type=int, help="每处理多少张拼图打印一次指标；0 表示关闭")
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


def _float_metric(value: float) -> str:
    return f"{float(value):.10g}"


def _ground_truth_metrics(prediction: np.ndarray, target: np.ndarray, grid: int) -> dict[str, float | int]:
    relationships = compute_relationship_counts(prediction, target, grid)
    correct_pieces = int(np.count_nonzero(prediction == target))
    horizontal_correct = int(relationships["horizontal_correct"])
    horizontal_total = int(relationships["horizontal_total"])
    vertical_correct = int(relationships["vertical_correct"])
    vertical_total = int(relationships["vertical_total"])
    return {
        "correct_pieces": correct_pieces,
        "total_pieces": grid * grid,
        "PA": int(correct_pieces == grid * grid),
        "AA": correct_pieces / (grid * grid),
        "horizontal_correct": horizontal_correct,
        "horizontal_total": horizontal_total,
        "horizontal_SRA": horizontal_correct / horizontal_total,
        "vertical_correct": vertical_correct,
        "vertical_total": vertical_total,
        "vertical_SRA": vertical_correct / vertical_total,
        "SRA": (horizontal_correct + vertical_correct) / (horizontal_total + vertical_total),
    }


def _aggregate_metrics(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot aggregate an empty metric list")
    samples = len(rows)
    horizontal_correct = sum(int(row["horizontal_correct"]) for row in rows)
    horizontal_total = sum(int(row["horizontal_total"]) for row in rows)
    vertical_correct = sum(int(row["vertical_correct"]) for row in rows)
    vertical_total = sum(int(row["vertical_total"]) for row in rows)
    return {
        "samples": samples,
        "perfect_puzzles": sum(int(row["PA"]) for row in rows),
        "correct_pieces": sum(int(row["correct_pieces"]) for row in rows),
        "total_pieces": sum(int(row["total_pieces"]) for row in rows),
        "PA": sum(int(row["PA"]) for row in rows) / samples,
        "AA": sum(float(row["AA"]) for row in rows) / samples,
        "horizontal_SRA": horizontal_correct / horizontal_total,
        "vertical_SRA": vertical_correct / vertical_total,
        "SRA": (horizontal_correct + vertical_correct) / (horizontal_total + vertical_total),
    }


def _write_pa_case_lists(output_dir: Path, rows: list[dict]) -> tuple[list[str], list[str]]:
    repaired = [
        f"sample_{int(row['sample_index']):06d}_{row['lsej_id']}.png"
        for row in rows
        if int(row["baseline_PA"]) == 0 and int(row["selected_PA"]) == 1
    ]
    broken = [
        f"sample_{int(row['sample_index']):06d}_{row['lsej_id']}.png"
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
    initial_solver_name: str = "Gallagher",
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
        "Official shuffled input",
        f"S1A {initial_solver_name} baseline (AA {np.count_nonzero(baseline == target)}/{grid * grid})",
        "Last component seed (cyan)",
        f"S1A3 final (AA {np.count_nonzero(selected == target)}/{grid * grid})",
        "Ground truth",
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
                axis.add_patch(
                    Rectangle(
                        (column * piece_size + 1.5, row * piece_size + 1.5),
                        piece_size - 4,
                        piece_size - 4,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.8,
                    )
                )
        if panel == 2:
            for piece in component_set:
                row, column = divmod(int(seed_prediction[piece]), grid)
                axis.add_patch(
                    Rectangle(
                        (column * piece_size + 1.5, row * piece_size + 1.5),
                        piece_size - 4,
                        piece_size - 4,
                        fill=False,
                        edgecolor="#15aabf",
                        linewidth=2.0,
                    )
                )
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _default_output_dir(
    args: argparse.Namespace,
    checkpoint: Path,
    end_index: int,
    dataset_length: int,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    initial_solver_tag: str = "gallagher",
) -> Path:
    range_tag = (
        "all"
        if args.start_index == 0 and end_index == dataset_length
        else f"start{args.start_index}_n{end_index - args.start_index}"
    )
    solver_part = "" if initial_solver_tag == "gallagher" else f"init{initial_solver_tag}_"
    folder = (
        f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{args.task}_{args.split}_{range_tag}_"
        f"{solver_part}"
        f"k{args.mutual_top_k}_c{args.component_index}_min{args.min_component_size}_"
        f"r{args.refinement_rounds}_selecte1"
    )
    return Path(output_root) / DEFAULT_DATASET_NAME / folder


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
    validate_checkpoint(args.checkpoint, args.task, args.loss_type)

    dataset = load_official_dataset_class()(args.data_root, split=args.split, task=args.task, normalize=False)
    if args.start_index >= len(dataset):
        raise ValueError(f"start-index {args.start_index} is outside dataset length {len(dataset)}")
    end_index = len(dataset) if args.max_samples == 0 else min(len(dataset), args.start_index + args.max_samples)
    sample_indices = list(range(args.start_index, end_index))
    grid = int(dataset.config["puzzle"]["grid_rows"])
    if grid != 10:
        raise ValueError("S1A3 iterative component reassembly currently supports only grid10")

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
            end_index,
            len(dataset),
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
    lsej_ids: list[str] = []
    status_counts: dict[str, int] = {}
    started = time.perf_counter()

    for evaluation_number, sample_index in enumerate(sample_indices, start=1):
        sample = dataset[sample_index]
        lsej_id = str(sample["lsej_id"])
        pieces = list(chw_uint8_to_hwc_numpy(sample["pieces"]))
        target = sample["permutation"].detach().cpu().numpy().astype(np.int32, copy=False)
        e1_scores = scorer.score_pieces(pieces)
        baseline = initial_solver(pieces, e1_scores, grid, verbose=args.verbose_solver)

        if not is_complete_prediction(baseline, grid):
            status = f"incomplete_{initial_solver_tag}_baseline"
            selected = baseline.copy()
            result = None
            display_component = np.empty(0, dtype=np.int32)
            display_seed = baseline.copy()
            rounds_attempted = 0
            rounds_accepted = 0
            total_candidates = 0
            initial_e1_score = None
            final_e1_score = None
            final_mutual_top1 = None
        else:
            result = iterative_component_reassembly(
                e1_scores,
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
            total_candidates = sum(len(round_record["candidates"]) for round_record in result["rounds"])
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
                round_rows.append(
                    {
                        "sample_index": sample_index,
                        "lsej_id": lsej_id,
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
                    candidate_metric = _ground_truth_metrics(candidate["prediction"], target, grid)
                    candidate_rows.append(
                        {
                            "sample_index": sample_index,
                            "lsej_id": lsej_id,
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
                            **candidate_metric,
                        }
                    )

        baseline_metric = _ground_truth_metrics(baseline, target, grid)
        selected_metric = _ground_truth_metrics(selected, target, grid)
        baseline_metrics.append(baseline_metric)
        selected_metrics.append(selected_metric)
        status_counts[status] = status_counts.get(status, 0) + 1
        sample_rows.append(
            {
                "sample_index": sample_index,
                "lsej_id": lsej_id,
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
                "baseline_PA": baseline_metric["PA"],
                "selected_PA": selected_metric["PA"],
                "baseline_AA": baseline_metric["AA"],
                "selected_AA": selected_metric["AA"],
                "baseline_horizontal_SRA": baseline_metric["horizontal_SRA"],
                "selected_horizontal_SRA": selected_metric["horizontal_SRA"],
                "baseline_vertical_SRA": baseline_metric["vertical_SRA"],
                "selected_vertical_SRA": selected_metric["vertical_SRA"],
                "baseline_SRA": baseline_metric["SRA"],
                "selected_SRA": selected_metric["SRA"],
            }
        )
        baseline_predictions.append(baseline.copy())
        selected_predictions.append(selected.copy())
        targets.append(target.copy())
        component_mask = np.zeros(grid * grid, dtype=np.uint8)
        component_mask[display_component] = 1
        selected_component_masks.append(component_mask)
        lsej_ids.append(lsej_id)

        # 不完整的 Gallagher 初始解同样需要可视化；缺失 piece 会显示为空位，便于诊断失败模式。
        if (
            args.save_visuals
            and (args.visual_limit == 0 or evaluation_number <= args.visual_limit)
        ):
            _save_visual(
                visual_dir / f"sample_{sample_index:06d}_{lsej_id}.png",
                pieces,
                baseline,
                display_seed,
                selected,
                target,
                display_component,
                grid,
                f"{args.task} | {lsej_id} | accepted {rounds_accepted}/{rounds_attempted} | {status}",
                initial_solver_name=initial_solver_name,
            )

        if args.progress_interval > 0 and (
            evaluation_number % args.progress_interval == 0 or evaluation_number == len(sample_indices)
        ):
            running_baseline = _aggregate_metrics(baseline_metrics)
            running_selected = _aggregate_metrics(selected_metrics)
            print(
                f"[{evaluation_number}/{len(sample_indices)}] {lsej_id}: {status}, "
                f"rounds={rounds_accepted}/{rounds_attempted}, "
                f"PA {running_baseline['PA']:.2%}->{running_selected['PA']:.2%}, "
                f"AA {running_baseline['AA']:.2%}->{running_selected['AA']:.2%}, "
                f"SRA {running_baseline['SRA']:.2%}->{running_selected['SRA']:.2%}"
            )

    elapsed_seconds = time.perf_counter() - started
    baseline_aggregate = _aggregate_metrics(baseline_metrics)
    selected_aggregate = _aggregate_metrics(selected_metrics)
    piece_changes = [int(row["piece_change"]) for row in sample_rows]
    comparison = {
        "improved_puzzles": sum(change > 0 for change in piece_changes),
        "damaged_puzzles": sum(change < 0 for change in piece_changes),
        "unchanged_puzzles": sum(change == 0 for change in piece_changes),
        "changed_predictions": sum(int(row["prediction_changed"]) for row in sample_rows),
        "repaired_to_perfect": sum(
            int(row["baseline_PA"]) == 0 and int(row["selected_PA"]) == 1 for row in sample_rows
        ),
        "broken_perfect": sum(
            int(row["baseline_PA"]) == 1 and int(row["selected_PA"]) == 0 for row in sample_rows
        ),
        "accepted_rounds": sum(int(row["rounds_accepted"]) for row in sample_rows),
    }
    repaired_images, broken_images = _write_pa_case_lists(args.output_dir, sample_rows)
    payload = {
        "method": f"s1a3_iterative_component_reassembly_{initial_solver_tag}_initialization",
        "initial_solver": initial_solver_name,
        "uses_training": False,
        "uses_e2": False,
        "uses_e5": False,
        "uses_ground_truth_for_selection": False,
        "selection_rule": "strictly minimize E1 mean adjacency score; maximize mutual-Top1 edges only as tie-break",
        "task": args.task,
        "split": args.split,
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
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for filename, rows in (
        ("sample_results.csv", sample_rows),
        ("round_results.csv", round_rows),
        ("candidates.csv", candidate_rows),
    ):
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
            lsej_ids=np.asarray(lsej_ids),
            baseline_predictions=np.stack(baseline_predictions),
            selected_predictions=np.stack(selected_predictions),
            targets=np.stack(targets),
            selected_component_masks=np.stack(selected_component_masks),
        )

    summary_lines = [
        f"ImageNet-LSEJ S1A3 Iterative Component Reassembly ({initial_solver_name} initialization)",
        "=" * 80,
        f"checkpoint: {args.checkpoint}",
        f"task/split: {args.task} / {args.split}",
        f"samples: [{args.start_index}, {end_index}) = {len(sample_indices)}",
        f"configuration: k={args.mutual_top_k}, component={args.component_index}, "
        f"min_size={args.min_component_size}, rounds={args.refinement_rounds}",
        f"selection: {payload['selection_rule']}",
        f"status: {status_counts}",
        "",
        f"S1A {initial_solver_name} baseline:",
        f"  PA={baseline_aggregate['PA']:.4%}",
        f"  AA={baseline_aggregate['AA']:.4%}",
        f"  Horizontal SRA={baseline_aggregate['horizontal_SRA']:.4%}",
        f"  Vertical SRA={baseline_aggregate['vertical_SRA']:.4%}",
        f"  SRA={baseline_aggregate['SRA']:.4%}",
        "",
        "S1A3:",
        f"  PA={selected_aggregate['PA']:.4%}",
        f"  AA={selected_aggregate['AA']:.4%}",
        f"  Horizontal SRA={selected_aggregate['horizontal_SRA']:.4%}",
        f"  Vertical SRA={selected_aggregate['vertical_SRA']:.4%}",
        f"  SRA={selected_aggregate['SRA']:.4%}",
        "",
        f"Comparison: {comparison}",
        f"PA repaired image list: pa_repaired_images.txt ({len(repaired_images)})",
        f"PA broken image list: pa_broken_images.txt ({len(broken_images)})",
        f"elapsed_seconds: {_float_metric(elapsed_seconds)}",
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
