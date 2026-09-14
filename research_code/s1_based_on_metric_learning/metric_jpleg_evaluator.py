"""
@file: metric_jpleg_evaluator.py
@description: JPLEG 的通用求解器评估、PA/AA/HA/VA/SRA/NA 统计、结果保存与可视化工具。
@author: Changxin Ye
@created: 2026-07-26
@version: 1.0
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .experiment_runtime_info import collect_runtime_environment, format_runtime_environment
    from .metric_jpleg_data import (
        JPLEG_CONFIGS,
        image_to_pieces,
        label_to_target_positions,
        load_jpleg_arrays,
    )
except ImportError:
    from experiment_runtime_info import collect_runtime_environment, format_runtime_environment
    from metric_jpleg_data import (
        JPLEG_CONFIGS,
        image_to_pieces,
        label_to_target_positions,
        load_jpleg_arrays,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GALLAGHER_IMPLEMENTATION_DIR = PROJECT_ROOT / "baselines" / "PuzzleDemoMGC_CVPR2012_Python"
if str(GALLAGHER_IMPLEMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(GALLAGHER_IMPLEMENTATION_DIR))

from mgc import do_all_assembly_of_puzzle  # noqa: E402


@dataclass
class JPLEGMetrics:
    dataset: str
    split: str
    start_index: int
    samples: int
    perfect_puzzles: int
    correct_pieces: int
    total_pieces: int
    correct_horizontal_relationships: int
    total_horizontal_relationships: int
    correct_vertical_relationships: int
    total_vertical_relationships: int
    correct_neighbor_relationships: int
    total_neighbor_relationships: int
    visualized_samples: int
    elapsed_seconds: float

    @property
    def pa(self) -> float:
        return self.perfect_puzzles / self.samples if self.samples else 0.0

    @property
    def aa(self) -> float:
        return self.correct_pieces / self.total_pieces if self.total_pieces else 0.0

    @property
    def ha(self) -> float:
        total = self.total_horizontal_relationships
        return self.correct_horizontal_relationships / total if total else 0.0

    @property
    def va(self) -> float:
        total = self.total_vertical_relationships
        return self.correct_vertical_relationships / total if total else 0.0

    @property
    def sra(self) -> float:
        correct = self.correct_horizontal_relationships + self.correct_vertical_relationships
        total = self.total_horizontal_relationships + self.total_vertical_relationships
        return correct / total if total else 0.0

    @property
    def na(self) -> float:
        total = self.total_neighbor_relationships
        return self.correct_neighbor_relationships / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "start_index": self.start_index,
            "samples": self.samples,
            "perfect_puzzles": self.perfect_puzzles,
            "correct_pieces": self.correct_pieces,
            "total_pieces": self.total_pieces,
            "correct_horizontal_relationships": self.correct_horizontal_relationships,
            "total_horizontal_relationships": self.total_horizontal_relationships,
            "correct_vertical_relationships": self.correct_vertical_relationships,
            "total_vertical_relationships": self.total_vertical_relationships,
            "correct_neighbor_relationships": self.correct_neighbor_relationships,
            "total_neighbor_relationships": self.total_neighbor_relationships,
            "visualized_samples": self.visualized_samples,
            "elapsed_seconds": self.elapsed_seconds,
            "PA": self.pa,
            "AA": self.aa,
            "HA": self.ha,
            "VA": self.va,
            "SRA": self.sra,
            "NA": self.na,
        }


def solver_grid_to_positions(grid_indices: np.ndarray, grid: int) -> np.ndarray:
    prediction = np.full(grid * grid, -1, dtype=np.int32)
    grid_indices = np.asarray(grid_indices)
    for row in range(min(grid, grid_indices.shape[0])):
        for column in range(min(grid, grid_indices.shape[1])):
            piece_id = int(grid_indices[row, column])
            if 1 <= piece_id <= grid * grid:
                prediction[piece_id - 1] = row * grid + column
    return prediction


def is_complete_prediction(prediction: np.ndarray, grid: int) -> bool:
    prediction = np.asarray(prediction, dtype=np.int64)
    return (
        prediction.shape == (grid * grid,)
        and np.array_equal(np.sort(prediction), np.arange(grid * grid, dtype=np.int64))
    )


@contextlib.contextmanager
def maybe_silence_solver(verbose: bool):
    if verbose:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            yield


def solve_with_gallagher(
    pieces: list[np.ndarray],
    scores: np.ndarray,
    grid: int,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """运行仓库中的 Gallagher 求解器并返回 piece 到位置的映射。"""

    position_key = np.arange(1, grid * grid + 1, dtype=np.int32)
    with maybe_silence_solver(verbose):
        grid_indices, *_ = do_all_assembly_of_puzzle(
            pieces,
            scores,
            nr=grid,
            nc=grid,
            rot_flag=0,
            position_key=position_key,
        )
    return solver_grid_to_positions(grid_indices, grid)


def compute_relationship_counts(prediction: np.ndarray, target: np.ndarray, grid: int) -> dict[str, int]:
    """统计方向敏感的 HA/VA/SRA，以及方向无关的 NA。"""

    num_pieces = grid * grid
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != (num_pieces,) or target.shape != (num_pieces,):
        raise ValueError(f"预测和标签都应为 ({num_pieces},)，实际为 {prediction.shape} 和 {target.shape}")
    if not np.array_equal(np.sort(target), np.arange(num_pieces)):
        raise ValueError("标签必须是全部拼图位置的一个排列")

    piece_at_target_position = np.empty(num_pieces, dtype=np.int64)
    piece_at_target_position[target] = np.arange(num_pieces)
    horizontal_correct = horizontal_total = 0
    vertical_correct = vertical_total = 0
    neighbor_correct = neighbor_total = 0

    def predicted_coordinates(piece_index: int) -> tuple[int, int] | None:
        position = int(prediction[piece_index])
        return divmod(position, grid) if 0 <= position < num_pieces else None

    for row in range(grid):
        for column in range(grid):
            target_position = row * grid + column
            current_piece = int(piece_at_target_position[target_position])
            current_coordinates = predicted_coordinates(current_piece)
            if column + 1 < grid:
                right_piece = int(piece_at_target_position[target_position + 1])
                right_coordinates = predicted_coordinates(right_piece)
                horizontal_total += 1
                neighbor_total += 1
                if current_coordinates is not None and right_coordinates is not None:
                    row_a, column_a = current_coordinates
                    row_b, column_b = right_coordinates
                    horizontal_correct += int(row_b == row_a and column_b - column_a == 1)
                    neighbor_correct += int(abs(row_b - row_a) + abs(column_b - column_a) == 1)
            if row + 1 < grid:
                bottom_piece = int(piece_at_target_position[target_position + grid])
                bottom_coordinates = predicted_coordinates(bottom_piece)
                vertical_total += 1
                neighbor_total += 1
                if current_coordinates is not None and bottom_coordinates is not None:
                    row_a, column_a = current_coordinates
                    row_b, column_b = bottom_coordinates
                    vertical_correct += int(row_b - row_a == 1 and column_b == column_a)
                    neighbor_correct += int(abs(row_b - row_a) + abs(column_b - column_a) == 1)

    return {
        "horizontal_correct": horizontal_correct,
        "horizontal_total": horizontal_total,
        "vertical_correct": vertical_correct,
        "vertical_total": vertical_total,
        "neighbor_correct": neighbor_correct,
        "neighbor_total": neighbor_total,
    }


def sample_ground_truth_metrics(prediction: np.ndarray, target: np.ndarray, grid: int) -> dict[str, float | int]:
    relationships = compute_relationship_counts(prediction, target, grid)
    correct_pieces = int(np.count_nonzero(np.asarray(prediction) == np.asarray(target)))
    horizontal_correct = int(relationships["horizontal_correct"])
    horizontal_total = int(relationships["horizontal_total"])
    vertical_correct = int(relationships["vertical_correct"])
    vertical_total = int(relationships["vertical_total"])
    neighbor_correct = int(relationships["neighbor_correct"])
    neighbor_total = int(relationships["neighbor_total"])
    return {
        "correct_pieces": correct_pieces,
        "total_pieces": grid * grid,
        "PA": int(correct_pieces == grid * grid),
        "AA": correct_pieces / (grid * grid),
        "horizontal_correct": horizontal_correct,
        "horizontal_total": horizontal_total,
        "HA": horizontal_correct / horizontal_total if horizontal_total else 0.0,
        "vertical_correct": vertical_correct,
        "vertical_total": vertical_total,
        "VA": vertical_correct / vertical_total if vertical_total else 0.0,
        "SRA": (horizontal_correct + vertical_correct) / (horizontal_total + vertical_total),
        "neighbor_correct": neighbor_correct,
        "neighbor_total": neighbor_total,
        "NA": neighbor_correct / neighbor_total if neighbor_total else 0.0,
    }


def aggregate_ground_truth_metrics(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("不能汇总空指标列表")
    samples = len(rows)
    horizontal_correct = sum(int(row["horizontal_correct"]) for row in rows)
    horizontal_total = sum(int(row["horizontal_total"]) for row in rows)
    vertical_correct = sum(int(row["vertical_correct"]) for row in rows)
    vertical_total = sum(int(row["vertical_total"]) for row in rows)
    neighbor_correct = sum(int(row["neighbor_correct"]) for row in rows)
    neighbor_total = sum(int(row["neighbor_total"]) for row in rows)
    correct_pieces = sum(int(row["correct_pieces"]) for row in rows)
    total_pieces = sum(int(row["total_pieces"]) for row in rows)
    return {
        "samples": samples,
        "perfect_puzzles": sum(int(row["PA"]) for row in rows),
        "correct_pieces": correct_pieces,
        "total_pieces": total_pieces,
        "PA": sum(int(row["PA"]) for row in rows) / samples,
        "AA": correct_pieces / total_pieces,
        "horizontal_correct": horizontal_correct,
        "horizontal_total": horizontal_total,
        "HA": horizontal_correct / horizontal_total if horizontal_total else 0.0,
        "vertical_correct": vertical_correct,
        "vertical_total": vertical_total,
        "VA": vertical_correct / vertical_total if vertical_total else 0.0,
        "SRA": (horizontal_correct + vertical_correct) / (horizontal_total + vertical_total),
        "neighbor_correct": neighbor_correct,
        "neighbor_total": neighbor_total,
        "NA": neighbor_correct / neighbor_total if neighbor_total else 0.0,
    }


def compose_puzzle(pieces: list[np.ndarray], positions: np.ndarray, grid: int) -> np.ndarray:
    piece_size = int(pieces[0].shape[0])
    image = np.zeros((grid * piece_size, grid * piece_size, 3), dtype=np.uint8)
    for piece_index, restored_position in enumerate(np.asarray(positions)):
        restored_position = int(restored_position)
        if 0 <= restored_position < grid * grid:
            row, column = divmod(restored_position, grid)
            image[row * piece_size : (row + 1) * piece_size, column * piece_size : (column + 1) * piece_size] = pieces[piece_index]
    return image


def save_evaluation_visual(
    path: Path,
    pieces: list[np.ndarray],
    prediction: np.ndarray,
    target: np.ndarray,
    grid: int,
    title: str,
    method_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    piece_size = int(pieces[0].shape[0])
    images = [
        compose_puzzle(pieces, np.arange(grid * grid), grid),
        compose_puzzle(pieces, prediction, grid),
        compose_puzzle(pieces, target, grid),
    ]
    names = [
        "输入乱序拼图",
        f"{method_name}（AA {int(np.count_nonzero(prediction == target))}/{grid * grid}）",
        "真实拼图",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for panel, (axis, image, name) in enumerate(zip(axes, images, names)):
        axis.imshow(image)
        axis.set_title(name)
        axis.axis("off")
        for line in range(piece_size, grid * piece_size, piece_size):
            axis.axhline(line - 0.5, color="white", linewidth=0.6, alpha=0.8)
            axis.axvline(line - 0.5, color="white", linewidth=0.6, alpha=0.8)
        if panel == 1:
            for piece_index, position in enumerate(prediction):
                if not 0 <= int(position) < grid * grid:
                    continue
                row, column = divmod(int(position), grid)
                color = "#2fb344" if int(position) == int(target[piece_index]) else "#e03131"
                axis.add_patch(Rectangle(
                    (column * piece_size + 1.5, row * piece_size + 1.5),
                    piece_size - 4,
                    piece_size - 4,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.8,
                ))
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_jpleg_split(
    *,
    scorer,
    data_root: str | Path,
    dataset: str,
    split: str,
    solver,
    output_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    max_samples: int = 0,
    start_index: int = 0,
    progress_interval: int = 25,
    verbose_solver: bool = False,
    save_visuals: bool = True,
    visual_limit: int = 20,
    save_predictions: bool = True,
    method_name: str = "S1A",
    configuration: dict | None = None,
) -> JPLEGMetrics:
    if dataset not in JPLEG_CONFIGS:
        raise ValueError(f"未知 JPLEG 数据集：{dataset}")
    if max_samples < 0 or start_index < 0:
        raise ValueError("max_samples 和 start_index 必须非负")
    if visual_limit < 0 or progress_interval < 0:
        raise ValueError("visual_limit 和 progress_interval 必须非负")

    config = JPLEG_CONFIGS[dataset]
    images, labels = load_jpleg_arrays(data_root, dataset, split)
    if start_index >= len(images):
        raise ValueError(f"start_index={start_index} 超出数据集长度 {len(images)}")
    end_index = len(images) if max_samples == 0 else min(len(images), start_index + max_samples)
    sample_count = end_index - start_index
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_indices: list[int] = []
    visualized = 0
    started = time.perf_counter()
    print(f"在 {dataset.upper()} {split} 上评估 {method_name}：samples={sample_count}, grid={config.grid}x{config.grid}")

    for done, index in enumerate(range(start_index, end_index), start=1):
        image = np.asarray(images[index])
        pieces = list(image_to_pieces(image, config.grid))
        target = label_to_target_positions(np.asarray(labels[index]), config.grid).astype(np.int32)
        scores = scorer.score_pieces(pieces, rot_flag=0)
        prediction = np.asarray(
            solver(pieces, scores, config.grid, verbose=verbose_solver),
            dtype=np.int32,
        )
        row = sample_ground_truth_metrics(prediction, target, config.grid)
        metric_rows.append(row)
        if save_predictions:
            predictions.append(prediction.copy())
            targets.append(target.copy())
            sample_indices.append(index)
        if output_dir is not None and save_visuals and visualized < visual_limit:
            save_evaluation_visual(
                output_dir / "visuals" / f"sample_{index:06d}.png",
                pieces,
                prediction,
                target,
                config.grid,
                f"{dataset.upper()} {split} | sample {index}",
                method_name,
            )
            visualized += 1
        if progress_interval > 0 and (done == 1 or done % progress_interval == 0 or done == sample_count):
            running = aggregate_ground_truth_metrics(metric_rows)
            print(
                f"  [{done}/{sample_count}] PA={running['PA']:.2%}, AA={running['AA']:.2%}, "
                f"HA={running['HA']:.2%}, VA={running['VA']:.2%}, "
                f"SRA={running['SRA']:.2%}, NA={running['NA']:.2%}"
            )

    aggregate = aggregate_ground_truth_metrics(metric_rows)
    metrics = JPLEGMetrics(
        dataset=dataset,
        split=split,
        start_index=start_index,
        samples=sample_count,
        perfect_puzzles=int(aggregate["perfect_puzzles"]),
        correct_pieces=int(aggregate["correct_pieces"]),
        total_pieces=int(aggregate["total_pieces"]),
        correct_horizontal_relationships=int(aggregate["horizontal_correct"]),
        total_horizontal_relationships=int(aggregate["horizontal_total"]),
        correct_vertical_relationships=int(aggregate["vertical_correct"]),
        total_vertical_relationships=int(aggregate["vertical_total"]),
        correct_neighbor_relationships=int(aggregate["neighbor_correct"]),
        total_neighbor_relationships=int(aggregate["neighbor_total"]),
        visualized_samples=visualized,
        elapsed_seconds=time.perf_counter() - started,
    )

    if output_dir is not None:
        runtime_environment = collect_runtime_environment(
            requested_device=(configuration or {}).get("device"),
            resolved_device=getattr(scorer, "device", None),
            gpu_id=(configuration or {}).get("gpu_id"),
        )
        payload = metrics.to_dict()
        payload.update({
            "method": method_name,
            "configuration": configuration or {},
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "data_root": str(data_root),
                "runtime_environment": runtime_environment,
        })
        (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if save_predictions:
            np.savez_compressed(
                output_dir / "predictions.npz",
                sample_indices=np.asarray(sample_indices, dtype=np.int32),
                predictions=np.asarray(predictions, dtype=np.int32),
                targets=np.asarray(targets, dtype=np.int32),
                grid_size=np.asarray(config.grid, dtype=np.int32),
            )
        summary = [
            f"JPLEG {method_name} 评估结果",
            "=" * 80,
            f"checkpoint: {checkpoint}",
            f"data_root: {data_root}",
            f"dataset/split: {dataset} / {split}",
            f"samples: [{start_index}, {end_index}) = {sample_count}",
            f"configuration: {configuration or {}}",
            "",
            f"PA: {metrics.pa:.4%} ({metrics.perfect_puzzles}/{metrics.samples})",
            f"AA: {metrics.aa:.4%} ({metrics.correct_pieces}/{metrics.total_pieces})",
            f"HA: {metrics.ha:.4%}",
            f"VA: {metrics.va:.4%}",
            f"SRA: {metrics.sra:.4%}",
            f"NA: {metrics.na:.4%}",
            f"visualized_samples: {metrics.visualized_samples}",
            f"elapsed_seconds: {metrics.elapsed_seconds:.3f}",
            "",
            *format_runtime_environment(runtime_environment),
        ]
        (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return metrics


__all__ = [
    "JPLEGMetrics",
    "aggregate_ground_truth_metrics",
    "compose_puzzle",
    "compute_relationship_counts",
    "evaluate_jpleg_split",
    "is_complete_prediction",
    "sample_ground_truth_metrics",
    "save_evaluation_visual",
    "solve_with_gallagher",
    "solver_grid_to_positions",
]
