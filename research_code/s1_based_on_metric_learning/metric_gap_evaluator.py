"""
@file: metric_gap_evaluator.py
@description: GAP-3/GAP-5 的 S1A 兼容性评分、Gallagher 重组、检索指标、拼图指标与 RGBA 可视化。
@author: Changxin Ye
@created: 2026-07-27
@version: 1.0
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .experiment_runtime_info import collect_runtime_environment, format_runtime_environment
    from .metric_gap_data import GAPPuzzleDataset
    from .metric_jpleg_evaluator import (
        aggregate_ground_truth_metrics,
        sample_ground_truth_metrics,
        solve_with_gallagher,
    )
    from .metric_lsej_evaluator import NeighborRetrievalMetrics, compute_true_neighbor_ranks
except ImportError:
    from experiment_runtime_info import collect_runtime_environment, format_runtime_environment
    from metric_gap_data import GAPPuzzleDataset
    from metric_jpleg_evaluator import (
        aggregate_ground_truth_metrics,
        sample_ground_truth_metrics,
        solve_with_gallagher,
    )
    from metric_lsej_evaluator import NeighborRetrievalMetrics, compute_true_neighbor_ranks


DEFAULT_NEIGHBOR_RECALL_KS = (1, 3, 5, 10, 15)


@dataclass
class GAPMetrics:
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
    neighbor_retrieval: NeighborRetrievalMetrics
    complete_predictions: int
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
        return (
            self.correct_horizontal_relationships / self.total_horizontal_relationships
            if self.total_horizontal_relationships else 0.0
        )

    @property
    def va(self) -> float:
        return (
            self.correct_vertical_relationships / self.total_vertical_relationships
            if self.total_vertical_relationships else 0.0
        )

    @property
    def sra(self) -> float:
        correct = self.correct_horizontal_relationships + self.correct_vertical_relationships
        total = self.total_horizontal_relationships + self.total_vertical_relationships
        return correct / total if total else 0.0

    @property
    def na(self) -> float:
        return (
            self.correct_neighbor_relationships / self.total_neighbor_relationships
            if self.total_neighbor_relationships else 0.0
        )

    def to_dict(self) -> dict:
        payload = {
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
            "complete_predictions": self.complete_predictions,
            "visualized_samples": self.visualized_samples,
            "elapsed_seconds": self.elapsed_seconds,
            "PA": self.pa,
            "AA": self.aa,
            "HA": self.ha,
            "VA": self.va,
            "SRA": self.sra,
            "NA": self.na,
            # Exact aliases used by PuzzleFlow's native GAP evaluator.
            "exact_match_accuracy": self.pa,
            "position_accuracy": self.aa,
            "neighbor_accuracy": self.na,
        }
        payload.update(self.neighbor_retrieval.to_dict())
        return payload


def is_complete_prediction(prediction: np.ndarray, grid: int) -> bool:
    prediction = np.asarray(prediction, dtype=np.int64)
    return prediction.shape == (grid * grid,) and np.array_equal(
        np.sort(prediction), np.arange(grid * grid, dtype=np.int64)
    )


def rgba_to_rgb(piece: np.ndarray, background: int = 255) -> np.ndarray:
    piece = np.asarray(piece, dtype=np.uint8)
    if piece.ndim != 3 or piece.shape[2] != 4:
        raise ValueError(f"Expected HWC RGBA piece, got {piece.shape}")
    alpha = piece[..., 3:4].astype(np.float32) / 255.0
    rgb = piece[..., :3].astype(np.float32)
    return np.clip(rgb * alpha + float(background) * (1.0 - alpha), 0, 255).astype(np.uint8)


def compose_puzzle(pieces: list[np.ndarray], positions: np.ndarray, grid: int) -> np.ndarray:
    rgb_pieces = [rgba_to_rgb(piece) for piece in pieces]
    height, width = rgb_pieces[0].shape[:2]
    canvas = np.full((grid * height, grid * width, 3), 255, dtype=np.uint8)
    for piece_index, position in enumerate(np.asarray(positions, dtype=np.int64)):
        if not 0 <= int(position) < grid * grid:
            continue
        row, column = divmod(int(position), grid)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = rgb_pieces[piece_index]
    return canvas


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

    input_positions = np.arange(grid * grid, dtype=np.int32)
    panels = [
        (compose_puzzle(pieces, input_positions, grid), "Official shuffled input"),
        (compose_puzzle(pieces, prediction, grid), f"{method_name} solved"),
        (compose_puzzle(pieces, target, grid), "Ground truth"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for axis, (image, panel_title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(panel_title)
        axis.axis("off")
    piece_height, piece_width = pieces[0].shape[:2]
    solved_axis = axes[1]
    for piece_index, position in enumerate(np.asarray(prediction, dtype=np.int64)):
        if not 0 <= int(position) < grid * grid:
            continue
        row, column = divmod(int(position), grid)
        color = "#20b15a" if int(position) == int(target[piece_index]) else "#e53935"
        solved_axis.add_patch(
            Rectangle(
                (column * piece_width, row * piece_height),
                piece_width,
                piece_height,
                fill=False,
                edgecolor=color,
                linewidth=1.8,
            )
        )
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_gap_split(
    *,
    scorer,
    solver=solve_with_gallagher,
    data_root: str | Path,
    dataset: str,
    split: str,
    output_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    max_samples: int = 0,
    start_index: int = 0,
    progress_interval: int = 25,
    verbose_solver: bool = False,
    save_visuals: bool = True,
    visual_limit: int = 20,
    visual_interval: int = 1,
    save_predictions: bool = True,
    method_name: str = "S1A",
    configuration: dict | None = None,
) -> GAPMetrics:
    if max_samples < 0 or start_index < 0:
        raise ValueError("max_samples and start_index must be non-negative")
    if visual_limit < 0 or visual_interval <= 0 or progress_interval < 0:
        raise ValueError("invalid visual/progress settings")
    gap_data = GAPPuzzleDataset(
        data_root=data_root,
        dataset=dataset,
        split=split,
        start_index=start_index,
        max_samples=max_samples or None,
    )
    grid = gap_data.config.grid
    num_pieces = gap_data.config.num_pieces
    recall_ks = tuple(k for k in DEFAULT_NEIGHBOR_RECALL_KS if k <= num_pieces - 1)
    rank_histogram = np.zeros(num_pieces, dtype=np.int64)
    horizontal_rank_histogram = np.zeros(num_pieces, dtype=np.int64)
    vertical_rank_histogram = np.zeros(num_pieces, dtype=np.int64)
    metric_rows: list[dict] = []
    predictions = []
    targets = []
    sample_indices = []
    complete_predictions = 0
    visualized = 0
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print(f"Evaluating {method_name} on {gap_data.dataset_name} {split}: samples={len(gap_data)}, grid={grid}x{grid}")
    for done in range(1, len(gap_data) + 1):
        sample = gap_data[done - 1]
        pieces = list(sample["pieces"])
        target = np.asarray(sample["target"], dtype=np.int32)
        scores = scorer.score_pieces(pieces, rot_flag=0)
        ranks = compute_true_neighbor_ranks(scores, target, grid)
        rank_histogram += np.bincount(ranks["all"], minlength=num_pieces)[:num_pieces]
        horizontal_rank_histogram += np.bincount(ranks["horizontal"], minlength=num_pieces)[:num_pieces]
        vertical_rank_histogram += np.bincount(ranks["vertical"], minlength=num_pieces)[:num_pieces]
        # Match the S1A3/S1A4 GAP evaluation path exactly: keep RGBA for metric
        # scoring, but pass an RGB copy to every assembly solver.  Gallagher's
        # legacy renderer requires RGB, while LP and Pomeranz ignore the images.
        solver_pieces = [rgba_to_rgb(piece) for piece in pieces]
        prediction = np.asarray(
            solver(solver_pieces, scores, grid, verbose=verbose_solver), dtype=np.int32
        )
        complete_predictions += int(is_complete_prediction(prediction, grid))
        metric_rows.append(sample_ground_truth_metrics(prediction, target, grid))
        if save_predictions:
            predictions.append(prediction.copy())
            targets.append(target.copy())
            sample_indices.append(int(sample["sample_index"]))
        if (
            output_path is not None
            and save_visuals
            and visualized < visual_limit
            and (done - 1) % visual_interval == 0
        ):
            save_evaluation_visual(
                output_path / "visuals" / f"sample_{int(sample['sample_index']):06d}.png",
                pieces,
                prediction,
                target,
                grid,
                f"{gap_data.dataset_name} {split} | sample {int(sample['sample_index'])}",
                method_name,
            )
            visualized += 1
        if progress_interval > 0 and (done == 1 or done % progress_interval == 0 or done == len(gap_data)):
            running = aggregate_ground_truth_metrics(metric_rows)
            print(
                f"  [{done}/{len(gap_data)}] PA={running['PA']:.2%}, AA={running['AA']:.2%}, "
                f"SRA={running['SRA']:.2%}, NA={running['NA']:.2%}"
            )

    aggregate = aggregate_ground_truth_metrics(metric_rows)
    retrieval = NeighborRetrievalMetrics(
        recall_ks=recall_ks,
        rank_histogram=rank_histogram,
        horizontal_rank_histogram=horizontal_rank_histogram,
        vertical_rank_histogram=vertical_rank_histogram,
    )
    metrics = GAPMetrics(
        dataset=gap_data.dataset_name,
        split=split,
        start_index=start_index,
        samples=len(gap_data),
        perfect_puzzles=int(aggregate["perfect_puzzles"]),
        correct_pieces=int(aggregate["correct_pieces"]),
        total_pieces=int(aggregate["total_pieces"]),
        correct_horizontal_relationships=int(aggregate["horizontal_correct"]),
        total_horizontal_relationships=int(aggregate["horizontal_total"]),
        correct_vertical_relationships=int(aggregate["vertical_correct"]),
        total_vertical_relationships=int(aggregate["vertical_total"]),
        correct_neighbor_relationships=int(aggregate["neighbor_correct"]),
        total_neighbor_relationships=int(aggregate["neighbor_total"]),
        neighbor_retrieval=retrieval,
        complete_predictions=complete_predictions,
        visualized_samples=visualized,
        elapsed_seconds=time.perf_counter() - started,
    )
    if output_path is not None:
        runtime_environment = collect_runtime_environment(
            requested_device=(configuration or {}).get("device"),
            resolved_device=getattr(scorer, "device", None),
            gpu_id=(configuration or {}).get("gpu_id"),
        )
        payload = metrics.to_dict()
        payload.update(
            {
                "method": method_name,
                "configuration": configuration or {},
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "data_root": str(Path(data_root).expanduser().resolve()),
                "runtime_environment": runtime_environment,
            }
        )
        (output_path / "metrics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if save_predictions:
            np.savez_compressed(
                output_path / "predictions.npz",
                sample_indices=np.asarray(sample_indices, dtype=np.int32),
                predictions=np.asarray(predictions, dtype=np.int32),
                targets=np.asarray(targets, dtype=np.int32),
                grid_size=np.asarray(grid, dtype=np.int32),
                dataset=np.asarray(gap_data.dataset_name),
                split=np.asarray(split),
            )
        summary = [
            f"GAP {method_name} Evaluation Summary",
            "=" * 80,
            f"checkpoint: {checkpoint}",
            f"data_root: {Path(data_root).expanduser().resolve()}",
            f"dataset/split: {gap_data.dataset_name} / {split}",
            f"samples: {len(gap_data)}",
            f"complete_predictions: {metrics.complete_predictions}/{metrics.samples}",
            f"configuration: {configuration or {}}",
            "",
            f"PA: {metrics.pa:.4%} ({metrics.perfect_puzzles}/{metrics.samples})",
            f"AA: {metrics.aa:.4%} ({metrics.correct_pieces}/{metrics.total_pieces})",
            f"HA: {metrics.ha:.4%}",
            f"VA: {metrics.va:.4%}",
            f"SRA: {metrics.sra:.4%}",
            f"NA / PuzzleFlow neighbor_accuracy: {metrics.na:.4%}",
            f"True-Neighbor MRR: {retrieval.mrr:.6f}",
            f"Mean True-Neighbor Rank: {retrieval.mean_rank:.3f}",
        ]
        for k in retrieval.recall_ks:
            summary.append(f"True-Neighbor Recall@{k}: {retrieval.recall_at(k):.4%}")
        summary.extend(
            [
                f"visualized_samples: {metrics.visualized_samples}",
                f"elapsed_seconds: {metrics.elapsed_seconds:.3f}",
                "",
                *format_runtime_environment(runtime_environment),
            ]
        )
        (output_path / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return metrics


__all__ = [
    "DEFAULT_NEIGHBOR_RECALL_KS",
    "GAPMetrics",
    "compose_puzzle",
    "evaluate_gap_split",
    "is_complete_prediction",
    "rgba_to_rgb",
    "save_evaluation_visual",
    "solve_with_gallagher",
]
