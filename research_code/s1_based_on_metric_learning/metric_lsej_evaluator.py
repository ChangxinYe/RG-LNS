"""
@file: metric_lsej_evaluator.py
@description: s1a 在 ImageNet-LSEJ 上的 Gallagher 求解、PA/AA/SRA 统计、结果保存与可视化工具。
@author: Changxin Ye
@created: 2026-07-17
@version: 1.1
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
    from .metric_lsej_data import chw_uint8_to_hwc_numpy, load_official_dataset_class
    from .metric_geometry import BOTTOM, DIRECTIONS, LEFT, RIGHT, TOP
except ImportError:
    from experiment_runtime_info import collect_runtime_environment, format_runtime_environment
    from metric_lsej_data import chw_uint8_to_hwc_numpy, load_official_dataset_class
    from metric_geometry import BOTTOM, DIRECTIONS, LEFT, RIGHT, TOP


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GALLAGHER_IMPLEMENTATION_DIR = PROJECT_ROOT / "baselines" / "PuzzleDemoMGC_CVPR2012_Python"
if str(GALLAGHER_IMPLEMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(GALLAGHER_IMPLEMENTATION_DIR))

from mgc import do_all_assembly_of_puzzle  # noqa: E402


DEFAULT_NEIGHBOR_RECALL_KS = (1, 3, 5, 10, 15)
DIRECTION_OFFSETS = {
    TOP: (-1, 0),
    RIGHT: (0, 1),
    BOTTOM: (1, 0),
    LEFT: (0, -1),
}


@dataclass
class NeighborRetrievalMetrics:
    """Aggregate true-neighbor ranks before Gallagher assembly.

    Each query is an ordered ``(anchor piece, direction)`` relation with one
    ground-truth neighbor. Rank histograms use index 0 as an unused sentinel,
    so index ``r`` stores the number of queries whose true neighbor ranked
    exactly ``r`` among all other pieces.
    """

    recall_ks: tuple[int, ...]
    rank_histogram: np.ndarray
    horizontal_rank_histogram: np.ndarray
    vertical_rank_histogram: np.ndarray

    @staticmethod
    def _query_count(histogram: np.ndarray) -> int:
        return int(np.asarray(histogram, dtype=np.int64).sum())

    @staticmethod
    def _recall_at(histogram: np.ndarray, k: int) -> float:
        query_count = NeighborRetrievalMetrics._query_count(histogram)
        if query_count == 0:
            return 0.0
        upper = min(int(k), len(histogram) - 1)
        return float(np.asarray(histogram, dtype=np.int64)[1 : upper + 1].sum() / query_count)

    @staticmethod
    def _mean_rank(histogram: np.ndarray) -> float:
        query_count = NeighborRetrievalMetrics._query_count(histogram)
        if query_count == 0:
            return 0.0
        ranks = np.arange(len(histogram), dtype=np.float64)
        return float(np.dot(ranks, histogram) / query_count)

    @staticmethod
    def _median_rank(histogram: np.ndarray) -> float:
        query_count = NeighborRetrievalMetrics._query_count(histogram)
        if query_count == 0:
            return 0.0
        cumulative = np.cumsum(np.asarray(histogram, dtype=np.int64))
        lower_position = (query_count - 1) // 2 + 1
        upper_position = query_count // 2 + 1
        lower_rank = int(np.searchsorted(cumulative, lower_position, side="left"))
        upper_rank = int(np.searchsorted(cumulative, upper_position, side="left"))
        return 0.5 * (lower_rank + upper_rank)

    @staticmethod
    def _mrr(histogram: np.ndarray) -> float:
        query_count = NeighborRetrievalMetrics._query_count(histogram)
        if query_count == 0:
            return 0.0
        ranks = np.arange(1, len(histogram), dtype=np.float64)
        return float(np.sum(histogram[1:] / ranks) / query_count)

    @property
    def queries(self) -> int:
        return self._query_count(self.rank_histogram)

    @property
    def horizontal_queries(self) -> int:
        return self._query_count(self.horizontal_rank_histogram)

    @property
    def vertical_queries(self) -> int:
        return self._query_count(self.vertical_rank_histogram)

    @property
    def mean_rank(self) -> float:
        return self._mean_rank(self.rank_histogram)

    @property
    def median_rank(self) -> float:
        return self._median_rank(self.rank_histogram)

    @property
    def mrr(self) -> float:
        return self._mrr(self.rank_histogram)

    def recall_at(self, k: int) -> float:
        return self._recall_at(self.rank_histogram, k)

    def horizontal_recall_at(self, k: int) -> float:
        return self._recall_at(self.horizontal_rank_histogram, k)

    def vertical_recall_at(self, k: int) -> float:
        return self._recall_at(self.vertical_rank_histogram, k)

    def to_dict(self) -> dict:
        payload = {
            "neighbor_queries": self.queries,
            "horizontal_neighbor_queries": self.horizontal_queries,
            "vertical_neighbor_queries": self.vertical_queries,
            "neighbor_mrr": self.mrr,
            "mean_true_neighbor_rank": self.mean_rank,
            "median_true_neighbor_rank": self.median_rank,
        }
        for k in self.recall_ks:
            payload[f"neighbor_recall_at_{k}"] = self.recall_at(k)
            payload[f"horizontal_neighbor_recall_at_{k}"] = self.horizontal_recall_at(k)
            payload[f"vertical_neighbor_recall_at_{k}"] = self.vertical_recall_at(k)
        return payload


@dataclass
class LSEJMetrics:
    task: str
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
    neighbor_retrieval: NeighborRetrievalMetrics
    visualized_samples: int
    elapsed_seconds: float

    @property
    def pa(self) -> float:
        return self.perfect_puzzles / self.samples if self.samples else 0.0

    @property
    def aa(self) -> float:
        return self.correct_pieces / self.total_pieces if self.total_pieces else 0.0

    @property
    def horizontal_sra(self) -> float:
        total = self.total_horizontal_relationships
        return self.correct_horizontal_relationships / total if total else 0.0

    @property
    def vertical_sra(self) -> float:
        total = self.total_vertical_relationships
        return self.correct_vertical_relationships / total if total else 0.0

    @property
    def sra(self) -> float:
        correct = self.correct_horizontal_relationships + self.correct_vertical_relationships
        total = self.total_horizontal_relationships + self.total_vertical_relationships
        return correct / total if total else 0.0

    def to_dict(self) -> dict:
        payload = {
            "task": self.task,
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
            "visualized_samples": self.visualized_samples,
            "elapsed_seconds": self.elapsed_seconds,
            "PA": self.pa,
            "AA": self.aa,
            "Horizontal SRA": self.horizontal_sra,
            "Vertical SRA": self.vertical_sra,
            "SRA": self.sra,
        }
        payload.update(self.neighbor_retrieval.to_dict())
        return payload


def solver_grid_to_positions(grid_indices: np.ndarray, grid: int) -> np.ndarray:
    prediction = np.full(grid * grid, -1, dtype=np.int32)
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


def solve_with_gallagher(
    pieces: list[np.ndarray],
    scores: np.ndarray,
    grid: int,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Run the repository's Gallagher greedy assembly solver and return piece positions."""

    position_key = np.arange(1, grid * grid + 1, dtype=np.int32)
    with maybe_silence_solver(verbose):
        grid_indices, *_ = do_all_assembly_of_puzzle(
            pieces,
            scores,
            grid,
            grid,
            0,
            position_key,
        )
    return solver_grid_to_positions(grid_indices, grid)


def compute_relationship_counts(prediction: np.ndarray, target: np.ndarray, grid: int) -> dict[str, int]:
    """Count direction-sensitive right and bottom relationships."""

    num_pieces = grid * grid
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != (num_pieces,) or target.shape != (num_pieces,):
        raise ValueError(
            f"Expected prediction and target shape ({num_pieces},), got {prediction.shape} and {target.shape}"
        )
    if not np.array_equal(np.sort(target), np.arange(num_pieces)):
        raise ValueError("Target positions must be a permutation of all grid positions")

    piece_at_target_position = np.empty(num_pieces, dtype=np.int64)
    piece_at_target_position[target] = np.arange(num_pieces)
    horizontal_correct = 0
    horizontal_total = grid * (grid - 1)
    vertical_correct = 0
    vertical_total = grid * (grid - 1)

    def coordinates(piece_index: int):
        predicted_position = int(prediction[piece_index])
        if not 0 <= predicted_position < num_pieces:
            return None
        return divmod(predicted_position, grid)

    for row in range(grid):
        for column in range(grid):
            target_position = row * grid + column
            current_piece = int(piece_at_target_position[target_position])
            current_coordinates = coordinates(current_piece)
            if current_coordinates is None:
                continue

            if column + 1 < grid:
                right_piece = int(piece_at_target_position[target_position + 1])
                right_coordinates = coordinates(right_piece)
                if right_coordinates is not None:
                    horizontal_correct += int(
                        right_coordinates[0] - current_coordinates[0] == 0
                        and right_coordinates[1] - current_coordinates[1] == 1
                    )

            if row + 1 < grid:
                bottom_piece = int(piece_at_target_position[target_position + grid])
                bottom_coordinates = coordinates(bottom_piece)
                if bottom_coordinates is not None:
                    vertical_correct += int(
                        bottom_coordinates[0] - current_coordinates[0] == 1
                        and bottom_coordinates[1] - current_coordinates[1] == 0
                    )

    return {
        "horizontal_correct": horizontal_correct,
        "horizontal_total": horizontal_total,
        "vertical_correct": vertical_correct,
        "vertical_total": vertical_total,
    }


def compute_true_neighbor_ranks(scores: np.ndarray, target: np.ndarray, grid: int) -> dict[str, np.ndarray]:
    """Rank every ground-truth directed neighbor in the pre-assembly score matrix.

    Lower compatibility scores are better. Each valid ``(anchor, direction)``
    query has exactly one relevant candidate. If candidates have exactly equal
    scores, the true neighbor receives the worst rank within that tie group so
    Recall@K is not inflated by arbitrary candidate ordering.
    """

    num_pieces = int(grid) * int(grid)
    scores = np.asarray(scores, dtype=np.float32)
    target = np.asarray(target, dtype=np.int64)
    if scores.shape != (num_pieces, num_pieces, 4):
        raise ValueError(
            f"Expected scores shape ({num_pieces}, {num_pieces}, 4), got {scores.shape}"
        )
    if target.shape != (num_pieces,):
        raise ValueError(f"Expected target shape ({num_pieces},), got {target.shape}")
    if not np.array_equal(np.sort(target), np.arange(num_pieces)):
        raise ValueError("Target positions must be a permutation of all grid positions")
    if np.isnan(scores).any():
        raise ValueError("Compatibility scores contain NaN values")

    piece_at_target_position = np.empty(num_pieces, dtype=np.int64)
    piece_at_target_position[target] = np.arange(num_pieces)
    all_ranks = []
    horizontal_ranks = []
    vertical_ranks = []

    for row in range(grid):
        for column in range(grid):
            target_position = row * grid + column
            anchor_piece = int(piece_at_target_position[target_position])
            candidate_mask = np.ones(num_pieces, dtype=bool)
            candidate_mask[anchor_piece] = False

            for direction in DIRECTIONS:
                row_offset, column_offset = DIRECTION_OFFSETS[int(direction)]
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset
                if not (0 <= neighbor_row < grid and 0 <= neighbor_column < grid):
                    continue

                neighbor_position = neighbor_row * grid + neighbor_column
                true_neighbor = int(piece_at_target_position[neighbor_position])
                direction_scores = scores[anchor_piece, :, int(direction)]
                true_score = float(direction_scores[true_neighbor])
                if not np.isfinite(true_score):
                    raise ValueError(
                        "Ground-truth neighbor has a non-finite compatibility score: "
                        f"anchor={anchor_piece}, neighbor={true_neighbor}, direction={direction}"
                    )

                # Worst rank within an exact tie: number of non-self candidates
                # whose score is no greater than the ground-truth score.
                rank = int(np.count_nonzero(candidate_mask & (direction_scores <= true_score)))
                if not 1 <= rank <= num_pieces - 1:
                    raise RuntimeError(f"Resolved invalid true-neighbor rank: {rank}")
                all_ranks.append(rank)
                if int(direction) in {LEFT, RIGHT}:
                    horizontal_ranks.append(rank)
                else:
                    vertical_ranks.append(rank)

    expected_axis_queries = 2 * grid * (grid - 1)
    expected_all_queries = 2 * expected_axis_queries
    if (
        len(all_ranks) != expected_all_queries
        or len(horizontal_ranks) != expected_axis_queries
        or len(vertical_ranks) != expected_axis_queries
    ):
        raise RuntimeError(
            "Unexpected directed-neighbor query counts: "
            f"all={len(all_ranks)}, horizontal={len(horizontal_ranks)}, "
            f"vertical={len(vertical_ranks)}"
        )

    return {
        "all": np.asarray(all_ranks, dtype=np.int32),
        "horizontal": np.asarray(horizontal_ranks, dtype=np.int32),
        "vertical": np.asarray(vertical_ranks, dtype=np.int32),
    }


@contextlib.contextmanager
def maybe_silence_solver(verbose: bool):
    if verbose:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            yield


def compose_puzzle(pieces: list[np.ndarray], positions: np.ndarray, grid: int) -> np.ndarray:
    piece_size = int(pieces[0].shape[0])
    image = np.zeros((grid * piece_size, grid * piece_size, 3), dtype=np.uint8)
    for current_index, restored_position in enumerate(positions):
        restored_position = int(restored_position)
        if 0 <= restored_position < grid * grid:
            row, column = divmod(restored_position, grid)
            image[
                row * piece_size : (row + 1) * piece_size,
                column * piece_size : (column + 1) * piece_size,
            ] = pieces[current_index]
    return image


def save_evaluation_visual(
    path: Path,
    pieces: list[np.ndarray],
    prediction: np.ndarray,
    target: np.ndarray,
    grid: int,
    title: str,
    method_name: str = "s1a",
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
        "Official shuffled input",
        f"{method_name} solved (AA {int((prediction == target).sum())}/{grid * grid})",
        "Ground truth",
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
            for current_index, restored_position in enumerate(prediction):
                restored_position = int(restored_position)
                if 0 <= restored_position < grid * grid:
                    row, column = divmod(restored_position, grid)
                    color = "#2fb344" if restored_position == int(target[current_index]) else "#e03131"
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
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_evaluation_summary(
    path: Path,
    metrics: LSEJMetrics,
    checkpoint: str | Path | None,
    data_root: str | Path,
    max_samples: int,
    method_name: str = "s1a",
    configuration: dict | None = None,
    runtime_environment: dict | None = None,
) -> None:
    lines = [
        f"ImageNet-LSEJ {method_name} Metric-Learning Evaluation Summary",
        "=" * 80,
        f"checkpoint: {checkpoint if checkpoint is not None else 'in-memory model'}",
        f"data_root: {data_root}",
        f"task: {metrics.task}",
        f"split: {metrics.split}",
        f"start_index: {metrics.start_index}",
        f"max_samples: {max_samples}",
    ]
    if configuration:
        lines.append("configuration:")
        lines.extend(f"  {key}: {value}" for key, value in configuration.items())
    lines.extend(
        [
        "",
        f"samples: {metrics.samples}",
        f"PA: {metrics.pa:.2%} ({metrics.perfect_puzzles}/{metrics.samples})",
        f"AA: {metrics.aa:.2%} ({metrics.correct_pieces}/{metrics.total_pieces})",
        f"Horizontal SRA: {metrics.horizontal_sra:.2%} "
        f"({metrics.correct_horizontal_relationships}/{metrics.total_horizontal_relationships})",
        f"Vertical SRA: {metrics.vertical_sra:.2%} "
        f"({metrics.correct_vertical_relationships}/{metrics.total_vertical_relationships})",
        f"SRA: {metrics.sra:.2%} "
        f"({metrics.correct_horizontal_relationships + metrics.correct_vertical_relationships}/"
        f"{metrics.total_horizontal_relationships + metrics.total_vertical_relationships})",
        "",
        f"True-Neighbor queries: {metrics.neighbor_retrieval.queries}",
        ]
    )
    for k in metrics.neighbor_retrieval.recall_ks:
        lines.append(
            f"True-Neighbor Recall@{k}: {metrics.neighbor_retrieval.recall_at(k):.2%}"
        )
    lines.extend(
        [
            f"True-Neighbor MRR: {metrics.neighbor_retrieval.mrr:.6f}",
            f"Mean True-Neighbor Rank: {metrics.neighbor_retrieval.mean_rank:.3f}",
            f"Median True-Neighbor Rank: {metrics.neighbor_retrieval.median_rank:.3f}",
            "",
        ]
    )
    for k in metrics.neighbor_retrieval.recall_ks:
        lines.append(
            f"Horizontal True-Neighbor Recall@{k}: "
            f"{metrics.neighbor_retrieval.horizontal_recall_at(k):.2%}"
        )
    for k in metrics.neighbor_retrieval.recall_ks:
        lines.append(
            f"Vertical True-Neighbor Recall@{k}: "
            f"{metrics.neighbor_retrieval.vertical_recall_at(k):.2%}"
        )
    lines.extend(
        [
            "",
            f"visualized_samples: {metrics.visualized_samples}",
            f"elapsed_seconds: {metrics.elapsed_seconds:.3f}",
        ]
    )
    if runtime_environment is not None:
        lines.extend(["", *format_runtime_environment(runtime_environment)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_lsej_split(
    *,
    scorer,
    data_root: str | Path,
    task: str,
    split: str,
    output_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    max_samples: int = 0,
    start_index: int = 0,
    progress_interval: int = 1,
    verbose_solver: bool = False,
    save_visuals: bool = True,
    visual_limit: int = 10,
    save_predictions: bool = True,
    method_name: str = "s1a",
    solver=None,
    configuration: dict | None = None,
    dataset_override=None,
    progress_callback=None,
) -> LSEJMetrics:
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if visual_limit < 0:
        raise ValueError("visual_limit must be non-negative")
    if solver is None:
        solver = solve_with_gallagher

    if dataset_override is None:
        dataset_class = load_official_dataset_class()
        dataset = dataset_class(data_root, split=split, task=task, normalize=False)
    else:
        dataset = dataset_override
    grid = int(dataset.config["puzzle"]["grid_rows"])
    if start_index < 0 or start_index >= len(dataset):
        raise ValueError(f"start_index={start_index} is outside dataset length {len(dataset)}")
    end_index = len(dataset) if max_samples == 0 else min(len(dataset), start_index + max_samples)
    sample_count = end_index - start_index
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    targets = []
    lsej_ids = []
    correct_pieces = 0
    perfect_puzzles = 0
    horizontal_correct = 0
    horizontal_total = 0
    vertical_correct = 0
    vertical_total = 0
    max_neighbor_rank = grid * grid - 1
    neighbor_rank_histogram = np.zeros(max_neighbor_rank + 1, dtype=np.int64)
    horizontal_neighbor_rank_histogram = np.zeros(max_neighbor_rank + 1, dtype=np.int64)
    vertical_neighbor_rank_histogram = np.zeros(max_neighbor_rank + 1, dtype=np.int64)
    visualized = 0
    started = time.perf_counter()
    print(f"Evaluating {method_name} on {task} {split}: samples={sample_count}, grid={grid}x{grid}")

    for done, index in enumerate(range(start_index, end_index), start=1):
        sample = dataset[index]
        pieces = list(chw_uint8_to_hwc_numpy(sample["pieces"]))
        target = sample["permutation"].numpy().astype(np.int32, copy=False)
        scores = scorer.score_pieces(pieces, rot_flag=0)
        neighbor_ranks = compute_true_neighbor_ranks(scores, target, grid)
        np.add.at(neighbor_rank_histogram, neighbor_ranks["all"], 1)
        np.add.at(horizontal_neighbor_rank_histogram, neighbor_ranks["horizontal"], 1)
        np.add.at(vertical_neighbor_rank_histogram, neighbor_ranks["vertical"], 1)
        prediction = solver(pieces, scores, grid, verbose=verbose_solver)
        correct = int((prediction == target).sum())
        correct_pieces += correct
        perfect_puzzles += int(correct == grid * grid)
        relationship_counts = compute_relationship_counts(prediction, target, grid)
        horizontal_correct += relationship_counts["horizontal_correct"]
        horizontal_total += relationship_counts["horizontal_total"]
        vertical_correct += relationship_counts["vertical_correct"]
        vertical_total += relationship_counts["vertical_total"]

        if save_predictions:
            predictions.append(prediction)
            targets.append(target.copy())
            lsej_ids.append(sample["lsej_id"])
        if output_dir is not None and save_visuals and visualized < visual_limit:
            save_evaluation_visual(
                output_dir / "visuals" / f"sample_{index:06d}_{sample['lsej_id']}.png",
                pieces,
                prediction,
                target,
                grid,
                f"{task} | {sample['lsej_id']}",
                method_name=method_name,
            )
            visualized += 1

        should_print_progress = progress_interval > 0 and (
            done == 1 or done % progress_interval == 0 or done == sample_count
        )
        if progress_callback is not None:
            progress_callback(
                done=done,
                total=sample_count,
                dataset_index=index,
                sample=sample,
                prediction=prediction,
                target=target,
                should_print=should_print_progress,
            )
        elif should_print_progress:
            relationship_correct = horizontal_correct + vertical_correct
            relationship_total = horizontal_total + vertical_total
            neighbor_queries = int(neighbor_rank_histogram.sum())
            neighbor_recall_at_1 = (
                float(neighbor_rank_histogram[1] / neighbor_queries) if neighbor_queries else 0.0
            )
            neighbor_recall_at_5 = (
                float(neighbor_rank_histogram[1 : min(5, max_neighbor_rank) + 1].sum() / neighbor_queries)
                if neighbor_queries
                else 0.0
            )
            print(
                f"  [{done}/{sample_count}] PA={perfect_puzzles / done:.2%}, "
                f"AA={correct_pieces / (done * grid * grid):.2%}, "
                f"SRA={relationship_correct / relationship_total:.2%}, "
                f"Neighbor R@1={neighbor_recall_at_1:.2%}, "
                f"R@5={neighbor_recall_at_5:.2%}"
            )

    neighbor_retrieval = NeighborRetrievalMetrics(
        recall_ks=DEFAULT_NEIGHBOR_RECALL_KS,
        rank_histogram=neighbor_rank_histogram,
        horizontal_rank_histogram=horizontal_neighbor_rank_histogram,
        vertical_rank_histogram=vertical_neighbor_rank_histogram,
    )
    metrics = LSEJMetrics(
        task=task,
        split=split,
        start_index=start_index,
        samples=sample_count,
        perfect_puzzles=perfect_puzzles,
        correct_pieces=correct_pieces,
        total_pieces=sample_count * grid * grid,
        correct_horizontal_relationships=horizontal_correct,
        total_horizontal_relationships=horizontal_total,
        correct_vertical_relationships=vertical_correct,
        total_vertical_relationships=vertical_total,
        neighbor_retrieval=neighbor_retrieval,
        visualized_samples=visualized,
        elapsed_seconds=time.perf_counter() - started,
    )
    recalls = ", ".join(
        f"R@{k}={neighbor_retrieval.recall_at(k):.2%}"
        for k in neighbor_retrieval.recall_ks
    )
    print(
        f"True-Neighbor retrieval: {recalls}, MRR={neighbor_retrieval.mrr:.4f}, "
        f"mean_rank={neighbor_retrieval.mean_rank:.2f}, "
        f"median_rank={neighbor_retrieval.median_rank:.2f}"
    )

    if output_dir is not None:
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
                "data_root": str(data_root),
                "runtime_environment": runtime_environment,
            }
        )
        (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if save_predictions:
            np.savez_compressed(
                output_dir / "predictions.npz",
                predictions=np.asarray(predictions),
                targets=np.asarray(targets),
                lsej_ids=np.asarray(lsej_ids),
                grid_size=np.asarray(grid),
                start_index=np.asarray(start_index),
            )
        write_evaluation_summary(
            output_dir / "summary.txt",
            metrics,
            checkpoint,
            data_root,
            max_samples,
            method_name=method_name,
            configuration=configuration,
            runtime_environment=runtime_environment,
        )

    return metrics


__all__ = [
    "DEFAULT_NEIGHBOR_RECALL_KS",
    "LSEJMetrics",
    "NeighborRetrievalMetrics",
    "compute_relationship_counts",
    "compute_true_neighbor_ranks",
    "evaluate_lsej_split",
    "solver_grid_to_positions",
]
