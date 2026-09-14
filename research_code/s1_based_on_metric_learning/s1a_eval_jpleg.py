"""
@file: s1a_eval_jpleg.py
@description: 使用 Gallagher 求解器评估 puzzle-level hard triplet checkpoint 的 PA、AA、HA、VA、SRA 和 NA。
@author: Changxin Ye
@created: 2026-07-10
@version: 1.3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from .experiment_runtime_info import append_runtime_environment
    from .metric_jpleg_data import DEFAULT_DATA_ROOT, JPLEG_CONFIGS, SPLIT_ALIASES
    from .metric_scorer import MetricCompatibilityScorer, resolve_device
except ImportError:
    from experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from experiment_runtime_info import append_runtime_environment
    from metric_jpleg_data import DEFAULT_DATA_ROOT, JPLEG_CONFIGS, SPLIT_ALIASES
    from metric_scorer import MetricCompatibilityScorer, resolve_device


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EDGE2VEC_DIR = PROJECT_ROOT / "baselines" / "Edge2Vec_arxiv_2022"
GALLAGHER_IMPLEMENTATION_DIR = PROJECT_ROOT / "baselines" / "PuzzleDemoMGC_CVPR2012_Python"
for path in (EDGE2VEC_DIR, GALLAGHER_IMPLEMENTATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jpleg_data import PIECE_SIZE, image_to_pieces, label_to_target_positions, load_jpleg_arrays  # noqa: E402
from mgc import do_all_assembly_of_puzzle  # noqa: E402


DEFAULT_LOG_ROOT = script_output_root(SCRIPT_DIR / "s1a_train_jpleg.py", "logs")
LEGACY_LOG_ROOT = SCRIPT_DIR / "logs" / "puzzle_hard_triplet_jpleg"
DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "jpleg3"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_SAVE_VISUALS = True
DEFAULT_VISUAL_LIMIT = 10
DEFAULT_RUN_NAMES = {
    "jpleg3": "hard_triplet_d128_s224_2026-07-27-08-14-53",
    "jpleg5": "hard_triplet_d128_s224_2026-07-28-23-27-31",
    "all": "",
}
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]
DEFAULT_RUN_DIR = None
DEFAULT_CHECKPOINT_NAME = "best.pth"


@dataclass
class DatasetMetrics:
    dataset: str
    split: str
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
        return (
            self.correct_horizontal_relationships / self.total_horizontal_relationships
            if self.total_horizontal_relationships
            else 0.0
        )

    @property
    def va(self) -> float:
        return (
            self.correct_vertical_relationships / self.total_vertical_relationships
            if self.total_vertical_relationships
            else 0.0
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
            if self.total_neighbor_relationships
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
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
            "PA": self.pa,
            "AA": self.aa,
            "HA": self.ha,
            "VA": self.va,
            "SRA": self.sra,
            "NA": self.na,
            "elapsed_seconds": self.elapsed_seconds,
        }


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected True or False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate puzzle-level hard triplet metric compatibility with the Gallagher solver on JPLEG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None, type=Path, help="Exact checkpoint path.")
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Run folder name under logs/s1a_train_jpleg/<dataset>. Empty string means auto-pick latest.",
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="Exact run directory containing checkpoints/.")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="Checkpoint file inside checkpoints/.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=["all", *JPLEG_CONFIGS.keys()])
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=SPLIT_ALIASES.keys())
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path, help="评估结果目录；留空时按评估脚本、数据集和实验自动分类")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="0 means evaluate the full split")
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--progress-interval", default=25, type=int)
    parser.add_argument("--save-visuals", default=DEFAULT_SAVE_VISUALS, type=parse_bool)
    parser.add_argument("--visual-limit", default=DEFAULT_VISUAL_LIMIT, type=int)
    parser.add_argument("--visual-interval", default=1, type=int)
    parser.add_argument("--save-predictions", default=True, type=parse_bool)
    parser.add_argument("--postprocess", default=True, type=parse_bool)
    parser.add_argument("--score-batch-size", default=128, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--verbose-solver", action="store_true")
    return parser.parse_args()


def checkpoint_from_run_dir(run_dir: Path, checkpoint_name: str) -> Path:
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if checkpoint.exists():
        return checkpoint
    available = []
    dataset_root = run_dir.parent
    if dataset_root.exists():
        available = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    suffix = f" Available runs under {dataset_root}: {available}" if available else ""
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}.{suffix}")


def resolve_default_checkpoint(dataset: str, checkpoint_name: str, log_root: Path = DEFAULT_LOG_ROOT) -> Path:
    log_roots = [log_root]
    if log_root == DEFAULT_LOG_ROOT and LEGACY_LOG_ROOT != DEFAULT_LOG_ROOT:
        log_roots.append(LEGACY_LOG_ROOT)
    search_roots = [root if dataset == "all" else root / dataset for root in log_roots]
    candidates = sorted(
        (path for root in search_roots for path in root.glob(f"**/checkpoints/{checkpoint_name}")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            (path for root in search_roots for path in root.glob("**/checkpoints/last.pth")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "No default metric-learning checkpoint was found. "
        f"Searched under: {search_roots} for {checkpoint_name} and last.pth. "
        "Train first with s1a_train_jpleg.py, or pass --checkpoint explicitly."
    )


def resolve_checkpoint_arg(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint).resolve()
    if args.run_dir is not None:
        checkpoint = checkpoint_from_run_dir(Path(args.run_dir), args.checkpoint_name)
        print(f"Using checkpoint from run dir: {checkpoint}")
        return checkpoint.resolve()
    run_name = (args.run_name or "").strip()
    if run_name:
        if args.dataset == "all":
            raise ValueError("--run-name requires a concrete --dataset, not --dataset all")
        run_dirs = [
            DEFAULT_LOG_ROOT / args.dataset / run_name,
            LEGACY_LOG_ROOT / args.dataset / run_name,
        ]
        checkpoint = next(
            (run_dir / "checkpoints" / args.checkpoint_name for run_dir in run_dirs
             if (run_dir / "checkpoints" / args.checkpoint_name).exists()),
            None,
        )
        if checkpoint is None:
            searched = [run_dir / "checkpoints" / args.checkpoint_name for run_dir in run_dirs]
            raise FileNotFoundError(f"Checkpoint not found. Searched: {searched}")
        print(f"Using checkpoint from run name: {checkpoint}")
        return checkpoint.resolve()
    checkpoint = resolve_default_checkpoint(args.dataset, args.checkpoint_name)
    print(f"Using latest default checkpoint: {checkpoint}")
    return checkpoint.resolve()


def default_output_dir(checkpoint: Path, dataset: str, split: str) -> Path:
    folder_name = f"{checkpoint_run_name(checkpoint)}_{Path(checkpoint).stem}_{split}"
    return DEFAULT_OUTPUT_ROOT / dataset / folder_name


@contextlib.contextmanager
def maybe_silence(verbose: bool):
    if verbose:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            yield


def gi_to_predicted_positions(gi: np.ndarray, grid: int) -> np.ndarray:
    num_pieces = grid * grid
    pred = np.full(num_pieces, -1, dtype=np.int32)
    gi = np.asarray(gi)
    rows = min(grid, gi.shape[0])
    cols = min(grid, gi.shape[1])
    for row in range(rows):
        for col in range(cols):
            piece_id = int(gi[row, col])
            if 1 <= piece_id <= num_pieces:
                pred[piece_id - 1] = row * grid + col
    return pred


def compute_relationship_counts(pred: np.ndarray, target: np.ndarray, grid: int) -> dict[str, int]:
    """计算与 TEN 一致的方向敏感 HA/VA 和方向无关 NA 关系计数。"""

    num_pieces = grid * grid
    pred = np.asarray(pred, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if pred.shape != (num_pieces,) or target.shape != (num_pieces,):
        raise ValueError(
            f"Expected prediction and target shape ({num_pieces},), got {pred.shape} and {target.shape}"
        )
    if not np.array_equal(np.sort(target), np.arange(num_pieces)):
        raise ValueError("Target positions must be a permutation of all grid positions")

    piece_at_target_position = np.empty(num_pieces, dtype=np.int64)
    piece_at_target_position[target] = np.arange(num_pieces)
    horizontal_correct = horizontal_total = 0
    vertical_correct = vertical_total = 0
    neighbor_correct = neighbor_total = 0

    def predicted_coordinates(piece_index: int) -> tuple[int, int] | None:
        predicted_position = int(pred[piece_index])
        if not 0 <= predicted_position < num_pieces:
            return None
        return divmod(predicted_position, grid)

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
                    horizontal_correct += int(row_b - row_a == 0 and column_b - column_a == 1)
                    neighbor_correct += int(abs(row_b - row_a) + abs(column_b - column_a) == 1)

            if row + 1 < grid:
                lower_piece = int(piece_at_target_position[target_position + grid])
                lower_coordinates = predicted_coordinates(lower_piece)
                vertical_total += 1
                neighbor_total += 1
                if current_coordinates is not None and lower_coordinates is not None:
                    row_a, column_a = current_coordinates
                    row_b, column_b = lower_coordinates
                    vertical_correct += int(row_b - row_a == 1 and column_b - column_a == 0)
                    neighbor_correct += int(abs(row_b - row_a) + abs(column_b - column_a) == 1)

    return {
        "horizontal_correct": horizontal_correct,
        "horizontal_total": horizontal_total,
        "vertical_correct": vertical_correct,
        "vertical_total": vertical_total,
        "neighbor_correct": neighbor_correct,
        "neighbor_total": neighbor_total,
    }


def positions_to_image(image: np.ndarray, positions: np.ndarray, grid: int) -> np.ndarray:
    pieces = image_to_pieces(image, grid)
    restored = np.zeros_like(image)
    for current_position, restored_position in enumerate(np.asarray(positions, dtype=np.int32)):
        if restored_position < 0 or restored_position >= grid * grid:
            continue
        row, col = divmod(int(restored_position), grid)
        restored[row * PIECE_SIZE : (row + 1) * PIECE_SIZE, col * PIECE_SIZE : (col + 1) * PIECE_SIZE, :] = pieces[current_position]
    return restored


def draw_grid(ax, grid: int) -> None:
    limit = grid * PIECE_SIZE
    for pos in range(PIECE_SIZE, limit, PIECE_SIZE):
        ax.axhline(pos - 0.5, color="white", linewidth=0.6, alpha=0.8)
        ax.axvline(pos - 0.5, color="white", linewidth=0.6, alpha=0.8)


def draw_prediction_boxes(ax, pred: np.ndarray, target: np.ndarray, grid: int) -> None:
    from matplotlib.patches import Rectangle

    inset = 3.0
    box_size = PIECE_SIZE - 2 * inset
    for current_position, restored_position in enumerate(np.asarray(pred, dtype=np.int32)):
        if restored_position < 0 or restored_position >= grid * grid:
            continue
        row, col = divmod(int(restored_position), grid)
        correct = current_position < len(target) and int(restored_position) == int(target[current_position])
        color = "#2fb344" if correct else "#e03131"
        ax.add_patch(
            Rectangle(
                (col * PIECE_SIZE + inset - 0.5, row * PIECE_SIZE + inset - 0.5),
                box_size,
                box_size,
                fill=False,
                edgecolor=color,
                linewidth=2.8,
                zorder=5,
            )
        )


def save_visual_comparison(
    path: Path,
    input_image: np.ndarray,
    predicted_image: np.ndarray,
    target_image: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    grid: int,
    sample_index: int,
    correct_pieces: int,
    perfect: bool,
    method_name: str = "Puzzle hard triplet",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    images = [input_image, predicted_image, target_image]
    titles = [
        f"Input scrambled\nsample {sample_index}",
        f"{method_name} solved\nAA {correct_pieces}/{grid * grid}, PA {perfect}",
        "Ground truth",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for idx, (ax, image, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(image)
        draw_grid(ax, grid)
        if idx == 1:
            draw_prediction_boxes(ax, pred, target, grid)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def solve_with_metric(image: np.ndarray, grid: int, scorer: MetricCompatibilityScorer, verbose_solver: bool = False) -> np.ndarray:
    pieces_array = image_to_pieces(image, grid)
    pieces = [pieces_array[idx] for idx in range(pieces_array.shape[0])]
    position_key = np.arange(1, grid * grid + 1, dtype=np.int32)
    scores = scorer.score_pieces(pieces, rot_flag=0)
    with maybe_silence(verbose_solver):
        gi, _gr, _solved, _results = do_all_assembly_of_puzzle(
            pieces,
            scores,
            nr=grid,
            nc=grid,
            rot_flag=0,
            position_key=position_key,
        )
    return gi


def evaluate_sample(image: np.ndarray, label: np.ndarray, grid: int, scorer: MetricCompatibilityScorer, verbose_solver: bool = False):
    gi = solve_with_metric(image, grid, scorer=scorer, verbose_solver=verbose_solver)
    pred = gi_to_predicted_positions(gi, grid)
    target = label_to_target_positions(label, grid).astype(np.int32)
    correct = pred == target
    return int(correct.sum()), bool(correct.all()), pred, target, gi


def evaluate_dataset(
    dataset: str,
    split: str,
    data_root: Path,
    scorer: MetricCompatibilityScorer,
    max_samples: int | None,
    start_index: int,
    progress_interval: int,
    output_dir: Path | None,
    save_visuals: bool,
    visual_limit: int,
    visual_interval: int,
    save_predictions: bool,
    verbose_solver: bool,
    method_name: str = "S1A hard triplet",
) -> DatasetMetrics:
    config = JPLEG_CONFIGS[dataset]
    images, labels = load_jpleg_arrays(data_root, dataset, split)
    if start_index < 0 or start_index >= len(images):
        raise ValueError(f"start-index must be in [0, {len(images) - 1}], got {start_index}")
    end_index = len(images) if max_samples is None else min(len(images), start_index + max_samples)
    sample_count = end_index - start_index
    correct_pieces = 0
    perfect_puzzles = 0
    horizontal_correct = horizontal_total = 0
    vertical_correct = vertical_total = 0
    neighbor_correct = neighbor_total = 0
    visualized_samples = 0
    predictions = []
    targets = []
    started = time.perf_counter()
    visual_dir = output_dir / "visuals" / dataset if output_dir is not None and save_visuals else None

    print("")
    print(f"Evaluating {method_name} on {dataset.upper()} {split}: samples={sample_count}, grid={config.grid}x{config.grid}")

    for done, index in enumerate(range(start_index, end_index), start=1):
        image = np.asarray(images[index])
        label = np.asarray(labels[index])
        correct, perfect, pred, target, _gi = evaluate_sample(
            image=image,
            label=label,
            grid=config.grid,
            scorer=scorer,
            verbose_solver=verbose_solver,
        )
        correct_pieces += correct
        perfect_puzzles += int(perfect)
        relationship_counts = compute_relationship_counts(pred, target, config.grid)
        horizontal_correct += relationship_counts["horizontal_correct"]
        horizontal_total += relationship_counts["horizontal_total"]
        vertical_correct += relationship_counts["vertical_correct"]
        vertical_total += relationship_counts["vertical_total"]
        neighbor_correct += relationship_counts["neighbor_correct"]
        neighbor_total += relationship_counts["neighbor_total"]
        if save_predictions:
            predictions.append(pred)
            targets.append(target)
        if visual_dir is not None and visualized_samples < visual_limit and (done - 1) % visual_interval == 0:
            save_visual_comparison(
                visual_dir / f"sample_{index:06d}.png",
                input_image=image,
                predicted_image=positions_to_image(image, pred, config.grid),
                target_image=positions_to_image(image, target, config.grid),
                pred=pred,
                target=target,
                grid=config.grid,
                sample_index=index,
                correct_pieces=correct,
                perfect=perfect,
                method_name=method_name,
            )
            visualized_samples += 1
        if progress_interval > 0 and (done == 1 or done % progress_interval == 0 or done == sample_count):
            sra_correct = horizontal_correct + vertical_correct
            sra_total = horizontal_total + vertical_total
            print(
                f"  [{done:>{len(str(sample_count))}}/{sample_count}] "
                f"PA={perfect_puzzles / done:.2%}, "
                f"AA={correct_pieces / (done * config.num_pieces):.2%}, "
                f"HA={horizontal_correct / horizontal_total:.2%}, "
                f"VA={vertical_correct / vertical_total:.2%}, "
                f"SRA={sra_correct / sra_total:.2%}, "
                f"NA={neighbor_correct / neighbor_total:.2%}"
            )

    metrics = DatasetMetrics(
        dataset=dataset,
        split=split,
        samples=sample_count,
        perfect_puzzles=perfect_puzzles,
        correct_pieces=correct_pieces,
        total_pieces=sample_count * config.num_pieces,
        correct_horizontal_relationships=horizontal_correct,
        total_horizontal_relationships=horizontal_total,
        correct_vertical_relationships=vertical_correct,
        total_vertical_relationships=vertical_total,
        correct_neighbor_relationships=neighbor_correct,
        total_neighbor_relationships=neighbor_total,
        visualized_samples=visualized_samples,
        elapsed_seconds=time.perf_counter() - started,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_base = output_dir / f"{dataset}_{split}"
        if save_predictions:
            np.savez_compressed(
                out_base.with_suffix(".npz"),
                predictions=np.asarray(predictions, dtype=np.int32),
                targets=np.asarray(targets, dtype=np.int32),
                grid_size=np.asarray(config.grid, dtype=np.int32),
                start_index=np.asarray(start_index, dtype=np.int32),
            )
        out_base.with_suffix(".json").write_text(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def print_summary(metrics_list: list[DatasetMetrics]) -> None:
    print("")
    print("=============== JPLEG PUZZLE-LEVEL HARD TRIPLET RESULT SUMMARY ===============")
    for metrics in metrics_list:
        print(
            f"{metrics.dataset.upper()} {metrics.split}: "
            f"PA={metrics.pa:.2%} ({metrics.perfect_puzzles}/{metrics.samples}), "
            f"AA={metrics.aa:.2%} ({metrics.correct_pieces}/{metrics.total_pieces}), "
            f"HA={metrics.ha:.2%}, VA={metrics.va:.2%}, "
            f"SRA={metrics.sra:.2%}, NA={metrics.na:.2%}, "
            f"time={metrics.elapsed_seconds:.1f}s"
        )
    print("===============================================================================")


def summary_filename(dataset: str, split: str) -> str:
    return "summary.txt"


def write_summary(path: Path, metrics_list: list[DatasetMetrics], args: argparse.Namespace, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "JPLEG Puzzle-Level Hard Triplet Metric Evaluation Summary",
        f"checkpoint: {args.checkpoint}",
        f"data_root: {args.data_root}",
        f"split: {split}",
        f"dataset: {args.dataset}",
        f"start_index: {args.start_index}",
        f"max_samples: {args.max_samples}",
        f"postprocess: {args.postprocess}",
        "",
    ]
    for metrics in metrics_list:
        lines.extend(
            [
                f"[{metrics.dataset.upper()} {metrics.split}]",
                f"samples: {metrics.samples}",
                f"PA: {metrics.pa:.2%} ({metrics.perfect_puzzles}/{metrics.samples})",
                f"AA: {metrics.aa:.2%} ({metrics.correct_pieces}/{metrics.total_pieces})",
                f"HA: {metrics.ha:.2%} ({metrics.correct_horizontal_relationships}/{metrics.total_horizontal_relationships})",
                f"VA: {metrics.va:.2%} ({metrics.correct_vertical_relationships}/{metrics.total_vertical_relationships})",
                f"SRA: {metrics.sra:.2%} ({metrics.correct_horizontal_relationships + metrics.correct_vertical_relationships}/{metrics.total_horizontal_relationships + metrics.total_vertical_relationships})",
                f"NA: {metrics.na:.2%} ({metrics.correct_neighbor_relationships}/{metrics.total_neighbor_relationships})",
                f"visualized_samples: {metrics.visualized_samples}",
                f"elapsed_seconds: {metrics.elapsed_seconds:.3f}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.checkpoint = resolve_checkpoint_arg(args)
    split = SPLIT_ALIASES[args.split]
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.checkpoint, args.dataset, split)
    args.output_dir = prepare_unique_output_dir(args.output_dir)
    device = resolve_device(args.device, args.gpu_id)
    scorer = MetricCompatibilityScorer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
    )
    datasets = list(JPLEG_CONFIGS) if args.dataset == "all" else [args.dataset]
    metrics_list = []
    max_samples = args.max_samples or None
    for dataset in datasets:
        metrics_list.append(
            evaluate_dataset(
                dataset=dataset,
                split=split,
                data_root=args.data_root,
                scorer=scorer,
                max_samples=max_samples,
                start_index=args.start_index,
                progress_interval=args.progress_interval,
                output_dir=args.output_dir,
                save_visuals=args.save_visuals,
                visual_limit=args.visual_limit,
                visual_interval=args.visual_interval,
                save_predictions=args.save_predictions,
                verbose_solver=args.verbose_solver,
            )
        )
    print_summary(metrics_list)
    summary_path = args.output_dir / summary_filename(args.dataset, split)
    write_summary(summary_path, metrics_list, args, split)
    append_runtime_environment(
        summary_path,
        requested_device=args.device,
        resolved_device=device,
        gpu_id=args.gpu_id,
    )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
