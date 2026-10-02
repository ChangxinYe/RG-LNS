from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compatibility.scorer import MetricCompatibilityScorer
from data_loaders.jpleg import (
    JPLEG_CONFIGS,
    SPLIT_ALIASES,
    image_to_pieces,
    label_to_target_positions,
    load_jpleg_arrays,
)
from initial_solvers.gallagher.mgc import do_all_assembly_of_puzzle


@dataclass
class AssemblyMetrics:
    dataset: str
    split: str
    samples: int
    perfect_puzzles: int
    correct_pieces: int
    total_pieces: int
    elapsed_seconds: float

    @property
    def pa(self) -> float:
        return self.perfect_puzzles / self.samples if self.samples else 0.0

    @property
    def aa(self) -> float:
        return self.correct_pieces / self.total_pieces if self.total_pieces else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "samples": self.samples,
            "perfect_puzzles": self.perfect_puzzles,
            "correct_pieces": self.correct_pieces,
            "total_pieces": self.total_pieces,
            "pa": self.pa,
            "aa": self.aa,
            "elapsed_seconds": self.elapsed_seconds,
        }


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


def solve_with_edge2vec(
    image: np.ndarray,
    grid: int,
    scorer: MetricCompatibilityScorer,
    verbose_solver: bool = False,
) -> np.ndarray:
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


def evaluate_assembly_sample(
    image: np.ndarray,
    label: np.ndarray,
    grid: int,
    scorer: MetricCompatibilityScorer,
    verbose_solver: bool = False,
) -> tuple[int, bool]:
    gi = solve_with_edge2vec(image, grid, scorer=scorer, verbose_solver=verbose_solver)
    pred = gi_to_predicted_positions(gi, grid)
    target = label_to_target_positions(label, grid).astype(np.int32)
    correct = pred == target
    return int(correct.sum()), bool(correct.all())


def evaluate_assembly_split(
    dataset: str,
    split: str,
    data_root: str | Path,
    scorer: MetricCompatibilityScorer,
    max_samples: int | None = None,
    start_index: int = 0,
    progress_interval: int = 0,
    verbose_solver: bool = False,
) -> AssemblyMetrics:
    split = SPLIT_ALIASES[split]
    config = JPLEG_CONFIGS[dataset]
    images, labels = load_jpleg_arrays(data_root, dataset, split)
    if start_index < 0 or start_index >= len(images):
        raise ValueError(f"start-index must be in [0, {len(images) - 1}], got {start_index}")
    end_index = len(images) if max_samples is None else min(len(images), start_index + int(max_samples))
    sample_count = end_index - start_index
    correct_pieces = 0
    perfect_puzzles = 0
    started = time.perf_counter()

    for done, index in enumerate(range(start_index, end_index), start=1):
        correct, perfect = evaluate_assembly_sample(
            image=np.asarray(images[index]),
            label=np.asarray(labels[index]),
            grid=config.grid,
            scorer=scorer,
            verbose_solver=verbose_solver,
        )
        correct_pieces += correct
        perfect_puzzles += int(perfect)
        if progress_interval > 0 and (done == 1 or done % progress_interval == 0 or done == sample_count):
            pa_now = perfect_puzzles / done
            aa_now = correct_pieces / (done * config.num_pieces)
            print(
                f"    assembly {dataset} {split} [{done:>{len(str(sample_count))}}/{sample_count}] "
                f"PA={pa_now:.2%}, AA={aa_now:.2%}"
            )

    return AssemblyMetrics(
        dataset=dataset,
        split=split,
        samples=sample_count,
        perfect_puzzles=perfect_puzzles,
        correct_pieces=correct_pieces,
        total_pieces=sample_count * config.num_pieces,
        elapsed_seconds=time.perf_counter() - started,
    )
