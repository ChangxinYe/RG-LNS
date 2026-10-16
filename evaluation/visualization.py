"""Render the input, baseline, RG-LNS result, and ground truth together."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np


def save_comparison_visual(
    path: Path,
    pieces: list[np.ndarray],
    baseline: np.ndarray,
    selected: np.ndarray,
    target: np.ndarray,
    grid: int,
    title: str,
    initial_solver_name: str,
    *,
    compose_puzzle: Callable,
) -> None:
    """Show two method results with input and ground-truth reference panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    layouts = (np.arange(grid * grid), baseline, selected, target)
    titles = (
        "Input",
        f"Baseline (ViT-T + {initial_solver_name})\n"
        f"AA {np.count_nonzero(baseline == target)}/{grid * grid}",
        f"Baseline + RG-LNS\nAA {np.count_nonzero(selected == target)}/{grid * grid}",
        "Ground truth",
    )
    piece_height, piece_width = pieces[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(20, 6))
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.02, top=0.83, wspace=0.025)
    for index, (axis, layout, label) in enumerate(zip(axes, layouts, titles)):
        axis.imshow(compose_puzzle(pieces, layout, grid))
        axis.set_title(label, fontsize=12)
        axis.axis("off")
        if index in {1, 2}:
            for piece, position in enumerate(layout):
                if not 0 <= int(position) < grid * grid:
                    continue
                row, column = divmod(int(position), grid)
                axis.add_patch(Rectangle(
                    (column * piece_width + 1.5, row * piece_height + 1.5),
                    piece_width - 4,
                    piece_height - 4,
                    fill=False,
                    edgecolor="#2fb344" if int(position) == int(target[piece]) else "#e03131",
                    linewidth=1.8,
                ))
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)
