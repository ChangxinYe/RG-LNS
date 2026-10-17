"""Dispatch the exact dataset/initial-solver evaluation entry points."""

from __future__ import annotations

import argparse
import runpy
import sys


def evaluator_module(dataset: str, solver: str, method: str) -> str:
    solver_token = {
        "gallagher": "_gallagher",
        "pomeranz": "_pomeranz",
        "linear-programming": "_linear_programming",
    }[solver]
    if method == "initial":
        return f"evaluation.baselines.evaluate_{dataset}{solver_token}"
    if method == "greedy-rg-lns":
        return f"experiments.comparisons.evaluate_{dataset}{solver_token}_greedy_rg_lns"
    if method != "rg-lns":
        raise ValueError(f"Unknown evaluation method: {method}")
    return f"evaluation.commands.evaluate_{dataset}{solver_token}_rg_lns"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a ViT-T baseline with Baseline + RG-LNS.",
        epilog="Each rg-lns run reports both Baseline and Baseline + RG-LNS.",
    )
    parser.add_argument(
        "dataset", nargs="?", default="GAP-5",
        choices=("GAP-3", "GAP-5", "JPwLEG-5",
                 "ImageNet-LSEJ-10x10-2px", "ImageNet-LSEJ-10x10-5px"),
    )
    parser.add_argument("solver", nargs="?", default="gallagher", choices=("gallagher", "pomeranz", "linear-programming"))
    # Preserve earlier dispatcher commands without presenting comparison variants
    # as choices required for normal evaluation.
    parser.add_argument(
        "method", nargs="?", default="rg-lns",
        choices=("initial", "greedy-rg-lns", "rg-lns"), metavar="{rg-lns}",
    )
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    # Keep downstream options and their values out of optional positionals.
    arguments = sys.argv[1:]
    option_start = next((i for i, value in enumerate(arguments) if value.startswith("-")), len(arguments))
    args, remaining = parser.parse_known_args(arguments[:option_start])
    remaining.extend(arguments[option_start:])
    if any(arg.split("=", 1)[0] in ("--dataset", "--task") for arg in remaining):
        parser.error("Choose the concrete dataset/task as the first argument; omit --dataset and --task.")
    dataset = args.dataset
    if dataset.startswith("GAP-"):
        args.dataset = "gap"
        remaining = ["--dataset", dataset, *remaining]
    elif dataset.startswith("JPwLEG-"):
        args.dataset = "jpwleg"
        remaining = ["--dataset", dataset.lower().replace("-", ""), *remaining]
    elif dataset.startswith("ImageNet-LSEJ-"):
        args.dataset = "lsej"
        task = {
            "ImageNet-LSEJ-10x10-2px": "lsej_grid10_erode2",
            "ImageNet-LSEJ-10x10-5px": "lsej_grid10_erode5",
        }[dataset]
        remaining = ["--task", task, *remaining]
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module(
        evaluator_module(args.dataset, args.solver, args.method),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
