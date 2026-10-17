"""Dispatch the exact dataset/initial-solver evaluation entry points."""

from __future__ import annotations

import argparse
import runpy
import sys


def evaluator_module(dataset: str, solver: str, method: str) -> str:
    solver_token = {
        "gallagher": "",
        "pomeranz": "_pomeranz",
        "linear-programming": "_linear_programming",
    }[solver]
    if method == "initial":
        initial_token = "_gallagher" if solver == "gallagher" else solver_token
        return f"evaluation.baselines.evaluate_{dataset}{initial_token}"
    if method == "greedy-rg-lns":
        return f"experiments.comparisons.evaluate_{dataset}{solver_token}_greedy_rg_lns"
    if method != "rg-lns":
        raise ValueError(f"Unknown evaluation method: {method}")
    return f"evaluation.commands.evaluate_{dataset}{solver_token}_rg_lns"


def main() -> None:
    help_parser = argparse.ArgumentParser(
        description="Compare a ViT-T baseline with Baseline + RG-LNS.",
        epilog="Each rg-lns run reports both Baseline and Baseline + RG-LNS.",
    )
    help_parser.add_argument("dataset", nargs="?", default="gap", choices=("gap", "jpwleg", "lsej"))
    help_parser.add_argument("solver", nargs="?", default="gallagher", choices=("gallagher", "pomeranz", "linear-programming"))
    help_parser.add_argument("method", nargs="?", default="rg-lns", choices=("rg-lns",))
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        help_parser.print_help()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("dataset", nargs="?", default="gap", choices=("gap", "jpwleg", "lsej"))
    parser.add_argument("solver", nargs="?", default="gallagher", choices=("gallagher", "pomeranz", "linear-programming"))
    # Preserve earlier dispatcher commands without presenting comparison variants
    # as choices required for normal evaluation.
    parser.add_argument("method", nargs="?", default="rg-lns", choices=("initial", "greedy-rg-lns", "rg-lns"))
    # Keep downstream options and their values out of optional positionals.
    arguments = sys.argv[1:]
    option_start = next((i for i, value in enumerate(arguments) if value.startswith("-")), len(arguments))
    args, remaining = parser.parse_known_args(arguments[:option_start])
    remaining.extend(arguments[option_start:])
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module(
        evaluator_module(args.dataset, args.solver, args.method),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
