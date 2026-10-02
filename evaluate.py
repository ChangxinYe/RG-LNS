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
        return f"evaluation.commands.evaluate_{dataset}{initial_token}"
    method_token = {
        "greedy-rg-lns": "_greedy_rg_lns",
        "rg-lns": "_rg_lns",
    }[method]
    return f"evaluation.commands.evaluate_{dataset}{solver_token}{method_token}"


def main() -> None:
    help_parser = argparse.ArgumentParser(description="Evaluate initial layouts or RG-LNS.")
    help_parser.add_argument("dataset", choices=("gap", "jpleg", "lsej"))
    help_parser.add_argument("solver", choices=("gallagher", "pomeranz", "linear-programming"))
    help_parser.add_argument("method", choices=("initial", "greedy-rg-lns", "rg-lns"))
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        help_parser.print_help()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("dataset", choices=("gap", "jpleg", "lsej"))
    parser.add_argument("solver", choices=("gallagher", "pomeranz", "linear-programming"))
    parser.add_argument("method", choices=("initial", "greedy-rg-lns", "rg-lns"))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module(
        evaluator_module(args.dataset, args.solver, args.method),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
