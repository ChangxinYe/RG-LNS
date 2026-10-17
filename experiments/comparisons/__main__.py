"""Run an optional single-path comparison outside the normal evaluation workflow."""

from __future__ import annotations

import argparse
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a standalone greedy comparison for experimental analysis.",
    )
    parser.add_argument("dataset", choices=("gap", "jpwleg", "lsej"))
    parser.add_argument("solver", choices=("gallagher", "pomeranz", "linear-programming"))
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    # Forward --help after dataset/solver to the selected comparison's arguments.
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("dataset", choices=("gap", "jpwleg", "lsej"))
    selector.add_argument("solver", choices=("gallagher", "pomeranz", "linear-programming"))
    args, remaining = selector.parse_known_args()
    suffix = {"gallagher": "_gallagher", "pomeranz": "_pomeranz", "linear-programming": "_linear_programming"}[args.solver]
    module = f"experiments.comparisons.evaluate_{args.dataset}{suffix}_greedy_rg_lns"
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
