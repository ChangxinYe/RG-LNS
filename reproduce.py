"""Dispatch paper-level ablation, sensitivity, and runtime experiments."""

from __future__ import annotations

import argparse
import runpy
import sys


EXPERIMENTS = {
    "ablation": "experiments.ablation",
    "sensitivity": "experiments.sensitivity",
    "runtime": "experiments.runtime",
}


def main() -> None:
    help_parser = argparse.ArgumentParser(description="Reproduce a paper experiment.")
    help_parser.add_argument("experiment", choices=EXPERIMENTS)
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        help_parser.print_help()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module(EXPERIMENTS[args.experiment], run_name="__main__")


if __name__ == "__main__":
    main()
