"""Dispatch Stage-I compatibility training without duplicating trainer logic."""

from __future__ import annotations

import argparse
import runpy
import sys


TRAINERS = {
    "gap": "compatibility.training.train_gap",
    "jpleg": "compatibility.training.train_jpleg",
    "lsej": "compatibility.training.train_lsej",
}


def main() -> None:
    help_parser = argparse.ArgumentParser(description="Train directional compatibility.")
    help_parser.add_argument("dataset", choices=TRAINERS)
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        help_parser.print_help()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("dataset", choices=TRAINERS)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module(TRAINERS[args.dataset], run_name="__main__")


if __name__ == "__main__":
    main()
