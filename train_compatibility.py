"""Dispatch Stage-I compatibility training without duplicating trainer logic."""

from __future__ import annotations

import argparse
import importlib
import sys


TRAINERS = {
    "gap": "compatibility.training.train_gap",
    "jpwleg": "compatibility.training.train_jpwleg",
    "lsej": "compatibility.training.train_lsej",
}


def main() -> None:
    help_parser = argparse.ArgumentParser(description="Train directional compatibility.")
    help_parser.add_argument("dataset", nargs="?", default="gap", choices=TRAINERS)
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        help_parser.print_help()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("dataset", nargs="?", default="gap", choices=TRAINERS)
    # Keep downstream options and their values out of the optional positional.
    arguments = sys.argv[1:]
    option_start = next((i for i, value in enumerate(arguments) if value.startswith("-")), len(arguments))
    args, remaining = parser.parse_known_args(arguments[:option_start])
    remaining.extend(arguments[option_start:])
    sys.argv = [sys.argv[0], *remaining]
    trainer = importlib.import_module(TRAINERS[args.dataset])
    trainer.main()


if __name__ == "__main__":
    main()
