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
    parser = argparse.ArgumentParser(description="Train directional compatibility.")
    parser.add_argument(
        "dataset", nargs="?", default="GAP-3",
        choices=("GAP-3", "GAP-5", "JPwLEG-5",
                 "ImageNet-LSEJ-10x10-2px", "ImageNet-LSEJ-10x10-5px"),
    )
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    # Keep downstream options and their values out of the optional positional.
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
    trainer = importlib.import_module(TRAINERS[args.dataset])
    trainer.main()


if __name__ == "__main__":
    main()
