"""
@file: evaluate_gap_pomeranz.py
@description: Evaluate GAP with the unified learned compatibility distance and the pure-Python Pomeranz solver.
@author: Changxin Ye
@created: 2026-08-21
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

from initial_solvers.pomeranz import solve_with_pomeranz
from evaluation.output import script_output_root
from data_loaders.gap import GAP_CONFIGS
from evaluation.common.gap_inputs import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_DATASET,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_SPLIT,
    build_parser,
)
from evaluation.baselines.evaluate_gap_gallagher import print_metrics, run_evaluation
from evaluation.common.solver_options import add_pomeranz_arguments, pomeranz_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = "Evaluate the learned compatibility baseline on GAP-3/GAP-5 with Pomeranz"
    parser.set_defaults(
        dataset=DEFAULT_DATASET,
        split=DEFAULT_SPLIT,
        max_samples=DEFAULT_MAX_SAMPLES,
        run_name=None,
        checkpoint_name=DEFAULT_CHECKPOINT_NAME,
        postprocess=True,
    )
    add_pomeranz_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = pomeranz_config_from_args(args)
    config.validate(GAP_CONFIGS[args.dataset].num_pieces)
    solver = partial(solve_with_pomeranz, config=config)
    solver_tag = (
        f"pomeranzr{config.restarts}m{config.max_refinement_rounds}"
        f"s{config.random_seed}"
    )
    solver_config = {
        "implementation": "pomeranz_python",
        "postprocess": args.postprocess,
        **asdict(config),
    }
    print(f"Pomeranz configuration: {asdict(config)}")
    metrics, output_dir = run_evaluation(
        args,
        solver=solver,
        solver_name="Pomeranz",
        method_name="Baseline (ViT-T + Pomeranz)",
        solver_configuration=solver_config,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_suffix=solver_tag,
    )
    print_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()
