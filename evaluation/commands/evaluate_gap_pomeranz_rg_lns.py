"""
@file: evaluate_gap_pomeranz_rg_lns.py
@description: Generate initial GAP-3/GAP-5 layouts with the E1 compatibility distance and Pomeranz CVPR 2011 solver,
              then run profiled large-neighborhood reassembly and compare it strictly with the greedy
              single-path result from the same Pomeranz initial layout.
@author: Changxin Ye
@created: 2026-08-01
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

try:
    from initial_solvers.pomeranz import solve_with_pomeranz
    from evaluation.output import script_output_root
    from data_loaders.gap import GAP_CONFIGS
    from evaluation.commands.evaluate_gap_rg_lns import (
        build_parser,
        run_evaluation,
    )
    from evaluation.commands.evaluate_gap_gallagher import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_NAMES
    from evaluation.commands.evaluate_gap_pomeranz import add_pomeranz_arguments, pomeranz_config_from_args
except ImportError:
    from initial_solvers.pomeranz import solve_with_pomeranz
    from evaluation.output import script_output_root
    from data_loaders.gap import GAP_CONFIGS
    from evaluation.commands.evaluate_gap_rg_lns import (
        build_parser,
        run_evaluation,
    )
    from evaluation.commands.evaluate_gap_gallagher import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_NAMES
    from evaluation.commands.evaluate_gap_pomeranz import add_pomeranz_arguments, pomeranz_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "GAP-5"  # "GAP-3" or "GAP-5"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]


def parse_args():
    parser = build_parser()
    parser.description = (
        "Evaluate profiled RG-LNS on GAP-3/GAP-5 with Pomeranz CVPR 2011 initialization"
    )
    parser.set_defaults(
        dataset=DEFAULT_DATASET,
        split=DEFAULT_SPLIT,
        max_samples=DEFAULT_MAX_SAMPLES,
        run_name=None,
        checkpoint_name=DEFAULT_CHECKPOINT_NAME,
    )
    add_pomeranz_arguments(parser)
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = DEFAULT_RUN_NAMES[args.dataset]
    return args


def main() -> None:
    args = parse_args()
    config = pomeranz_config_from_args(args)
    config.validate(GAP_CONFIGS[args.dataset].num_pieces)
    solver = partial(solve_with_pomeranz, config=config)
    solver_tag = (
        f"pomeranzr{config.restarts}m{config.max_refinement_rounds}"
        f"s{config.random_seed}"
    )
    print(f"Pomeranz configuration: {asdict(config)}")
    run_evaluation(
        args,
        initial_solver=solver,
        initial_solver_name="Pomeranz CVPR 2011",
        initial_solver_tag=solver_tag,
        initial_solver_config={
            "implementation": "paper_guided_python_reimplementation_with_s1a_e1",
            "postprocess": args.postprocess,
            **asdict(config),
        },
        output_root=DEFAULT_OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
