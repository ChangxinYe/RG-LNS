"""
@file: evaluate_gap_linear_programming_greedy_rg_lns.py
@description: Generate E1-compatibility initial GAP-3/GAP-5 layouts with the pure-Python BMVC 2016 Type-1 LP solver,
              then run greedy single-path component reassembly. Output diagnostics before and after reassembly,
              jigsaw metrics, neighbor-retrieval metrics, predictions, and visualizations consistent with the Gallagher version.
@author: Changxin Ye
@created: 2026-08-01
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

from initial_solvers.linear_programming import solve_with_lp
from evaluation.output import script_output_root
from data_loaders.gap import GAP_CONFIGS
from evaluation.common.gap_search import build_parser
from experiments.comparisons.evaluate_gap_greedy_rg_lns import run_evaluation
from evaluation.common.gap_inputs import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_NAMES
from evaluation.common.solver_options import add_lp_arguments, lp_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "GAP-5"  # "GAP-3" or "GAP-5"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]


def parse_args():
    parser = build_parser()
    parser.description = (
        "Evaluate greedy component reassembly on GAP-3/GAP-5 with faithful BMVC 2016 Type-1 LP initialization"
    )
    # The faithful LP setting uses the raw compatibility scores directly and does not apply extra compatibility post-processing by default.
    parser.set_defaults(
        dataset=DEFAULT_DATASET,
        split=DEFAULT_SPLIT,
        max_samples=DEFAULT_MAX_SAMPLES,
        run_name=None,
        checkpoint_name=DEFAULT_CHECKPOINT_NAME,
        postprocess=False,
    )
    add_lp_arguments(parser)
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = DEFAULT_RUN_NAMES[args.dataset]
    return args


def main() -> None:
    args = parse_args()
    config = lp_config_from_args(args)
    config.validate(GAP_CONFIGS[args.dataset].num_pieces)
    solver = partial(solve_with_lp, config=config)
    print(f"LP configuration: {asdict(config)}")
    run_evaluation(
        args,
        initial_solver=solver,
        initial_solver_name="faithful Python LP",
        initial_solver_tag="lpfaithful",
        initial_solver_config={
            "implementation": "faithful_bmvc2016_type1_python_port",
            "postprocess": args.postprocess,
            **asdict(config),
        },
        output_root=DEFAULT_OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
