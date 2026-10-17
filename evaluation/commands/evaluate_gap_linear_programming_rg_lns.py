"""
@file: evaluate_gap_linear_programming_rg_lns.py
@description: Generate E1-compatibility initial GAP-3/GAP-5 layouts with the pure-Python BMVC 2016 Type-1 LP solver,
              then run profiled large-neighborhood reassembly and compare it strictly with the greedy
              single-path result from the same LP initial layout.
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
from evaluation.commands.evaluate_gap_rg_lns import build_parser, run_evaluation
from evaluation.common.gap_inputs import DEFAULT_CHECKPOINT_NAME
from evaluation.common.solver_options import add_lp_arguments, lp_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "GAP-5"  # "GAP-3" or "GAP-5"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0


def parse_args():
    parser = build_parser()
    parser.description = (
        "Evaluate RG-LNS on GAP-3/GAP-5 with BMVC 2016 Type-1 LP initialization"
    )
    parser.set_defaults(
        dataset=DEFAULT_DATASET,
        split=DEFAULT_SPLIT,
        max_samples=DEFAULT_MAX_SAMPLES,
        run_name=None,
        checkpoint_name=DEFAULT_CHECKPOINT_NAME,
        postprocess=False,
    )
    add_lp_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = lp_config_from_args(args)
    config.validate(GAP_CONFIGS[args.dataset].num_pieces)
    solver = partial(solve_with_lp, config=config)
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
