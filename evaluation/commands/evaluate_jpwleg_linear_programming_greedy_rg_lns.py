"""
@file: evaluate_jpwleg_linear_programming_greedy_rg_lns.py
@description: Generate an initial JPwLEG layout with the pure-Python BMVC 2016 Type-1 LP solver, then run our single-component greedy refinement.
@author: Changxin Ye
@created: 2026-07-26
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

try:
    from initial_solvers.linear_programming import solve_with_lp
    from evaluation.output import script_output_root
    from data_loaders.jpwleg import JPWLEG_CONFIGS
    from evaluation.commands.evaluate_jpwleg_greedy_rg_lns import build_parser, run_evaluation
    from evaluation.commands.evaluate_jpwleg_linear_programming import add_lp_arguments, lp_config_from_args
except ImportError:
    from initial_solvers.linear_programming import solve_with_lp
    from evaluation.output import script_output_root
    from data_loaders.jpwleg import JPWLEG_CONFIGS
    from evaluation.commands.evaluate_jpwleg_greedy_rg_lns import build_parser, run_evaluation
    from evaluation.commands.evaluate_jpwleg_linear_programming import add_lp_arguments, lp_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = "Use the pure-Python BMVC 2016 Type-1 LP solver as the greedy initial solver on JPwLEG"
    parser.set_defaults(postprocess=False)
    add_lp_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = lp_config_from_args(args)
    config.validate(JPWLEG_CONFIGS[args.dataset].num_pieces)
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
