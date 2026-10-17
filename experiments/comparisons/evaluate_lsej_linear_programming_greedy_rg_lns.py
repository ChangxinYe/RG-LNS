"""
@file: evaluate_lsej_linear_programming_greedy_rg_lns.py
@description: Generate an initial layout with the faithful pure-Python BMVC 2016 Type-1 port, then run greedy high-confidence component reassembly.
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

from evaluation.output import script_output_root
from initial_solvers.linear_programming import solve_with_lp
from experiments.comparisons.evaluate_lsej_gallagher_greedy_rg_lns import build_parser, run_evaluation
from evaluation.common.solver_options import add_lp_arguments, lp_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = "Use the faithful pure-Python BMVC 2016 Type-1 LP port as the greedy initial solver"
    parser.set_defaults(postprocess=False)
    add_lp_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = lp_config_from_args(args)
    grid = int(str(args.task).split("_", 1)[0].removeprefix("grid"))
    config.validate(grid * grid)
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
