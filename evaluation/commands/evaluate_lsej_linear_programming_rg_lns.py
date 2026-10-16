"""
@file: evaluate_lsej_linear_programming_rg_lns.py
@description: Generate an ImageNet-LSEJ initial layout with the faithful pure-Python BMVC 2016 Type-1 port,
              then run profiled RG-LNS with multi-path completion.
              Retain multiple remaining-piece fill paths at every reliable-component pose and select a representative
              using the full-layout E1 objective; beam=1 strictly reduces to the matching greedy single-path reassembly.
@author: Changxin Ye
@created: 2026-08-01
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

try:
    from initial_solvers.linear_programming import solve_with_lp
    from evaluation.output import script_output_root
    from evaluation.commands.evaluate_lsej_rg_lns import (
        build_parser,
        run_evaluation,
    )
    from evaluation.commands.evaluate_lsej_linear_programming import add_lp_arguments, lp_config_from_args
except ImportError:
    from initial_solvers.linear_programming import solve_with_lp
    from evaluation.output import script_output_root
    from evaluation.commands.evaluate_lsej_rg_lns import (
        build_parser,
        run_evaluation,
    )
    from evaluation.commands.evaluate_lsej_linear_programming import add_lp_arguments, lp_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = (
        "Evaluate RG-LNS on ImageNet-LSEJ with BMVC 2016 Type-1 LP initialization"
    )
    # Keep the same convention as the existing compatibility/greedy LP entry points: LP uses raw E1 by default
    # and does not apply the compatibility post-processing enabled by the Gallagher version.
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
