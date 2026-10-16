"""
@file: evaluate_lsej_pomeranz_rg_lns.py
@description: Generate an ImageNet-LSEJ initial layout with E1 compatibility and the Pomeranz CVPR 2011 solver,
              then run profiled RG-LNS with multi-path completion.
              Compare multiple complete fills at each reliable-component pose and accept a new layout only
              when the global E1 objective strictly improves; beam=1 reduces to greedy reassembly.
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
    from evaluation.commands.evaluate_lsej_rg_lns import (
        build_parser,
        run_evaluation,
    )
    from evaluation.commands.evaluate_lsej_pomeranz import (
        add_pomeranz_arguments,
        pomeranz_config_from_args,
    )
except ImportError:
    from initial_solvers.pomeranz import solve_with_pomeranz
    from evaluation.output import script_output_root
    from evaluation.commands.evaluate_lsej_rg_lns import (
        build_parser,
        run_evaluation,
    )
    from evaluation.commands.evaluate_lsej_pomeranz import (
        add_pomeranz_arguments,
        pomeranz_config_from_args,
    )


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = (
        "Evaluate RG-LNS on ImageNet-LSEJ with Pomeranz CVPR 2011 initialization"
    )
    add_pomeranz_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = pomeranz_config_from_args(args)
    grid = int(str(args.task).split("_", 1)[0].removeprefix("grid"))
    config.validate(grid * grid)
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
