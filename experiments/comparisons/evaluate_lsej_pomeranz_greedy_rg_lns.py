"""
@file: evaluate_lsej_pomeranz_greedy_rg_lns.py
@description: Generate an LSEJ initial solution with learned compatibility + Pomeranz, then run greedy component reassembly.
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

from data_loaders.lsej import lsej_grid_size

from dataclasses import asdict
from functools import partial

from initial_solvers.pomeranz import solve_with_pomeranz
from evaluation.output import script_output_root
from experiments.comparisons.evaluate_lsej_gallagher_greedy_rg_lns import build_parser, run_evaluation
from evaluation.common.solver_options import add_pomeranz_arguments, pomeranz_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = "Use the complete learned compatibility + Pomeranz solution as the greedy initial layout"
    add_pomeranz_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = pomeranz_config_from_args(args)
    grid = lsej_grid_size(args.task)
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
        initial_solver_name="Pomeranz",
        initial_solver_tag=solver_tag,
        initial_solver_config={
            "implementation": "pomeranz_python",
            "postprocess": args.postprocess,
            **asdict(config),
        },
        output_root=DEFAULT_OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
