"""
@file: evaluate_gap_pomeranz_greedy_rg_lns.py
@description: Generate initial GAP-3/GAP-5 layouts with the learned compatibility distance and Pomeranz solver,
              then run greedy single-path component reassembly. Output complete metrics, search diagnostics,
              predictions, PA repair/degradation lists, and visualizations with correctness boxes.
@author: Changxin Ye
@created: 2026-08-01
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

from initial_solvers.pomeranz import solve_with_pomeranz
from evaluation.output import script_output_root
from data_loaders.gap import GAP_CONFIGS
from evaluation.common.gap_search import build_parser
from experiments.comparisons.evaluate_gap_gallagher_greedy_rg_lns import run_evaluation
from evaluation.common.gap_inputs import DEFAULT_CHECKPOINT_NAME
from evaluation.common.solver_options import add_pomeranz_arguments, pomeranz_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "GAP-5"  # "GAP-3" or "GAP-5"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0
DEFAULT_RUN_NAME = None


def parse_args():
    parser = build_parser()
    parser.description = (
        "Evaluate greedy component reassembly on GAP-3/GAP-5 with Pomeranz initialization"
    )
    parser.set_defaults(
        dataset=DEFAULT_DATASET,
        split=DEFAULT_SPLIT,
        max_samples=DEFAULT_MAX_SAMPLES,
        run_name=DEFAULT_RUN_NAME,
        checkpoint_name=DEFAULT_CHECKPOINT_NAME,
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
