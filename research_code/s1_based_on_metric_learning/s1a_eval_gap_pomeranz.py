"""
@file: s1a_eval_gap_pomeranz.py
@description: 使用统一 S1A E1 距离和 Pomeranz CVPR 2011 纯 Python 求解器评估 GAP。
@author: Changxin Ye
@created: 2026-08-21
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

try:
    from .assembly_solvers.pomeranz_cvpr2011 import solve_with_pomeranz
    from .experiment_naming import script_output_root
    from .metric_gap_data import GAP_CONFIGS
    from .s1a_eval_gap import (
        DEFAULT_CHECKPOINT_NAME,
        DEFAULT_DATASET,
        DEFAULT_MAX_SAMPLES,
        DEFAULT_SPLIT,
        build_parser,
        print_metrics,
        run_evaluation,
    )
    from .s1a_eval_lsej_pomeranz import add_pomeranz_arguments, pomeranz_config_from_args
except ImportError:
    from assembly_solvers.pomeranz_cvpr2011 import solve_with_pomeranz
    from experiment_naming import script_output_root
    from metric_gap_data import GAP_CONFIGS
    from s1a_eval_gap import (
        DEFAULT_CHECKPOINT_NAME,
        DEFAULT_DATASET,
        DEFAULT_MAX_SAMPLES,
        DEFAULT_SPLIT,
        build_parser,
        print_metrics,
        run_evaluation,
    )
    from s1a_eval_lsej_pomeranz import add_pomeranz_arguments, pomeranz_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = "Evaluate S1A on GAP-3/GAP-5 with Pomeranz CVPR 2011"
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
        "implementation": "paper_guided_python_reimplementation_with_s1a_e1",
        "postprocess": args.postprocess,
        **asdict(config),
    }
    print(f"Pomeranz configuration: {asdict(config)}")
    metrics, output_dir = run_evaluation(
        args,
        solver=solver,
        solver_name="Pomeranz CVPR 2011",
        method_name="S1A E1 + Pomeranz CVPR 2011",
        solver_configuration=solver_config,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_suffix=solver_tag,
    )
    print_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()
