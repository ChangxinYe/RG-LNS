"""
@file: s1a_eval_gap_lp.py
@description: 使用统一 S1A E1 距离和 BMVC 2016 Type-1 纯 Python LP 求解器评估 GAP。
@author: Changxin Ye
@created: 2026-08-21
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

try:
    from .assembly_solvers.lp_bmvc2016 import solve_with_lp
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
    from .s1a_eval_jpleg_lp import add_lp_arguments, lp_config_from_args
except ImportError:
    from assembly_solvers.lp_bmvc2016 import solve_with_lp
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
    from s1a_eval_jpleg_lp import add_lp_arguments, lp_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = "Evaluate S1A on GAP-3/GAP-5 with faithful BMVC 2016 Type-1 LP"
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
    solver_config = {
        "implementation": "faithful_bmvc2016_type1_python_port",
        "postprocess": args.postprocess,
        **asdict(config),
    }
    print(f"LP configuration: {asdict(config)}")
    metrics, output_dir = run_evaluation(
        args,
        solver=solver,
        solver_name="faithful Python LP",
        method_name="S1A E1 + faithful Python LP",
        solver_configuration=solver_config,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_suffix=f"lpfaithful_k{config.top_k}_r{config.refinement_iterations}",
    )
    print_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()
