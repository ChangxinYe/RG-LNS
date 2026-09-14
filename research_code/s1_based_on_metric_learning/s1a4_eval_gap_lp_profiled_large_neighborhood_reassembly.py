"""
@file: s1a4_eval_gap_lp_profiled_large_neighborhood_reassembly.py
@description: 在 GAP-3/GAP-5 上使用 BMVC 2016 Type-1 纯 Python LP 生成 S1A E1
              初始布局，再执行 S1A4 profiled large-neighborhood reassembly，并与同一
              LP 初始布局下的 S1A3 单路径结果进行严格对照。
@author: Changxin Ye
@created: 2026-08-01
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

try:
    from .assembly_solvers.lp_bmvc2016 import solve_with_lp
    from .experiment_naming import script_output_root
    from .metric_gap_data import GAP_CONFIGS
    from .s1a4_eval_gap_profiled_large_neighborhood_reassembly import (
        build_parser,
        run_evaluation,
    )
    from .s1a_eval_gap import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_NAMES
    from .s1a_eval_gap_lp import add_lp_arguments, lp_config_from_args
except ImportError:
    from assembly_solvers.lp_bmvc2016 import solve_with_lp
    from experiment_naming import script_output_root
    from metric_gap_data import GAP_CONFIGS
    from s1a4_eval_gap_profiled_large_neighborhood_reassembly import (
        build_parser,
        run_evaluation,
    )
    from s1a_eval_gap import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_NAMES
    from s1a_eval_gap_lp import add_lp_arguments, lp_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "GAP-5"  # "GAP-3" or "GAP-5"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]


def parse_args():
    parser = build_parser()
    parser.description = (
        "Evaluate S1A4 on GAP-3/GAP-5 with faithful BMVC 2016 Type-1 LP initialization"
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
