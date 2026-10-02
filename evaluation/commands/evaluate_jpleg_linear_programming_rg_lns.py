"""
@file: s1a4_eval_jpleg_lp_profiled_large_neighborhood_reassembly.py
@description: 在 JPwLEG 上使用 BMVC 2016 Type-1 纯 Python LP 生成初始布局，
              再执行 S1A4 基于多路径回填的 profiled large-neighborhood reassembly。
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
    from data_loaders.jpleg import JPLEG_CONFIGS
    from evaluation.commands.evaluate_jpleg_rg_lns import (
        build_parser,
        run_evaluation,
    )
    from evaluation.commands.evaluate_jpleg_linear_programming import add_lp_arguments, lp_config_from_args
except ImportError:
    from initial_solvers.linear_programming import solve_with_lp
    from evaluation.output import script_output_root
    from data_loaders.jpleg import JPLEG_CONFIGS
    from evaluation.commands.evaluate_jpleg_rg_lns import (
        build_parser,
        run_evaluation,
    )
    from evaluation.commands.evaluate_jpleg_linear_programming import add_lp_arguments, lp_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = "在 JPwLEG 上以 BMVC 2016 Type-1 LP 作为 S1A4 初始求解器"
    parser.set_defaults(postprocess=False)
    add_lp_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = lp_config_from_args(args)
    config.validate(JPLEG_CONFIGS[args.dataset].num_pieces)
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
