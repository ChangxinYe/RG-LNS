"""
@file: s1a4_eval_lsej_lp_profiled_large_neighborhood_reassembly.py
@description: 使用 BMVC 2016 Type-1 纯 Python 忠实移植生成 ImageNet-LSEJ 初始布局，
              再执行 S1A4 基于多路径回填的 profiled large-neighborhood reassembly。
              每个可靠组件放置位置均保留多条剩余 piece 回填路径，并由完整布局的
              E1 目标选择代表解；beam=1 时严格退化为同配置的 S1A3 单路径重组。
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
    from .s1a4_eval_lsej_profiled_large_neighborhood_reassembly import (
        build_parser,
        run_evaluation,
    )
    from .s1a_eval_lsej_lp import add_lp_arguments, lp_config_from_args
except ImportError:
    from assembly_solvers.lp_bmvc2016 import solve_with_lp
    from experiment_naming import script_output_root
    from s1a4_eval_lsej_profiled_large_neighborhood_reassembly import (
        build_parser,
        run_evaluation,
    )
    from s1a_eval_lsej_lp import add_lp_arguments, lp_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = (
        "以作者 BMVC 2016 Type-1 LP 的纯 Python 忠实移植作为 S1A4 初始求解器"
    )
    # 与已有 S1A/S1A3 LP 入口保持相同口径：LP 默认直接使用原始 E1，
    # 不额外执行 Gallagher 版本默认开启的兼容性后处理。
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
