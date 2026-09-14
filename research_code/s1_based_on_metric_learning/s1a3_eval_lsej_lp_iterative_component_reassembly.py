"""
@file: s1a3_eval_lsej_lp_iterative_component_reassembly.py
@description: 使用 BMVC 2016 Type-1 纯 Python 忠实移植生成初始布局，再执行 S1A3 迭代式高置信组件重组。
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

try:
    from .experiment_naming import script_output_root
    from .assembly_solvers.lp_bmvc2016 import solve_with_lp
    from .s1a3_eval_lsej_iterative_component_reassembly import build_parser, run_evaluation
    from .s1a_eval_lsej_lp import add_lp_arguments, lp_config_from_args
except ImportError:
    from experiment_naming import script_output_root
    from assembly_solvers.lp_bmvc2016 import solve_with_lp
    from s1a3_eval_lsej_iterative_component_reassembly import build_parser, run_evaluation
    from s1a_eval_lsej_lp import add_lp_arguments, lp_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = "以作者 BMVC 2016 Type-1 LP 的纯 Python 忠实移植作为 S1A3 初始求解器"
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
