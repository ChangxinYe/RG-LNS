"""
@file: s1a4_eval_lsej_pomeranz_profiled_large_neighborhood_reassembly.py
@description: 使用 S1A E1 与 Pomeranz CVPR 2011 求解器生成 ImageNet-LSEJ 初始布局，
              再执行 S1A4 基于多路径回填的 profiled large-neighborhood reassembly。
              方法显式比较每个可靠组件放置位置下的多条完整回填结果，并且只在
              全局 E1 目标严格改善时接受新布局；beam=1 时退化为 S1A3。
@author: Changxin Ye
@created: 2026-08-01
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

try:
    from .assembly_solvers.pomeranz_cvpr2011 import solve_with_pomeranz
    from .experiment_naming import script_output_root
    from .s1a4_eval_lsej_profiled_large_neighborhood_reassembly import (
        build_parser,
        run_evaluation,
    )
    from .s1a_eval_lsej_pomeranz import (
        add_pomeranz_arguments,
        pomeranz_config_from_args,
    )
except ImportError:
    from assembly_solvers.pomeranz_cvpr2011 import solve_with_pomeranz
    from experiment_naming import script_output_root
    from s1a4_eval_lsej_profiled_large_neighborhood_reassembly import (
        build_parser,
        run_evaluation,
    )
    from s1a_eval_lsej_pomeranz import (
        add_pomeranz_arguments,
        pomeranz_config_from_args,
    )


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = (
        "以 S1A E1 + Pomeranz CVPR 2011 完整解作为 S1A4 初始布局"
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
