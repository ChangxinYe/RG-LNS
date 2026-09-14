"""
@file: s1a3_eval_jpleg_pomeranz_iterative_component_reassembly.py
@description: 使用 S1A E1 + Pomeranz 生成 JPLEG 初始解，再执行我们的迭代式组件重组。
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

try:
    from .assembly_solvers.pomeranz_cvpr2011 import solve_with_pomeranz
    from .experiment_naming import script_output_root
    from .metric_jpleg_data import JPLEG_CONFIGS
    from .s1a3_eval_jpleg_iterative_component_reassembly import build_parser, run_evaluation
    from .s1a_eval_lsej_pomeranz import add_pomeranz_arguments, pomeranz_config_from_args
except ImportError:
    from assembly_solvers.pomeranz_cvpr2011 import solve_with_pomeranz
    from experiment_naming import script_output_root
    from metric_jpleg_data import JPLEG_CONFIGS
    from s1a3_eval_jpleg_iterative_component_reassembly import build_parser, run_evaluation
    from s1a_eval_lsej_pomeranz import add_pomeranz_arguments, pomeranz_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")


def parse_args():
    parser = build_parser()
    parser.description = "在 JPLEG 中以 S1A E1 + Pomeranz CVPR 2011 完整解作为 S1A3 初始布局"
    add_pomeranz_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = pomeranz_config_from_args(args)
    config.validate(JPLEG_CONFIGS[args.dataset].num_pieces)
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
