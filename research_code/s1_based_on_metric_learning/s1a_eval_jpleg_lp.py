"""
@file: s1a_eval_jpleg_lp.py
@description: 加载 JPLEG 的 S1A E1 checkpoint，并使用 BMVC 2016 Type-1 纯 Python LP 求解器评估。
@author: Changxin Ye
@created: 2026-07-26
@version: 1.0
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
from pathlib import Path

try:
    from .assembly_solvers.lp_bmvc2016 import LPSolverConfig, solve_with_lp
    from .experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from .metric_jpleg_data import DEFAULT_DATA_ROOT, JPLEG_CONFIGS, SPLIT_ALIASES
    from .metric_jpleg_evaluator import evaluate_jpleg_split
    from .metric_scorer import MetricCompatibilityScorer, resolve_device
    from .s1a_eval_jpleg import (
        DEFAULT_CHECKPOINT_NAME,
        DEFAULT_RUN_DIR,
        parse_bool,
        resolve_checkpoint_arg,
    )
except ImportError:
    from assembly_solvers.lp_bmvc2016 import LPSolverConfig, solve_with_lp
    from experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from metric_jpleg_data import DEFAULT_DATA_ROOT, JPLEG_CONFIGS, SPLIT_ALIASES
    from metric_jpleg_evaluator import evaluate_jpleg_split
    from metric_scorer import MetricCompatibilityScorer, resolve_device
    from s1a_eval_jpleg import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_DIR, parse_bool, resolve_checkpoint_arg


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "jpleg5"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_RUN_NAMES = {
    "jpleg3": "hard_triplet_d128_s224_2026-07-27-08-14-53",
    "jpleg5": "hard_triplet_d128_s224_2026-07-28-23-27-31",
}
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]

# PuzzleParametersSet.m 的 Type-1 默认配置。
DEFAULT_LP_TOP_K = 1
DEFAULT_LP_REFINEMENT_ITERATIONS = 5
DEFAULT_LP_PROBABILITY_LAMBDA = 5.0
DEFAULT_LP_MINIMUM_WEIGHT = 1e-15
DEFAULT_LP_MATCH_TOLERANCE = 1e-4
DEFAULT_LP_RIGID_COMPONENTS = True
DEFAULT_LP_BUDDY_CHECK = True
DEFAULT_LP_LOOP_CHECK = True
DEFAULT_LP_GREEDY_STOP_THRESHOLD = 1.25
DEFAULT_LP_TIME_LIMIT = 0.0


def add_lp_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--lp-top-k", default=DEFAULT_LP_TOP_K, type=int, help="每个 piece 在每个方向保留的候选数 kk")
    parser.add_argument("--lp-refinement-iterations", default=DEFAULT_LP_REFINEMENT_ITERATIONS, type=int, help="初始 LP 之后最多执行多少轮刚性组件优化")
    parser.add_argument("--lp-probability-lambda", default=DEFAULT_LP_PROBABILITY_LAMBDA, type=float, help="概率权重 exp(-lambda * ratio^2) 中的 lambda")
    parser.add_argument("--lp-minimum-weight", default=DEFAULT_LP_MINIMUM_WEIGHT, type=float, help="作者代码中的数值下限 t；达到该下限的候选会被删除")
    parser.add_argument("--lp-match-tolerance", default=DEFAULT_LP_MATCH_TOLERANCE, type=float, help="判断 LP 邻接等式成立的绝对误差阈值")
    parser.add_argument("--lp-rigid-components", default=DEFAULT_LP_RIGID_COMPONENTS, type=parse_bool, help="是否用硬等式约束保持上一轮连通组件内部形状")
    parser.add_argument("--lp-buddy-check", default=DEFAULT_LP_BUDDY_CHECK, type=parse_bool, help="是否只保留具有反向镜像记录的候选")
    parser.add_argument("--lp-loop-check", default=DEFAULT_LP_LOOP_CHECK, type=parse_bool, help="是否执行作者定义的八种四步闭环检查")
    parser.add_argument("--lp-greedy-stop-threshold", default=DEFAULT_LP_GREEDY_STOP_THRESHOLD, type=float, help="组件级 Gallagher 合并的归一化分数停止阈值")
    parser.add_argument("--lp-time-limit", default=DEFAULT_LP_TIME_LIMIT, type=float, help="每次 HiGHS 求解的时间上限（秒）；0 表示不限制")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "使用纯 Python BMVC 2016 Type-1 LP 求解器评估 JPLEG 的 S1A E1 兼容性",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=JPLEG_CONFIGS.keys(), help="要评估的 JPLEG 数据集；不同数据集使用各自 checkpoint")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=SPLIT_ALIASES.keys(), help="要评估的数据划分")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path, help="MET_Dataset 数据集根目录")
    parser.add_argument("--checkpoint", default=None, type=Path, help="精确 checkpoint 路径；设置后优先级最高")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="logs/s1a_train_jpleg/<dataset> 下的实验文件夹名")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="直接指定包含 checkpoints 的实验目录")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="要读取的 checkpoint 文件名")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="最多评估多少张拼图；0 表示完整 split")
    parser.add_argument("--start-index", default=0, type=int, help="从 split 中第几张拼图开始")
    parser.add_argument("--score-batch-size", default=128, type=int, help="提取边缘 embedding 时的前向批量")
    parser.add_argument("--postprocess", default=False, type=parse_bool, help="是否在 LP 前额外执行 S1A 分数后处理；忠实复现默认关闭")
    parser.add_argument("--device", default="auto", help="运行设备，例如 auto、cpu、cuda:0")
    parser.add_argument("--gpu-id", default=0, type=int, help="device=auto 时使用的 GPU 编号")
    parser.add_argument("--progress-interval", default=25, type=int, help="每处理多少张拼图打印一次指标；0 表示关闭")
    parser.add_argument("--verbose-solver", action="store_true", help="显示每轮 LP 的候选、组件和目标值")
    parser.add_argument("--output-dir", default=None, type=Path, help="结果保存目录；留空时自动生成")
    parser.add_argument("--save-visuals", default=True, type=parse_bool, help="是否保存拼图可视化")
    parser.add_argument("--visual-limit", default=10, type=int, help="最多保存多少张可视化图片")
    parser.add_argument("--save-predictions", default=True, type=parse_bool, help="是否保存 predictions.npz")
    return add_lp_arguments(parser)


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def lp_config_from_args(args: argparse.Namespace) -> LPSolverConfig:
    return LPSolverConfig(
        top_k=args.lp_top_k,
        refinement_iterations=args.lp_refinement_iterations,
        probability_lambda=args.lp_probability_lambda,
        minimum_weight=args.lp_minimum_weight,
        match_tolerance=args.lp_match_tolerance,
        rigid_components=args.lp_rigid_components,
        buddy_check=args.lp_buddy_check,
        loop_check=args.lp_loop_check,
        greedy_stop_threshold=args.lp_greedy_stop_threshold,
        highs_time_limit=args.lp_time_limit,
    )


def default_output_dir(checkpoint: Path, args: argparse.Namespace, split: str) -> Path:
    folder = (
        f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{split}_"
        f"lpfaithful_k{args.lp_top_k}_r{args.lp_refinement_iterations}"
    )
    return DEFAULT_OUTPUT_ROOT / args.dataset / folder


def validate_args(args: argparse.Namespace, config: LPSolverConfig) -> None:
    if args.start_index < 0 or args.max_samples < 0:
        raise ValueError("start-index 和 max-samples 必须非负")
    if args.score_batch_size <= 0:
        raise ValueError("score-batch-size 必须为正整数")
    if args.visual_limit < 0 or args.progress_interval < 0:
        raise ValueError("visual-limit 和 progress-interval 必须非负")
    config.validate(JPLEG_CONFIGS[args.dataset].num_pieces)


def main() -> None:
    args = parse_args()
    config = lp_config_from_args(args)
    validate_args(args, config)
    args.checkpoint = resolve_checkpoint_arg(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    split = SPLIT_ALIASES[args.split]
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.checkpoint, args, split)
    args.output_dir = prepare_unique_output_dir(args.output_dir)

    device = resolve_device(args.device, args.gpu_id)
    scorer = MetricCompatibilityScorer.from_checkpoint(
        args.checkpoint,
        device=device,
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
    )
    solver = partial(solve_with_lp, config=config)
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"LP configuration: {asdict(config)}")
    metrics = evaluate_jpleg_split(
        scorer=scorer,
        data_root=args.data_root,
        dataset=args.dataset,
        split=split,
        solver=solver,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        max_samples=args.max_samples,
        start_index=args.start_index,
        progress_interval=args.progress_interval,
        verbose_solver=args.verbose_solver,
        save_visuals=args.save_visuals,
        visual_limit=args.visual_limit,
        save_predictions=args.save_predictions,
        method_name="S1A E1 + faithful Python LP",
        configuration={
            "implementation": "faithful_bmvc2016_type1_python_port",
            "postprocess": args.postprocess,
            **asdict(config),
        },
    )
    print(
        f"Result: PA={metrics.pa:.2%}, AA={metrics.aa:.2%}, HA={metrics.ha:.2%}, "
        f"VA={metrics.va:.2%}, SRA={metrics.sra:.2%}, NA={metrics.na:.2%}, "
        f"time={metrics.elapsed_seconds:.1f}s"
    )
    print(f"Saved: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
