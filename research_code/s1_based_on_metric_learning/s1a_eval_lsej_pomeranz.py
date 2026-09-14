"""
@file: s1a_eval_lsej_pomeranz.py
@description: 使用统一 S1A E1 距离和 Pomeranz CVPR 2011 纯 Python 求解器评估 ImageNet-LSEJ。
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
from pathlib import Path

try:
    from .assembly_solvers.pomeranz_cvpr2011 import PomeranzSolverConfig, solve_with_pomeranz
    from .experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from .metric_lsej_data import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS
    from .metric_lsej_evaluator import evaluate_lsej_split
    from .metric_scorer import MetricCompatibilityScorer, resolve_device
    from .s1a_eval_lsej import (
        DEFAULT_CHECKPOINT_NAME,
        DEFAULT_RUN_DIR,
        LOSS_TYPES,
        parse_bool,
        resolve_checkpoint_arg,
        validate_checkpoint,
    )
except ImportError:
    from assembly_solvers.pomeranz_cvpr2011 import PomeranzSolverConfig, solve_with_pomeranz
    from experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from metric_lsej_data import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS
    from metric_lsej_evaluator import evaluate_lsej_split
    from metric_scorer import MetricCompatibilityScorer, resolve_device
    from s1a_eval_lsej import (
        DEFAULT_CHECKPOINT_NAME,
        DEFAULT_RUN_DIR,
        LOSS_TYPES,
        parse_bool,
        resolve_checkpoint_arg,
        validate_checkpoint,
    )


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET_NAME = "ImageNet_LSEJ"
DEFAULT_TASK = "grid10_erode2"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0
DEFAULT_LOSS_TYPE = "hard_triplet"
DEFAULT_RUN_NAME = None
if DEFAULT_LOSS_TYPE == "hard_triplet":
    if DEFAULT_TASK == "grid10_erode2":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-07-25-10-22-41"
    elif DEFAULT_TASK == "grid10_erode5":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-08-02-21-16-37"
    elif DEFAULT_TASK == "grid10_erode8":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-08-02-21-17-06"

DEFAULT_POMERANZ_RESTARTS = 10
DEFAULT_POMERANZ_MAX_REFINEMENT_ROUNDS = 0
DEFAULT_POMERANZ_RANDOM_SEED = 0


def add_pomeranz_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--pomeranz-restarts",
        default=DEFAULT_POMERANZ_RESTARTS,
        type=int,
        help="使用多少个不同初始 seed 独立运行 Pomeranz，并按最终 BBM 选择结果",
    )
    parser.add_argument(
        "--pomeranz-max-refinement-rounds",
        default=DEFAULT_POMERANZ_MAX_REFINEMENT_ROUNDS,
        type=int,
        help="每个 seed 最多执行多少轮重组；0 表示直到 BBM 不再严格提升",
    )
    parser.add_argument(
        "--pomeranz-random-seed",
        default=DEFAULT_POMERANZ_RANDOM_SEED,
        type=int,
        help="生成多次重启初始 piece 的固定随机种子",
    )
    return parser


def pomeranz_config_from_args(args: argparse.Namespace) -> PomeranzSolverConfig:
    return PomeranzSolverConfig(
        restarts=args.pomeranz_restarts,
        max_refinement_rounds=args.pomeranz_max_refinement_rounds,
        random_seed=args.pomeranz_random_seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "使用 S1A E1 距离和 Pomeranz CVPR 2011 求解器评估 ImageNet-LSEJ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK, choices=OFFICIAL_TASKS, help="要评估的官方 LSEJ 任务")
    parser.add_argument("--loss-type", default=DEFAULT_LOSS_TYPE, choices=LOSS_TYPES, help="checkpoint 对应的训练损失")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["train", "val", "test"], help="要评估的数据划分")
    parser.add_argument("--data-root", default=DEFAULT_LSEJ_DATA_ROOT, type=Path, help="ImageNet-LSEJ 数据集根目录")
    parser.add_argument("--checkpoint", default=None, type=Path, help="精确 checkpoint 路径；设置后优先级最高")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="当前 task 日志目录下的实验文件夹名")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="直接指定包含 checkpoints 的实验目录")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="要读取的 checkpoint 文件名")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="最多评估多少张拼图；0 表示完整 split")
    parser.add_argument("--start-index", default=0, type=int, help="从 split 中第几张拼图开始")
    parser.add_argument("--score-batch-size", default=128, type=int, help="提取边缘 embedding 时的前向批量")
    parser.add_argument("--postprocess", default=True, type=parse_bool, help="是否对 E1 距离矩阵执行现有 S1A 后处理")
    parser.add_argument("--device", default="auto", help="运行设备，例如 auto、cpu、cuda:0")
    parser.add_argument("--gpu-id", default=0, type=int, help="device=auto 时使用的 GPU 编号")
    parser.add_argument("--progress-interval", default=1, type=int, help="每处理多少张拼图打印一次指标；0 表示关闭")
    parser.add_argument("--verbose-solver", action="store_true", help="打印 Pomeranz 每次重启和重组轮次摘要")
    parser.add_argument("--output-dir", default=None, type=Path, help="结果保存目录；留空时自动生成")
    parser.add_argument("--save-visuals", default=True, type=parse_bool, help="是否保存拼图结果可视化")
    parser.add_argument("--visual-limit", default=10, type=int, help="最多保存多少张可视化图片")
    parser.add_argument("--save-predictions", default=True, type=parse_bool, help="是否保存 predictions.npz")
    return add_pomeranz_arguments(parser)


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def default_output_dir(checkpoint: Path, args: argparse.Namespace) -> Path:
    folder = (
        f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{args.task}_{args.split}_"
        f"pomeranz_r{args.pomeranz_restarts}_m{args.pomeranz_max_refinement_rounds}_"
        f"s{args.pomeranz_random_seed}"
    )
    return DEFAULT_OUTPUT_ROOT / DEFAULT_DATASET_NAME / folder


def validate_args(args: argparse.Namespace, config: PomeranzSolverConfig) -> None:
    if args.start_index < 0 or args.max_samples < 0:
        raise ValueError("start-index 和 max-samples 必须非负")
    if args.score_batch_size <= 0:
        raise ValueError("score-batch-size 必须为正整数")
    if args.visual_limit < 0 or args.progress_interval < 0:
        raise ValueError("visual-limit 和 progress-interval 必须非负")
    grid = int(str(args.task).split("_", 1)[0].removeprefix("grid"))
    config.validate(grid * grid)


def main() -> None:
    args = parse_args()
    config = pomeranz_config_from_args(args)
    validate_args(args, config)
    args.checkpoint = resolve_checkpoint_arg(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    validate_checkpoint(args.checkpoint, args.task, args.loss_type)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.checkpoint, args)
    args.output_dir = prepare_unique_output_dir(args.output_dir)

    device = resolve_device(args.device, args.gpu_id)
    scorer = MetricCompatibilityScorer.from_checkpoint(
        args.checkpoint,
        device=device,
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
    )
    solver = partial(solve_with_pomeranz, config=config)
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"Pomeranz configuration: {asdict(config)}")
    metrics = evaluate_lsej_split(
        scorer=scorer,
        data_root=args.data_root,
        task=args.task,
        split=args.split,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        max_samples=args.max_samples,
        start_index=args.start_index,
        progress_interval=args.progress_interval,
        verbose_solver=args.verbose_solver,
        save_visuals=args.save_visuals,
        visual_limit=args.visual_limit,
        save_predictions=args.save_predictions,
        method_name="S1A E1 + Pomeranz CVPR 2011",
        solver=solver,
        configuration={
            "implementation": "paper_guided_python_reimplementation_with_s1a_e1",
            "postprocess": args.postprocess,
            **asdict(config),
        },
    )
    print(
        f"Result: PA={metrics.pa:.2%}, AA={metrics.aa:.2%}, "
        f"Horizontal SRA={metrics.horizontal_sra:.2%}, "
        f"Vertical SRA={metrics.vertical_sra:.2%}, SRA={metrics.sra:.2%}, "
        f"time={metrics.elapsed_seconds:.1f}s"
    )
    print(f"Saved: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
