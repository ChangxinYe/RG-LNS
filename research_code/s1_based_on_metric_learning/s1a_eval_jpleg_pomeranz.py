"""
@file: s1a_eval_jpleg_pomeranz.py
@description: 使用统一 S1A E1 距离和 Pomeranz CVPR 2011 纯 Python 求解器评估 JPLEG。
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
    from .assembly_solvers.pomeranz_cvpr2011 import solve_with_pomeranz
    from .experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from .metric_jpleg_data import DEFAULT_DATA_ROOT, JPLEG_CONFIGS, SPLIT_ALIASES
    from .metric_jpleg_evaluator import evaluate_jpleg_split
    from .metric_scorer import MetricCompatibilityScorer, resolve_device
    from .s1a_eval_jpleg import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_DIR, parse_bool, resolve_checkpoint_arg
    from .s1a_eval_lsej_pomeranz import add_pomeranz_arguments, pomeranz_config_from_args
except ImportError:
    from assembly_solvers.pomeranz_cvpr2011 import solve_with_pomeranz
    from experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from metric_jpleg_data import DEFAULT_DATA_ROOT, JPLEG_CONFIGS, SPLIT_ALIASES
    from metric_jpleg_evaluator import evaluate_jpleg_split
    from metric_scorer import MetricCompatibilityScorer, resolve_device
    from s1a_eval_jpleg import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_DIR, parse_bool, resolve_checkpoint_arg
    from s1a_eval_lsej_pomeranz import add_pomeranz_arguments, pomeranz_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "jpleg3"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_RUN_NAMES = {
    "jpleg3": "hard_triplet_d128_s224_2026-07-27-08-14-53",
    "jpleg5": "hard_triplet_d128_s224_2026-07-28-23-27-31",
}
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "使用 S1A E1 距离和 Pomeranz CVPR 2011 求解器评估 JPLEG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=JPLEG_CONFIGS.keys(), help="要评估的 JPLEG 数据集")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=SPLIT_ALIASES.keys(), help="要评估的数据划分")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path, help="MET_Dataset 数据集根目录")
    parser.add_argument("--checkpoint", default=None, type=Path, help="精确 checkpoint 路径；设置后优先级最高")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="logs/s1a_train_jpleg/<dataset> 下的实验文件夹名")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="直接指定包含 checkpoints 的实验目录")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="要读取的 checkpoint 文件名")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="最多评估多少张拼图；0 表示完整 split")
    parser.add_argument("--start-index", default=0, type=int, help="从 split 中第几张拼图开始")
    parser.add_argument("--score-batch-size", default=128, type=int, help="提取边缘 embedding 时的前向批量")
    parser.add_argument("--postprocess", default=True, type=parse_bool, help="是否对 E1 距离矩阵执行现有 S1A 后处理")
    parser.add_argument("--device", default="auto", help="运行设备，例如 auto、cpu、cuda:0")
    parser.add_argument("--gpu-id", default=2, type=int, help="device=auto 时使用的 GPU 编号")
    parser.add_argument("--progress-interval", default=25, type=int, help="每处理多少张拼图打印一次指标；0 表示关闭")
    parser.add_argument("--verbose-solver", action="store_true", help="打印 Pomeranz 每次重启和重组轮次摘要")
    parser.add_argument("--output-dir", default=None, type=Path, help="结果保存目录；留空时自动生成")
    parser.add_argument("--save-visuals", default=True, type=parse_bool, help="是否保存拼图结果可视化")
    parser.add_argument("--visual-limit", default=10, type=int, help="最多保存多少张可视化图片")
    parser.add_argument("--save-predictions", default=True, type=parse_bool, help="是否保存 predictions.npz")
    return add_pomeranz_arguments(parser)


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def default_output_dir(checkpoint: Path, args: argparse.Namespace, split: str) -> Path:
    folder = (
        f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{split}_"
        f"pomeranz_r{args.pomeranz_restarts}_m{args.pomeranz_max_refinement_rounds}_"
        f"s{args.pomeranz_random_seed}"
    )
    return DEFAULT_OUTPUT_ROOT / args.dataset / folder


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0 or args.max_samples < 0:
        raise ValueError("start-index 和 max-samples 必须非负")
    if args.score_batch_size <= 0:
        raise ValueError("score-batch-size 必须为正整数")
    if args.visual_limit < 0 or args.progress_interval < 0:
        raise ValueError("visual-limit 和 progress-interval 必须非负")
    pomeranz_config_from_args(args).validate(JPLEG_CONFIGS[args.dataset].num_pieces)


def main() -> None:
    args = parse_args()
    config = pomeranz_config_from_args(args)
    validate_args(args)
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
    solver = partial(solve_with_pomeranz, config=config)
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"Pomeranz configuration: {asdict(config)}")
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
        method_name="S1A E1 + Pomeranz CVPR 2011",
        configuration={
            "implementation": "paper_guided_python_reimplementation_with_s1a_e1",
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
