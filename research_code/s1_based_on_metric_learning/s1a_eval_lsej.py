"""
@file: s1a_eval_lsej.py
@description: 加载 s1a checkpoint 并使用 Gallagher 求解器评估官方 ImageNet-LSEJ 任务的脚本。
@author: Changxin Ye
@created: 2026-07-17
@version: 1.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from .metric_lsej_data import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS
    from .metric_lsej_evaluator import evaluate_lsej_split
    from .metric_scorer import MetricCompatibilityScorer, load_torch_checkpoint, resolve_device
except ImportError:
    from experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from metric_lsej_data import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS
    from metric_lsej_evaluator import evaluate_lsej_split
    from metric_scorer import MetricCompatibilityScorer, load_torch_checkpoint, resolve_device


SCRIPT_DIR = Path(__file__).resolve().parent
LOSS_TYPES = (
    "hard_triplet",
    "semi_hard_triplet",
    "random_triplet",
    "distance_weighted_triplet",
    "infonce",
)
DEFAULT_LOSS_TYPE = "hard_triplet"
DEFAULT_LOG_ROOT = script_output_root(SCRIPT_DIR / "s1a_train_lsej.py", "logs")
DEFAULT_LOG_ROOTS = {loss_type: DEFAULT_LOG_ROOT for loss_type in LOSS_TYPES}
LEGACY_LOG_ROOTS = {
    "hard_triplet": SCRIPT_DIR / "logs" / "puzzle_hard_triplet_lsej",
    "semi_hard_triplet": SCRIPT_DIR / "logs" / "puzzle_semi_hard_triplet_lsej",
    "random_triplet": SCRIPT_DIR / "logs" / "puzzle_random_triplet_lsej",
    "distance_weighted_triplet": SCRIPT_DIR / "logs" / "puzzle_distance_weighted_triplet_lsej",
    "infonce": SCRIPT_DIR / "logs" / "puzzle_infonce_lsej",
}
DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET_NAME = "ImageNet_LSEJ"
DEFAULT_TASK = "grid10_erode2"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0
if DEFAULT_LOSS_TYPE == "hard_triplet":
    if DEFAULT_TASK == "grid10_erode2":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-07-25-10-22-41"
    elif DEFAULT_TASK == "grid10_erode5":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-08-02-21-16-37"
    elif DEFAULT_TASK == "grid10_erode8":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-08-02-21-17-06"
elif DEFAULT_LOSS_TYPE == "semi_hard_triplet":
    # 留空时自动选择当前 task 最新的 semi-hard 实验；训练完成后也可改成精确 run name。
    DEFAULT_RUN_NAME = "grid10_erode2_vit_tiny_patch16_224_pretrained_puzzle_semi_hard_triplet_d128_s224_kall_2026-07-18-15-51-36"
elif DEFAULT_LOSS_TYPE == "random_triplet":
    DEFAULT_RUN_NAME = "grid10_erode2_vit_tiny_patch16_224_pretrained_puzzle_random_triplet_d128_s224_kall_2026-07-18-22-33-41"
elif DEFAULT_LOSS_TYPE == "distance_weighted_triplet":
    DEFAULT_RUN_NAME = "grid10_erode2_vit_tiny_patch16_224_pretrained_puzzle_distance_weighted_triplet_c0.5_n1.4_d128_s224_kall_2026-07-18-22-34-25"
elif DEFAULT_LOSS_TYPE == "infonce":
    DEFAULT_RUN_NAME = "grid10_erode2_vit_tiny_patch16_224_pretrained_puzzle_infonce_tau0.1_d128_s224_kall_2026-07-18-11-34-05"
else:
    raise ValueError(f"DEFAULT_LOSS_TYPE must be one of {LOSS_TYPES}, got {DEFAULT_LOSS_TYPE!r}")
DEFAULT_RUN_DIR = None
DEFAULT_CHECKPOINT_NAME = "best.pth"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("请输入 True 或 False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "在 ImageNet-LSEJ 上使用 Gallagher 求解器评估 s1a 度量兼容性",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK, choices=OFFICIAL_TASKS, help="要评估的官方 LSEJ 任务")
    parser.add_argument("--loss-type", default=DEFAULT_LOSS_TYPE, choices=LOSS_TYPES, help="选择默认 checkpoint 和结果目录对应的训练目标")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["train", "val", "test"], help="要评估的数据划分")
    parser.add_argument("--data-root", default=DEFAULT_LSEJ_DATA_ROOT, type=Path, help="ImageNet-LSEJ 数据集根目录")
    parser.add_argument("--checkpoint", default=None, type=Path, help="精确 checkpoint 路径；设置后优先级最高")
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="task 日志目录下的实验文件夹名；留空时只在当前 task 中选择最新实验",
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="直接指定包含 checkpoints 子目录的实验目录")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="从实验 checkpoints 目录中读取的文件名")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="最多评估多少张 puzzle；0 表示完整 split")
    parser.add_argument("--start-index", default=0, type=int, help="从 split 中第几张 puzzle 开始评估")
    parser.add_argument("--score-batch-size", default=128, type=int, help="分批提取边缘 embedding 时每个前向批次的图片数")
    parser.add_argument("--postprocess", default=True, type=parse_bool, help="是否对兼容性矩阵做逐行归一化和双向对称化")
    parser.add_argument("--device", default="auto", help="运行设备，例如 auto、cpu、cuda:0；auto 会优先使用 GPU")
    parser.add_argument("--gpu-id", default=0, type=int, help="device=auto 时使用的 GPU 编号")
    parser.add_argument("--progress-interval", default=1, type=int, help="每评估多少张 puzzle 打印一次 PA/AA/SRA；0 表示关闭")
    parser.add_argument("--verbose-solver", action="store_true", help="显示 Gallagher 求解器内部输出；默认静默")
    parser.add_argument("--output-dir", default=None, type=Path, help="评估结果目录；留空时将实验、checkpoint、task 和 split 合并为单个文件夹名")
    parser.add_argument("--save-visuals", default=True, type=parse_bool, help="是否保存带正确/错误边框的拼图可视化")
    parser.add_argument("--visual-limit", default=10, type=int, help="最多保存多少张可视化图片")
    parser.add_argument("--save-predictions", default=True, type=parse_bool, help="是否保存 predictions.npz 供后续重算指标")
    return parser.parse_args()


def checkpoint_from_run_dir(run_dir: Path, checkpoint_name: str) -> Path:
    checkpoint = Path(run_dir) / "checkpoints" / checkpoint_name
    if checkpoint.is_file():
        return checkpoint
    available = sorted(path.name for path in (Path(run_dir) / "checkpoints").glob("*.pth"))
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Available checkpoint files: {available}")


def resolve_default_checkpoint(
    task: str,
    checkpoint_name: str,
    loss_type: str,
    log_root: Path | None = None,
) -> Path:
    if log_root is None:
        log_root = DEFAULT_LOG_ROOTS[loss_type]
    log_roots = [Path(log_root)]
    if Path(log_root) == DEFAULT_LOG_ROOTS[loss_type]:
        log_roots.append(LEGACY_LOG_ROOTS[loss_type])
    search_roots = [root / task for root in log_roots]
    candidates = sorted(
        (path for root in search_roots for path in root.glob(f"*/checkpoints/{checkpoint_name}")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            (path for root in search_roots for path in root.glob("*/checkpoints/last.pth")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No s1a LSEJ {loss_type} checkpoint found for task {task}. Searched under {search_roots}. "
        "Train this task first or pass --checkpoint explicitly."
    )


def resolve_checkpoint_arg(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint).resolve()
    if args.run_dir is not None:
        return checkpoint_from_run_dir(Path(args.run_dir), args.checkpoint_name).resolve()
    run_name = (args.run_name or "").strip()
    log_root = DEFAULT_LOG_ROOTS[args.loss_type]
    if run_name:
        run_dirs = [
            log_root / args.task / run_name,
            LEGACY_LOG_ROOTS[args.loss_type] / args.task / run_name,
        ]
        for run_dir in run_dirs:
            checkpoint = run_dir / "checkpoints" / args.checkpoint_name
            if checkpoint.is_file():
                return checkpoint.resolve()
        searched = [run_dir / "checkpoints" / args.checkpoint_name for run_dir in run_dirs]
        raise FileNotFoundError(f"Checkpoint not found. Searched: {searched}")
    return resolve_default_checkpoint(args.task, args.checkpoint_name, args.loss_type, log_root).resolve()


def default_output_dir(checkpoint: Path, task: str, split: str) -> Path:
    checkpoint = Path(checkpoint)
    folder_name = f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{task}_{split}"
    return DEFAULT_OUTPUT_ROOT / DEFAULT_DATASET_NAME / folder_name


def checkpoint_loss_type(checkpoint: dict) -> str:
    loss_config = checkpoint.get("loss_config", {})
    loss_type = loss_config.get("loss_type")
    if loss_type in LOSS_TYPES:
        return str(loss_type)
    loss_name = str(loss_config.get("loss", "puzzle_level_sampled_hard_triplet")).lower()
    if "distance_weighted" in loss_name:
        return "distance_weighted_triplet"
    if "random_triplet" in loss_name:
        return "random_triplet"
    if "semi_hard" in loss_name:
        return "semi_hard_triplet"
    return "infonce" if "infonce" in loss_name else "hard_triplet"


def validate_checkpoint(checkpoint_path: Path, task: str, loss_type: str) -> None:
    checkpoint = load_torch_checkpoint(checkpoint_path, resolve_device("cpu", 0))
    checkpoint_task = checkpoint.get("task") or checkpoint.get("args", {}).get("task")
    if checkpoint_task is not None and str(checkpoint_task) != task:
        raise ValueError(
            f"Checkpoint was trained for task {checkpoint_task}, but evaluation requested {task}. "
            "Cross-task checkpoint guessing is disabled."
        )
    checkpoint_loss = checkpoint_loss_type(checkpoint)
    if checkpoint_loss != loss_type:
        raise ValueError(
            f"Checkpoint uses loss_type={checkpoint_loss}, but evaluation requested {loss_type}. "
            "Pass the matching --loss-type or select another checkpoint."
        )


def main() -> None:
    args = parse_args()
    args.checkpoint = resolve_checkpoint_arg(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")
    validate_checkpoint(args.checkpoint, args.task, args.loss_type)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.checkpoint, args.task, args.split)
    args.output_dir = prepare_unique_output_dir(args.output_dir)

    device = resolve_device(args.device, args.gpu_id)
    scorer = MetricCompatibilityScorer.from_checkpoint(
        args.checkpoint,
        device=device,
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
    )
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
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
    )
    print(
        f"Result: PA={metrics.pa:.2%}, AA={metrics.aa:.2%}, "
        f"Horizontal SRA={metrics.horizontal_sra:.2%}, Vertical SRA={metrics.vertical_sra:.2%}, "
        f"SRA={metrics.sra:.2%}, time={metrics.elapsed_seconds:.1f}s"
    )
    print(f"Saved: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
