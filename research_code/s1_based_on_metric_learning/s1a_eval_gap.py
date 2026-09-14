"""
@file: s1a_eval_gap.py
@description: 加载指定的 S1A GAP checkpoint，计算 RGBA piece 兼容性矩阵并用 Gallagher Solver
              重组 GAP-3/GAP-5，输出 PA、AA、HA、VA、SRA、NA、Recall@K、预测与可视化。
@author: Changxin Ye
@created: 2026-07-27
@version: 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from .metric_gap_data import DEFAULT_GAP_DATA_ROOT, GAP_CONFIGS, GAP_SPLITS
    from .metric_gap_evaluator import GAPMetrics, evaluate_gap_split, solve_with_gallagher
    from .metric_scorer import MetricCompatibilityScorer, load_metric_checkpoint, resolve_device
except ImportError:
    from experiment_naming import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from metric_gap_data import DEFAULT_GAP_DATA_ROOT, GAP_CONFIGS, GAP_SPLITS
    from metric_gap_evaluator import GAPMetrics, evaluate_gap_split, solve_with_gallagher
    from metric_scorer import MetricCompatibilityScorer, load_metric_checkpoint, resolve_device


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_ROOT = script_output_root(SCRIPT_DIR / "s1a_train_gap.py", "logs")
DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "GAP-5"  # "GAP-3" or "GAP-5"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0  # 0 means the full split
DEFAULT_SAVE_VISUALS = True
DEFAULT_VISUAL_LIMIT = 10
DEFAULT_RUN_NAMES = {
    "GAP-3": "hard_triplet_d128_s224_kall_2026-07-31-20-21-04",
    "GAP-5": "hard_triplet_d128_s224_k15_2026-07-31-20-43-26",
}
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Evaluate an S1A metric checkpoint on GAP-3/GAP-5 with Gallagher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=GAP_CONFIGS.keys())
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=GAP_SPLITS)
    parser.add_argument("--data-root", default=DEFAULT_GAP_DATA_ROOT, type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path, help="Exact checkpoint path; highest priority")
    parser.add_argument(
        "--run-name",
        default=None,
        help="logs/s1a_train_gap/<dataset> 下的实验名；留空时按 dataset 使用 DEFAULT_RUN_NAMES",
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int)
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--progress-interval", default=25, type=int)
    parser.add_argument("--save-visuals", default=DEFAULT_SAVE_VISUALS, type=parse_bool)
    parser.add_argument("--visual-limit", default=DEFAULT_VISUAL_LIMIT, type=int)
    parser.add_argument("--visual-interval", default=1, type=int)
    parser.add_argument("--save-predictions", default=True, type=parse_bool)
    parser.add_argument("--postprocess", default=True, type=parse_bool)
    parser.add_argument("--score-batch-size", default=128, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--verbose-solver", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def checkpoint_from_run_dir(run_dir: Path, checkpoint_name: str) -> Path:
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if checkpoint.is_file():
        return checkpoint.resolve()
    available = sorted(path.name for path in (run_dir / "checkpoints").glob("*.pth")) \
        if (run_dir / "checkpoints").is_dir() else []
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Available: {available}")


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint
    if args.run_dir is not None:
        return checkpoint_from_run_dir(Path(args.run_dir).expanduser().resolve(), args.checkpoint_name)
    dataset_root = Path(args.log_root).expanduser().resolve() / args.dataset
    if args.run_name.strip():
        return checkpoint_from_run_dir(dataset_root / args.run_name.strip(), args.checkpoint_name)
    candidates = sorted(
        dataset_root.glob(f"*/checkpoints/{args.checkpoint_name}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        available = sorted(path.name for path in dataset_root.iterdir() if path.is_dir()) \
            if dataset_root.is_dir() else []
        raise FileNotFoundError(
            f"No {args.checkpoint_name} found under {dataset_root}. "
            f"Train with s1a_train_gap.py or set DEFAULT_RUN_NAMES. Available runs: {available}"
        )
    print(f"DEFAULT_RUN_NAME is empty; using latest checkpoint: {candidates[0]}")
    return candidates[0].resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 0 or args.start_index < 0:
        raise ValueError("max_samples and start_index must be non-negative")
    if args.score_batch_size <= 0 or args.visual_limit < 0 or args.visual_interval <= 0:
        raise ValueError("invalid score/visual settings")


def run_evaluation(
    args: argparse.Namespace,
    *,
    solver=solve_with_gallagher,
    solver_name: str = "Gallagher",
    method_name: str = "S1A",
    solver_configuration: dict | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output_suffix: str = "",
) -> tuple[GAPMetrics, Path]:
    """Run a standalone GAP baseline through the same scorer/solver path as S1A4."""

    validate_args(args)
    if args.run_name is None:
        args.run_name = DEFAULT_RUN_NAMES[args.dataset]
    device = resolve_device(args.device, args.gpu_id)
    checkpoint = resolve_checkpoint(args)
    model, model_config, checkpoint_payload = load_metric_checkpoint(checkpoint, device)
    checkpoint_dataset = checkpoint_payload.get("dataset") or checkpoint_payload.get("args", {}).get("dataset")
    if checkpoint_dataset not in {None, args.dataset}:
        raise ValueError(f"Checkpoint dataset {checkpoint_dataset} does not match requested {args.dataset}")
    if int(checkpoint_payload.get("model_config", {}).get("input_channels", 3)) != 4:
        raise ValueError("Selected checkpoint is not an RGBA GAP metric checkpoint")
    scorer = MetricCompatibilityScorer(
        model=model,
        device=device,
        input_size=model_config["input_size"],
        normalization=model_config["normalization"],
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
        score_metric=model_config["score_metric"],
        canonical_edge=model_config["canonical_edge"],
        geometry_mode=model_config["geometry_mode"],
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else Path(output_root)
        / args.dataset
        / (
            f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{args.split}"
            f"{('_' + output_suffix) if output_suffix else ''}"
        )
    )
    output_dir = prepare_unique_output_dir(output_dir)
    configuration = {
        "solver": solver_name,
        "postprocess": args.postprocess,
        "score_batch_size": args.score_batch_size,
        "score_metric": model_config["score_metric"],
        "normalization": model_config["normalization"],
        "geometry_mode": model_config["geometry_mode"],
        "input_channels": 4,
        "rgba_adapter": "learned_conv1x1_4_to_3",
    }
    if solver_configuration is not None:
        configuration["initial_solver"] = solver_configuration
    print(f"dataset/split: {args.dataset} / {args.split}")
    print(f"data_root: {Path(args.data_root).expanduser().resolve()}")
    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    print(f"output: {output_dir}")
    metrics = evaluate_gap_split(
        scorer=scorer,
        solver=solver,
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        output_dir=output_dir,
        checkpoint=checkpoint,
        max_samples=args.max_samples,
        start_index=args.start_index,
        progress_interval=args.progress_interval,
        verbose_solver=args.verbose_solver,
        save_visuals=args.save_visuals,
        visual_limit=args.visual_limit,
        visual_interval=args.visual_interval,
        save_predictions=args.save_predictions,
        method_name=method_name,
        configuration=configuration,
    )
    return metrics, output_dir


def print_metrics(metrics: GAPMetrics, output_dir: Path) -> None:
    print("\nGAP evaluation")
    print("=" * 72)
    print(f"PA:  {metrics.pa:.4%}")
    print(f"AA:  {metrics.aa:.4%}")
    print(f"HA:  {metrics.ha:.4%}")
    print(f"VA:  {metrics.va:.4%}")
    print(f"SRA: {metrics.sra:.4%}")
    print(f"NA:  {metrics.na:.4%} (PuzzleFlow neighbor_accuracy)")
    for k in metrics.neighbor_retrieval.recall_ks:
        print(f"Recall@{k}: {metrics.neighbor_retrieval.recall_at(k):.4%}")
    print(f"Summary: {output_dir / 'summary.txt'}")


def main() -> None:
    metrics, output_dir = run_evaluation(parse_args())
    print_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()
