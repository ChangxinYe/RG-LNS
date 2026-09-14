"""
@file: s1a_train_jpleg.py
@description: 基于 handwritten ViT 和同图合法 hard negative mining 的 JPLEG 训练脚本。
@author: Changxin Ye
@created: 2026-07-10
@version: 1.1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .experiment_naming import build_run_name, script_output_root
    from .metric_jpleg_data import DEFAULT_DATA_ROOT, JPLEG_CONFIGS, JPLEGPuzzleHardTripletDataset
    from .metric_logger import append_epoch_log, write_metric_outputs, write_summary, write_training_args
    from .metric_losses import puzzle_level_hard_triplet_loss
    from .metric_scorer import MetricCompatibilityScorer, load_torch_checkpoint, resolve_device
    from .metric_vit_model import ViTMetricEncoder, model_config_dict
except ImportError:
    from experiment_naming import build_run_name, script_output_root
    from metric_jpleg_data import DEFAULT_DATA_ROOT, JPLEG_CONFIGS, JPLEGPuzzleHardTripletDataset
    from metric_logger import append_epoch_log, write_metric_outputs, write_summary, write_training_args
    from metric_losses import puzzle_level_hard_triplet_loss
    from metric_scorer import MetricCompatibilityScorer, load_torch_checkpoint, resolve_device
    from metric_vit_model import ViTMetricEncoder, model_config_dict


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_LOG_ROOT = script_output_root(__file__, "logs")
EDGE2VEC_DIR = PROJECT_ROOT / "baselines" / "Edge2Vec_arxiv_2022"
if str(EDGE2VEC_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE2VEC_DIR))

from edge2vec_assembly_eval import evaluate_assembly_split  # noqa: E402


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected True or False")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train a puzzle-level hard triplet metric-learning baseline on JPLEG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="jpleg3", choices=JPLEG_CONFIGS.keys())
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_LOG_ROOT, type=Path)
    parser.add_argument("--run-name", default="")

    parser.add_argument("--backbone", default="vit-t", type=str, help="handwritten ViT backbone, e.g. vit-t, vit-s, vit-b")
    parser.add_argument("--pretrained-backbone", "--pretrained_backbone", default=True, type=parse_bool)
    parser.add_argument("--pretrained-weights-path", "--pretrained_weights_path", default=None, type=Path)
    parser.add_argument("--input-size", "--input_size", default=224, type=int)
    parser.add_argument("--embedding-dim", "--embedding_dim", default=128, type=int)
    parser.add_argument("--normalization", default="vit", choices=["zero_one", "vit", "imagenet", "fragment"])
    parser.add_argument("--score-metric", default="euclidean", choices=["cosine", "euclidean"])

    parser.add_argument("--triplets-per-puzzle", default=8, type=int)
    parser.add_argument("--permute-pieces", default=True, type=parse_bool)
    parser.add_argument("--margin", default=0.2, type=float)
    parser.add_argument("--loss-distance", default="euclidean", choices=["euclidean", "cosine"])

    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--steps-per-epoch", default=0, type=int, help="0 means use the full DataLoader")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Legacy global learning-rate override; normally use --backbone-lr and --head-lr",
    )
    parser.add_argument("--backbone-lr", default=1e-5, type=float)
    parser.add_argument("--head-lr", default=1e-4, type=float)
    parser.add_argument("--warmup-epochs", default=5, type=int)
    parser.add_argument("--lr-factor", default=0.5, type=float)
    parser.add_argument("--lr-patience", default=3, type=int)
    parser.add_argument("--grad-clip-norm", default=1.0, type=float, help="0 disables gradient clipping")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-train-samples", default=0, type=int)
    parser.add_argument("--max-val-samples", default=0, type=int)
    parser.add_argument("--max-test-samples", default=0, type=int)

    parser.add_argument("--assembly-eval", default=True, type=parse_bool)
    parser.add_argument("--assembly-eval-max-train-samples", default=0, type=int, help="0 means full train split")
    parser.add_argument("--assembly-eval-max-val-samples", default=0, type=int, help="0 means full valid split")
    parser.add_argument("--assembly-eval-max-test-samples", default=0, type=int, help="0 means full test split")
    parser.add_argument("--assembly-eval-start-index", default=0, type=int)
    parser.add_argument("--assembly-eval-progress-interval", default=0, type=int)
    parser.add_argument("--assembly-eval-score-batch-size", default=128, type=int)
    parser.add_argument("--assembly-eval-postprocess", default=True, type=parse_bool)
    parser.add_argument("--assembly-eval-verbose-solver", action="store_true")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print-freq", default=25, type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name or build_run_name(
        "hard_triplet",
        f"d{args.embedding_dim}",
        f"s{args.input_size}",
    )
    output_dir = Path(args.output_dir) / args.dataset / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def checkpoint_args_dict(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["anchor"].to(device, non_blocking=True),
        batch["positive"].to(device, non_blocking=True),
        batch["negatives"].to(device, non_blocking=True),
    )


def average_stats(totals: dict[str, float], steps: int) -> dict[str, float]:
    return {key: value / max(steps, 1) for key, value in totals.items()}


METRIC_KEYS = [
    "loss",
    "triplet_loss",
    "triplet_acc",
    "pos_dist",
    "hard_neg_dist",
    "margin_gap",
    "negatives_per_anchor",
    "hard_neg_index",
    "embedding_std",
    "embedding_norm",
]


def resolve_learning_rates(args: argparse.Namespace) -> None:
    if args.lr is not None:
        args.backbone_lr = args.lr
        args.head_lr = args.lr
    if args.backbone_lr <= 0 or args.head_lr <= 0:
        raise ValueError("backbone_lr and head_lr must be positive")
    if args.pretrained_backbone and args.backbone_lr > 1e-4:
        raise ValueError(
            f"backbone_lr={args.backbone_lr:g} is unsafe for a pretrained ViT. "
            "Remove the legacy --lr argument and use the defaults, or set --backbone-lr <= 1e-4 explicitly."
        )
    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")


def make_optimizer(model: ViTMetricEncoder, args: argparse.Namespace) -> Adam:
    return Adam(
        [
            {
                "params": model.vit.parameters(),
                "lr": args.backbone_lr,
                "target_lr": args.backbone_lr,
                "name": "backbone",
            },
            {
                "params": model.project.parameters(),
                "lr": args.head_lr,
                "target_lr": args.head_lr,
                "name": "projection_head",
            },
        ],
    )


def apply_linear_warmup(optimizer: torch.optim.Optimizer, epoch: int, warmup_epochs: int) -> None:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    scale = epoch / warmup_epochs
    for group in optimizer.param_groups:
        group["lr"] = float(group["target_lr"]) * scale


def learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {str(group.get("name", index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}


def embed_triplet_batch(
    model: torch.nn.Module,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, negative_count = negatives.shape[:2]
    negative_flat = negatives.flatten(0, 1)
    inputs = torch.cat([anchor, positive, negative_flat], dim=0)
    embeddings = model(inputs)
    emb_anchor = embeddings[:batch_size]
    emb_positive = embeddings[batch_size : batch_size * 2]
    emb_negatives = embeddings[batch_size * 2 :].reshape(batch_size, negative_count, -1)
    return emb_anchor, emb_positive, emb_negatives


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals = {key: 0.0 for key in METRIC_KEYS}
    steps = 0
    iterator = tqdm(loader, total=args.steps_per_epoch or len(loader), leave=False)
    for batch in iterator:
        anchor, positive, negatives = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        emb_anchor, emb_positive, emb_negatives = embed_triplet_batch(model, anchor, positive, negatives)
        loss, stats = puzzle_level_hard_triplet_loss(
            emb_anchor,
            emb_positive,
            emb_negatives,
            margin=args.margin,
            distance=args.loss_distance,
        )
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        steps += 1
        for key in totals:
            totals[key] += stats[key]
        avg = average_stats(totals, steps)
        iterator.set_description(f"epoch {epoch:03d} loss={avg['loss']:.4f} acc={avg['triplet_acc']:.3f}")
        if args.print_freq > 0 and steps % args.print_freq == 0:
            print(
                f"epoch {epoch:03d} train step {steps:04d}: "
                f"loss={avg['loss']:.4f}, triplet_acc={avg['triplet_acc']:.3f}, "
                f"pos={avg['pos_dist']:.4f}, hard_neg={avg['hard_neg_dist']:.4f}, gap={avg['margin_gap']:.4f}"
            )
        if args.steps_per_epoch > 0 and steps >= args.steps_per_epoch:
            break
    return average_stats(totals, steps)


@torch.no_grad()
def evaluate_triplet_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in METRIC_KEYS}
    steps = 0
    for batch in tqdm(loader, total=len(loader), leave=False):
        anchor, positive, negatives = move_batch(batch, device)
        emb_anchor, emb_positive, emb_negatives = embed_triplet_batch(model, anchor, positive, negatives)
        _loss, stats = puzzle_level_hard_triplet_loss(
            emb_anchor,
            emb_positive,
            emb_negatives,
            margin=args.margin,
            distance=args.loss_distance,
        )
        steps += 1
        for key in totals:
            totals[key] += stats[key]
    return average_stats(totals, steps)


def evaluate_assembly_epoch(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> dict[str, dict]:
    if not args.assembly_eval:
        return {}
    was_training = model.training
    model.eval()
    scorer = MetricCompatibilityScorer(
        model=model,
        device=device,
        input_size=args.input_size,
        normalization=args.normalization,
        batch_size=args.assembly_eval_score_batch_size,
        postprocess=args.assembly_eval_postprocess,
        score_metric=args.score_metric,
    )
    max_by_split = {
        "train": args.assembly_eval_max_train_samples or None,
        "valid": args.assembly_eval_max_val_samples or None,
        "test": args.assembly_eval_max_test_samples or None,
    }
    out = {}
    for split_name in ("train", "valid", "test"):
        metrics = evaluate_assembly_split(
            dataset=args.dataset,
            split=split_name,
            data_root=args.data_root,
            scorer=scorer,
            max_samples=max_by_split[split_name],
            start_index=args.assembly_eval_start_index,
            progress_interval=args.assembly_eval_progress_interval,
            verbose_solver=args.assembly_eval_verbose_solver,
        )
        out[split_name] = metrics.to_dict()
    if was_training:
        model.train()
    return out


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    args: argparse.Namespace,
    epoch: int,
    history: list[dict],
    best_val_loss: float,
    best_val_acc: float,
    best_val_assembly_aa: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_val_assembly_aa": best_val_assembly_aa,
            "model_config": {
                **model_config_dict(model),
                "normalization": args.normalization,
                "score_metric": args.score_metric,
                "canonical_edge": "right",
                "geometry_mode": "right_edge_role_independent",
            },
            "loss_config": {
                "loss": "puzzle_level_hard_triplet",
                "margin": args.margin,
                "loss_distance": args.loss_distance,
                "triplets_per_puzzle": args.triplets_per_puzzle,
            },
            "optimization_config": {
                "optimizer": "Adam",
                "parameter_groups": ["backbone", "projection_head"],
                "backbone_lr": args.backbone_lr,
                "head_lr": args.head_lr,
                "warmup_epochs": args.warmup_epochs,
                "lr_factor": args.lr_factor,
                "lr_patience": args.lr_patience,
                "grad_clip_norm": args.grad_clip_norm,
            },
            "normalization": args.normalization,
            "score_metric": args.score_metric,
            "args": checkpoint_args_dict(args),
        },
        path,
    )
    print(f"saved checkpoint: {path}")


def load_resume(
    resume_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, list[dict], float, float, float]:
    if not resume_path:
        return 1, [], float("inf"), -float("inf"), -float("inf")
    checkpoint = load_torch_checkpoint(resume_path, device)
    geometry_mode = checkpoint.get("model_config", {}).get("geometry_mode")
    if geometry_mode != "right_edge_role_independent":
        raise ValueError(
            "This checkpoint uses a legacy role-aware edge canonicalization and cannot be resumed "
            "with the current four-embedding role-independent pipeline. Start a new run instead."
        )
    embedding_normalization = checkpoint.get("model_config", {}).get("embedding_normalization", "none")
    if embedding_normalization != "l2":
        raise ValueError(
            "This checkpoint was trained without unit-normalized hyperspherical embeddings and cannot be resumed "
            "with the current L2-normalized encoder. Start a new run instead."
        )
    checkpoint_group_count = len(checkpoint.get("optimizer", {}).get("param_groups", []))
    if checkpoint_group_count != len(optimizer.param_groups):
        raise ValueError(
            "This checkpoint predates the separate backbone/projection-head learning rates and cannot be resumed "
            "safely. Start a new run with the current anti-collapse optimizer settings instead."
        )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    history = list(checkpoint.get("history", []))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_val_acc = float(checkpoint.get("best_val_acc", -float("inf")))
    best_val_assembly_aa = float(checkpoint.get("best_val_assembly_aa", -float("inf")))
    print(f"resumed from {resume_path} at epoch {start_epoch}")
    return start_epoch, history, best_val_loss, best_val_acc, best_val_assembly_aa


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def make_dataset(args: argparse.Namespace, split: str, max_samples: int | None):
    return JPLEGPuzzleHardTripletDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        split=split,
        input_size=args.input_size,
        normalization=args.normalization,
        triplets_per_puzzle=args.triplets_per_puzzle,
        permute_pieces=args.permute_pieces and split == "train",
        max_samples=max_samples,
    )


def main() -> None:
    args = parse_args()
    resolve_learning_rates(args)
    seed_everything(args.seed)
    device = resolve_device(args.device, args.gpu_id)
    output_dir = prepare_output_dir(args)
    args.output_dir = output_dir
    print(f"device: {device}")
    print(f"output_dir: {output_dir}")

    train_dataset = make_dataset(args, "train", args.max_train_samples or None)
    train_eval_dataset = make_dataset(args, "train", args.max_train_samples or None)
    val_dataset = make_dataset(args, "valid", args.max_val_samples or None)
    test_dataset = make_dataset(args, "test", args.max_test_samples or None)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, pin_memory)
    train_eval_loader = make_loader(train_eval_dataset, args.batch_size, False, args.num_workers, pin_memory)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, pin_memory)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, pin_memory)

    model = ViTMetricEncoder(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        image_size=args.input_size,
        pretrained=args.pretrained_backbone,
        pretrained_weights_path=args.pretrained_weights_path,
    ).to(device)
    optimizer = make_optimizer(model, args)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )
    start_epoch, history, best_val_loss, best_val_acc, best_val_assembly_aa = load_resume(
        args.resume, model, optimizer, scheduler, device
    )

    (output_dir / "training_args.json").write_text(
        json.dumps({key: str(value) for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )
    write_training_args(args, output_dir / "training_args.txt")
    train_log_path = output_dir / "train_log.txt"
    if not args.resume:
        train_log_path.write_text("", encoding="utf-8")
    print(
        f"train triplets: {len(train_dataset)}, valid triplets: {len(val_dataset)}, test triplets: {len(test_dataset)}, "
        f"negatives per anchor: {train_dataset.negatives_per_anchor}"
    )
    print(
        f"optimizer: backbone_lr={args.backbone_lr:g}, head_lr={args.head_lr:g}, "
        f"warmup_epochs={args.warmup_epochs}, margin={args.margin:g}, grad_clip_norm={args.grad_clip_norm:g}"
    )

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        apply_linear_warmup(optimizer, epoch, args.warmup_epochs)
        epoch_lrs = learning_rates(optimizer)
        train_optimization_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
        )
        train_stats = evaluate_triplet_loss(model, train_eval_loader, device, args)
        val_stats = evaluate_triplet_loss(model, val_loader, device, args)
        test_stats = evaluate_triplet_loss(model, test_loader, device, args)
        assembly_stats = evaluate_assembly_epoch(args=args, model=model, device=device)
        if epoch > args.warmup_epochs:
            scheduler.step(val_stats["loss"])
        record = {
            "epoch": epoch,
            "learning_rate": epoch_lrs["projection_head"],
            "backbone_learning_rate": epoch_lrs["backbone"],
            "head_learning_rate": epoch_lrs["projection_head"],
            "train_optimization": train_optimization_stats,
            "train": train_stats,
            "valid": val_stats,
            "test": test_stats,
            "assembly": assembly_stats,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        epoch_line = (
            f"epoch {epoch:03d}: "
            f"opt_loss={train_optimization_stats['loss']:.4f}, opt_acc={train_optimization_stats['triplet_acc']:.3f}, "
            f"train_loss={train_stats['loss']:.4f}, train_acc={train_stats['triplet_acc']:.3f}, "
            f"valid_loss={val_stats['loss']:.4f}, valid_acc={val_stats['triplet_acc']:.3f}, "
            f"valid_emb_std={val_stats['embedding_std']:.3e}"
        )
        if test_stats:
            epoch_line += f", test_loss={test_stats['loss']:.4f}, test_acc={test_stats['triplet_acc']:.3f}"
        if assembly_stats:
            epoch_line += (
                f", train_AA={assembly_stats['train']['aa']:.3f}, train_PA={assembly_stats['train']['pa']:.3f}, "
                f"valid_AA={assembly_stats['valid']['aa']:.3f}, valid_PA={assembly_stats['valid']['pa']:.3f}, "
                f"test_AA={assembly_stats['test']['aa']:.3f}, test_PA={assembly_stats['test']['pa']:.3f}"
            )
        print(epoch_line)

        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        append_epoch_log(record, train_log_path)
        write_metric_outputs(history, output_dir)
        save_checkpoint(
            output_dir / "checkpoints" / "last.pth",
            model,
            optimizer,
            scheduler,
            args,
            epoch,
            history,
            best_val_loss,
            best_val_acc,
            best_val_assembly_aa,
        )
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            save_checkpoint(
                output_dir / "checkpoints" / "best_loss.pth",
                model,
                optimizer,
                scheduler,
                args,
                epoch,
                history,
                best_val_loss,
                best_val_acc,
                best_val_assembly_aa,
            )
        if val_stats["triplet_acc"] > best_val_acc:
            best_val_acc = val_stats["triplet_acc"]
            checkpoint_names = ["best_acc.pth"]
            if not args.assembly_eval:
                checkpoint_names.append("best.pth")
            for name in checkpoint_names:
                save_checkpoint(
                    output_dir / "checkpoints" / name,
                    model,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    history,
                    best_val_loss,
                    best_val_acc,
                    best_val_assembly_aa,
                )
        valid_assembly_aa = assembly_stats.get("valid", {}).get("aa") if isinstance(assembly_stats, dict) else None
        if valid_assembly_aa is not None and valid_assembly_aa > best_val_assembly_aa:
            best_val_assembly_aa = valid_assembly_aa
            for name in ("best_assembly_aa.pth", "best.pth"):
                save_checkpoint(
                    output_dir / "checkpoints" / name,
                    model,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    history,
                    best_val_loss,
                    best_val_acc,
                    best_val_assembly_aa,
                )
    total_time = time.perf_counter() - started
    write_summary(args=args, history=history, total_time=total_time, path=output_dir / "train_summary.txt")
    print(f"Training finished in {total_time:.1f}s. Summary: {output_dir / 'train_summary.txt'}")


if __name__ == "__main__":
    main()
