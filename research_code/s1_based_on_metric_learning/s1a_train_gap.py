"""
@file: s1a_train_gap.py
@description: 在 GAP-3/GAP-5 不规则 RGBA 拼图上训练 S1A piece-to-piece 度量模型。
              模型保留预训练 handwritten ViT、右边缘规范化、同图困难负样本和 Hard Triplet；
              RGBA 先经过可学习的 1×1 通道适配器。训练仅使用 train/val，固定跑满 epoch，
              test 仅由独立评估脚本使用。
@author: Changxin Ye
@created: 2026-07-27
@version: 1.0
"""

from __future__ import annotations

import argparse
import json
import random
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
    from .metric_gap_data import DEFAULT_GAP_DATA_ROOT, GAP_CONFIGS, MetricGAPTripletDataset
    from .metric_logger import append_epoch_log, write_metric_outputs, write_summary, write_training_args
    from .metric_losses import puzzle_level_hard_triplet_loss
    from .metric_scorer import load_torch_checkpoint, resolve_device
    from .metric_vit_model import ViTMetricEncoder, model_config_dict
except ImportError:
    from experiment_naming import build_run_name, script_output_root
    from metric_gap_data import DEFAULT_GAP_DATA_ROOT, GAP_CONFIGS, MetricGAPTripletDataset
    from metric_logger import append_epoch_log, write_metric_outputs, write_summary, write_training_args
    from metric_losses import puzzle_level_hard_triplet_loss
    from metric_scorer import load_torch_checkpoint, resolve_device
    from metric_vit_model import ViTMetricEncoder, model_config_dict


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = "GAP-3"  # "GAP-3" or "GAP-5"
if DEFAULT_DATASET == "GAP-3":
    DEFAULT_NEGATIVES_PER_ANCHOR = 0  # 0 means all 7 legal negatives
    DEFAULT_BATCH_SIZE = 16
elif DEFAULT_DATASET == "GAP-5":
    DEFAULT_NEGATIVES_PER_ANCHOR = 15
    DEFAULT_BATCH_SIZE = 8
else:
    raise ValueError(f"Unsupported DEFAULT_DATASET: {DEFAULT_DATASET}")

DEFAULT_LOG_ROOT = script_output_root(__file__, "logs")
DEFAULT_LOSS_TYPE = "hard_triplet"
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
        "Train S1A hard-triplet metric learning on GAP-3/GAP-5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=GAP_CONFIGS.keys())
    parser.add_argument("--data-root", default=DEFAULT_GAP_DATA_ROOT, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_LOG_ROOT, type=Path)
    parser.add_argument("--run-name", default="")

    parser.add_argument("--backbone", default="vit-t", help="handwritten ViT: vit-t, vit-s or vit-b")
    parser.add_argument("--pretrained-backbone", default=True, type=parse_bool)
    parser.add_argument("--pretrained-weights-path", default=None, type=Path)
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--embedding-dim", default=128, type=int)
    parser.add_argument("--normalization", default="vit", choices=["zero_one", "vit", "imagenet", "fragment"])
    parser.add_argument("--score-metric", default="euclidean", choices=["cosine", "euclidean"])

    parser.add_argument("--triplets-per-puzzle", default=4, type=int)
    parser.add_argument("--val-triplets-per-puzzle", default=2, type=int)
    parser.add_argument("--negatives-per-anchor", default=DEFAULT_NEGATIVES_PER_ANCHOR, type=int)
    parser.add_argument("--permute-pieces", default=True, type=parse_bool)
    parser.add_argument("--margin", default=0.2, type=float)
    parser.add_argument("--loss-distance", default="euclidean", choices=["euclidean", "cosine"])

    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--steps-per-epoch", default=0, type=int, help="0 means full train loader")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--backbone-lr", default=1e-5, type=float)
    parser.add_argument("--head-lr", default=1e-4, type=float, help="projection head and RGBA adapter learning rate")
    parser.add_argument("--warmup-epochs", default=5, type=int)
    parser.add_argument("--lr-factor", default=0.5, type=float)
    parser.add_argument("--lr-patience", default=3, type=int)
    parser.add_argument("--grad-clip-norm", default=1.0, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--max-train-samples", default=0, type=int)
    parser.add_argument("--max-val-samples", default=0, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--print-freq", default=25, type=int)
    args = parser.parse_args()
    args.loss_type = DEFAULT_LOSS_TYPE
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.input_size <= 0 or args.input_size % 16:
        raise ValueError("input_size must be positive and divisible by 16")
    if args.embedding_dim <= 0 or args.triplets_per_puzzle <= 0 or args.val_triplets_per_puzzle <= 0:
        raise ValueError("embedding_dim and triplet counts must be positive")
    if args.negatives_per_anchor < 0:
        raise ValueError("negatives_per_anchor must be non-negative")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
    if args.steps_per_epoch < 0 or args.max_train_samples < 0 or args.max_val_samples < 0:
        raise ValueError("step/sample limits must be non-negative")
    if args.backbone_lr <= 0 or args.head_lr <= 0:
        raise ValueError("learning rates must be positive")
    if args.pretrained_backbone and args.backbone_lr > 1e-4:
        raise ValueError("backbone_lr above 1e-4 is unsafe for a pretrained ViT")
    if args.warmup_epochs < 0 or args.lr_patience < 0 or not 0 < args.lr_factor < 1:
        raise ValueError("invalid warmup or ReduceLROnPlateau settings")
    max_legal = GAP_CONFIGS[args.dataset].num_pieces - 2
    if args.negatives_per_anchor > max_legal:
        raise ValueError(
            f"{args.dataset} has only {max_legal} legal negatives per anchor; "
            "set negatives_per_anchor=0 for all candidates"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_output_dir(args: argparse.Namespace) -> Path:
    negative_tag = "kall" if args.negatives_per_anchor == 0 else f"k{args.negatives_per_anchor}"
    run_name = args.run_name or build_run_name(
        "hard_triplet",
        f"d{args.embedding_dim}",
        f"s{args.input_size}",
        negative_tag,
    )
    output_dir = Path(args.output_dir).expanduser().resolve() / args.dataset / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_dataset(
    args: argparse.Namespace,
    split: str,
    triplets_per_puzzle: int,
    permute_pieces: bool,
    max_samples: int | None,
) -> MetricGAPTripletDataset:
    return MetricGAPTripletDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        split=split,
        input_size=args.input_size,
        normalization=args.normalization,
        triplets_per_puzzle=triplets_per_puzzle,
        negatives_per_anchor=args.negatives_per_anchor,
        permute_pieces=permute_pieces,
        max_samples=max_samples,
        seed=args.seed,
    )


def make_loader(
    dataset: MetricGAPTripletDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def embed_triplet_batch(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor = batch["anchor"].flatten(0, 1).to(device, non_blocking=True)
    positive = batch["positive"].flatten(0, 1).to(device, non_blocking=True)
    negatives = batch["negatives"].flatten(0, 1).to(device, non_blocking=True)
    relation_count, negative_count = negatives.shape[:2]
    inputs = torch.cat([anchor, positive, negatives.flatten(0, 1)], dim=0)
    embeddings = model(inputs)
    return (
        embeddings[:relation_count],
        embeddings[relation_count : 2 * relation_count],
        embeddings[2 * relation_count :].reshape(relation_count, negative_count, -1),
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in METRIC_KEYS}
    steps = 0
    context = torch.enable_grad() if training else torch.no_grad()
    iterator = tqdm(loader, total=max_steps or len(loader), leave=False)
    with context:
        for batch in iterator:
            if training:
                optimizer.zero_grad(set_to_none=True)
            anchor, positive, negatives = embed_triplet_batch(model, batch, device)
            loss, stats = puzzle_level_hard_triplet_loss(
                anchor,
                positive,
                negatives,
                margin=args.margin,
                distance=args.loss_distance,
            )
            if training:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
            steps += 1
            for key in METRIC_KEYS:
                totals[key] += float(stats[key])
            averages = {key: value / steps for key, value in totals.items()}
            label = "train" if training else "valid"
            iterator.set_description(
                f"epoch {epoch:03d} {label} loss={averages['loss']:.4f} acc={averages['triplet_acc']:.3f}"
            )
            if training and args.print_freq > 0 and steps % args.print_freq == 0:
                print(
                    f"epoch {epoch:03d} train step {steps:04d}: "
                    f"loss={averages['loss']:.4f}, acc={averages['triplet_acc']:.3f}, "
                    f"pos={averages['pos_dist']:.4f}, hard_neg={averages['hard_neg_dist']:.4f}"
                )
            if max_steps > 0 and steps >= max_steps:
                break
    if steps == 0:
        raise RuntimeError("DataLoader produced no optimization/evaluation steps")
    return {key: value / steps for key, value in totals.items()}


def make_optimizer(model: ViTMetricEncoder, args: argparse.Namespace) -> Adam:
    head_parameters = list(model.project.parameters())
    if model.channel_adapter is not None:
        head_parameters.extend(model.channel_adapter.parameters())
    return Adam(
        [
            {
                "params": model.vit.parameters(),
                "lr": args.backbone_lr,
                "target_lr": args.backbone_lr,
                "name": "backbone",
            },
            {
                "params": head_parameters,
                "lr": args.head_lr,
                "target_lr": args.head_lr,
                "name": "projection_and_rgba_adapter",
            },
        ]
    )


def apply_linear_warmup(optimizer: torch.optim.Optimizer, epoch: int, warmup_epochs: int) -> None:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    scale = epoch / warmup_epochs
    for group in optimizer.param_groups:
        group["lr"] = float(group["target_lr"]) * scale


def checkpoint_args_dict(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def save_checkpoint(
    path: Path,
    model: ViTMetricEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    args: argparse.Namespace,
    epoch: int,
    history: list[dict],
    best_val_loss: float,
    best_val_score: float,
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
            "best_val_acc": best_val_score,
            "best_val_score": best_val_score,
            "selection_metric": "triplet_acc",
            "dataset": args.dataset,
            "benchmark": "GAP",
            "model_config": {
                **model_config_dict(model),
                "normalization": args.normalization,
                "score_metric": args.score_metric,
                "canonical_edge": "right",
                "geometry_mode": "right_edge_role_independent",
            },
            "loss_config": {
                "loss": "puzzle_level_sampled_hard_triplet",
                "loss_type": "hard_triplet",
                "margin": args.margin,
                "loss_distance": args.loss_distance,
                "triplets_per_puzzle": args.triplets_per_puzzle,
                "negatives_per_anchor": args.effective_negatives_per_anchor,
                "negatives_per_anchor_requested": args.negatives_per_anchor,
                "selection_metric": "triplet_acc",
            },
            "optimization_config": {
                "optimizer": "Adam",
                "parameter_groups": ["backbone", "projection_and_rgba_adapter"],
                "backbone_lr": args.backbone_lr,
                "head_lr": args.head_lr,
                "warmup_epochs": args.warmup_epochs,
                "lr_factor": args.lr_factor,
                "lr_patience": args.lr_patience,
                "grad_clip_norm": args.grad_clip_norm,
                "fixed_epoch_training": True,
                "early_stopping": False,
            },
            "data_config": {
                "source": "official_gap_hdf5",
                "dataset": args.dataset,
                "input_channels": 4,
                "rgba_adapter": "learned_conv1x1_4_to_3",
                "runtime_train_piece_permutation": args.permute_pieces,
                "test_used_during_training": False,
            },
            "normalization": args.normalization,
            "score_metric": args.score_metric,
            "args": checkpoint_args_dict(args),
        },
        path,
    )


def load_resume(
    path: str,
    model: ViTMetricEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, list[dict], float, float]:
    if not path:
        return 1, [], float("inf"), -float("inf")
    checkpoint = load_torch_checkpoint(path, device)
    if checkpoint.get("dataset") != args.dataset:
        raise ValueError(f"Cannot resume {checkpoint.get('dataset')} checkpoint as {args.dataset}")
    config = checkpoint.get("model_config", {})
    if int(config.get("input_channels", 3)) != 4:
        raise ValueError("Resume checkpoint is not an RGBA GAP metric checkpoint")
    if config.get("geometry_mode") != "right_edge_role_independent":
        raise ValueError("Resume checkpoint uses an incompatible geometry mode")
    checkpoint_negatives = checkpoint.get("loss_config", {}).get("negatives_per_anchor")
    if checkpoint_negatives is not None and int(checkpoint_negatives) != args.effective_negatives_per_anchor:
        raise ValueError("Resume checkpoint uses a different effective negative count")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        int(checkpoint.get("epoch", 0)) + 1,
        list(checkpoint.get("history", [])),
        float(checkpoint.get("best_val_loss", float("inf"))),
        float(checkpoint.get("best_val_score", checkpoint.get("best_val_acc", -float("inf")))),
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device, args.gpu_id)
    train_dataset = make_dataset(
        args,
        "train",
        args.triplets_per_puzzle,
        args.permute_pieces,
        args.max_train_samples or None,
    )
    val_dataset = make_dataset(
        args,
        "val",
        args.val_triplets_per_puzzle,
        False,
        args.max_val_samples or None,
    )
    if train_dataset.negatives_per_anchor != val_dataset.negatives_per_anchor:
        raise RuntimeError("train and validation resolved different negative counts")
    args.effective_negatives_per_anchor = train_dataset.negatives_per_anchor
    output_dir = prepare_output_dir(args)
    args.output_dir = output_dir
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device.type == "cuda", args.seed
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, device.type == "cuda", args.seed + 1
    )

    model = ViTMetricEncoder(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        image_size=args.input_size,
        input_channels=4,
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
    start_epoch, history, best_val_loss, best_val_score = load_resume(
        args.resume, model, optimizer, scheduler, args, device
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(checkpoint_args_dict(args), indent=2), encoding="utf-8"
    )
    write_training_args(args, output_dir / "training_args.txt")
    train_log_path = output_dir / "train_log.txt"
    if not args.resume:
        train_log_path.write_text("", encoding="utf-8")

    print(f"device: {device}")
    print(f"data_root: {train_dataset.puzzles.data_root}")
    print(f"output_dir: {output_dir}")
    print(train_dataset)
    print(val_dataset)
    print(
        f"RGBA adapter: learned Conv2d(4,3,1); negatives={args.effective_negatives_per_anchor}; "
        f"fixed epochs={args.epochs}; test is not loaded"
    )

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        apply_linear_warmup(optimizer, epoch, args.warmup_epochs)
        learning_rates = {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            args,
            epoch=epoch,
            optimizer=optimizer,
            max_steps=args.steps_per_epoch,
        )
        valid_stats = run_epoch(model, val_loader, device, args, epoch=epoch)
        if epoch > args.warmup_epochs:
            scheduler.step(valid_stats["loss"])

        improved_loss = valid_stats["loss"] < best_val_loss
        improved_score = valid_stats["triplet_acc"] > best_val_score
        if improved_loss:
            best_val_loss = valid_stats["loss"]
        if improved_score:
            best_val_score = valid_stats["triplet_acc"]
        record = {
            "epoch": epoch,
            "loss_type": "hard_triplet",
            "selection_metric": "triplet_acc",
            "learning_rate": learning_rates["projection_and_rgba_adapter"],
            "backbone_learning_rate": learning_rates["backbone"],
            "head_learning_rate": learning_rates["projection_and_rgba_adapter"],
            "train_optimization": train_stats,
            "train": train_stats,
            "valid": valid_stats,
            "test": {},
            "assembly": {},
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)

        save_checkpoint(
            output_dir / "checkpoints" / "last.pth",
            model, optimizer, scheduler, args, epoch, history, best_val_loss, best_val_score,
        )
        if improved_loss:
            save_checkpoint(
                output_dir / "checkpoints" / "best_loss.pth",
                model, optimizer, scheduler, args, epoch, history, best_val_loss, best_val_score,
            )
        if improved_score:
            save_checkpoint(
                output_dir / "checkpoints" / "best.pth",
                model, optimizer, scheduler, args, epoch, history, best_val_loss, best_val_score,
            )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        append_epoch_log(record, train_log_path)
        write_metric_outputs(history, output_dir, loss_type="hard_triplet")
        print(
            f"epoch {epoch:03d}: train_loss={train_stats['loss']:.4f}, "
            f"train_acc={train_stats['triplet_acc']:.3f}, valid_loss={valid_stats['loss']:.4f}, "
            f"valid_acc={valid_stats['triplet_acc']:.3f}"
        )

    total_time = time.perf_counter() - started
    write_summary(args=args, history=history, total_time=total_time, path=output_dir / "train_summary.txt")
    print(f"Training finished in {total_time:.1f}s. Summary: {output_dir / 'train_summary.txt'}")


if __name__ == "__main__":
    main()
