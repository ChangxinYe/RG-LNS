"""
@file: s1a_train_lsej.py
@description: 基于 handwritten ViT 和采样式同图候选的 ImageNet-LSEJ 度量学习训练脚本。
@author: Changxin Ye
@created: 2026-07-17
@version: 1.4
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
    from .metric_logger import append_epoch_log, write_metric_outputs, write_summary, write_training_args
    from .metric_losses import (
        puzzle_level_distance_weighted_triplet_loss,
        puzzle_level_hard_triplet_loss,
        puzzle_level_info_nce_loss,
        puzzle_level_random_triplet_loss,
        puzzle_level_semi_hard_triplet_loss,
    )
    from .metric_lsej_data import (
        DEFAULT_LSEJ_DATA_ROOT,
        OFFICIAL_TASKS,
        MetricLSEJTripletDataset,
    )
    from .metric_scorer import load_torch_checkpoint, resolve_device
    from .metric_vit_model import ViTMetricEncoder, model_config_dict
except ImportError:
    from experiment_naming import build_run_name, script_output_root
    from metric_logger import append_epoch_log, write_metric_outputs, write_summary, write_training_args
    from metric_losses import (
        puzzle_level_distance_weighted_triplet_loss,
        puzzle_level_hard_triplet_loss,
        puzzle_level_info_nce_loss,
        puzzle_level_random_triplet_loss,
        puzzle_level_semi_hard_triplet_loss,
    )
    from metric_lsej_data import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS, MetricLSEJTripletDataset
    from metric_scorer import load_torch_checkpoint, resolve_device
    from metric_vit_model import ViTMetricEncoder, model_config_dict


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOSS_TYPE = "hard_triplet"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_DISTANCE_WEIGHTED_CUTOFF = 0.5
DEFAULT_DISTANCE_WEIGHTED_NONZERO_CUTOFF = 1.4
LOSS_TYPES = (
    "hard_triplet",
    "semi_hard_triplet",
    "random_triplet",
    "distance_weighted_triplet",
    "infonce",
)
DEFAULT_NEGATIVES_PER_ANCHOR = {
    "hard_triplet": 15,
    "semi_hard_triplet": 0,
    "random_triplet": 0,
    "distance_weighted_triplet": 0,
    "infonce": 0,
}[DEFAULT_LOSS_TYPE]
DEFAULT_BATCH_SIZE = {
    "hard_triplet": 4,
    "semi_hard_triplet": 1,
    "random_triplet": 1,
    "distance_weighted_triplet": 1,
    "infonce": 1,
}[DEFAULT_LOSS_TYPE]
DEFAULT_LOG_ROOT = script_output_root(__file__, "logs")
DEFAULT_LOG_ROOTS = {loss_type: DEFAULT_LOG_ROOT for loss_type in LOSS_TYPES}
TRIPLET_METRIC_KEYS = [
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
SEMI_HARD_METRIC_KEYS = [
    "loss",
    "triplet_loss",
    "triplet_acc",
    "pos_dist",
    "hard_neg_dist",
    "selected_neg_dist",
    "margin_gap",
    "selected_margin_gap",
    "semi_hard_fraction",
    "fallback_fraction",
    "active_triplet_fraction",
    "negatives_per_anchor",
    "selected_neg_index",
    "embedding_std",
    "embedding_norm",
]
SAMPLED_TRIPLET_METRIC_KEYS = [
    "loss",
    "triplet_loss",
    "triplet_acc",
    "pos_dist",
    "hard_neg_dist",
    "selected_neg_dist",
    "margin_gap",
    "selected_margin_gap",
    "hard_sample_fraction",
    "semi_hard_sample_fraction",
    "easy_sample_fraction",
    "active_triplet_fraction",
    "eligible_candidate_fraction",
    "sampling_entropy",
    "hard_negative_count",
    "selected_negative_rank",
    "negatives_per_anchor",
    "selected_neg_index",
    "embedding_std",
    "embedding_norm",
]
INFONCE_METRIC_KEYS = [
    "loss",
    "contrastive_loss",
    "candidate_top1_acc",
    "positive_rank",
    "mrr",
    "positive_similarity",
    "hardest_negative_similarity",
    "similarity_gap",
    "negatives_per_anchor",
    "temperature",
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
        "在 ImageNet-LSEJ 上训练 s1a puzzle-level 度量学习模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default="grid10_erode2", choices=OFFICIAL_TASKS, help="官方 LSEJ 任务名称")
    parser.add_argument("--data-root", default=DEFAULT_LSEJ_DATA_ROOT, type=Path, help="ImageNet-LSEJ 数据集根目录")
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="训练日志和 checkpoint 根目录；留空时根据 loss-type 选择独立目录",
    )
    parser.add_argument("--run-name", default="", help="本次实验名称；留空时根据配置和时间自动生成")

    parser.add_argument("--backbone", default="vit-t", type=str, help="ViT 主干，可选 vit-t、vit-s、vit-b")
    parser.add_argument(
        "--pretrained-backbone",
        "--pretrained_backbone",
        default=True,
        type=parse_bool,
        help="是否加载本地 ImageNet 预训练 ViT 权重；开启时 input-size 必须为 224",
    )
    parser.add_argument(
        "--pretrained-weights-path",
        "--pretrained_weights_path",
        default=None,
        type=Path,
        help="预训练权重路径；留空时使用 models 目录中的默认权重",
    )
    parser.add_argument("--input-size", "--input_size", default=224, type=int, help="送入 ViT 的正方形 piece 尺寸，必须能被 16 整除")
    parser.add_argument("--embedding-dim", "--embedding_dim", default=128, type=int, help="单位超球面 embedding 的维度")
    parser.add_argument(
        "--normalization",
        default="vit",
        choices=["zero_one", "vit", "imagenet", "fragment"],
        help="piece resize 后采用的像素归一化方式",
    )
    parser.add_argument(
        "--score-metric",
        default=None,
        choices=["cosine", "euclidean"],
        help="checkpoint 在拼图兼容性评分阶段使用的距离类型；留空时 InfoNCE 用 cosine，triplet 用 euclidean",
    )

    parser.add_argument(
        "--loss-type",
        default=DEFAULT_LOSS_TYPE,
        choices=LOSS_TYPES,
        help="训练目标；支持 hard、semi-hard、随机、距离加权 triplet 和 InfoNCE",
    )
    parser.add_argument("--temperature", default=DEFAULT_TEMPERATURE, type=float, help="InfoNCE cosine logits 的温度参数")
    parser.add_argument("--triplets-per-puzzle", default=4, type=int, help="每张训练 puzzle 一次采样的 anchor-positive 关系数量")
    parser.add_argument("--val-triplets-per-puzzle", default=2, type=int, help="每张验证 puzzle 采用的确定性关系数量")
    parser.add_argument(
        "--negatives-per-anchor",
        default=DEFAULT_NEGATIVES_PER_ANCHOR,
        type=int,
        help="每个 anchor 无放回采样的同图错误候选数；0 表示使用该 puzzle 的全部合法负样本",
    )
    parser.add_argument("--permute-pieces", default=True, type=parse_bool, help="训练时是否在官方固定打乱基础上再次随机重排全部 pieces")
    parser.add_argument("--margin", default=0.2, type=float, help="Triplet loss 的间隔 margin")
    parser.add_argument(
        "--loss-distance",
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Triplet loss 内部采用的距离类型",
    )
    parser.add_argument(
        "--distance-weighted-cutoff",
        default=DEFAULT_DISTANCE_WEIGHTED_CUTOFF,
        type=float,
        help="距离加权采样的最小欧氏距离截断；更近的负样本共享该距离的权重以抑制高方差",
    )
    parser.add_argument(
        "--distance-weighted-nonzero-cutoff",
        default=DEFAULT_DISTANCE_WEIGHTED_NONZERO_CUTOFF,
        type=float,
        help="距离加权采样允许的最大负样本距离；默认沿用论文官方实现的 1.4",
    )

    parser.add_argument("--epochs", default=150, type=int, help="最多训练 epoch 数")
    parser.add_argument("--steps-per-epoch", default=0, type=int, help="每个 epoch 最多更新多少步；设为 0 表示遍历完整训练集")
    parser.add_argument(
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        help="每个 batch 的 puzzle 数，不是 triplet 数；每张 puzzle 会产生 triplets-per-puzzle 个 triplet",
    )
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="旧版统一学习率覆盖项；一般不要使用，建议分别设置 backbone-lr 和 head-lr",
    )
    parser.add_argument("--backbone-lr", default=1e-5, type=float, help="预训练 ViT 主干的目标学习率")
    parser.add_argument("--head-lr", default=1e-4, type=float, help="embedding 投影头的目标学习率")
    parser.add_argument("--warmup-epochs", default=5, type=int, help="学习率从低值线性升到目标值的 epoch 数")
    parser.add_argument("--lr-factor", default=0.5, type=float, help="验证 loss 停滞时学习率的衰减倍率")
    parser.add_argument("--lr-patience", default=3, type=int, help="验证 loss 连续多少轮未改善后衰减学习率")
    parser.add_argument("--grad-clip-norm", default=1.0, type=float, help="梯度范数裁剪阈值；设为 0 表示关闭")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader 并行读取进程数")
    parser.add_argument("--max-train-samples", default=0, type=int, help="最多使用多少张训练 puzzle；0 表示不限制")
    parser.add_argument("--max-val-samples", default=250, type=int, help="最多使用多少张验证 puzzle；0 表示完整验证集")
    parser.add_argument("--device", default="auto", help="运行设备，例如 auto、cpu、cuda:0；auto 会优先使用 GPU")
    parser.add_argument("--gpu-id", default=0, type=int, help="device=auto 时使用的 GPU 编号")
    parser.add_argument("--seed", default=42, type=int, help="Python、NumPy 和 PyTorch 随机种子")
    parser.add_argument("--resume", default="", type=str, help="继续训练的 checkpoint 路径；留空表示新实验")
    parser.add_argument("--print-freq", default=25, type=int, help="每隔多少个训练 step 输出一次中间统计；0 表示不输出")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        args.output_dir = DEFAULT_LOG_ROOTS[args.loss_type]
    if args.score_metric is None:
        args.score_metric = "cosine" if args.loss_type == "infonce" else "euclidean"
    if args.lr is not None:
        args.backbone_lr = args.lr
        args.head_lr = args.lr
    if args.pretrained_backbone and args.input_size != 224:
        raise ValueError("The provided pretrained ViT weights require --input-size 224")
    if args.input_size <= 0 or args.input_size % 16 != 0:
        raise ValueError("input_size must be positive and divisible by the ViT patch size 16")
    if args.pretrained_backbone and args.backbone_lr > 1e-4:
        raise ValueError(
            f"backbone_lr={args.backbone_lr:g} is unsafe for a pretrained ViT. "
            "Remove the legacy --lr argument and use the separate defaults."
        )
    if args.backbone_lr <= 0 or args.head_lr <= 0:
        raise ValueError("backbone_lr and head_lr must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.triplets_per_puzzle <= 0 or args.val_triplets_per_puzzle <= 0:
        raise ValueError("triplets_per_puzzle values must be positive")
    if args.negatives_per_anchor < 0:
        raise ValueError("negatives_per_anchor must be non-negative; use 0 for all legal negatives")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.loss_type != "infonce" and args.margin <= 0:
        raise ValueError("margin must be positive for triplet losses")
    if args.loss_type == "distance_weighted_triplet":
        if args.loss_distance != "euclidean":
            raise ValueError("distance_weighted_triplet requires --loss-distance euclidean")
        if not 0 < args.distance_weighted_cutoff < 2:
            raise ValueError("distance_weighted_cutoff must be in (0, 2)")
        if not args.distance_weighted_cutoff < args.distance_weighted_nonzero_cutoff < 2:
            raise ValueError(
                "distance_weighted_nonzero_cutoff must be greater than distance_weighted_cutoff and less than 2"
            )
    if args.max_train_samples < 0 or args.max_val_samples < 0:
        raise ValueError("max sample limits must be non-negative")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_output_dir(args: argparse.Namespace) -> Path:
    if args.loss_type == "infonce":
        loss_tag = f"infonce_tau{args.temperature:g}"
    elif args.loss_type == "semi_hard_triplet":
        loss_tag = "semi_hard_triplet"
    elif args.loss_type == "random_triplet":
        loss_tag = "random_triplet"
    elif args.loss_type == "distance_weighted_triplet":
        loss_tag = (
            f"distance_weighted_triplet_c{args.distance_weighted_cutoff:g}_"
            f"n{args.distance_weighted_nonzero_cutoff:g}"
        )
    else:
        loss_tag = "hard_triplet"
    negative_tag = "kall" if args.negatives_per_anchor == 0 else f"k{args.negatives_per_anchor}"
    run_name = args.run_name or build_run_name(
        loss_tag,
        f"d{args.embedding_dim}",
        f"s{args.input_size}",
        negative_tag,
    )
    output_dir = Path(args.output_dir) / args.task / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_dataset(
    args: argparse.Namespace,
    split: str,
    triplets_per_puzzle: int,
    permute_pieces: bool,
    max_samples: int | None,
) -> MetricLSEJTripletDataset:
    return MetricLSEJTripletDataset(
        data_root=args.data_root,
        task=args.task,
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
    dataset: MetricLSEJTripletDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(dataset.seed)
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


def flatten_and_move_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor = batch["anchor"].flatten(0, 1).to(device, non_blocking=True)
    positive = batch["positive"].flatten(0, 1).to(device, non_blocking=True)
    negatives = batch["negatives"].flatten(0, 1).to(device, non_blocking=True)
    return anchor, positive, negatives


def embed_triplet_batch(
    model: torch.nn.Module,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, negative_count = negatives.shape[:2]
    inputs = torch.cat([anchor, positive, negatives.flatten(0, 1)], dim=0)
    embeddings = model(inputs)
    emb_anchor = embeddings[:batch_size]
    emb_positive = embeddings[batch_size : batch_size * 2]
    emb_negatives = embeddings[batch_size * 2 :].reshape(batch_size, negative_count, -1)
    return emb_anchor, emb_positive, emb_negatives


def average_stats(totals: dict[str, float], steps: int) -> dict[str, float]:
    return {key: value / max(steps, 1) for key, value in totals.items()}


def selection_metric_name(loss_type: str) -> str:
    return "candidate_top1_acc" if loss_type == "infonce" else "triplet_acc"


def metric_keys(loss_type: str) -> list[str]:
    if loss_type == "infonce":
        return INFONCE_METRIC_KEYS
    if loss_type == "semi_hard_triplet":
        return SEMI_HARD_METRIC_KEYS
    if loss_type in {"random_triplet", "distance_weighted_triplet"}:
        return SAMPLED_TRIPLET_METRIC_KEYS
    return TRIPLET_METRIC_KEYS


def compute_metric_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    args: argparse.Namespace,
    training: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if args.loss_type == "infonce":
        return puzzle_level_info_nce_loss(
            anchor,
            positive,
            negatives,
            temperature=args.temperature,
        )
    if args.loss_type == "semi_hard_triplet":
        return puzzle_level_semi_hard_triplet_loss(
            anchor,
            positive,
            negatives,
            margin=args.margin,
            distance=args.loss_distance,
        )
    if args.loss_type == "random_triplet":
        return puzzle_level_random_triplet_loss(
            anchor,
            positive,
            negatives,
            margin=args.margin,
            distance=args.loss_distance,
            training=training,
        )
    if args.loss_type == "distance_weighted_triplet":
        return puzzle_level_distance_weighted_triplet_loss(
            anchor,
            positive,
            negatives,
            margin=args.margin,
            cutoff=args.distance_weighted_cutoff,
            nonzero_loss_cutoff=args.distance_weighted_nonzero_cutoff,
            training=training,
        )
    return puzzle_level_hard_triplet_loss(
        anchor,
        positive,
        negatives,
        margin=args.margin,
        distance=args.loss_distance,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    max_steps: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    keys = metric_keys(args.loss_type)
    totals = {key: 0.0 for key in keys}
    accuracy_key = selection_metric_name(args.loss_type)
    steps = 0
    context = torch.enable_grad() if training else torch.no_grad()
    iterator = tqdm(loader, total=max_steps or len(loader), leave=False)
    with context:
        for batch in iterator:
            anchor, positive, negatives = flatten_and_move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            emb_anchor, emb_positive, emb_negatives = embed_triplet_batch(
                model,
                anchor,
                positive,
                negatives,
            )
            loss, stats = compute_metric_loss(
                emb_anchor,
                emb_positive,
                emb_negatives,
                args,
                training=training,
            )
            if training:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()

            steps += 1
            for key in totals:
                totals[key] += stats[key]
            averages = average_stats(totals, steps)
            label = "train" if training else "valid"
            iterator.set_description(
                f"epoch {epoch:03d} {label} loss={averages['loss']:.4f} acc={averages[accuracy_key]:.3f}"
            )
            if training and args.print_freq > 0 and steps % args.print_freq == 0:
                if args.loss_type == "infonce":
                    detail = (
                        f"pos_sim={averages['positive_similarity']:.4f}, "
                        f"hard_neg_sim={averages['hardest_negative_similarity']:.4f}, "
                        f"mrr={averages['mrr']:.3f}"
                    )
                elif args.loss_type == "semi_hard_triplet":
                    detail = (
                        f"pos={averages['pos_dist']:.4f}, "
                        f"selected_neg={averages['selected_neg_dist']:.4f}, "
                        f"semi_hard={averages['semi_hard_fraction']:.3f}, "
                        f"fallback={averages['fallback_fraction']:.3f}"
                    )
                elif args.loss_type in {"random_triplet", "distance_weighted_triplet"}:
                    detail = (
                        f"pos={averages['pos_dist']:.4f}, "
                        f"selected_neg={averages['selected_neg_dist']:.4f}, "
                        f"rank={averages['selected_negative_rank']:.2f}, "
                        f"active={averages['active_triplet_fraction']:.3f}"
                    )
                else:
                    detail = (
                        f"pos={averages['pos_dist']:.4f}, "
                        f"hard_neg={averages['hard_neg_dist']:.4f}"
                    )
                print(
                    f"epoch {epoch:03d} train step {steps:04d}: "
                    f"loss={averages['loss']:.4f}, acc={averages[accuracy_key]:.3f}, "
                    f"{detail}, embedding_std={averages['embedding_std']:.3e}"
                )
            if max_steps > 0 and steps >= max_steps:
                break
    return average_stats(totals, steps)


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
    return {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def checkpoint_args_dict(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def checkpoint_loss_name(loss_type: str) -> str:
    if loss_type == "infonce":
        return "puzzle_level_sampled_infonce"
    if loss_type == "semi_hard_triplet":
        return "puzzle_level_sampled_semi_hard_triplet"
    if loss_type == "random_triplet":
        return "puzzle_level_random_triplet"
    if loss_type == "distance_weighted_triplet":
        return "puzzle_level_distance_weighted_triplet"
    return "puzzle_level_sampled_hard_triplet"


def loss_config_dict(args: argparse.Namespace) -> dict:
    effective_negatives = int(getattr(args, "effective_negatives_per_anchor", args.negatives_per_anchor))
    config = {
        "loss": checkpoint_loss_name(args.loss_type),
        "loss_type": args.loss_type,
        "triplets_per_puzzle": args.triplets_per_puzzle,
        "negatives_per_anchor": effective_negatives,
        "negatives_per_anchor_requested": args.negatives_per_anchor,
        "selection_metric": selection_metric_name(args.loss_type),
    }
    if args.loss_type == "infonce":
        config.update(
            {
                "similarity": "cosine",
                "temperature": args.temperature,
            }
        )
    else:
        config.update(
            {
                "margin": args.margin,
                "loss_distance": args.loss_distance,
            }
        )
        if args.loss_type == "distance_weighted_triplet":
            config.update(
                {
                    "distance_weighted_cutoff": args.distance_weighted_cutoff,
                    "distance_weighted_nonzero_cutoff": args.distance_weighted_nonzero_cutoff,
                    "distance_weighted_reference": "Wu et al. ICCV 2017",
                }
            )
    return config


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
            # Keep best_val_acc for compatibility with checkpoints written by
            # the original hard-triplet-only trainer.
            "best_val_acc": best_val_score,
            "best_val_score": best_val_score,
            "selection_metric": selection_metric_name(args.loss_type),
            "task": args.task,
            "dataset": "ImageNet-LSEJ",
            "model_config": {
                **model_config_dict(model),
                "normalization": args.normalization,
                "score_metric": args.score_metric,
                "canonical_edge": "right",
                "geometry_mode": "right_edge_role_independent",
            },
            "loss_config": loss_config_dict(args),
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
            "data_config": {
                "source": "official_imagenet_lsej_v1",
                "task": args.task,
                "official_erosion_and_permutation": True,
                "runtime_train_piece_permutation": args.permute_pieces,
                "negatives_per_anchor": int(
                    getattr(args, "effective_negatives_per_anchor", args.negatives_per_anchor)
                ),
                "negatives_per_anchor_requested": args.negatives_per_anchor,
            },
            "normalization": args.normalization,
            "score_metric": args.score_metric,
            "args": checkpoint_args_dict(args),
        },
        path,
    )


def load_resume(
    resume_path: str,
    model: ViTMetricEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, list[dict], float, float]:
    if not resume_path:
        return 1, [], float("inf"), -float("inf")
    checkpoint = load_torch_checkpoint(resume_path, device)
    checkpoint_task = checkpoint.get("task") or checkpoint.get("args", {}).get("task")
    if str(checkpoint_task) != args.task:
        raise ValueError(f"Cannot resume task {checkpoint_task} as {args.task}")
    config = checkpoint.get("model_config", {})
    if config.get("geometry_mode") != "right_edge_role_independent":
        raise ValueError("Resume checkpoint does not use the current role-independent right-edge geometry")
    if config.get("embedding_normalization", "none") != "l2":
        raise ValueError("Resume checkpoint does not use L2-normalized embeddings")
    checkpoint_group_count = len(checkpoint.get("optimizer", {}).get("param_groups", []))
    if checkpoint_group_count != len(optimizer.param_groups):
        raise ValueError("Resume checkpoint does not use separate backbone/projection-head learning rates")
    checkpoint_negatives = checkpoint.get("data_config", {}).get("negatives_per_anchor")
    requested_effective_negatives = int(
        getattr(args, "effective_negatives_per_anchor", args.negatives_per_anchor)
    )
    if checkpoint_negatives is not None and int(checkpoint_negatives) != requested_effective_negatives:
        raise ValueError(
            f"Resume checkpoint used {checkpoint_negatives} effective negatives per anchor, "
            f"requested {requested_effective_negatives}"
        )
    checkpoint_loss_config = checkpoint.get("loss_config", {})
    checkpoint_loss_type = checkpoint_loss_config.get("loss_type")
    if checkpoint_loss_type is None:
        checkpoint_loss_name_value = checkpoint_loss_config.get("loss", "puzzle_level_sampled_hard_triplet")
        checkpoint_loss_name = str(checkpoint_loss_name_value).lower()
        if "distance_weighted" in checkpoint_loss_name:
            checkpoint_loss_type = "distance_weighted_triplet"
        elif "random_triplet" in checkpoint_loss_name:
            checkpoint_loss_type = "random_triplet"
        elif "semi_hard" in checkpoint_loss_name:
            checkpoint_loss_type = "semi_hard_triplet"
        elif "infonce" in checkpoint_loss_name:
            checkpoint_loss_type = "infonce"
        else:
            checkpoint_loss_type = "hard_triplet"
    if checkpoint_loss_type != args.loss_type:
        raise ValueError(f"Cannot resume {checkpoint_loss_type} checkpoint with loss_type={args.loss_type}")
    if args.loss_type == "infonce":
        checkpoint_temperature = checkpoint_loss_config.get("temperature")
        if checkpoint_temperature is not None and not np.isclose(float(checkpoint_temperature), args.temperature):
            raise ValueError(
                f"Resume checkpoint used temperature={checkpoint_temperature}, requested {args.temperature}"
            )
    elif args.loss_type == "distance_weighted_triplet":
        checkpoint_cutoff = checkpoint_loss_config.get("distance_weighted_cutoff")
        checkpoint_nonzero_cutoff = checkpoint_loss_config.get("distance_weighted_nonzero_cutoff")
        if checkpoint_cutoff is not None and not np.isclose(float(checkpoint_cutoff), args.distance_weighted_cutoff):
            raise ValueError(
                f"Resume checkpoint used distance_weighted_cutoff={checkpoint_cutoff}, "
                f"requested {args.distance_weighted_cutoff}"
            )
        if checkpoint_nonzero_cutoff is not None and not np.isclose(
            float(checkpoint_nonzero_cutoff),
            args.distance_weighted_nonzero_cutoff,
        ):
            raise ValueError(
                f"Resume checkpoint used distance_weighted_nonzero_cutoff={checkpoint_nonzero_cutoff}, "
                f"requested {args.distance_weighted_nonzero_cutoff}"
            )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scheduler") is not None:
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
        raise RuntimeError("train and validation datasets resolved different negative counts")
    args.effective_negatives_per_anchor = train_dataset.negatives_per_anchor
    output_dir = prepare_output_dir(args)
    args.output_dir = output_dir
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, pin_memory)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, pin_memory)

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
    start_epoch, history, best_val_loss, best_val_score = load_resume(
        args.resume,
        model,
        optimizer,
        scheduler,
        args,
        device,
    )
    selection_metric = selection_metric_name(args.loss_type)
    (output_dir / "training_args.json").write_text(
        json.dumps(checkpoint_args_dict(args), indent=2),
        encoding="utf-8",
    )
    write_training_args(args, output_dir / "training_args.txt")
    train_log_path = output_dir / "train_log.txt"
    if not args.resume:
        train_log_path.write_text("", encoding="utf-8")

    print(f"device: {device}")
    print(f"output_dir: {output_dir}")
    print(train_dataset)
    print(val_dataset)
    if args.negatives_per_anchor == 0:
        inputs_per_step = (
            args.batch_size
            * args.triplets_per_puzzle
            * (args.effective_negatives_per_anchor + 2)
        )
        print(
            f"negatives_per_anchor=0 resolved to all {args.effective_negatives_per_anchor} legal negatives; "
            f"the current training batch sends {inputs_per_step} images through ViT per optimization step"
        )
    if args.loss_type == "infonce":
        loss_detail = f"temperature={args.temperature:g}"
    elif args.loss_type == "distance_weighted_triplet":
        loss_detail = (
            f"margin={args.margin:g}, distance=euclidean, "
            f"cutoff={args.distance_weighted_cutoff:g}, "
            f"nonzero_cutoff={args.distance_weighted_nonzero_cutoff:g}"
        )
    else:
        loss_detail = f"margin={args.margin:g}, distance={args.loss_distance}"
    print(
        f"optimizer: backbone_lr={args.backbone_lr:g}, head_lr={args.head_lr:g}, "
        f"warmup_epochs={args.warmup_epochs}, loss={args.loss_type}, {loss_detail}, "
        f"grad_clip_norm={args.grad_clip_norm:g}"
    )

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        apply_linear_warmup(optimizer, epoch, args.warmup_epochs)
        epoch_lrs = learning_rates(optimizer)
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            args,
            optimizer=optimizer,
            epoch=epoch,
            max_steps=args.steps_per_epoch,
        )
        valid_stats = run_epoch(model, val_loader, device, args, epoch=epoch)
        if epoch > args.warmup_epochs:
            scheduler.step(valid_stats["loss"])

        record = {
            "epoch": epoch,
            "loss_type": args.loss_type,
            "selection_metric": selection_metric,
            "learning_rate": epoch_lrs["projection_head"],
            "backbone_learning_rate": epoch_lrs["backbone"],
            "head_learning_rate": epoch_lrs["projection_head"],
            "train_optimization": train_stats,
            "train": train_stats,
            "valid": valid_stats,
            "test": {},
            "assembly": {},
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)

        improved_loss = valid_stats["loss"] < best_val_loss
        improved_acc = valid_stats[selection_metric] > best_val_score
        if improved_loss:
            best_val_loss = valid_stats["loss"]
        if improved_acc:
            best_val_score = valid_stats[selection_metric]

        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        append_epoch_log(record, train_log_path)
        write_metric_outputs(history, output_dir, loss_type=args.loss_type)
        save_checkpoint(
            output_dir / "checkpoints" / "last.pth",
            model,
            optimizer,
            scheduler,
            args,
            epoch,
            history,
            best_val_loss,
            best_val_score,
        )
        if improved_loss:
            save_checkpoint(
                output_dir / "checkpoints" / "best_loss.pth",
                model,
                optimizer,
                scheduler,
                args,
                epoch,
                history,
                best_val_loss,
                best_val_score,
            )
        if improved_acc:
            for checkpoint_name in ("best_acc.pth", "best.pth"):
                save_checkpoint(
                    output_dir / "checkpoints" / checkpoint_name,
                    model,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    history,
                    best_val_loss,
                    best_val_score,
                )

        print(
            f"epoch {epoch:03d}: train_loss={train_stats['loss']:.4f}, train_acc={train_stats[selection_metric]:.3f}, "
            f"valid_loss={valid_stats['loss']:.4f}, valid_acc={valid_stats[selection_metric]:.3f}, "
            f"valid_emb_std={valid_stats['embedding_std']:.3e}"
        )
    total_time = time.perf_counter() - started
    write_summary(args=args, history=history, total_time=total_time, path=output_dir / "train_summary.txt")
    print(f"Training finished in {total_time:.1f}s. Summary: {output_dir / 'train_summary.txt'}")


if __name__ == "__main__":
    main()
