"""
@file: metric_losses.py
@description: 面向拼图同图候选集合的 triplet、InfoNCE 与双向部分分配损失函数。
@author: Changxin Ye
@created: 2026-07-10
@version: 1.5
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _embedding_statistics(*embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    all_embeddings = torch.cat(
        [embedding.flatten(0, -2) for embedding in embeddings],
        dim=0,
    )
    embedding_std = all_embeddings.var(dim=0, unbiased=False).mean().clamp_min(0.0).sqrt()
    embedding_norm = torch.linalg.vector_norm(all_embeddings, dim=1).mean()
    return embedding_std, embedding_norm


def puzzle_level_hard_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    margin: float = 1.0,
    distance: str = "euclidean",
    negative_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select the hardest legal same-puzzle negative for each anchor.

    Args:
        anchor: Tensor with shape [B, D].
        positive: Tensor with shape [B, D].
        negatives: Tensor with shape [B, K, D].
        margin: Triplet margin.
        distance: `euclidean` or `cosine`. Smaller values mean more compatible.
        negative_mask: Optional boolean tensor [B, K]. False entries are
            padding and are excluded from mining and statistics.
    """

    if negatives.ndim != 3:
        raise ValueError(f"Expected negatives shape [B, K, D], got {tuple(negatives.shape)}")
    if distance not in {"euclidean", "cosine"}:
        raise ValueError("distance must be 'euclidean' or 'cosine'")
    if negative_mask is not None:
        if negative_mask.shape != negatives.shape[:2]:
            raise ValueError(
                f"Expected negative_mask shape {tuple(negatives.shape[:2])}, got {tuple(negative_mask.shape)}"
            )
        negative_mask = negative_mask.to(device=negatives.device, dtype=torch.bool)
        if not bool(negative_mask.any(dim=1).all()):
            raise ValueError("Every anchor must have at least one valid negative")

    if distance == "cosine":
        anchor_score = F.normalize(anchor, dim=1)
        positive_score = F.normalize(positive, dim=1)
        negatives_score = F.normalize(negatives, dim=2)
        pos_dist = 1.0 - (anchor_score * positive_score).sum(dim=1)
        neg_distances = 1.0 - (anchor_score[:, None, :] * negatives_score).sum(dim=2)
    else:
        pos_dist = torch.linalg.vector_norm(anchor - positive, dim=1)
        neg_distances = torch.linalg.vector_norm(anchor[:, None, :] - negatives, dim=2)

    if negative_mask is not None:
        neg_distances = neg_distances.masked_fill(~negative_mask, float("inf"))
    hard_neg_dist, hard_neg_index = neg_distances.min(dim=1)
    triplet = torch.relu(pos_dist - hard_neg_dist + float(margin)).mean()
    valid_negatives = negatives if negative_mask is None else negatives[negative_mask]
    embedding_std, embedding_norm = _embedding_statistics(anchor, positive, valid_negatives)
    loss = triplet

    stats = {
        "loss": float(loss.detach().cpu()),
        "triplet_loss": float(triplet.detach().cpu()),
        "pos_dist": float(pos_dist.mean().detach().cpu()),
        "hard_neg_dist": float(hard_neg_dist.mean().detach().cpu()),
        "margin_gap": float((hard_neg_dist - pos_dist).mean().detach().cpu()),
        "triplet_acc": float((pos_dist < hard_neg_dist).float().mean().detach().cpu()),
        "negatives_per_anchor": float(
            negatives.shape[1]
            if negative_mask is None
            else negative_mask.sum(dim=1).float().mean().detach().cpu()
        ),
        "hard_neg_index": float(hard_neg_index.float().mean().detach().cpu()),
        # A unit-normalized encoder should keep embedding_norm close to 1.  In
        # contrast, embedding_std approaches 0 when every image is mapped to
        # the same point on the hypersphere, so it is a direct collapse alarm.
        "embedding_std": float(embedding_std.detach().cpu()),
        "embedding_norm": float(embedding_norm.detach().cpu()),
    }
    return loss, stats


def puzzle_level_semi_hard_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    margin: float = 1.0,
    distance: str = "euclidean",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mine the nearest negative farther than the positive for each anchor.

    A selected negative is semi-hard when ``d(a, p) < d(a, n) < d(a, p) + margin``.
    If no negative is farther than the positive, the farthest candidate is used
    as the least-hard fallback. This avoids falling back to the globally hardest
    candidate, which is especially brittle for eroded or repetitive puzzle edges.

    ``triplet_acc`` is still measured against the globally hardest candidate so
    checkpoint selection remains strict and comparable with hard-triplet runs.
    """

    if anchor.ndim != 2 or positive.ndim != 2:
        raise ValueError("anchor and positive must have shape [B, D]")
    if negatives.ndim != 3:
        raise ValueError(f"Expected negatives shape [B, K, D], got {tuple(negatives.shape)}")
    if anchor.shape != positive.shape:
        raise ValueError(f"anchor shape {tuple(anchor.shape)} does not match positive shape {tuple(positive.shape)}")
    if negatives.shape[0] != anchor.shape[0] or negatives.shape[2] != anchor.shape[1]:
        raise ValueError("negatives must share the anchor batch and embedding dimensions")
    if negatives.shape[1] < 1:
        raise ValueError("Semi-hard triplet loss requires at least one negative per anchor")
    if margin <= 0:
        raise ValueError("margin must be positive")
    if distance not in {"euclidean", "cosine"}:
        raise ValueError("distance must be 'euclidean' or 'cosine'")

    if distance == "cosine":
        anchor_score = F.normalize(anchor, dim=1)
        positive_score = F.normalize(positive, dim=1)
        negatives_score = F.normalize(negatives, dim=2)
        pos_dist = 1.0 - (anchor_score * positive_score).sum(dim=1)
        neg_distances = 1.0 - (anchor_score[:, None, :] * negatives_score).sum(dim=2)
    else:
        pos_dist = torch.linalg.vector_norm(anchor - positive, dim=1)
        neg_distances = torch.linalg.vector_norm(anchor[:, None, :] - negatives, dim=2)

    farther_mask = neg_distances > pos_dist[:, None]
    has_farther = farther_mask.any(dim=1)
    nearest_farther_dist, nearest_farther_index = neg_distances.masked_fill(
        ~farther_mask,
        float("inf"),
    ).min(dim=1)
    fallback_dist, fallback_index = neg_distances.max(dim=1)
    selected_neg_dist = torch.where(has_farther, nearest_farther_dist, fallback_dist)
    selected_neg_index = torch.where(has_farther, nearest_farther_index, fallback_index)

    per_anchor_loss = torch.relu(pos_dist - selected_neg_dist + float(margin))
    triplet = per_anchor_loss.mean()
    hard_neg_dist, _hard_neg_index = neg_distances.min(dim=1)
    semi_hard_mask = has_farther & (selected_neg_dist < pos_dist + float(margin))
    fallback_mask = ~has_farther
    embedding_std, embedding_norm = _embedding_statistics(anchor, positive, negatives)
    loss = triplet

    stats = {
        "loss": float(loss.detach().cpu()),
        "triplet_loss": float(triplet.detach().cpu()),
        # Strict candidate top-1 accuracy against the global hardest negative.
        "triplet_acc": float((pos_dist < hard_neg_dist).float().mean().detach().cpu()),
        "pos_dist": float(pos_dist.mean().detach().cpu()),
        "hard_neg_dist": float(hard_neg_dist.mean().detach().cpu()),
        "selected_neg_dist": float(selected_neg_dist.mean().detach().cpu()),
        "margin_gap": float((hard_neg_dist - pos_dist).mean().detach().cpu()),
        "selected_margin_gap": float((selected_neg_dist - pos_dist).mean().detach().cpu()),
        "semi_hard_fraction": float(semi_hard_mask.float().mean().detach().cpu()),
        "fallback_fraction": float(fallback_mask.float().mean().detach().cpu()),
        "active_triplet_fraction": float((per_anchor_loss > 0).float().mean().detach().cpu()),
        "negatives_per_anchor": float(negatives.shape[1]),
        "selected_neg_index": float(selected_neg_index.float().mean().detach().cpu()),
        "embedding_std": float(embedding_std.detach().cpu()),
        "embedding_norm": float(embedding_norm.detach().cpu()),
    }
    return loss, stats


def _sampled_triplet_statistics(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    pos_dist: torch.Tensor,
    neg_distances: torch.Tensor,
    probabilities: torch.Tensor,
    margin: float,
    training: bool,
    eligible_candidate_fraction: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply a sampled triplet objective or its deterministic expectation."""

    probabilities = probabilities.detach()
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    per_candidate_loss = torch.relu(pos_dist[:, None] - neg_distances + float(margin))

    if training:
        selected_index = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        selected_neg_dist = neg_distances.gather(1, selected_index[:, None]).squeeze(1)
        per_anchor_loss = per_candidate_loss.gather(1, selected_index[:, None]).squeeze(1)
        selected_probabilities = torch.zeros_like(probabilities).scatter(1, selected_index[:, None], 1.0)
    else:
        candidate_indices = torch.arange(
            negatives.shape[1],
            device=negatives.device,
            dtype=probabilities.dtype,
        )
        selected_index = (probabilities * candidate_indices[None, :]).sum(dim=1)
        selected_neg_dist = (probabilities * neg_distances).sum(dim=1)
        per_anchor_loss = (probabilities * per_candidate_loss).sum(dim=1)
        selected_probabilities = probabilities

    triplet = per_anchor_loss.mean()
    hard_neg_dist, _hard_neg_index = neg_distances.min(dim=1)
    hard_mask = neg_distances <= pos_dist[:, None]
    semi_hard_mask = (neg_distances > pos_dist[:, None]) & (
        neg_distances < pos_dist[:, None] + float(margin)
    )
    easy_mask = neg_distances >= pos_dist[:, None] + float(margin)
    active_mask = per_candidate_loss > 0
    candidate_ranks = 1 + (
        neg_distances[:, None, :] < neg_distances[:, :, None]
    ).sum(dim=2)
    selected_rank = (selected_probabilities * candidate_ranks.to(probabilities.dtype)).sum(dim=1)

    if probabilities.shape[1] > 1:
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        entropy = entropy / torch.log(
            torch.tensor(float(probabilities.shape[1]), device=probabilities.device)
        )
    else:
        entropy = torch.zeros(probabilities.shape[0], device=probabilities.device)

    embedding_std, embedding_norm = _embedding_statistics(anchor, positive, negatives)
    loss = triplet
    stats = {
        "loss": float(loss.detach().cpu()),
        "triplet_loss": float(triplet.detach().cpu()),
        "triplet_acc": float((pos_dist < hard_neg_dist).float().mean().detach().cpu()),
        "pos_dist": float(pos_dist.mean().detach().cpu()),
        "hard_neg_dist": float(hard_neg_dist.mean().detach().cpu()),
        "selected_neg_dist": float(selected_neg_dist.mean().detach().cpu()),
        "margin_gap": float((hard_neg_dist - pos_dist).mean().detach().cpu()),
        "selected_margin_gap": float((selected_neg_dist - pos_dist).mean().detach().cpu()),
        "hard_sample_fraction": float(
            (selected_probabilities * hard_mask.float()).sum(dim=1).mean().detach().cpu()
        ),
        "semi_hard_sample_fraction": float(
            (selected_probabilities * semi_hard_mask.float()).sum(dim=1).mean().detach().cpu()
        ),
        "easy_sample_fraction": float(
            (selected_probabilities * easy_mask.float()).sum(dim=1).mean().detach().cpu()
        ),
        "active_triplet_fraction": float(
            (selected_probabilities * active_mask.float()).sum(dim=1).mean().detach().cpu()
        ),
        "eligible_candidate_fraction": float(eligible_candidate_fraction.mean().detach().cpu()),
        "sampling_entropy": float(entropy.mean().detach().cpu()),
        "hard_negative_count": float(hard_mask.float().sum(dim=1).mean().detach().cpu()),
        "selected_negative_rank": float(selected_rank.mean().detach().cpu()),
        "negatives_per_anchor": float(negatives.shape[1]),
        "selected_neg_index": float(selected_index.float().mean().detach().cpu()),
        "embedding_std": float(embedding_std.detach().cpu()),
        "embedding_norm": float(embedding_norm.detach().cpu()),
    }
    return loss, stats


def puzzle_level_random_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    margin: float = 1.0,
    distance: str = "euclidean",
    training: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Uniformly sample one legal negative per anchor.

    Validation uses the exact expectation over the uniform distribution so the
    validation loss and checkpoint selection are reproducible.
    """

    if anchor.ndim != 2 or positive.ndim != 2:
        raise ValueError("anchor and positive must have shape [B, D]")
    if anchor.shape != positive.shape:
        raise ValueError(f"anchor shape {tuple(anchor.shape)} does not match positive shape {tuple(positive.shape)}")
    if negatives.ndim != 3 or negatives.shape[1] < 1:
        raise ValueError(f"Expected negatives shape [B, K, D] with K >= 1, got {tuple(negatives.shape)}")
    if negatives.shape[0] != anchor.shape[0] or negatives.shape[2] != anchor.shape[1]:
        raise ValueError("negatives must share the anchor batch and embedding dimensions")
    if distance not in {"euclidean", "cosine"}:
        raise ValueError("distance must be 'euclidean' or 'cosine'")
    if margin <= 0:
        raise ValueError("margin must be positive")

    if distance == "cosine":
        anchor_score = F.normalize(anchor, dim=1)
        positive_score = F.normalize(positive, dim=1)
        negatives_score = F.normalize(negatives, dim=2)
        pos_dist = 1.0 - (anchor_score * positive_score).sum(dim=1)
        neg_distances = 1.0 - (anchor_score[:, None, :] * negatives_score).sum(dim=2)
    else:
        pos_dist = torch.linalg.vector_norm(anchor - positive, dim=1)
        neg_distances = torch.linalg.vector_norm(anchor[:, None, :] - negatives, dim=2)

    probabilities = torch.full_like(neg_distances, 1.0 / negatives.shape[1])
    eligible_fraction = torch.ones(anchor.shape[0], device=anchor.device)
    return _sampled_triplet_statistics(
        anchor,
        positive,
        negatives,
        pos_dist,
        neg_distances,
        probabilities,
        margin,
        training,
        eligible_fraction,
    )


def puzzle_level_distance_weighted_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    margin: float = 1.0,
    cutoff: float = 0.5,
    nonzero_loss_cutoff: float = 1.4,
    training: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sample negatives using inverse unit-hypersphere distance density.

    This follows Wu et al., *Sampling Matters in Deep Embedding Learning*.
    Distances below ``cutoff`` share the same clipped weight to suppress noisy
    extreme negatives. Candidates must also induce non-zero triplet loss and be
    below ``nonzero_loss_cutoff``. Stable fallbacks preserve a valid distribution
    when either filter would otherwise remove every candidate.
    """

    if anchor.ndim != 2 or positive.ndim != 2:
        raise ValueError("anchor and positive must have shape [B, D]")
    if anchor.shape != positive.shape:
        raise ValueError(f"anchor shape {tuple(anchor.shape)} does not match positive shape {tuple(positive.shape)}")
    if negatives.ndim != 3 or negatives.shape[1] < 1:
        raise ValueError(f"Expected negatives shape [B, K, D] with K >= 1, got {tuple(negatives.shape)}")
    if negatives.shape[0] != anchor.shape[0] or negatives.shape[2] != anchor.shape[1]:
        raise ValueError("negatives must share the anchor batch and embedding dimensions")
    if margin <= 0:
        raise ValueError("margin must be positive")
    if not 0 < cutoff < 2:
        raise ValueError("cutoff must be in (0, 2) for L2-normalized embeddings")
    if not cutoff < nonzero_loss_cutoff < 2:
        raise ValueError("nonzero_loss_cutoff must be greater than cutoff and less than 2")

    anchor_score = F.normalize(anchor, dim=1)
    positive_score = F.normalize(positive, dim=1)
    negatives_score = F.normalize(negatives, dim=2)
    pos_dist = torch.linalg.vector_norm(anchor_score - positive_score, dim=1)
    neg_distances = torch.linalg.vector_norm(anchor_score[:, None, :] - negatives_score, dim=2)

    sampling_distances = neg_distances.detach().clamp(min=float(cutoff), max=2.0 - 1e-6)
    embedding_dim = float(anchor.shape[1])
    log_weights = (2.0 - embedding_dim) * sampling_distances.log()
    log_weights -= ((embedding_dim - 3.0) / 2.0) * torch.log(
        (1.0 - 0.25 * sampling_distances.square()).clamp_min(1e-8)
    )

    active_mask = neg_distances.detach() < pos_dist.detach()[:, None] + float(margin)
    cutoff_mask = neg_distances.detach() < float(nonzero_loss_cutoff)
    eligible_mask = active_mask & cutoff_mask
    has_eligible = eligible_mask.any(dim=1, keepdim=True)
    # If the fixed cutoff removes all active candidates, retain the active set.
    eligible_mask = torch.where(has_eligible, eligible_mask, active_mask)
    has_active = eligible_mask.any(dim=1, keepdim=True)
    # A fully separated anchor has zero triplet loss for every candidate; use a
    # uniform distribution so diagnostics remain finite and deterministic.
    eligible_mask = torch.where(has_active, eligible_mask, torch.ones_like(eligible_mask))

    masked_log_weights = log_weights.masked_fill(~eligible_mask, -torch.inf)
    weighted_probabilities = torch.softmax(masked_log_weights, dim=1)
    uniform_probabilities = torch.full_like(weighted_probabilities, 1.0 / negatives.shape[1])
    probabilities = torch.where(has_active, weighted_probabilities, uniform_probabilities)
    eligible_fraction = eligible_mask.float().mean(dim=1)
    return _sampled_triplet_statistics(
        anchor_score,
        positive_score,
        negatives_score,
        pos_dist,
        neg_distances,
        probabilities,
        margin,
        training,
        eligible_fraction,
    )


def puzzle_level_info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Classify the true same-puzzle neighbor among sampled legal candidates.

    Args:
        anchor: Tensor with shape [B, D].
        positive: Tensor with shape [B, D].
        negatives: Tensor with shape [B, K, D].
        temperature: Positive cosine-logit temperature.

    The positive candidate is placed at column zero. All remaining columns are
    legal wrong candidates sampled from the same puzzle and canonical direction.
    """

    if anchor.ndim != 2 or positive.ndim != 2:
        raise ValueError("anchor and positive must have shape [B, D]")
    if negatives.ndim != 3:
        raise ValueError(f"Expected negatives shape [B, K, D], got {tuple(negatives.shape)}")
    if anchor.shape != positive.shape:
        raise ValueError(f"anchor shape {tuple(anchor.shape)} does not match positive shape {tuple(positive.shape)}")
    if negatives.shape[0] != anchor.shape[0] or negatives.shape[2] != anchor.shape[1]:
        raise ValueError("negatives must share the anchor batch and embedding dimensions")
    if negatives.shape[1] < 1:
        raise ValueError("InfoNCE requires at least one negative per anchor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    anchor_score = F.normalize(anchor, dim=1)
    positive_score = F.normalize(positive, dim=1)
    negatives_score = F.normalize(negatives, dim=2)
    positive_similarity = (anchor_score * positive_score).sum(dim=1)
    negative_similarities = (anchor_score[:, None, :] * negatives_score).sum(dim=2)
    candidate_similarities = torch.cat([positive_similarity[:, None], negative_similarities], dim=1)
    logits = candidate_similarities / float(temperature)
    labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    contrastive = F.cross_entropy(logits, labels)

    hardest_negative_similarity, _ = negative_similarities.max(dim=1)
    positive_rank = 1 + (negative_similarities >= positive_similarity[:, None]).sum(dim=1)
    # Use a strict comparison so a fully collapsed/tied embedding does not get
    # artificial credit merely because the positive occupies column zero.
    candidate_top1_acc = (positive_similarity > hardest_negative_similarity).float().mean()
    reciprocal_rank = positive_rank.float().reciprocal().mean()
    similarity_gap = positive_similarity - hardest_negative_similarity
    embedding_std, embedding_norm = _embedding_statistics(anchor, positive, negatives)

    stats = {
        "loss": float(contrastive.detach().cpu()),
        "contrastive_loss": float(contrastive.detach().cpu()),
        "candidate_top1_acc": float(candidate_top1_acc.detach().cpu()),
        "positive_rank": float(positive_rank.float().mean().detach().cpu()),
        "mrr": float(reciprocal_rank.detach().cpu()),
        "positive_similarity": float(positive_similarity.mean().detach().cpu()),
        "hardest_negative_similarity": float(hardest_negative_similarity.mean().detach().cpu()),
        "similarity_gap": float(similarity_gap.mean().detach().cpu()),
        "negatives_per_anchor": float(negatives.shape[1]),
        "temperature": float(temperature),
        "embedding_std": float(embedding_std.detach().cpu()),
        "embedding_norm": float(embedding_norm.detach().cpu()),
    }
    return contrastive, stats


def puzzle_bidirectional_assignment_loss(
    anchor_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    row_targets: torch.Tensor,
    column_targets: torch.Tensor,
    row_negative_indices: torch.Tensor,
    column_negative_indices: torch.Tensor,
    invalid_pair_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
    assignment_weight: float = 0.1,
    hard_triplet_weight: float = 1.0,
    margin: float = 0.2,
    distance: str = "euclidean",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimize a puzzle-specific bidirectional partial assignment.

    Rows are directed anchor edges and columns are candidate edges from the
    same puzzle and axis. ``row_targets`` maps each matched row to its true
    candidate column. ``column_targets`` maps matched candidate columns back to
    their unique true anchor row; unmatched columns use -1 and remain valid row
    negatives. The two cross-entropies therefore learn both directions of the
    puzzle's partial one-to-one adjacency structure.

    A K-candidate bidirectional hard-triplet term is retained so S6A can be
    compared against S1A-style metric training while adding only the structured
    assignment objective. Smaller distances mean greater compatibility.
    """

    if anchor_embeddings.ndim != 3 or candidate_embeddings.ndim != 3:
        raise ValueError("anchor_embeddings and candidate_embeddings must have shape [B, N, D]")
    if anchor_embeddings.shape[0] != candidate_embeddings.shape[0]:
        raise ValueError("anchor and candidate embeddings must share the puzzle batch dimension")
    if anchor_embeddings.shape[2] != candidate_embeddings.shape[2]:
        raise ValueError("anchor and candidate embeddings must share the embedding dimension")
    batch_size, row_count, embedding_dim = anchor_embeddings.shape
    candidate_count = candidate_embeddings.shape[1]
    if row_targets.shape != (batch_size, row_count):
        raise ValueError(f"Expected row_targets shape {(batch_size, row_count)}, got {tuple(row_targets.shape)}")
    if column_targets.shape != (batch_size, candidate_count):
        raise ValueError(
            f"Expected column_targets shape {(batch_size, candidate_count)}, got {tuple(column_targets.shape)}"
        )
    if row_negative_indices.ndim != 3 or row_negative_indices.shape[:2] != (batch_size, row_count):
        raise ValueError("row_negative_indices must have shape [B, rows, K]")
    if column_negative_indices.ndim != 3 or column_negative_indices.shape[:2] != (
        batch_size,
        candidate_count,
    ):
        raise ValueError("column_negative_indices must have shape [B, columns, K]")
    if row_negative_indices.shape[2] != column_negative_indices.shape[2]:
        raise ValueError("row and column hard-negative tensors must use the same K")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if assignment_weight < 0 or hard_triplet_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if assignment_weight == 0 and hard_triplet_weight == 0:
        raise ValueError("at least one loss weight must be positive")
    if hard_triplet_weight > 0 and margin <= 0:
        raise ValueError("margin must be positive when hard_triplet_weight is non-zero")
    if distance not in {"euclidean", "cosine"}:
        raise ValueError("distance must be 'euclidean' or 'cosine'")

    row_targets = row_targets.to(device=anchor_embeddings.device, dtype=torch.long)
    column_targets = column_targets.to(device=anchor_embeddings.device, dtype=torch.long)
    row_negative_indices = row_negative_indices.to(device=anchor_embeddings.device, dtype=torch.long)
    column_negative_indices = column_negative_indices.to(
        device=anchor_embeddings.device,
        dtype=torch.long,
    )
    row_valid = row_targets >= 0
    column_valid = column_targets >= 0
    if not bool(row_valid.any()) or not bool(column_valid.any()):
        raise ValueError("bidirectional assignment requires matched rows and columns")

    if distance == "cosine":
        anchor_score = F.normalize(anchor_embeddings, dim=2)
        candidate_score = F.normalize(candidate_embeddings, dim=2)
        distances = 1.0 - torch.matmul(anchor_score, candidate_score.transpose(1, 2))
    else:
        distances = torch.cdist(anchor_embeddings, candidate_embeddings, p=2)

    if invalid_pair_mask is None:
        invalid_pair_mask = torch.zeros_like(distances, dtype=torch.bool)
    else:
        if invalid_pair_mask.shape != distances.shape:
            raise ValueError(
                f"Expected invalid_pair_mask shape {tuple(distances.shape)}, "
                f"got {tuple(invalid_pair_mask.shape)}"
            )
        invalid_pair_mask = invalid_pair_mask.to(device=distances.device, dtype=torch.bool)

    batch_indices, row_indices = torch.where(row_valid)
    positive_columns = row_targets[batch_indices, row_indices]
    if bool(invalid_pair_mask[batch_indices, row_indices, positive_columns].any()):
        raise ValueError("a ground-truth row assignment was marked invalid")
    column_batch_indices, column_indices = torch.where(column_valid)
    positive_rows = column_targets[column_batch_indices, column_indices]
    if bool(invalid_pair_mask[column_batch_indices, positive_rows, column_indices].any()):
        raise ValueError("a ground-truth column assignment was marked invalid")

    logits = (-distances / float(temperature)).masked_fill(invalid_pair_mask, -torch.inf)
    if not bool(torch.isfinite(logits[batch_indices, row_indices]).any(dim=1).all()):
        raise ValueError("every matched row must retain at least one valid candidate")
    transposed_logits = logits.transpose(1, 2)
    if not bool(
        torch.isfinite(transposed_logits[column_batch_indices, column_indices]).any(dim=1).all()
    ):
        raise ValueError("every matched column must retain at least one valid anchor")

    row_logits = logits[batch_indices, row_indices]
    column_logits = transposed_logits[column_batch_indices, column_indices]
    row_assignment_loss = F.cross_entropy(row_logits, positive_columns)
    column_assignment_loss = F.cross_entropy(column_logits, positive_rows)
    assignment_loss = 0.5 * (row_assignment_loss + column_assignment_loss)

    row_predictions = logits.argmax(dim=2)
    column_predictions = transposed_logits.argmax(dim=2)
    row_correct = row_predictions[batch_indices, row_indices] == positive_columns
    column_correct = column_predictions[column_batch_indices, column_indices] == positive_rows
    reverse_prediction_for_rows = column_predictions[batch_indices, positive_columns]
    mutual_correct = row_correct & (reverse_prediction_for_rows == row_indices)

    row_positive_distances = distances[batch_indices, row_indices, positive_columns]
    column_positive_distances = distances[column_batch_indices, positive_rows, column_indices]
    row_competitor_distances = distances[batch_indices, row_indices]
    row_positive_mask = torch.zeros_like(row_competitor_distances, dtype=torch.bool)
    row_positive_mask.scatter_(1, positive_columns[:, None], True)
    row_eligible = ~invalid_pair_mask[batch_indices, row_indices] & ~row_positive_mask
    row_rank = 1 + (
        row_eligible & (row_competitor_distances <= row_positive_distances[:, None])
    ).sum(dim=1)

    column_distance_matrix = distances.transpose(1, 2)
    column_competitor_distances = column_distance_matrix[column_batch_indices, column_indices]
    column_positive_mask = torch.zeros_like(column_competitor_distances, dtype=torch.bool)
    column_positive_mask.scatter_(1, positive_rows[:, None], True)
    column_eligible = (
        ~invalid_pair_mask.transpose(1, 2)[column_batch_indices, column_indices]
        & ~column_positive_mask
    )
    column_rank = 1 + (
        column_eligible & (column_competitor_distances <= column_positive_distances[:, None])
    ).sum(dim=1)

    negative_count = row_negative_indices.shape[2]
    row_negative_valid = row_negative_indices >= 0
    column_negative_valid = column_negative_indices >= 0
    safe_row_negative_indices = row_negative_indices.clamp_min(0)
    safe_column_negative_indices = column_negative_indices.clamp_min(0)
    row_sampled_distances = distances.gather(2, safe_row_negative_indices)
    column_sampled_distances = column_distance_matrix.gather(2, safe_column_negative_indices)
    row_sampled_distances = row_sampled_distances.masked_fill(~row_negative_valid, torch.inf)
    column_sampled_distances = column_sampled_distances.masked_fill(
        ~column_negative_valid,
        torch.inf,
    )
    row_hard_negative = row_sampled_distances[batch_indices, row_indices].min(dim=1).values
    column_hard_negative = column_sampled_distances[
        column_batch_indices,
        column_indices,
    ].min(dim=1).values
    if hard_triplet_weight > 0 and (
        not bool(torch.isfinite(row_hard_negative).all())
        or not bool(torch.isfinite(column_hard_negative).all())
    ):
        raise ValueError("every matched row and column must contain a valid hard-negative candidate")

    row_triplet = torch.relu(row_positive_distances - row_hard_negative + float(margin)).mean()
    column_triplet = torch.relu(
        column_positive_distances - column_hard_negative + float(margin)
    ).mean()
    hard_triplet_loss = 0.5 * (row_triplet + column_triplet)
    loss = float(hard_triplet_weight) * hard_triplet_loss + float(assignment_weight) * assignment_loss

    positive_distance = torch.cat([row_positive_distances, column_positive_distances]).mean()
    hard_negative_distance = torch.cat([row_hard_negative, column_hard_negative]).mean()
    combined_rank = torch.cat([row_rank, column_rank]).float()
    embedding_std, embedding_norm = _embedding_statistics(
        anchor_embeddings,
        candidate_embeddings,
    )
    stats = {
        "loss": float(loss.detach().cpu()),
        "hard_triplet_loss": float(hard_triplet_loss.detach().cpu()),
        "assignment_loss": float(assignment_loss.detach().cpu()),
        "row_assignment_loss": float(row_assignment_loss.detach().cpu()),
        "column_assignment_loss": float(column_assignment_loss.detach().cpu()),
        "row_top1_acc": float(row_correct.float().mean().detach().cpu()),
        "column_top1_acc": float(column_correct.float().mean().detach().cpu()),
        "bidirectional_top1_acc": float(
            (0.5 * (row_correct.float().mean() + column_correct.float().mean())).detach().cpu()
        ),
        "mutual_top1_acc": float(mutual_correct.float().mean().detach().cpu()),
        "positive_rank": float(combined_rank.mean().detach().cpu()),
        "mrr": float(combined_rank.reciprocal().mean().detach().cpu()),
        "pos_dist": float(positive_distance.detach().cpu()),
        "hard_neg_dist": float(hard_negative_distance.detach().cpu()),
        "margin_gap": float((hard_negative_distance - positive_distance).detach().cpu()),
        "triplet_acc": float(
            (
                0.5
                * (
                    (row_positive_distances < row_hard_negative).float().mean()
                    + (column_positive_distances < column_hard_negative).float().mean()
                )
            )
            .detach()
            .cpu()
        ),
        "matched_edges": float(row_valid.sum(dim=1).float().mean().detach().cpu()),
        "candidates_per_edge": float(
            0.5
            * (
                row_eligible.sum(dim=1).float().mean()
                + column_eligible.sum(dim=1).float().mean()
            ).detach().cpu()
        ),
        "negatives_per_anchor": float(negative_count),
        "temperature": float(temperature),
        "assignment_weight": float(assignment_weight),
        "hard_triplet_weight": float(hard_triplet_weight),
        "embedding_std": float(embedding_std.detach().cpu()),
        "embedding_norm": float(embedding_norm.detach().cpu()),
    }
    return loss, stats
