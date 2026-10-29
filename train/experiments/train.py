"""Finetuning script for XTRBert (ColBERT / XTR-style late interaction)."""

from __future__ import annotations

import argparse
import math
import os

from hf_cache import configure_hf_cache

configure_hf_cache()
import shutil
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from data import (
    BekkoHardNegativeDataset,
    BGEDataset,
    BGEKDDataset,
    JSONLTripletDataset,
    KDCollator,
    KDContrastiveDataset,
    KDDataset,
    LanguageGroupedBatchSampler,
    RetrievalCollator,
    RetrievalDataset,
)
from feather import (
    activation_feather_stats,
    baked_feather_weights,
    enable_feather,
    set_feather_sparsity,
)
from model import (
    XTRBert,
    input_embedding_vocab_size_from_state,
    load_xtrbert_state_dict,
)


# ── scoring ──────────────────────────────────────────────────────────
@dataclass
class SparseBowThresholdStats:
    threshold: float
    sparsity: float
    num_values: int


@dataclass
class SparseBowBatchStats:
    q_active: float
    d_active: float
    q_l1: float
    d_l1: float
    flops: float


@dataclass
class TokenGateBatchStats:
    active: float
    target: float
    length: float
    keep_fraction: float
    gate_mean: float
    gate_min: float
    gate_max: float


def sigreg_loss(
    embeddings: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weak SIGREG: decorrelate embedding dimensions and equalize variance.

    Pushes covariance toward (trace/d)*I — off-diagonals to zero (decorrelation)
    and diagonal to uniform (equal variance per dimension). Both properties
    directly improve product quantization efficiency.
    """
    flat = embeddings.reshape(-1, embeddings.size(-1)).float()
    if mask is not None:
        flat = flat[mask.reshape(-1).bool()]
    flat = flat - flat.mean(dim=0, keepdim=True)
    d = flat.size(1)
    cov = (flat.T @ flat) / max(flat.size(0) - 1, 1)
    mean_var = cov.trace() / d
    target = mean_var * torch.eye(d, device=cov.device, dtype=cov.dtype)
    return ((cov - target) ** 2).sum() / (d * d)


@torch.no_grad()
def positive_activation_threshold_stats(
    tensors: list[torch.Tensor],
    masks: list[torch.Tensor | None],
    sparsity: float,
    *,
    max_threshold_samples: int | None = 2_000_000,
) -> SparseBowThresholdStats:
    """Choose a raw-logit threshold for a target fraction of zero activations."""
    sparsity = max(0.0, min(float(sparsity), 0.9999))
    pieces = []
    total_values = 0
    for tensor, mask in zip(tensors, masks):
        values = tensor.detach().float()
        if mask is not None:
            values = values.reshape(-1, values.size(-1))[mask.reshape(-1).bool()]
        values = values.reshape(-1).cpu()
        if values.numel() == 0:
            continue
        pieces.append(values)
        total_values += values.numel()

    if not pieces:
        return SparseBowThresholdStats(threshold=0.0, sparsity=0.0, num_values=0)
    if sparsity <= 0.0:
        return SparseBowThresholdStats(threshold=0.0, sparsity=0.0, num_values=total_values)

    if max_threshold_samples is None or total_values <= max_threshold_samples:
        sample = torch.cat(pieces)
    else:
        sampled = []
        for values in pieces:
            take = max(1, int(max_threshold_samples * values.numel() / total_values))
            if take >= values.numel():
                sampled.append(values)
            else:
                idx = torch.linspace(
                    0,
                    values.numel() - 1,
                    steps=take,
                    device=values.device,
                    dtype=torch.long,
                )
                sampled.append(values.index_select(0, idx))
        sample = torch.cat(sampled)

    threshold = sample.quantile(sparsity)
    sparse = sum(int((values <= threshold).sum().item()) for values in pieces)
    return SparseBowThresholdStats(
        threshold=float(threshold.detach().cpu()),
        sparsity=sparse / max(total_values, 1),
        num_values=total_values,
    )


@torch.no_grad()
def pooled_sparse_bow_threshold_stats(
    raw_tensors: list[torch.Tensor],
    masks: list[torch.Tensor],
    sparsity: float,
    *,
    pool: str = "max",
    max_threshold_samples: int | None = 2_000_000,
) -> SparseBowThresholdStats:
    """Choose a raw-logit threshold that sparsifies the pooled BoW vectors."""
    sparsity = max(0.0, min(float(sparsity), 0.9999))
    pieces = []
    total_values = 0
    for raw, mask in zip(raw_tensors, masks):
        values = raw.detach().float().masked_fill(~mask.unsqueeze(-1).bool(), -float("inf"))
        if pool == "max":
            pooled = values.max(dim=1).values
        elif pool == "sum":
            pooled = values.masked_fill(~torch.isfinite(values), 0.0).sum(dim=1)
        else:
            raise ValueError(f"unknown sparse BoW pooling mode: {pool}")
        pooled = pooled[torch.isfinite(pooled)].reshape(-1).cpu()
        if pooled.numel() == 0:
            continue
        pieces.append(pooled)
        total_values += pooled.numel()

    if not pieces:
        return SparseBowThresholdStats(threshold=0.0, sparsity=0.0, num_values=0)
    if sparsity <= 0.0:
        return SparseBowThresholdStats(threshold=0.0, sparsity=0.0, num_values=total_values)

    if max_threshold_samples is None or total_values <= max_threshold_samples:
        sample = torch.cat(pieces)
    else:
        sampled = []
        for values in pieces:
            take = max(1, int(max_threshold_samples * values.numel() / total_values))
            if take >= values.numel():
                sampled.append(values)
            else:
                idx = torch.linspace(
                    0,
                    values.numel() - 1,
                    steps=take,
                    dtype=torch.long,
                )
                sampled.append(values.index_select(0, idx))
        sample = torch.cat(sampled)

    threshold = sample.quantile(sparsity)
    sparse = sum(int((values <= threshold).sum().item()) for values in pieces)
    return SparseBowThresholdStats(
        threshold=float(threshold.detach().cpu()),
        sparsity=sparse / max(total_values, 1),
        num_values=total_values,
    )


def sparse_bow_contrastive_loss(
    query_vecs: torch.Tensor,
    doc_vecs: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    logits = query_vecs.float().matmul(doc_vecs.float().T) / temperature
    labels = torch.arange(query_vecs.size(0), device=query_vecs.device)
    return F.cross_entropy(logits, labels)


def sparse_bow_regularization(
    query_vecs: torch.Tensor,
    doc_vecs: torch.Tensor,
    *,
    l1_weight: float = 0.0,
    flops_weight: float = 0.0,
) -> tuple[torch.Tensor, SparseBowBatchStats]:
    q = query_vecs.float()
    d = doc_vecs.float()
    q_l1 = q.sum(dim=-1).mean()
    d_l1 = d.sum(dim=-1).mean()
    q_mean = q.mean(dim=0)
    d_mean = d.mean(dim=0)
    flops = q_mean.square().sum() + d_mean.square().sum()
    loss = q.new_tensor(0.0)
    if l1_weight > 0:
        loss = loss + l1_weight * (q_l1 + d_l1)
    if flops_weight > 0:
        loss = loss + flops_weight * flops
    stats = SparseBowBatchStats(
        q_active=float(q.gt(0).sum(dim=-1).float().mean().detach().cpu()),
        d_active=float(d.gt(0).sum(dim=-1).float().mean().detach().cpu()),
        q_l1=float(q_l1.detach().cpu()),
        d_l1=float(d_l1.detach().cpu()),
        flops=float(flops.detach().cpu()),
    )
    return loss, stats


def token_gate_budget_regularization(
    gate_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_mode: str = "sqrt",
    target_factor: float = 1.0,
    target_fraction: float = 0.25,
    min_keep: float = 1.0,
    progress: float = 1.0,
) -> tuple[torch.Tensor, TokenGateBatchStats]:
    """Keep expected active document tokens near a scheduled budget."""
    mask = mask.bool()
    probs = torch.sigmoid(gate_logits.float()).masked_fill(~mask, 0.0)
    lengths = mask.sum(dim=-1).float().clamp(min=1.0)

    if target_mode == "sqrt":
        final_target = target_factor * torch.sqrt(lengths)
    elif target_mode == "fraction":
        final_target = target_fraction * lengths
    elif target_mode == "fixed":
        final_target = torch.full_like(lengths, float(target_factor))
    else:
        raise ValueError(f"unknown token_gate target_mode: {target_mode}")
    final_target = torch.minimum(final_target.clamp_min(float(min_keep)), lengths)

    progress = max(0.0, min(float(progress), 1.0))
    target = lengths * (1.0 - progress) + final_target * progress
    active = probs.sum(dim=-1)
    # Normalizing by length keeps long documents from dominating the regularizer.
    loss = ((active - target) / lengths).square().mean()

    valid_probs = probs[mask]
    if valid_probs.numel() == 0:
        gate_mean = gate_min = gate_max = 0.0
    else:
        gate_mean = float(valid_probs.mean().detach().cpu())
        gate_min = float(valid_probs.min().detach().cpu())
        gate_max = float(valid_probs.max().detach().cpu())
    stats = TokenGateBatchStats(
        active=float(active.mean().detach().cpu()),
        target=float(target.mean().detach().cpu()),
        length=float(lengths.mean().detach().cpu()),
        keep_fraction=float((active / lengths).mean().detach().cpu()),
        gate_mean=gate_mean,
        gate_min=gate_min,
        gate_max=gate_max,
    )
    return loss, stats


def token_gate_target_counts(
    mask: torch.Tensor,
    *,
    target_mode: str = "sqrt",
    target_factor: float = 1.0,
    target_fraction: float = 0.25,
    min_keep: float = 1.0,
    progress: float = 1.0,
    prefix_factor: float = 1.0,
) -> torch.Tensor:
    mask = mask.bool()
    lengths = mask.sum(dim=-1).float().clamp(min=1.0)

    if target_mode == "sqrt":
        final_target = target_factor * torch.sqrt(lengths)
    elif target_mode == "fraction":
        final_target = target_fraction * lengths
    elif target_mode == "fixed":
        final_target = torch.full_like(lengths, float(target_factor))
    else:
        raise ValueError(f"unknown token_gate target_mode: {target_mode}")

    final_target = final_target * float(prefix_factor)
    final_target = torch.minimum(final_target.clamp_min(float(min_keep)), lengths)

    progress = max(0.0, min(float(progress), 1.0))
    return lengths * (1.0 - progress) + final_target * progress


def token_gate_prefix_mask(
    gate_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_mode: str = "sqrt",
    target_factor: float = 1.0,
    target_fraction: float = 0.25,
    min_keep: float = 1.0,
    prefix_factor: float = 1.0,
) -> torch.Tensor:
    """Hard keep-mask for a prefix of the current learned token ordering."""
    mask = mask.bool()
    keep_counts = token_gate_target_counts(
        mask,
        target_mode=target_mode,
        target_factor=target_factor,
        target_fraction=target_fraction,
        min_keep=min_keep,
        progress=1.0,
        prefix_factor=prefix_factor,
    ).ceil().long()

    out = torch.zeros_like(mask, dtype=torch.bool)
    scores = gate_logits.float().masked_fill(~mask, -float("inf"))
    for row in range(scores.size(0)):
        keep = int(keep_counts[row].item())
        available = int(mask[row].sum().item())
        if keep <= 0 or available <= 0:
            continue
        keep = min(keep, available)
        idx = scores[row].topk(keep).indices
        out[row, idx] = True
    return out & mask


def tokenizer_word_start_flags(tokenizer) -> torch.Tensor:
    """Return a vocab-sized bool tensor marking tokens that start lexical groups."""
    vocab_size = len(tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    special_ids = set(int(idx) for idx in getattr(tokenizer, "all_special_ids", []) if idx is not None)
    has_wordpiece_continuations = any(
        isinstance(token, str) and token.startswith("##") for token in tokens
    )
    has_space_markers = any(
        isinstance(token, str) and (token.startswith("Ġ") or token.startswith("▁"))
        for token in tokens
    )

    starts = []
    for idx, token in enumerate(tokens):
        if idx in special_ids or not isinstance(token, str) or token == "":
            starts.append(True)
        elif has_space_markers:
            starts.append(token.startswith("Ġ") or token.startswith("▁"))
        elif has_wordpiece_continuations:
            starts.append(not token.startswith("##"))
        else:
            starts.append(True)
    return torch.tensor(starts, dtype=torch.bool)


def token_gate_wordpiece_consistency_regularization(
    gate_logits: torch.Tensor,
    mask: torch.Tensor,
    word_start: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Encourage continuation wordpieces to receive the previous piece's gate."""
    if gate_logits.shape != word_start.shape or gate_logits.shape != mask.shape:
        raise ValueError(
            "wordpiece gate regularization expects matching gate/start/mask shapes: "
            f"{tuple(gate_logits.shape)}, {tuple(word_start.shape)}, {tuple(mask.shape)}"
        )
    if gate_logits.size(1) <= 1:
        return gate_logits.float().new_zeros(()), 0

    mask = mask.bool()
    word_start = word_start.to(device=gate_logits.device, dtype=torch.bool)
    pair_mask = mask[:, 1:] & mask[:, :-1] & ~word_start[:, 1:]
    pair_count = int(pair_mask.sum().detach().cpu())
    if pair_count == 0:
        return gate_logits.float().new_zeros(()), 0

    delta = gate_logits.float()[:, 1:] - gate_logits.float()[:, :-1]
    loss = F.smooth_l1_loss(delta[pair_mask], torch.zeros_like(delta[pair_mask]))
    return loss, pair_count


def token_gate_predictability_regularization(
    gate_logits: torch.Tensor,
    token_surprise: torch.Tensor,
    surprise_mask: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Penalize high gate probability on low-surprise, predictable tokens."""
    if gate_logits.shape != token_surprise.shape or gate_logits.shape != mask.shape:
        raise ValueError(
            "predictability gate regularization expects matching shapes: "
            f"{tuple(gate_logits.shape)}, {tuple(token_surprise.shape)}, {tuple(mask.shape)}"
        )
    valid = mask.bool() & surprise_mask.bool() & torch.isfinite(token_surprise)
    valid_count = int(valid.sum().detach().cpu())
    total_count = int(mask.bool().sum().detach().cpu())
    if valid_count == 0:
        return gate_logits.float().new_zeros(()), 0.0, 0.0

    surprise = token_surprise.float()
    row_min = surprise.masked_fill(~valid, float("inf")).amin(dim=1, keepdim=True)
    row_max = surprise.masked_fill(~valid, -float("inf")).amax(dim=1, keepdim=True)
    span = (row_max - row_min).clamp_min(1e-6)
    predictability = ((row_max - surprise) / span).clamp(0.0, 1.0)
    gate_probs = torch.sigmoid(gate_logits.float())
    loss = (gate_probs * predictability)[valid].mean()
    coverage = valid_count / max(total_count, 1)
    pred_mean = float(predictability[valid].mean().detach().cpu())
    return loss, coverage, pred_mean


def token_gate_maxsim_target_regularization(
    query_embs: torch.Tensor,
    q_mask: torch.Tensor,
    doc_embs: torch.Tensor,
    d_mask: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    target_mode: str = "sqrt",
    target_factor: float = 1.0,
    target_fraction: float = 0.25,
    min_keep: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Train gate ordering against tokens selected by paired-query MaxSim.

    The first B documents in the contrastive batch are the positives paired
    with the B queries.  We mark the top budgeted document tokens by full-token
    MaxSim salience as positives and train the gate with BCE.  Salience labels
    are detached: this is a gate-ordering target, not an embedding objective.
    """
    B = min(query_embs.size(0), doc_embs.size(0), gate_logits.size(0))
    if B == 0:
        return gate_logits.float().new_zeros(()), 0.0

    q_mask = q_mask[:B].bool()
    d_mask = d_mask[:B].bool()
    gates = gate_logits[:B]
    valid = d_mask & torch.isfinite(gates)
    if int(valid.sum().detach().cpu()) == 0:
        return gate_logits.float().new_zeros(()), 0.0

    with torch.no_grad():
        sim = torch.einsum(
            "bnd,bmd->bnm",
            query_embs[:B].detach().float(),
            doc_embs[:B].detach().float(),
        )
        sim = sim.masked_fill(~q_mask.unsqueeze(-1), -float("inf"))
        sim = sim.masked_fill(~d_mask.unsqueeze(1), -float("inf"))
        salience = sim.amax(dim=1).masked_fill(~d_mask, -float("inf"))
        keep_counts = token_gate_target_counts(
            d_mask,
            target_mode=target_mode,
            target_factor=target_factor,
            target_fraction=target_fraction,
            min_keep=min_keep,
            progress=1.0,
        ).ceil().long()
        labels = torch.zeros_like(gates, dtype=torch.float32)
        for row in range(B):
            available = int(d_mask[row].sum().item())
            if available <= 0:
                continue
            keep = min(int(keep_counts[row].item()), available)
            if keep <= 0:
                continue
            idx = salience[row].topk(keep).indices
            labels[row, idx] = 1.0

    logits = gates.float()
    loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid])
    positive_rate = float(labels[valid].mean().detach().cpu())
    return loss, positive_rate


def select_masked_sequence(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack selected sequence positions per row into a shorter padded tensor."""
    mask = mask.bool()
    lengths = mask.sum(dim=1)
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    if max_len == 0:
        max_len = 1
    out = values.new_zeros((values.size(0), max_len, values.size(-1)))
    out_mask = torch.zeros(
        (values.size(0), max_len),
        device=values.device,
        dtype=mask.dtype,
    )
    for row in range(values.size(0)):
        selected = values[row][mask[row]]
        count = selected.size(0)
        if count == 0:
            continue
        out[row, :count] = selected
        out_mask[row, :count] = True
    return out, out_mask.long()


def _xtr_topk_scores(
    sim: torch.Tensor,
    flat_dmask: torch.Tensor | None,
    q_mask: torch.Tensor | None,
    num_docs: int,
    doc_tokens: int,
    k_train: int,
    z_normalize: bool = True,
    z_mode: str = "retrieved",
    score_relu: bool = False,
    return_z: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute XTR scores from token similarities.

    ``sim`` has shape ``(B, n, num_docs * doc_tokens)``. ``flat_dmask`` may be
    either global ``(num_docs * doc_tokens,)`` or per-query
    ``(B, num_docs * doc_tokens)``. Top-k padding hits are explicitly ignored
    when building the retrieval-count normalizer.
    """
    if flat_dmask is not None:
        flat_dmask = flat_dmask.bool()
        if flat_dmask.dim() == 1:
            real_token_counts = flat_dmask.view(num_docs, doc_tokens).sum(dim=-1).to(sim.dtype).view(1, num_docs)
            sim = sim.masked_fill(~flat_dmask.view(1, 1, -1), torch.finfo(sim.dtype).min)
        elif flat_dmask.dim() == 2:
            real_token_counts = flat_dmask.view(sim.size(0), num_docs, doc_tokens).sum(dim=-1).to(sim.dtype)
            sim = sim.masked_fill(~flat_dmask.unsqueeze(1), torch.finfo(sim.dtype).min)
        else:
            raise ValueError("flat_dmask must be 1D or 2D")
    else:
        real_token_counts = torch.full((1, num_docs), doc_tokens, device=sim.device, dtype=sim.dtype)

    B, n, _ = sim.shape

    # If k covers the complete token pool, XTR's alignment mask is dense over
    # real tokens, so the score reduces to exact MaxSim without zero competitors.
    if k_train >= sim.size(-1):
        max_sim = sim.view(B, n, num_docs, doc_tokens).max(dim=-1).values
        has_real_tokens = real_token_counts.gt(0).unsqueeze(1)
        max_sim = torch.where(has_real_tokens, max_sim, torch.zeros_like(max_sim))

        if q_mask is not None:
            max_sim = max_sim * q_mask.unsqueeze(-1).float()

        score_sum = max_sim.sum(dim=1)

        if z_normalize:
            if z_mode == "retrieved":
                z_mask = has_real_tokens.expand(B, n, num_docs).float()
                if q_mask is not None:
                    z_mask = z_mask * q_mask.unsqueeze(-1).float()
            elif z_mode == "positive":
                z_mask = max_sim.gt(0).float()
            else:
                raise ValueError(f"unknown XTR z_mode: {z_mode}")

            Z = z_mask.sum(dim=1)
            doc_scores = torch.where(
                Z > 0,
                score_sum / Z.clamp(min=1e-3),
                torch.zeros_like(score_sum),
            )
        else:
            if q_mask is not None:
                Z = q_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
            else:
                Z = torch.full((B, 1), n, device=sim.device, dtype=sim.dtype)
            doc_scores = score_sum / Z

        if score_relu:
            doc_scores = doc_scores.clamp_min(0)

        if return_z:
            return doc_scores, Z
        return doc_scores

    topk_vals, topk_idx = sim.topk(min(k_train, sim.size(-1)), dim=-1)
    doc_idx = topk_idx // doc_tokens

    if flat_dmask is None:
        topk_valid = torch.ones_like(topk_idx, dtype=torch.bool)
    elif flat_dmask.dim() == 1:
        topk_valid = flat_dmask[topk_idx]
    else:
        expanded_mask = flat_dmask.unsqueeze(1).expand(-1, sim.size(1), -1)
        topk_valid = expanded_mask.gather(2, topk_idx)

    safe_doc_idx = doc_idx.masked_fill(~topk_valid, 0)
    safe_topk_vals = topk_vals.masked_fill(~topk_valid, -float("inf"))

    B, n, _ = sim.shape
    max_sim = torch.full((B, n, num_docs), -float("inf"), device=sim.device, dtype=sim.dtype)
    max_sim.scatter_reduce_(2, safe_doc_idx, safe_topk_vals, reduce="amax", include_self=False)

    selected_counts = torch.zeros((B, n, num_docs), device=sim.device, dtype=sim.dtype)
    selected_counts.scatter_add_(2, safe_doc_idx, topk_valid.to(sim.dtype))

    # XTR Eq. 2 multiplies similarities by the binary top-k alignment mask, so
    # non-selected real token slots contribute zero before the per-document max.
    # Padding is not a token slot.
    has_zero_competitor = selected_counts < real_token_counts.unsqueeze(1)
    max_sim = torch.where(has_zero_competitor, max_sim.clamp_min(0), max_sim)
    max_sim = torch.where(real_token_counts.gt(0).unsqueeze(1), max_sim, torch.zeros_like(max_sim))

    if q_mask is not None:
        max_sim = max_sim * q_mask.unsqueeze(-1).float()

    score_sum = max_sim.sum(dim=1)

    if z_normalize:
        if z_mode == "retrieved":
            z_mask = selected_counts.gt(0).float()
            if q_mask is not None:
                z_mask = z_mask * q_mask.unsqueeze(-1).float()
        elif z_mode == "positive":
            z_mask = max_sim.gt(0).float()
        else:
            raise ValueError(f"unknown XTR z_mode: {z_mode}")

        Z = z_mask.sum(dim=1)
        doc_scores = torch.where(
            Z > 0,
            score_sum / Z.clamp(min=1e-3),
            torch.zeros_like(score_sum),
        )
    else:
        if q_mask is not None:
            Z = q_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        else:
            Z = torch.full((B, 1), n, device=sim.device, dtype=sim.dtype)
        doc_scores = score_sum / Z

    if score_relu:
        doc_scores = doc_scores.clamp_min(0)

    if return_z:
        return doc_scores, Z
    return doc_scores


def contrastive_loss(
    query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
    q_mask: torch.Tensor,
    d_mask: torch.Tensor,
    temperature: float = 1.0,
    k_train: int = 64,
    use_maxsim: bool = False,
    z_normalize: bool = True,
    xtr_score_relu: bool = False,
    doc_gate_logits: torch.Tensor | None = None,
    doc_gate_score_scale: float = 0.0,
) -> torch.Tensor:
    """In-batch InfoNCE loss with XTR or MaxSim scoring.

    Fully batched — no Python loop over queries.

    XTR mode (default) follows the XTR paper (Eq. 2): top-k retrieval
    across the batch token pool, Z-normalization (only query tokens that
    matched ≥1 token from the scored document), no imputation.  MSE is
    inference-only per the paper.

    MaxSim mode is exact ColBERT scoring (Eq. 1): max over doc tokens
    per document, mean over all query tokens.
    """
    B, n, d = query_embs.shape
    _, m, _ = doc_embs.shape
    Bd = doc_embs.size(0)

    flat_doc = doc_embs.reshape(Bd * m, d)
    sim = torch.einsum("bnd,md->bnm", query_embs, flat_doc)

    if doc_gate_logits is not None and doc_gate_score_scale != 0.0:
        if tuple(doc_gate_logits.shape) != (Bd, m):
            raise ValueError(
                f"doc_gate_logits shape {tuple(doc_gate_logits.shape)} != {(Bd, m)}"
            )
        gate_bias = (
            float(doc_gate_score_scale)
            * F.logsigmoid(doc_gate_logits.float()).reshape(Bd * m)
        ).to(sim.dtype)
        sim = sim + gate_bias.view(1, 1, -1)

    if d_mask is not None:
        flat_dmask = d_mask.reshape(Bd * m).bool()
        sim = sim.masked_fill(
            ~flat_dmask.unsqueeze(0).unsqueeze(0),
            torch.finfo(sim.dtype).min,
        )

    if use_maxsim:
        # Exact MaxSim (ColBERT Eq. 1): max over doc tokens, mean over query tokens
        sim = sim.view(B, n, Bd, m)
        max_sim = sim.max(dim=-1).values  # (B, n, Bd)

        if q_mask is not None:
            max_sim = max_sim * q_mask.unsqueeze(-1).float()
            nq = q_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        else:
            nq = n

        doc_scores = max_sim.sum(dim=1) / nq  # (B, Bd)
    else:
        # XTR (Eq. 2): top-k retrieval + scatter MaxSim + Z-normalization.
        doc_scores = _xtr_topk_scores(
            sim,
            flat_dmask if d_mask is not None else None,
            q_mask,
            num_docs=Bd,
            doc_tokens=m,
            k_train=k_train,
            z_normalize=z_normalize,
            score_relu=xtr_score_relu,
        )

    doc_scores = doc_scores / temperature

    labels = torch.arange(B, device=query_embs.device)
    return F.cross_entropy(doc_scores, labels)


def kd_loss(
    query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
    q_mask: torch.Tensor,
    d_mask: torch.Tensor,
    teacher_scores: torch.Tensor,
    doc_counts: torch.Tensor,
    kd_temperature: float = 1.0,
    kd_student_temperature: float | None = None,
    k_train: int = 0,
    xtr_score_relu: bool = False,
    doc_gate_logits: torch.Tensor | None = None,
    doc_gate_score_scale: float = 0.0,
) -> torch.Tensor:
    """KL divergence between student scores and teacher score distribution.

    When k_train > 0, uses XTR-style top-k scoring (matching inference-time
    approximation) instead of exact MaxSim.  The XTR paper showed +2pp from
    this train/inference consistency.

    query_embs: (B, n, d)
    doc_embs: (B*C, m, d)  where C = num_candidates (constant)
    teacher_scores: (B*C,) flat teacher scores
    doc_counts: (B,) number of candidates per query
    """
    B, n, d = query_embs.shape
    C = doc_counts[0].item()
    m = doc_embs.shape[1]

    # Reshape docs to (B, C, m, d) for per-query scoring
    doc_grouped = doc_embs.view(B, C, m, d)

    if k_train > 0:
        # XTR Eq. 2 applies top-k over all document tokens in the mini-batch.
        # For KD, score the full B*C document pool, then slice each query's
        # own C teacher-scored candidates for the KL loss.
        flat_doc = doc_embs.reshape(B * C * m, d)
        sim = torch.einsum("bnd,md->bnm", query_embs, flat_doc)

        if doc_gate_logits is not None and doc_gate_score_scale != 0.0:
            if tuple(doc_gate_logits.shape) != (B * C, m):
                raise ValueError(
                    f"doc_gate_logits shape {tuple(doc_gate_logits.shape)} != {(B * C, m)}"
                )
            gate_bias = (
                float(doc_gate_score_scale)
                * F.logsigmoid(doc_gate_logits.float()).reshape(B * C * m)
            ).to(sim.dtype)
            sim = sim + gate_bias.view(1, 1, -1)

        # Mask padding doc tokens
        flat_dmask = None
        if d_mask is not None:
            flat_dmask = d_mask.reshape(B * C * m).bool()
            sim = sim.masked_fill(
                ~flat_dmask.unsqueeze(0).unsqueeze(0),
                torch.finfo(sim.dtype).min,
            )

        all_scores = _xtr_topk_scores(
            sim,
            flat_dmask,
            q_mask,
            num_docs=B * C,
            doc_tokens=m,
            k_train=k_train,
            score_relu=xtr_score_relu,
        )
        idx = (
            torch.arange(B, device=query_embs.device).unsqueeze(1) * C
            + torch.arange(C, device=query_embs.device)
        )
        student_scores = all_scores.gather(1, idx)
    else:
        # Exact MaxSim
        sim = torch.einsum("bnd,bcmd->bncm", query_embs, doc_grouped)

        if doc_gate_logits is not None and doc_gate_score_scale != 0.0:
            if tuple(doc_gate_logits.shape) != (B * C, m):
                raise ValueError(
                    f"doc_gate_logits shape {tuple(doc_gate_logits.shape)} != {(B * C, m)}"
                )
            gate_bias = (
                float(doc_gate_score_scale)
                * F.logsigmoid(doc_gate_logits.float()).view(B, C, m)
            ).to(sim.dtype)
            sim = sim + gate_bias.unsqueeze(1)

        if d_mask is not None:
            dmask_grouped = d_mask.view(B, C, m).bool()
            sim = sim.masked_fill(~dmask_grouped.unsqueeze(1), -float("inf"))

        max_sim = sim.max(dim=-1).values  # (B, n, C)

        if q_mask is not None:
            max_sim = max_sim * q_mask.unsqueeze(-1).float()
            nq = q_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        else:
            nq = n
        student_scores = max_sim.sum(dim=1) / nq  # (B, C)

    teacher = teacher_scores.view(B, C)

    # KL divergence: match student distribution to teacher distribution.
    # If kd_student_temperature is set, we intentionally reproduce the
    # asymmetric "old KD" runs where only the student scores were sharpened.
    student_temperature = kd_student_temperature or kd_temperature
    teacher_log_probs = F.log_softmax(teacher / kd_temperature, dim=-1)
    student_log_probs = F.log_softmax(student_scores / student_temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction="batchmean")
    return loss * (kd_temperature ** 2)


def _launch_eval(ckpt_path: str, model_name: str, dim: int) -> None:
    """Fork a background CPU eval on NFCorpus, log to output_dir (not ckpt dir)."""
    eval_script = os.path.join(os.path.dirname(__file__), "eval_nfcorpus.py")
    if not os.path.exists(eval_script):
        return
    # Write eval log next to the checkpoint dir, not inside it,
    # so checkpoint cleanup doesn't kill a running eval.
    ckpt_name = os.path.basename(ckpt_path.rstrip("/"))
    output_dir = os.path.dirname(ckpt_path.rstrip("/"))
    log_path = os.path.join(output_dir, f"eval_{ckpt_name}.txt")
    log_file = open(log_path, "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.Popen(
        [sys.executable, eval_script, "--model", model_name,
         "--dim", str(dim), "--checkpoint", ckpt_path],
        stdout=log_file, stderr=subprocess.STDOUT,
        env=env,
    )
    print(f"  eval forked → {log_path}")


def initialize_projection(model: XTRBert, mode: str) -> None:
    if mode not in {"first_dims", "identity_slice"}:
        raise ValueError(
            "projection_init must be one of: first_dims, identity_slice"
        )
    if not isinstance(model.linear, torch.nn.Linear):
        raise ValueError("projection_init requires a linear projection head")
    out_dim, hidden_dim = model.linear.weight.shape
    if out_dim > hidden_dim:
        raise ValueError(
            f"cannot initialize {out_dim} projection dims from {hidden_dim} hidden dims"
        )
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
        idx = torch.arange(out_dim, device=model.linear.weight.device)
        model.linear.weight[idx, idx] = 1.0
    print(f"projection initialized from first {out_dim} hidden dimensions")


def initialize_head_from_checkpoint(
    model: XTRBert,
    checkpoint_dir: str,
    *,
    copy_gate: bool,
    token_gate_init_bias: float,
) -> None:
    ckpt = os.path.join(checkpoint_dir, "xtrbert.pt")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)

    if "linear.weight" not in state:
        raise ValueError(f"{ckpt} does not contain linear.weight")
    if tuple(state["linear.weight"].shape) != tuple(model.linear.weight.shape):
        raise ValueError(
            "init_head_from linear.weight shape mismatch: "
            f"{tuple(state['linear.weight'].shape)} != {tuple(model.linear.weight.shape)}"
        )
    with torch.no_grad():
        model.linear.weight.copy_(state["linear.weight"])
        if "linear.bias" in state:
            if tuple(state["linear.bias"].shape) != tuple(model.linear.bias.shape):
                raise ValueError(
                    "init_head_from linear.bias shape mismatch: "
                    f"{tuple(state['linear.bias'].shape)} != {tuple(model.linear.bias.shape)}"
                )
            model.linear.bias.copy_(state["linear.bias"])
        else:
            model.linear.bias.zero_()

    copied = ["linear"]
    gate_keys = {"token_gate.weight", "token_gate.bias", "token_gate_norm.weight"}
    has_gate = gate_keys.issubset(state)
    if copy_gate and has_gate:
        if model.token_gate is None:
            model.add_token_gate(init_bias=token_gate_init_bias)
        with torch.no_grad():
            model_state = model.state_dict()
            for key in gate_keys | {"token_gate_norm.bias"}:
                if key not in state:
                    continue
                target = model_state.get(key)
                if target is None:
                    raise ValueError(f"target model does not contain {key}")
                if tuple(state[key].shape) != tuple(target.shape):
                    raise ValueError(
                        f"init_head_from {key} shape mismatch: "
                        f"{tuple(state[key].shape)} != {tuple(target.shape)}"
                    )
                target.copy_(state[key])
        copied.append("token_gate")
    elif copy_gate:
        print(f"init_head_from: {ckpt} has no complete token gate; copied projection only")

    print(f"initialized {' + '.join(copied)} from {ckpt}")


# ── device ───────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model: XTRBert, cfg: dict, head_lr_override: float | None = None) -> torch.optim.Optimizer:
    """Build AdamW with separate encoder/head learning rates and sane decay."""
    head_lr = head_lr_override or cfg.get("projection_lr", cfg["lr"])
    encoder_lr = cfg.get("encoder_lr", cfg["lr"])
    weight_decay = cfg.get("weight_decay", 0.01)

    grouped: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    counts: dict[tuple[float, float], int] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr = encoder_lr if name.startswith("encoder.") else head_lr
        lname = name.lower()
        wd = 0.0 if param.ndim <= 1 or lname.endswith(".bias") or "norm" in lname else weight_decay
        key = (lr, wd)
        grouped.setdefault(key, []).append(param)
        counts[key] = counts.get(key, 0) + param.numel()

    if not grouped:
        raise ValueError("no trainable parameters found")

    optimizer_groups = []
    for (lr, wd), params in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        print(f"optim group: lr={lr:.2e} wd={wd:.2e} params={counts[(lr, wd)]:,}")
        optimizer_groups.append({"params": params, "lr": lr, "weight_decay": wd})

    return torch.optim.AdamW(optimizer_groups)


def save_training_checkpoint(
    model: XTRBert,
    tokenizer,
    path: str,
    *,
    bake_feather: bool = False,
) -> None:
    ctx = baked_feather_weights(model) if bake_feather else nullcontext()
    with ctx:
        model.save_pretrained(path)
    tokenizer.save_pretrained(path)


# ── training ─────────────────────────────────────────────────────────
def train(cfg: dict) -> None:
    device = get_device()
    print(f"device: {device}")

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("tokenizer", model_name), trust_remote_code=True)
    doc_anchor_token = cfg.get("doc_anchor_tokens", cfg.get("doc_anchor_token"))
    doc_anchor_mode = cfg.get("doc_anchor_mode", "sentence")
    doc_anchor_context_sentences = int(cfg.get("doc_anchor_context_sentences", 0))
    doc_anchor_placement = cfg.get("doc_anchor_placement", "prefix")
    token_gate_raw_cfg = cfg.get("token_gate", False)
    token_gate_enabled = bool(token_gate_raw_cfg)
    token_gate_cfg = token_gate_raw_cfg if isinstance(token_gate_raw_cfg, dict) else {}
    token_gate_init_bias = float(token_gate_cfg.get("init_bias", 2.944))
    token_gate_wordpiece_boundaries = str(
        token_gate_cfg.get("wordpiece_boundaries", "vocab")
    )
    if token_gate_wordpiece_boundaries not in {"vocab", "offsets"}:
        raise ValueError(
            "token_gate.wordpiece_boundaries must be 'vocab' or 'offsets', got "
            f"{token_gate_wordpiece_boundaries!r}"
        )
    token_gate_predictability_cache = (
        token_gate_cfg.get("predictability_cache")
        if token_gate_enabled else None
    )
    doc_anchor_tokens = None
    doc_anchor_token_ids: list[int] = []
    if doc_anchor_token:
        doc_anchor_tokens = (
            [str(token) for token in doc_anchor_token]
            if isinstance(doc_anchor_token, list)
            else [str(doc_anchor_token)]
        )
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": doc_anchor_tokens}
        )
        doc_anchor_token_ids = [
            int(tokenizer.convert_tokens_to_ids(token))
            for token in doc_anchor_tokens
        ]
        doc_anchor_token = doc_anchor_tokens if len(doc_anchor_tokens) > 1 else doc_anchor_tokens[0]
        token_desc = ", ".join(
            f"{token}:{token_id}"
            for token, token_id in zip(doc_anchor_tokens, doc_anchor_token_ids)
        )
        print(
            "document anchors enabled: "
            f"tokens=[{token_desc}] mode={doc_anchor_mode} "
            f"context_sentences={doc_anchor_context_sentences} "
            f"placement={doc_anchor_placement} added={added}"
        )

    if cfg.get("from_colmodernvbert", False):
        print("loading ColModernVBert text encoder + LoRA merge...")
        model = XTRBert.from_colmodernvbert()
    else:
        model = XTRBert(
            model_name_or_path=model_name,
            dim=cfg.get("dim", 128),
            hash_buckets=cfg.get("hash_buckets"),
            num_hashes=cfg.get("num_hashes", 2),
        )
    if model.hash_embedding is not None:
        n = model.hash_embedding.num_buckets
        k = model.hash_embedding.num_hashes
        params = sum(p.numel() for p in model.hash_embedding.parameters())
        print(f"hash embeddings: {n} buckets x {k} hashes ({params:,} params)")
    if cfg.get("pretrained_projection"):
        if cfg.get("projection_init") or cfg.get("init_head_from"):
            raise ValueError(
                "pretrained_projection, projection_init, and init_head_from are mutually exclusive"
            )
        from huggingface_hub import hf_hub_download
        proj_path = hf_hub_download(
            repo_id=cfg["pretrained_projection"],
            filename="2_Dense/pytorch_model.bin",
        )
        proj_state = torch.load(proj_path, map_location="cpu", weights_only=True)
        model.linear.weight.data = proj_state["linear.weight"]
        if "linear.bias" in proj_state:
            model.linear.bias.data = proj_state["linear.bias"]
        else:
            model.linear.bias.data.zero_()
        print(f"loaded pretrained projection from {cfg['pretrained_projection']}")
    if cfg.get("projection_init"):
        if cfg.get("resume"):
            raise ValueError("projection_init should not be used together with resume")
        if cfg.get("init_head_from"):
            raise ValueError("projection_init and init_head_from are mutually exclusive")
        initialize_projection(model, str(cfg["projection_init"]))
    if cfg.get("init_head_from"):
        if cfg.get("resume"):
            raise ValueError("init_head_from should not be used together with resume")
        initialize_head_from_checkpoint(
            model,
            str(cfg["init_head_from"]),
            copy_gate=bool(cfg.get("init_head_copy_gate", True)),
            token_gate_init_bias=token_gate_init_bias,
        )
    if cfg.get("resume"):
        ckpt = os.path.join(cfg["resume"], "xtrbert.pt")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if any(key.startswith("token_gate") for key in state):
            model.add_token_gate(init_bias=token_gate_init_bias)
        state_vocab = input_embedding_vocab_size_from_state(model, state)
        if state_vocab is not None:
            old_vocab, new_vocab = model.resize_token_embeddings(state_vocab)
            if old_vocab != new_vocab:
                print(f"resized token embeddings before resume: {old_vocab} -> {new_vocab}")
        load_xtrbert_state_dict(
            model,
            state,
            ignore_mismatched_shapes=bool(cfg.get("ignore_mismatched_resume_shapes", False)),
        )
        print(f"resumed from {ckpt}")
    if token_gate_enabled and model.token_gate is None:
        model.add_token_gate(init_bias=token_gate_init_bias)
    if doc_anchor_token:
        current_vocab = model.encoder.get_input_embeddings().num_embeddings
        init_from_token_id = tokenizer.cls_token_id
        if doc_anchor_token_ids and doc_anchor_token_ids[0] < current_vocab:
            init_from_token_id = doc_anchor_token_ids[0]
        old_vocab, new_vocab = model.resize_token_embeddings(
            len(tokenizer),
            init_token_ids=doc_anchor_token_ids,
            init_from_token_id=init_from_token_id,
        )
        if old_vocab != new_vocab:
            print(
                "resized token embeddings for document anchors: "
                f"{old_vocab} -> {new_vocab}; initialized from token_id={init_from_token_id}"
            )
    if cfg.get("projection_mlp"):
        model.upgrade_to_mlp(intermediate=cfg["projection_mlp"])
    if cfg.get("transformer_head", False):
        model.add_transformer_head()

    embedding_feather_raw_cfg = cfg.get("embedding_feather", False)
    embedding_feather_enabled = bool(embedding_feather_raw_cfg)
    embedding_feather_cfg = (
        embedding_feather_raw_cfg
        if isinstance(embedding_feather_raw_cfg, dict)
        else {}
    )
    if embedding_feather_enabled:
        model.configure_embedding_feather(
            enabled=True,
            theta=embedding_feather_cfg.get("theta", 1.0),
            p=embedding_feather_cfg.get("p", 3.0),
            threshold=embedding_feather_cfg.get(
                "threshold",
                float(model.embedding_feather_threshold.detach().cpu()),
            ),
        )
        print(
            "Embedding Feather enabled: "
            f"target_sparsity={float(embedding_feather_cfg.get('target_sparsity', 0.5)):.3f}, "
            f"theta={model.embedding_feather_theta:.3f}, p={model.embedding_feather_p:.3f}"
        )

    embedding_binary_enabled = bool(cfg.get("embedding_binary", False))
    if embedding_binary_enabled:
        if embedding_feather_enabled:
            raise ValueError("embedding_binary and embedding_feather are mutually exclusive")
        embedding_binary_cfg = (
            cfg["embedding_binary"]
            if isinstance(cfg["embedding_binary"], dict)
            else {}
        )
        model.configure_embedding_binary(
            enabled=True,
            hadamard=bool(embedding_binary_cfg.get("hadamard", False)),
            seed=int(embedding_binary_cfg.get("seed", 0)),
        )
        msg = "Embedding binary enabled: projection components forced to -1/+1"
        if model.embedding_binary_hadamard:
            msg += f" after fixed Hadamard rotation (seed={model.embedding_binary_seed})"
        print(msg)

    sparse_bow_raw_cfg = cfg.get("sparse_bow", False)
    sparse_bow_enabled = bool(sparse_bow_raw_cfg)
    sparse_bow_cfg = sparse_bow_raw_cfg if isinstance(sparse_bow_raw_cfg, dict) else {}
    if sparse_bow_enabled:
        if embedding_feather_enabled or embedding_binary_enabled:
            raise ValueError("sparse_bow is mutually exclusive with embedding_feather/binary")
        model.configure_embedding_sparse_bow(
            enabled=True,
            threshold=float(sparse_bow_cfg.get("threshold", 0.0)),
            theta=float(sparse_bow_cfg.get("theta", 0.1)),
        )
        print(
            "Sparse BoW enabled: "
            f"dim={model.dim}, pool={sparse_bow_cfg.get('pool', 'max')}, "
            f"target_sparsity={float(sparse_bow_cfg.get('target_sparsity', 0.999)):.4f}, "
            f"theta={model.embedding_sparse_bow_theta:.3f}"
        )
    if token_gate_enabled:
        print(
            "Token gate enabled: "
            f"score_scale={float(token_gate_cfg.get('score_scale', 0.10)):.3f}, "
            f"target_mode={token_gate_cfg.get('target_mode', 'sqrt')}, "
            f"budget_weight={float(token_gate_cfg.get('budget_weight', 1.0)):.3e}"
        )
    token_gate_word_start_by_id = (
        tokenizer_word_start_flags(tokenizer).to(device)
        if token_gate_enabled and token_gate_wordpiece_boundaries == "vocab"
        else None
    )

    feather_cfg = cfg.get("feather", False)
    if feather_cfg:
        if not isinstance(feather_cfg, dict):
            feather_cfg = {}
        n_feather = enable_feather(
            model,
            include_patterns=feather_cfg.get("include_patterns"),
            exclude_patterns=feather_cfg.get("exclude_patterns"),
            theta=feather_cfg.get("theta", 1.0),
            p=feather_cfg.get("p", 3.0),
        )
        print(f"Feather enabled for {n_feather} Linear layers")

    model.to(device)

    if cfg.get("freeze_except_token_gate", False):
        if model.token_gate is None:
            raise ValueError("freeze_except_token_gate requires token_gate")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("token_gate")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"frozen base model — training token gate only ({trainable:,} params)")

    if cfg.get("freeze_encoder", False):
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("encoder frozen — training projection layer only")

    if cfg.get("unfreeze_top_n_layers"):
        model.unfreeze_top_n_layers(cfg["unfreeze_top_n_layers"])

    if cfg.get("qat", False):
        from qat import enable_qat
        enable_qat(model)

    # ── contrastive data ────────────────────────────────────────────
    loader = None
    parts = []

    if "triplet_file" in cfg:
        parts.append(JSONLTripletDataset(
            cfg["triplet_file"],
            max_samples=cfg.get("max_samples"),
        ))

    if "kd_contrastive" in cfg:
        kdc = cfg["kd_contrastive"]
        parts.append(KDContrastiveDataset(
            dataset_name=kdc["dataset"],
            num_candidates=kdc.get("num_candidates", 32),
            max_samples=kdc.get("max_samples"),
            use_normalized=kdc.get("use_normalized", True),
        ))

    if "bge_splits" in cfg:
        parts.append(BGEDataset(
            splits=cfg["bge_splits"],
            max_samples_per_split=cfg.get("bge_max_samples_per_split"),
        ))

    if "bekko_hard_negatives" in cfg:
        bhn = cfg["bekko_hard_negatives"]
        parts.append(BekkoHardNegativeDataset(
            configs=bhn["configs"],
            dataset_name=bhn.get(
                "dataset",
                "hotchpotch/bekko-embedding-v1-hard-negatives",
            ),
            split=bhn.get("split", "train"),
            max_samples_per_config=bhn.get("max_samples_per_config"),
        ))

    if "datasets" in cfg or "dataset" in cfg:
        if "datasets" in cfg:
            parts.append(RetrievalDataset.from_multi(
                cfg["datasets"], split=cfg.get("split", "train"),
            ))
        else:
            parts.append(RetrievalDataset(
                dataset_name=cfg["dataset"],
                split=cfg.get("split", "train"),
                dataset_config=cfg.get("dataset_config"),
                data_files=cfg.get("data_files"),
                query_col=cfg.get("query_col", "query"),
                pos_col=cfg.get("pos_col", "positive"),
                neg_col=cfg.get("neg_col"),
                max_samples=cfg.get("max_samples"),
            ))

    if len(parts) == 1:
        dataset = parts[0]
    elif len(parts) > 1:
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(parts)
        print(f"combined {len(parts)} datasets: {sum(len(p) for p in parts)} total samples")
    else:
        dataset = None

    if dataset is not None:
        collator = RetrievalCollator(
            tokenizer=tokenizer,
            query_max_length=cfg.get("query_max_length", 32),
            doc_max_length=cfg.get("doc_max_length", 256),
            doc_anchor_token=doc_anchor_token,
            doc_anchor_mode=doc_anchor_mode,
            doc_anchor_context_sentences=doc_anchor_context_sentences,
            doc_anchor_placement=doc_anchor_placement,
            token_surprise_cache_path=token_gate_predictability_cache,
            return_word_start_mask=(token_gate_wordpiece_boundaries == "offsets"),
        )
        if hasattr(dataset, "lang_ids"):
            sampler = LanguageGroupedBatchSampler(
                dataset.lang_ids, cfg["batch_size"], drop_last=True,
            )
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=collator,
                num_workers=0,
                pin_memory=False,
            )
            print(f"using language-grouped batches ({len(sampler.groups)} groups)")
        else:
            loader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                collate_fn=collator,
                num_workers=0,
                drop_last=True,
                pin_memory=False,
            )

    # ── KD data ──────────────────────────────────────────────────────
    kd_loader = None
    if "bge_kd_splits" in cfg:
        kd_dataset = BGEKDDataset(
            splits=cfg["bge_kd_splits"],
            num_candidates=cfg.get("bge_kd_candidates", 16),
            max_samples_per_split=cfg.get("bge_kd_max_samples_per_split"),
        )
    elif "kd_datasets" in cfg:
        kd_spec = cfg["kd_datasets"][0]
        kd_dataset = KDDataset(
            dataset_name=kd_spec["dataset"],
            num_candidates=kd_spec.get("num_candidates", 16),
            max_samples=kd_spec.get("max_samples"),
            use_normalized=kd_spec.get("use_normalized", True),
        )
    else:
        kd_dataset = None
    if kd_dataset is not None:
        kd_collator = KDCollator(
            tokenizer=tokenizer,
            query_max_length=cfg.get("query_max_length", 32),
            doc_max_length=cfg.get("doc_max_length", 256),
            doc_anchor_token=doc_anchor_token,
            doc_anchor_mode=doc_anchor_mode,
            doc_anchor_context_sentences=doc_anchor_context_sentences,
            doc_anchor_placement=doc_anchor_placement,
        )
        kd_loader = DataLoader(
            kd_dataset,
            batch_size=cfg.get("kd_batch_size", 32),
            shuffle=True,
            collate_fn=kd_collator,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )

    if loader is None and kd_loader is None:
        raise ValueError("no datasets or kd_datasets configured")
    if sparse_bow_enabled and kd_loader is not None:
        raise ValueError("sparse_bow training currently supports contrastive/triplet data only")

    # ── optimiser & schedule ──────────────────────────────────────
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    primary_loader = loader or kd_loader
    steps_per_epoch = math.ceil(len(primary_loader) / grad_accum)
    total_steps = cfg.get("max_steps", steps_per_epoch * cfg["epochs"])
    max_epochs = math.ceil(total_steps / steps_per_epoch)
    warmup = min(cfg.get("warmup_steps", int(0.05 * total_steps)), total_steps)

    freeze_encoder_steps = cfg.get("freeze_encoder_steps", 0)
    if freeze_encoder_steps > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"encoder frozen for first {freeze_encoder_steps} optimizer steps")

    optimizer = build_optimizer(model, cfg)
    sched_type = cfg.get("lr_schedule", "cosine")
    if sched_type == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup)
    elif sched_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    temperature = cfg.get("temperature", 0.05)
    kd_temperature = cfg.get("kd_temperature", 1.0)
    kd_student_temperature = cfg.get("kd_student_temperature")
    kd_weight = cfg.get("kd_weight", 0.5)
    sigreg_weight = cfg.get("sigreg_weight", 0.0)
    feather_enabled = bool(feather_cfg)
    feather_target = float(feather_cfg.get("target_sparsity", 0.5)) if feather_enabled else 0.0
    feather_start_step = int(feather_cfg.get("start_step", 0)) if feather_enabled else 0
    feather_ramp_steps = int(feather_cfg.get("ramp_steps", total_steps)) if feather_enabled else total_steps
    feather_update_every = max(int(feather_cfg.get("update_every", 10)), 1) if feather_enabled else 1
    feather_max_threshold_samples = feather_cfg.get("max_threshold_samples", 4_000_000) if feather_enabled else None
    feather_bake_checkpoints = bool(feather_cfg.get("bake_checkpoints", True)) if feather_enabled else False
    feather_stats = None
    feather_last_update = -1
    if feather_enabled:
        print(
            "Feather schedule: "
            f"target_sparsity={feather_target:.3f}, start_step={feather_start_step}, "
            f"ramp_steps={feather_ramp_steps}, update_every={feather_update_every}, "
            f"max_threshold_samples={feather_max_threshold_samples}, "
            f"bake_checkpoints={feather_bake_checkpoints}"
        )
    embedding_feather_target = (
        float(embedding_feather_cfg.get("target_sparsity", 0.5))
        if embedding_feather_enabled else 0.0
    )
    embedding_feather_start_step = (
        int(embedding_feather_cfg.get("start_step", 0))
        if embedding_feather_enabled else 0
    )
    embedding_feather_ramp_steps = (
        int(embedding_feather_cfg.get("ramp_steps", total_steps))
        if embedding_feather_enabled else total_steps
    )
    embedding_feather_update_every = (
        max(int(embedding_feather_cfg.get("update_every", 10)), 1)
        if embedding_feather_enabled else 1
    )
    embedding_feather_max_threshold_samples = (
        embedding_feather_cfg.get("max_threshold_samples", 4_000_000)
        if embedding_feather_enabled else None
    )
    embedding_feather_stats = None
    embedding_feather_last_update = -1
    if embedding_feather_enabled:
        print(
            "Embedding Feather schedule: "
            f"target_sparsity={embedding_feather_target:.3f}, "
            f"start_step={embedding_feather_start_step}, "
            f"ramp_steps={embedding_feather_ramp_steps}, "
            f"update_every={embedding_feather_update_every}, "
            f"max_threshold_samples={embedding_feather_max_threshold_samples}"
        )
    sparse_bow_target = (
        float(sparse_bow_cfg.get("target_sparsity", 0.999))
        if sparse_bow_enabled else 0.0
    )
    sparse_bow_start_step = (
        int(sparse_bow_cfg.get("start_step", 0))
        if sparse_bow_enabled else 0
    )
    sparse_bow_ramp_steps = (
        int(sparse_bow_cfg.get("ramp_steps", total_steps))
        if sparse_bow_enabled else total_steps
    )
    sparse_bow_update_every = (
        max(int(sparse_bow_cfg.get("update_every", 10)), 1)
        if sparse_bow_enabled else 1
    )
    sparse_bow_max_threshold_samples = (
        sparse_bow_cfg.get("max_threshold_samples", 2_000_000)
        if sparse_bow_enabled else None
    )
    sparse_bow_pool = sparse_bow_cfg.get("pool", "max")
    sparse_bow_l1_weight = float(sparse_bow_cfg.get("l1_weight", 0.0))
    sparse_bow_flops_weight = float(sparse_bow_cfg.get("flops_weight", 0.0))
    sparse_bow_stats = None
    sparse_bow_batch_stats = None
    sparse_bow_last_update = -1
    if sparse_bow_enabled:
        print(
            "Sparse BoW schedule: "
            f"target_sparsity={sparse_bow_target:.4f}, "
            f"start_step={sparse_bow_start_step}, "
            f"ramp_steps={sparse_bow_ramp_steps}, "
            f"update_every={sparse_bow_update_every}, "
            f"max_threshold_samples={sparse_bow_max_threshold_samples}, "
            f"l1_weight={sparse_bow_l1_weight:.2e}, "
            f"flops_weight={sparse_bow_flops_weight:.2e}"
        )
    token_gate_score_scale = float(token_gate_cfg.get("score_scale", 0.10)) if token_gate_enabled else 0.0
    token_gate_budget_weight = float(token_gate_cfg.get("budget_weight", 1.0)) if token_gate_enabled else 0.0
    token_gate_start_step = int(token_gate_cfg.get("start_step", 0)) if token_gate_enabled else 0
    token_gate_ramp_steps = int(token_gate_cfg.get("ramp_steps", total_steps)) if token_gate_enabled else total_steps
    token_gate_target_mode = str(token_gate_cfg.get("target_mode", "sqrt"))
    token_gate_target_factor = float(token_gate_cfg.get("target_factor", 1.0))
    token_gate_target_fraction = float(token_gate_cfg.get("target_fraction", 0.25))
    token_gate_min_keep = float(token_gate_cfg.get("min_keep", 1.0))
    token_gate_nested_factors = [
        float(x) for x in token_gate_cfg.get("nested_prefix_factors", [])
    ] if token_gate_enabled else []
    token_gate_nested_weights = [
        float(x) for x in token_gate_cfg.get("nested_prefix_weights", [])
    ] if token_gate_enabled else []
    if token_gate_nested_factors and not token_gate_nested_weights:
        token_gate_nested_weights = [1.0 for _ in token_gate_nested_factors]
    if len(token_gate_nested_weights) != len(token_gate_nested_factors):
        raise ValueError("token_gate nested_prefix_weights must match nested_prefix_factors")
    token_gate_all_token_loss_weight = (
        float(token_gate_cfg.get("all_token_loss_weight", 1.0))
        if token_gate_enabled else 1.0
    )
    token_gate_nested_prefix_weight = (
        float(token_gate_cfg.get("nested_prefix_weight", 0.0))
        if token_gate_enabled else 0.0
    )
    token_gate_nested_start_step = (
        int(token_gate_cfg.get("nested_start_step", token_gate_start_step))
        if token_gate_enabled else 0
    )
    token_gate_nested_ramp_steps = (
        int(token_gate_cfg.get("nested_ramp_steps", token_gate_ramp_steps))
        if token_gate_enabled else total_steps
    )
    token_gate_wordpiece_consistency_weight = (
        float(token_gate_cfg.get("wordpiece_consistency_weight", 0.0))
        if token_gate_enabled else 0.0
    )
    token_gate_wordpiece_start_step = (
        int(token_gate_cfg.get("wordpiece_start_step", token_gate_start_step))
        if token_gate_enabled else 0
    )
    token_gate_wordpiece_ramp_steps = (
        int(token_gate_cfg.get("wordpiece_ramp_steps", token_gate_ramp_steps))
        if token_gate_enabled else total_steps
    )
    token_gate_maxsim_target_weight = (
        float(token_gate_cfg.get("maxsim_target_weight", 0.0))
        if token_gate_enabled else 0.0
    )
    token_gate_maxsim_target_start_step = (
        int(token_gate_cfg.get("maxsim_target_start_step", token_gate_start_step))
        if token_gate_enabled else 0
    )
    token_gate_maxsim_target_ramp_steps = (
        int(token_gate_cfg.get("maxsim_target_ramp_steps", token_gate_ramp_steps))
        if token_gate_enabled else total_steps
    )
    token_gate_predictability_cache = (
        token_gate_cfg.get("predictability_cache")
        if token_gate_enabled else None
    )
    token_gate_predictability_weight = (
        float(token_gate_cfg.get("predictability_weight", 0.0))
        if token_gate_enabled else 0.0
    )
    token_gate_predictability_start_step = (
        int(token_gate_cfg.get("predictability_start_step", token_gate_start_step))
        if token_gate_enabled else 0
    )
    token_gate_predictability_ramp_steps = (
        int(token_gate_cfg.get("predictability_ramp_steps", token_gate_ramp_steps))
        if token_gate_enabled else total_steps
    )
    token_gate_batch_stats = None
    token_gate_wordpiece_loss_val = 0.0
    token_gate_wordpiece_pair_count = 0
    token_gate_maxsim_target_loss_val = 0.0
    token_gate_maxsim_target_positive_rate = 0.0
    token_gate_predictability_loss_val = 0.0
    token_gate_predictability_coverage = 0.0
    token_gate_predictability_mean = 0.0
    if token_gate_enabled:
        print(
            "Token gate schedule: "
            f"target_mode={token_gate_target_mode}, "
            f"target_factor={token_gate_target_factor:.3f}, "
            f"target_fraction={token_gate_target_fraction:.3f}, "
            f"min_keep={token_gate_min_keep:.1f}, "
            f"start_step={token_gate_start_step}, "
            f"ramp_steps={token_gate_ramp_steps}"
        )
        if token_gate_nested_factors:
            print(
                "Token gate nested prefixes: "
                f"factors={token_gate_nested_factors}, "
                f"weights={token_gate_nested_weights}, "
                f"all_token_loss_weight={token_gate_all_token_loss_weight:.3f}, "
                f"nested_prefix_weight={token_gate_nested_prefix_weight:.3f}, "
                f"start_step={token_gate_nested_start_step}, "
                f"ramp_steps={token_gate_nested_ramp_steps}"
            )
        if token_gate_wordpiece_consistency_weight > 0.0:
            print(
                "Token gate wordpiece consistency: "
                f"weight={token_gate_wordpiece_consistency_weight:.3e}, "
                f"start_step={token_gate_wordpiece_start_step}, "
                f"ramp_steps={token_gate_wordpiece_ramp_steps}"
            )
        if token_gate_maxsim_target_weight > 0.0:
            print(
                "Token gate MaxSim target: "
                f"weight={token_gate_maxsim_target_weight:.3e}, "
                f"start_step={token_gate_maxsim_target_start_step}, "
                f"ramp_steps={token_gate_maxsim_target_ramp_steps}"
            )
        if token_gate_predictability_weight > 0.0:
            if not token_gate_predictability_cache:
                raise ValueError("token_gate.predictability_weight requires predictability_cache")
            print(
                "Token gate predictability regularizer: "
                f"weight={token_gate_predictability_weight:.3e}, "
                f"cache={token_gate_predictability_cache}, "
                f"start_step={token_gate_predictability_start_step}, "
                f"ramp_steps={token_gate_predictability_ramp_steps}"
            )
    use_maxsim = cfg.get("use_maxsim", False)
    maxsim_warmup_steps = cfg.get("maxsim_warmup_steps", 0)
    if use_maxsim:
        print("using MaxSim scoring for loss")
    elif maxsim_warmup_steps > 0:
        print(f"using MaxSim scoring for first {maxsim_warmup_steps} optimizer steps, then XTR")
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    log_every = cfg.get("log_every", 50)
    save_every = cfg.get("save_every_steps", max(log_every * 100, 1))
    keep_last = cfg.get("keep_last_n_checkpoints", 1)
    output_dir = cfg.get("output_dir", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    print(f"output_dir: {output_dir}")
    try:
        git_info = subprocess.run(
            ["git", "show", "--stat", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if git_info.returncode == 0:
            with open(os.path.join(output_dir, "git_head.txt"), "w") as f:
                f.write(git_info.stdout)
    except Exception:
        pass

    # ── gradient checkpointing ───────────────────────────────────
    if cfg.get("gradient_checkpointing", False):
        model.encoder.gradient_checkpointing_enable()

    # ── mixed precision ──────────────────────────────────────────
    use_amp = (cfg.get("fp16", False) or cfg.get("bf16", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.get("bf16", False) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype)
        if use_amp
        else nullcontext()
    )
    if use_amp:
        print(f"using {('bf16' if amp_dtype == torch.bfloat16 else 'fp16')} mixed precision")

    # ── loss log ──────────────────────────────────────────────────
    loss_log_path = os.path.join(output_dir, "loss.tsv")
    loss_log = open(loss_log_path, "a")
    if os.path.getsize(loss_log_path) == 0:
        header = "epoch\tstep\tglobal_step\tloss\tavg_loss\tlr"
        if loader and kd_loader:
            header += "\tc_loss\tkd_loss"
        if feather_enabled:
            header += "\tfeather_sparsity\tfeather_threshold"
        if embedding_feather_enabled:
            header += "\tembedding_feather_sparsity\tembedding_feather_threshold"
        if sparse_bow_enabled:
            header += (
                "\tsparse_bow_sparsity\tsparse_bow_threshold"
                "\tsparse_bow_q_active\tsparse_bow_d_active"
                "\tsparse_bow_q_l1\tsparse_bow_d_l1\tsparse_bow_flops"
            )
        if token_gate_enabled:
            header += (
                "\ttoken_gate_active\ttoken_gate_target"
                "\ttoken_gate_keep_fraction\ttoken_gate_mean"
            )
            if token_gate_wordpiece_consistency_weight > 0.0:
                header += "\ttoken_gate_wordpiece_loss\ttoken_gate_wordpiece_pairs"
            if token_gate_maxsim_target_weight > 0.0:
                header += "\ttoken_gate_maxsim_target_loss\ttoken_gate_maxsim_target_positive_rate"
            if token_gate_predictability_weight > 0.0:
                header += (
                    "\ttoken_gate_predictability_loss"
                    "\ttoken_gate_predictability_coverage"
                    "\ttoken_gate_predictability_mean"
                )
        loss_log.write(header + "\n")
        loss_log.flush()

    # ── loop ─────────────────────────────────────────────────────
    model.train()
    global_step = 0
    saved_step_ckpts: list[str] = []
    n_batches = len(primary_loader)
    reported_doc_anchor_stats = False

    sample_counts = []
    if loader:
        sample_counts.append(f"{len(loader.dataset)} contrastive")
    if kd_loader:
        sample_counts.append(f"{len(kd_loader.dataset)} KD")
    print(
        f"{', '.join(sample_counts)} samples, {n_batches} batches/epoch, "
        f"{steps_per_epoch} optimizer steps/epoch, target_steps={total_steps}"
    )
    if loader and kd_loader:
        kd_temp_msg = f"kd_weight={kd_weight}, kd_temperature={kd_temperature}"
        if kd_student_temperature is not None:
            kd_temp_msg += f", kd_student_temperature={kd_student_temperature}"
        print(kd_temp_msg)

    # Save step_0 checkpoint (before any training) for baseline eval
    ckpt0 = os.path.join(output_dir, "step_0")
    if not os.path.exists(ckpt0):
        if feather_enabled:
            feather_stats = set_feather_sparsity(
                model,
                0.0,
                max_threshold_samples=feather_max_threshold_samples,
            )
            feather_last_update = 0
        save_training_checkpoint(
            model,
            tokenizer,
            ckpt0,
            bake_feather=feather_bake_checkpoints,
        )
        print(f"  → {ckpt0} (baseline)")

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, max_epochs + 1):
        running_loss = 0.0
        processed_batches = 0
        kd_iter = iter(kd_loader) if kd_loader else None

        for step, batch in enumerate(primary_loader, start=1):
            processed_batches = step
            if step == 1:
                print(f"epoch {epoch} — first batch loaded, starting training...")

            if feather_enabled and (
                feather_last_update < 0
                or global_step == 0
                or global_step - feather_last_update >= feather_update_every
            ):
                if global_step < feather_start_step:
                    current_sparsity = 0.0
                else:
                    progress = 1.0
                    if feather_ramp_steps > 0:
                        progress = min(
                            1.0,
                            (global_step - feather_start_step + 1) / feather_ramp_steps,
                        )
                    current_sparsity = feather_target * progress
                feather_stats = set_feather_sparsity(
                    model,
                    current_sparsity,
                    max_threshold_samples=feather_max_threshold_samples,
                )
                feather_last_update = global_step

            batch = {k: v.to(device) for k, v in batch.items()}
            if "doc_index_mask" in batch and not reported_doc_anchor_stats:
                doc_counts = batch["doc_index_mask"].sum(dim=1).float()
                msg = (
                    "document anchor stats: "
                    f"avg={doc_counts.mean().item():.2f} "
                    f"min={doc_counts.min().item():.0f} "
                    f"max={doc_counts.max().item():.0f}"
                )
                if "neg_index_mask" in batch:
                    neg_counts = batch["neg_index_mask"].sum(dim=1).float()
                    msg += (
                        f"; neg_avg={neg_counts.mean().item():.2f} "
                        f"neg_min={neg_counts.min().item():.0f} "
                        f"neg_max={neg_counts.max().item():.0f}"
                    )
                print(msg)
                reported_doc_anchor_stats = True

            with autocast_ctx:
                total_loss = torch.tensor(0.0, device=device)
                c_loss_val = 0.0
                kd_loss_val = 0.0
                sr_loss_val = 0.0
                c_score_mode = ""
                token_gate_wordpiece_loss_val = 0.0
                token_gate_wordpiece_pair_count = 0
                token_gate_maxsim_target_loss_val = 0.0
                token_gate_maxsim_target_positive_rate = 0.0
                token_gate_predictability_loss_val = 0.0
                token_gate_predictability_coverage = 0.0
                token_gate_predictability_mean = 0.0

                # ── contrastive loss ─────────────────────────────
                if loader:
                    if loader is not primary_loader:
                        # KD is primary — pull contrastive batch separately
                        raise NotImplementedError("KD-primary with contrastive secondary")
                    if sparse_bow_enabled:
                        query_raw = model(
                            batch["query_input_ids"],
                            batch["query_attention_mask"],
                            return_raw_projection_only=True,
                        )
                        doc_raw = model(
                            batch["doc_input_ids"],
                            batch["doc_attention_mask"],
                            return_raw_projection_only=True,
                        )
                        d_mask = batch.get("doc_index_mask", batch["doc_attention_mask"])
                        if "neg_input_ids" in batch:
                            neg_raw = model(
                                batch["neg_input_ids"],
                                batch["neg_attention_mask"],
                                return_raw_projection_only=True,
                            )
                            doc_raw = torch.cat([doc_raw, neg_raw], dim=0)
                            neg_mask = batch.get("neg_index_mask", batch["neg_attention_mask"])
                            d_mask = torch.cat(
                                [d_mask, neg_mask],
                                dim=0,
                            )

                        if (
                            sparse_bow_last_update < 0
                            or global_step == 0
                            or global_step - sparse_bow_last_update >= sparse_bow_update_every
                        ):
                            if global_step < sparse_bow_start_step:
                                current_sparsity = 0.0
                            else:
                                progress = 1.0
                                if sparse_bow_ramp_steps > 0:
                                    progress = min(
                                        1.0,
                                        (global_step - sparse_bow_start_step + 1)
                                        / sparse_bow_ramp_steps,
                                    )
                                current_sparsity = sparse_bow_target * progress
                            sparse_bow_stats = pooled_sparse_bow_threshold_stats(
                                [query_raw, doc_raw],
                                [batch["query_attention_mask"], d_mask],
                                current_sparsity,
                                pool=sparse_bow_pool,
                                max_threshold_samples=sparse_bow_max_threshold_samples,
                            )
                            model.set_embedding_sparse_bow_threshold(sparse_bow_stats.threshold)
                            sparse_bow_last_update = global_step

                        query_vecs = model.sparse_bow_from_raw(
                            query_raw,
                            batch["query_attention_mask"],
                            pool=sparse_bow_pool,
                        )
                        doc_vecs = model.sparse_bow_from_raw(
                            doc_raw,
                            d_mask,
                            pool=sparse_bow_pool,
                        )
                        c_loss = sparse_bow_contrastive_loss(
                            query_vecs,
                            doc_vecs,
                            temperature,
                        )
                        reg_loss, sparse_bow_batch_stats = sparse_bow_regularization(
                            query_vecs,
                            doc_vecs,
                            l1_weight=sparse_bow_l1_weight,
                            flops_weight=sparse_bow_flops_weight,
                        )
                        c_loss = c_loss + reg_loss
                        c_score_mode = "sparse-bow"
                    else:
                        use_raw = sigreg_weight > 0 or embedding_feather_enabled
                        q_out = model(
                            batch["query_input_ids"], batch["query_attention_mask"],
                            return_raw_projection=use_raw,
                        )
                        d_out = model(
                            batch["doc_input_ids"], batch["doc_attention_mask"],
                            return_raw_projection=use_raw,
                            return_gate_logits=token_gate_enabled,
                        )
                        if use_raw:
                            query_embs, query_raw = q_out
                            if token_gate_enabled:
                                doc_embs, doc_raw, doc_gate_logits = d_out
                            else:
                                doc_embs, doc_raw = d_out
                        else:
                            query_embs = q_out
                            if token_gate_enabled:
                                doc_embs, doc_gate_logits = d_out
                            else:
                                doc_embs = d_out
                        doc_loss_mask = batch.get("doc_index_mask", batch["doc_attention_mask"])
                        doc_token_ids = batch["doc_input_ids"]
                        doc_word_start = batch.get("doc_word_start_mask")
                        doc_surprise = batch.get("doc_token_surprise")
                        doc_surprise_mask = batch.get("doc_token_surprise_mask")

                        if "neg_input_ids" in batch:
                            n_out = model(
                                batch["neg_input_ids"], batch["neg_attention_mask"],
                                return_raw_projection=use_raw,
                                return_gate_logits=token_gate_enabled,
                            )
                            if use_raw:
                                if token_gate_enabled:
                                    neg_embs, neg_raw, neg_gate_logits = n_out
                                else:
                                    neg_embs, neg_raw = n_out
                                doc_raw = torch.cat([doc_raw, neg_raw], dim=0)
                            else:
                                if token_gate_enabled:
                                    neg_embs, neg_gate_logits = n_out
                                else:
                                    neg_embs = n_out
                            doc_embs = torch.cat([doc_embs, neg_embs], dim=0)
                            if token_gate_enabled:
                                doc_gate_logits = torch.cat(
                                    [doc_gate_logits, neg_gate_logits],
                                    dim=0,
                                )
                            doc_token_ids = torch.cat(
                                [doc_token_ids, batch["neg_input_ids"]],
                                dim=0,
                            )
                            if doc_word_start is not None:
                                doc_word_start = torch.cat(
                                    [doc_word_start, batch["neg_word_start_mask"]],
                                    dim=0,
                                )
                            if doc_surprise is not None and "neg_token_surprise" in batch:
                                doc_surprise = torch.cat(
                                    [doc_surprise, batch["neg_token_surprise"]],
                                    dim=0,
                                )
                                doc_surprise_mask = torch.cat(
                                    [doc_surprise_mask, batch["neg_token_surprise_mask"]],
                                    dim=0,
                                )
                            else:
                                doc_surprise = None
                                doc_surprise_mask = None
                            neg_loss_mask = batch.get("neg_index_mask", batch["neg_attention_mask"])
                            d_mask = torch.cat(
                                [doc_loss_mask, neg_loss_mask],
                                dim=0,
                            )
                        else:
                            d_mask = doc_loss_mask

                        if "doc_index_mask" in batch:
                            doc_select_mask = d_mask
                            doc_embs, d_mask = select_masked_sequence(doc_embs, doc_select_mask)
                            if use_raw:
                                doc_raw, _ = select_masked_sequence(doc_raw, doc_select_mask)
                            if token_gate_enabled:
                                packed_gate, _ = select_masked_sequence(
                                    doc_gate_logits.unsqueeze(-1),
                                    doc_select_mask,
                                )
                                doc_gate_logits = packed_gate.squeeze(-1)
                            packed_token_ids, _ = select_masked_sequence(
                                doc_token_ids.unsqueeze(-1),
                                doc_select_mask,
                            )
                            doc_token_ids = packed_token_ids.squeeze(-1).long()
                            if doc_word_start is not None:
                                packed_word_start, _ = select_masked_sequence(
                                    doc_word_start.unsqueeze(-1),
                                    doc_select_mask,
                                )
                                doc_word_start = packed_word_start.squeeze(-1).bool()
                            if doc_surprise is not None:
                                packed_surprise, _ = select_masked_sequence(
                                    doc_surprise.unsqueeze(-1),
                                    doc_select_mask,
                                )
                                packed_surprise_mask, _ = select_masked_sequence(
                                    doc_surprise_mask.unsqueeze(-1),
                                    doc_select_mask,
                                )
                                doc_surprise = packed_surprise.squeeze(-1)
                                doc_surprise_mask = packed_surprise_mask.squeeze(-1).long()

                        if embedding_feather_enabled and (
                            embedding_feather_last_update < 0
                            or global_step == 0
                            or global_step - embedding_feather_last_update >= embedding_feather_update_every
                        ):
                            if global_step < embedding_feather_start_step:
                                current_sparsity = 0.0
                            else:
                                progress = 1.0
                                if embedding_feather_ramp_steps > 0:
                                    progress = min(
                                        1.0,
                                        (global_step - embedding_feather_start_step + 1)
                                        / embedding_feather_ramp_steps,
                                    )
                                current_sparsity = embedding_feather_target * progress
                            embedding_feather_stats = activation_feather_stats(
                                [query_raw, doc_raw],
                                [batch["query_attention_mask"], d_mask],
                                current_sparsity,
                                max_threshold_samples=embedding_feather_max_threshold_samples,
                            )
                            model.set_embedding_feather_threshold(embedding_feather_stats.threshold)
                            embedding_feather_last_update = global_step
                            query_embs = model.normalize_projection(
                                query_raw,
                                batch["query_attention_mask"],
                            )
                            doc_embs = model.normalize_projection(doc_raw, d_mask)

                        contrastive_use_maxsim = use_maxsim or (
                            maxsim_warmup_steps > 0 and global_step < maxsim_warmup_steps
                        )
                        c_score_mode = "maxsim" if contrastive_use_maxsim else "xtr"
                        c_loss = contrastive_loss(
                            query_embs,
                            doc_embs,
                            batch["query_attention_mask"],
                            d_mask,
                            temperature,
                            k_train=cfg.get("k_train", 64),
                            use_maxsim=contrastive_use_maxsim,
                            z_normalize=cfg.get("z_normalize", True),
                            xtr_score_relu=cfg.get("xtr_score_relu", False),
                            doc_gate_logits=doc_gate_logits if token_gate_enabled else None,
                            doc_gate_score_scale=token_gate_score_scale,
                        )

                        if (
                            token_gate_enabled
                            and token_gate_nested_factors
                            and token_gate_nested_prefix_weight != 0.0
                        ):
                            if global_step < token_gate_nested_start_step:
                                token_gate_nested_progress = 0.0
                            else:
                                token_gate_nested_progress = 1.0
                                if token_gate_nested_ramp_steps > 0:
                                    token_gate_nested_progress = min(
                                        1.0,
                                        (global_step - token_gate_nested_start_step + 1)
                                        / token_gate_nested_ramp_steps,
                                    )
                            if token_gate_nested_progress > 0.0:
                                prefix_loss_sum = c_loss.new_tensor(0.0)
                                prefix_weight_sum = 0.0
                                for prefix_factor, prefix_weight in zip(
                                    token_gate_nested_factors,
                                    token_gate_nested_weights,
                                ):
                                    if prefix_weight == 0.0:
                                        continue
                                    prefix_d_mask = token_gate_prefix_mask(
                                        doc_gate_logits,
                                        d_mask,
                                        target_mode=token_gate_target_mode,
                                        target_factor=token_gate_target_factor,
                                        target_fraction=token_gate_target_fraction,
                                        min_keep=token_gate_min_keep,
                                        prefix_factor=prefix_factor,
                                    )
                                    prefix_loss = contrastive_loss(
                                        query_embs,
                                        doc_embs,
                                        batch["query_attention_mask"],
                                        prefix_d_mask,
                                        temperature,
                                        k_train=cfg.get("k_train", 64),
                                        use_maxsim=contrastive_use_maxsim,
                                        z_normalize=cfg.get("z_normalize", True),
                                        xtr_score_relu=cfg.get("xtr_score_relu", False),
                                        doc_gate_logits=doc_gate_logits,
                                        doc_gate_score_scale=token_gate_score_scale,
                                    )
                                    prefix_loss_sum = prefix_loss_sum + float(prefix_weight) * prefix_loss
                                    prefix_weight_sum += float(prefix_weight)
                                if prefix_weight_sum > 0.0:
                                    nested_loss = prefix_loss_sum / prefix_weight_sum
                                    base_weight = 1.0 - token_gate_nested_progress * (
                                        1.0 - token_gate_all_token_loss_weight
                                    )
                                    c_loss = (
                                        base_weight * c_loss
                                        + token_gate_nested_progress
                                        * token_gate_nested_prefix_weight
                                        * nested_loss
                                    )
                                    c_score_mode += "+gateprefix"

                        if token_gate_enabled:
                            if global_step < token_gate_start_step:
                                token_gate_progress = 0.0
                            else:
                                token_gate_progress = 1.0
                                if token_gate_ramp_steps > 0:
                                    token_gate_progress = min(
                                        1.0,
                                        (global_step - token_gate_start_step + 1)
                                        / token_gate_ramp_steps,
                                    )
                            gate_reg, token_gate_batch_stats = token_gate_budget_regularization(
                                doc_gate_logits,
                                d_mask,
                                target_mode=token_gate_target_mode,
                                target_factor=token_gate_target_factor,
                                target_fraction=token_gate_target_fraction,
                                min_keep=token_gate_min_keep,
                                progress=token_gate_progress,
                            )
                            c_loss = c_loss + token_gate_budget_weight * token_gate_progress * gate_reg
                            token_gate_wordpiece_loss_val = 0.0
                            token_gate_wordpiece_pair_count = 0
                            word_start = doc_word_start
                            if word_start is None and token_gate_word_start_by_id is not None:
                                word_start = token_gate_word_start_by_id[
                                    doc_token_ids.long().clamp(
                                        min=0,
                                        max=token_gate_word_start_by_id.numel() - 1,
                                    )
                                ]
                            if (
                                token_gate_wordpiece_consistency_weight > 0.0
                                and word_start is not None
                            ):
                                if global_step < token_gate_wordpiece_start_step:
                                    token_gate_wordpiece_progress = 0.0
                                else:
                                    token_gate_wordpiece_progress = 1.0
                                    if token_gate_wordpiece_ramp_steps > 0:
                                        token_gate_wordpiece_progress = min(
                                            1.0,
                                            (global_step - token_gate_wordpiece_start_step + 1)
                                            / token_gate_wordpiece_ramp_steps,
                                        )
                                if token_gate_wordpiece_progress > 0.0:
                                    wordpiece_loss, token_gate_wordpiece_pair_count = (
                                        token_gate_wordpiece_consistency_regularization(
                                            doc_gate_logits,
                                            d_mask,
                                            word_start,
                                        )
                                    )
                                    token_gate_wordpiece_loss_val = float(
                                        wordpiece_loss.detach().cpu()
                                    )
                                    c_loss = (
                                        c_loss
                                        + token_gate_wordpiece_consistency_weight
                                        * token_gate_wordpiece_progress
                                        * wordpiece_loss
                                    )
                            if token_gate_maxsim_target_weight > 0.0:
                                if global_step < token_gate_maxsim_target_start_step:
                                    token_gate_maxsim_target_progress = 0.0
                                else:
                                    token_gate_maxsim_target_progress = 1.0
                                    if token_gate_maxsim_target_ramp_steps > 0:
                                        token_gate_maxsim_target_progress = min(
                                            1.0,
                                            (global_step - token_gate_maxsim_target_start_step + 1)
                                            / token_gate_maxsim_target_ramp_steps,
                                        )
                                if token_gate_maxsim_target_progress > 0.0:
                                    (
                                        maxsim_target_loss,
                                        token_gate_maxsim_target_positive_rate,
                                    ) = token_gate_maxsim_target_regularization(
                                        query_embs,
                                        batch["query_attention_mask"],
                                        doc_embs,
                                        d_mask,
                                        doc_gate_logits,
                                        target_mode=token_gate_target_mode,
                                        target_factor=token_gate_target_factor,
                                        target_fraction=token_gate_target_fraction,
                                        min_keep=token_gate_min_keep,
                                    )
                                    token_gate_maxsim_target_loss_val = float(
                                        maxsim_target_loss.detach().cpu()
                                    )
                                    c_loss = (
                                        c_loss
                                        + token_gate_maxsim_target_weight
                                        * token_gate_maxsim_target_progress
                                        * maxsim_target_loss
                                    )
                            if (
                                token_gate_predictability_weight > 0.0
                                and doc_surprise is not None
                                and doc_surprise_mask is not None
                            ):
                                if global_step < token_gate_predictability_start_step:
                                    token_gate_predictability_progress = 0.0
                                else:
                                    token_gate_predictability_progress = 1.0
                                    if token_gate_predictability_ramp_steps > 0:
                                        token_gate_predictability_progress = min(
                                            1.0,
                                            (global_step - token_gate_predictability_start_step + 1)
                                            / token_gate_predictability_ramp_steps,
                                        )
                                if token_gate_predictability_progress > 0.0:
                                    (
                                        predictability_loss,
                                        token_gate_predictability_coverage,
                                        token_gate_predictability_mean,
                                    ) = token_gate_predictability_regularization(
                                        doc_gate_logits,
                                        doc_surprise,
                                        doc_surprise_mask,
                                        d_mask,
                                    )
                                    token_gate_predictability_loss_val = float(
                                        predictability_loss.detach().cpu()
                                    )
                                    c_loss = (
                                        c_loss
                                        + token_gate_predictability_weight
                                        * token_gate_predictability_progress
                                        * predictability_loss
                                    )

                        if use_raw:
                            sr = sigreg_loss(query_raw, batch["query_attention_mask"])
                            sr = sr + sigreg_loss(doc_raw, d_mask)
                            total_loss = total_loss + sigreg_weight * sr
                            sr_loss_val = sr.item()
                    w = (1.0 - kd_weight) if kd_loader else 1.0
                    total_loss = total_loss + w * c_loss
                    c_loss_val = c_loss.item()

                # ── KD loss ──────────────────────────────────────
                if kd_iter is not None:
                    try:
                        kd_batch = next(kd_iter)
                    except StopIteration:
                        kd_iter = iter(kd_loader)
                        kd_batch = next(kd_iter)
                    kd_batch = {k: v.to(device) for k, v in kd_batch.items()}

                    kd_q_embs = model(
                        kd_batch["kd_query_input_ids"],
                        kd_batch["kd_query_attention_mask"],
                    )
                    if token_gate_enabled:
                        kd_d_embs, kd_d_gate_logits = model(
                            kd_batch["kd_doc_input_ids"],
                            kd_batch["kd_doc_attention_mask"],
                            return_gate_logits=True,
                        )
                    else:
                        kd_d_embs = model(
                            kd_batch["kd_doc_input_ids"],
                            kd_batch["kd_doc_attention_mask"],
                        )
                        kd_d_gate_logits = None
                    kd_d_mask = kd_batch.get(
                        "kd_doc_index_mask",
                        kd_batch["kd_doc_attention_mask"],
                    )
                    if "kd_doc_index_mask" in kd_batch:
                        kd_d_embs, kd_d_mask = select_masked_sequence(kd_d_embs, kd_d_mask)
                        if token_gate_enabled:
                            packed_gate, _ = select_masked_sequence(
                                kd_d_gate_logits.unsqueeze(-1),
                                kd_batch["kd_doc_index_mask"],
                            )
                            kd_d_gate_logits = packed_gate.squeeze(-1)
                    kd_l = kd_loss(
                        kd_q_embs,
                        kd_d_embs,
                        kd_batch["kd_query_attention_mask"],
                        kd_d_mask,
                        kd_batch["kd_teacher_scores"],
                        kd_batch["kd_doc_counts"],
                        kd_temperature=kd_temperature,
                        kd_student_temperature=kd_student_temperature,
                        k_train=cfg.get("kd_k_train", 0),
                        xtr_score_relu=cfg.get("xtr_score_relu", False),
                        doc_gate_logits=kd_d_gate_logits,
                        doc_gate_score_scale=token_gate_score_scale,
                    )
                    w = kd_weight if loader else 1.0
                    total_loss = total_loss + w * kd_l
                    kd_loss_val = kd_l.item()

            scaler.scale(total_loss / grad_accum).backward()

            should_step = (step % grad_accum == 0) or (step == n_batches)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                if global_step + 1 < total_steps:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if freeze_encoder_steps > 0 and global_step == freeze_encoder_steps:
                    for p in model.encoder.parameters():
                        p.requires_grad = True
                    # Rebuild optimizer — head falls back to base lr (not projection_lr)
                    optimizer = build_optimizer(model, cfg, head_lr_override=cfg["lr"])
                    remaining = total_steps - global_step
                    if sched_type == "constant":
                        scheduler = get_constant_schedule_with_warmup(
                            optimizer, min(warmup, remaining),
                        )
                    elif sched_type == "linear":
                        scheduler = get_linear_schedule_with_warmup(
                            optimizer, min(warmup, remaining), remaining,
                        )
                    else:
                        scheduler = get_cosine_schedule_with_warmup(
                            optimizer, min(warmup, remaining), remaining,
                        )
                    print(f"\n  *** encoder unfrozen at step {global_step}, "
                          f"rebuilt optimizer for {remaining} remaining steps ***\n")

            running_loss += total_loss.item()

            if step % log_every == 0:
                avg = running_loss / step
                lr = max(scheduler.get_last_lr())
                row = f"{epoch}\t{step}\t{global_step}\t{total_loss.item():.6f}\t{avg:.6f}\t{lr:.2e}"
                if loader and kd_loader:
                    row += f"\t{c_loss_val:.6f}\t{kd_loss_val:.6f}"
                if feather_enabled and feather_stats is not None:
                    row += f"\t{feather_stats.sparsity:.6f}\t{feather_stats.threshold:.6e}"
                if embedding_feather_enabled and embedding_feather_stats is not None:
                    row += (
                        f"\t{embedding_feather_stats.sparsity:.6f}"
                        f"\t{embedding_feather_stats.threshold:.6e}"
                    )
                if sparse_bow_enabled:
                    row += (
                        f"\t{(sparse_bow_stats.sparsity if sparse_bow_stats else 0.0):.6f}"
                        f"\t{(sparse_bow_stats.threshold if sparse_bow_stats else 0.0):.6e}"
                    )
                    if sparse_bow_batch_stats is not None:
                        row += (
                            f"\t{sparse_bow_batch_stats.q_active:.3f}"
                            f"\t{sparse_bow_batch_stats.d_active:.3f}"
                            f"\t{sparse_bow_batch_stats.q_l1:.6f}"
                            f"\t{sparse_bow_batch_stats.d_l1:.6f}"
                            f"\t{sparse_bow_batch_stats.flops:.6f}"
                        )
                    else:
                        row += "\t0.000\t0.000\t0.000000\t0.000000\t0.000000"
                if token_gate_enabled:
                    if token_gate_batch_stats is not None:
                        row += (
                            f"\t{token_gate_batch_stats.active:.3f}"
                            f"\t{token_gate_batch_stats.target:.3f}"
                            f"\t{token_gate_batch_stats.keep_fraction:.6f}"
                            f"\t{token_gate_batch_stats.gate_mean:.6f}"
                        )
                    else:
                        row += "\t0.000\t0.000\t0.000000\t0.000000"
                    if token_gate_wordpiece_consistency_weight > 0.0:
                        row += (
                            f"\t{token_gate_wordpiece_loss_val:.6f}"
                            f"\t{token_gate_wordpiece_pair_count}"
                        )
                    if token_gate_maxsim_target_weight > 0.0:
                        row += (
                            f"\t{token_gate_maxsim_target_loss_val:.6f}"
                            f"\t{token_gate_maxsim_target_positive_rate:.6f}"
                        )
                    if token_gate_predictability_weight > 0.0:
                        row += (
                            f"\t{token_gate_predictability_loss_val:.6f}"
                            f"\t{token_gate_predictability_coverage:.6f}"
                            f"\t{token_gate_predictability_mean:.6f}"
                        )
                loss_log.write(row + "\n")
                loss_log.flush()
                loss_parts = f"loss={avg:.4f}"
                if loader and kd_loader:
                    loss_parts += f" (c={c_loss_val:.4f} kd={kd_loss_val:.4f})"
                elif c_score_mode:
                    loss_parts += f" ({c_score_mode})"
                if sigreg_weight > 0:
                    loss_parts += f" sr={sr_loss_val:.6f}"
                if feather_enabled and feather_stats is not None:
                    loss_parts += (
                        f" feather={100.0 * feather_stats.sparsity:.1f}%"
                        f" gth={feather_stats.threshold:.3e}"
                    )
                if embedding_feather_enabled and embedding_feather_stats is not None:
                    loss_parts += (
                        f" embfeather={100.0 * embedding_feather_stats.sparsity:.1f}%"
                        f" eth={embedding_feather_stats.threshold:.3e}"
                    )
                if sparse_bow_enabled and sparse_bow_stats is not None:
                    loss_parts += (
                        f" sbow={100.0 * sparse_bow_stats.sparsity:.2f}%"
                        f" th={sparse_bow_stats.threshold:.3e}"
                    )
                    if sparse_bow_batch_stats is not None:
                        loss_parts += (
                            f" qnnz={sparse_bow_batch_stats.q_active:.1f}"
                            f" dnnz={sparse_bow_batch_stats.d_active:.1f}"
                        )
                if token_gate_enabled and token_gate_batch_stats is not None:
                    loss_parts += (
                        f" gate={token_gate_batch_stats.active:.1f}"
                        f"/{token_gate_batch_stats.target:.1f}"
                        f" ({100.0 * token_gate_batch_stats.keep_fraction:.1f}%)"
                    )
                    if token_gate_wordpiece_consistency_weight > 0.0:
                        loss_parts += (
                            f" wpl={token_gate_wordpiece_loss_val:.4f}"
                            f"/{token_gate_wordpiece_pair_count}"
                        )
                    if token_gate_maxsim_target_weight > 0.0:
                        loss_parts += (
                            f" mxt={token_gate_maxsim_target_loss_val:.4f}"
                            f"/{100.0 * token_gate_maxsim_target_positive_rate:.1f}%"
                        )
                    if token_gate_predictability_weight > 0.0:
                        loss_parts += (
                            f" pred={token_gate_predictability_loss_val:.4f}"
                            f" cov={100.0 * token_gate_predictability_coverage:.1f}%"
                        )
                print(
                    f"  epoch {epoch}  step {step}/{n_batches}"
                    f"  {loss_parts}  lr={lr:.2e}"
                )

            if should_step and global_step % save_every == 0:
                ckpt = os.path.join(output_dir, f"step_{global_step}")
                save_training_checkpoint(
                    model,
                    tokenizer,
                    ckpt,
                    bake_feather=feather_bake_checkpoints,
                )
                shutil.copy2(loss_log_path, os.path.join(ckpt, "loss.tsv"))
                print(f"  → {ckpt}")
                saved_step_ckpts.append(ckpt)
                while keep_last >= 0 and len(saved_step_ckpts) > keep_last:
                    old_ckpt = saved_step_ckpts.pop(0)
                    shutil.rmtree(old_ckpt, ignore_errors=True)

            if global_step >= total_steps:
                break

        epoch_loss = running_loss / max(processed_batches, 1)
        print(f"epoch {epoch}  avg_loss={epoch_loss:.4f}")

        if cfg.get("save_each_epoch", True):
            ckpt = os.path.join(output_dir, f"epoch_{epoch}")
            save_training_checkpoint(
                model,
                tokenizer,
                ckpt,
                bake_feather=feather_bake_checkpoints,
            )
            shutil.copy2(loss_log_path, os.path.join(ckpt, "loss.tsv"))
            print(f"  → {ckpt}")

        if global_step >= total_steps:
            break

    loss_log.close()


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune XTRBert")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", help="Path to checkpoint dir to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.resume:
        cfg["resume"] = args.resume

    train(cfg)
