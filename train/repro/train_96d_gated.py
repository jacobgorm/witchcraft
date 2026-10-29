#!/usr/bin/env python3
"""Train the direct English 96d XTR model, then its document-token gate.

Usage: python train_96d_gated.py CHECKPOINT_DIR
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# ModernBERT may use an operation without a native Metal kernel. This must be
# set before torch initializes MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


# The complete recipe. Keep experiment machinery out of this file.
MODEL = "ibm-granite/granite-embedding-english-r2"
DATASET = "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
DATASET_CONFIG = "triplet"
QUERY_LENGTH = 32
DOC_LENGTH = 256
TEMPERATURE = 0.05
K_TRAIN = 256
SEED = 42
LOG_EVERY = 10

DIM = 96
BATCH_SIZE = 32
SAVE_EVERY = 250

XTR_STEPS = 1_000
XTR_LR = 3e-6
XTR_WARMUP = 100
# Stop at the observed optimum but preserve the LR trajectory of the original
# 2,000-step run through step 1,500.
GATE_STEPS = 1_500
GATE_SCHEDULE_STEPS = 2_000
GATE_LR = 1e-4
GATE_WARMUP = 50
GATE_TARGET = 0.30
GATE_RAMP = 500
GATE_SCORE_SCALE = 0.10
GATE_BUDGET_WEIGHT = 2.0
GATE_WORDPIECE_WEIGHT = 0.02
GATE_MAXSIM_WEIGHT = 0.05
PREFIX_FACTORS = (1.0, 2.0, 4.0)
PREFIX_WEIGHTS = (1.0, 0.75, 0.5)


def device_for_training() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class Triplets(Dataset):
    def __init__(self):
        self.rows = load_dataset(DATASET, DATASET_CONFIG, split="train")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def text(value):
        if isinstance(value, list):
            value = value[0]
        if isinstance(value, dict):
            value = value.get("text", value.get("title", str(value)))
        return value

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        return {
            "query": self.text(row["query"]),
            "positive": self.text(row["positive"]),
            "negative": self.text(row["negative"]),
        }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        queries = [row["query"] for row in rows]
        # Positives must precede negatives: target i is document i.
        documents = [row["positive"] for row in rows] + [row["negative"] for row in rows]
        q = self.tokenizer(
            queries, padding=True, truncation=True, max_length=QUERY_LENGTH,
            return_tensors="pt",
        )
        d = self.tokenizer(
            documents, padding=True, truncation=True, max_length=DOC_LENGTH,
            return_tensors="pt",
        )
        return {
            "query_ids": q.input_ids,
            "query_mask": q.attention_mask,
            "doc_ids": d.input_ids,
            "doc_mask": d.attention_mask,
        }


class XTRModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Granite advertises BF16 weights. MPS currently fails when a BF16
        # matmul feeds an FP32 accumulator, so materialize the encoder in FP32.
        self.encoder = AutoModel.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.float32
        )
        hidden = self.encoder.config.hidden_size
        self.linear = nn.Linear(hidden, DIM)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)
        self.token_gate_norm: nn.LayerNorm | None = None
        self.token_gate: nn.Linear | None = None

    def add_gate(self) -> None:
        if self.token_gate is not None:
            return
        hidden = self.encoder.config.hidden_size
        self.token_gate_norm = nn.LayerNorm(hidden)
        self.token_gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.token_gate.weight)
        nn.init.constant_(self.token_gate.bias, 2.944)  # sigmoid ~= 0.95

    def encode(self, ids: torch.Tensor, mask: torch.Tensor, gate: bool = False):
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        raw = self.linear(hidden)
        vectors = F.normalize(raw, dim=-1) * mask.unsqueeze(-1)
        if not gate:
            return vectors
        assert self.token_gate is not None and self.token_gate_norm is not None
        logits = self.token_gate(self.token_gate_norm(hidden)).squeeze(-1)
        return vectors, logits.masked_fill(~mask.bool(), -float("inf"))


def xtr_scores(
    queries: torch.Tensor,
    documents: torch.Tensor,
    query_mask: torch.Tensor,
    document_mask: torch.Tensor,
    gate_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """XTR top-k score with a simple MPS-compatible document loop."""
    batch, query_tokens, dim = queries.shape
    num_docs, doc_tokens, _ = documents.shape
    sim = torch.einsum("bqd,nd->bqn", queries, documents.reshape(-1, dim))
    flat_mask = document_mask.reshape(-1).bool()
    sim = sim.masked_fill(~flat_mask.view(1, 1, -1), torch.finfo(sim.dtype).min)

    if gate_logits is not None:
        bias = GATE_SCORE_SCALE * F.logsigmoid(gate_logits.float()).reshape(-1)
        sim = sim + bias.view(1, 1, -1).to(sim.dtype)

    k = min(K_TRAIN, sim.size(-1))
    values, indices = sim.topk(k, dim=-1)
    selected_docs = indices // doc_tokens
    selected_valid = flat_mask[indices]
    real_counts = document_mask.sum(dim=-1)

    per_document = []
    retrieved = []
    for doc in range(num_docs):
        chosen = selected_valid & selected_docs.eq(doc)
        best = values.masked_fill(~chosen, -float("inf")).amax(dim=-1)
        count = chosen.sum(dim=-1)
        # XTR's unselected real token slots are zero-valued competitors.
        best = torch.where(count < real_counts[doc], best.clamp_min(0), best)
        best = torch.where(real_counts[doc] > 0, best, torch.zeros_like(best))
        per_document.append(best)
        retrieved.append(count.gt(0))

    maxsim = torch.stack(per_document, dim=-1)
    zmask = torch.stack(retrieved, dim=-1) & query_mask.bool().unsqueeze(-1)
    maxsim = maxsim * query_mask.unsqueeze(-1)
    z = zmask.sum(dim=1).clamp_min(1)
    return maxsim.sum(dim=1) / z


def contrastive_loss(q, d, qmask, dmask, gates=None) -> torch.Tensor:
    scores = xtr_scores(q, d, qmask, dmask, gates) / TEMPERATURE
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(scores, labels)


def keep_counts(mask: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    lengths = mask.sum(dim=-1)
    return (lengths.float() * GATE_TARGET * factor).clamp(min=1).minimum(lengths).ceil().long()


def prefix_mask(logits: torch.Tensor, mask: torch.Tensor, factor: float) -> torch.Tensor:
    result = torch.zeros_like(mask, dtype=torch.bool)
    scores = logits.float().masked_fill(~mask.bool(), -float("inf"))
    counts = keep_counts(mask, factor)
    for row in range(scores.size(0)):
        result[row, scores[row].topk(int(counts[row].item())).indices] = True
    return result & mask.bool()


def budget_loss(logits: torch.Tensor, mask: torch.Tensor, progress: float):
    mask = mask.bool()
    lengths = mask.sum(dim=-1).float().clamp_min(1)
    final = (GATE_TARGET * lengths).clamp_min(1).minimum(lengths)
    target = lengths * (1 - progress) + final * progress
    active = torch.sigmoid(logits.float()).masked_fill(~mask, 0).sum(dim=-1)
    return (((active - target) / lengths) ** 2).mean(), (active / lengths).mean()


def word_start_flags(tokenizer) -> torch.Tensor:
    tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    special = set(tokenizer.all_special_ids)
    space_style = any(isinstance(t, str) and (t.startswith("Ġ") or t.startswith("▁")) for t in tokens)
    hash_style = any(isinstance(t, str) and t.startswith("##") for t in tokens)
    starts = []
    for index, token in enumerate(tokens):
        if index in special or not isinstance(token, str) or not token:
            starts.append(True)
        elif space_style:
            starts.append(token.startswith("Ġ") or token.startswith("▁"))
        elif hash_style:
            starts.append(not token.startswith("##"))
        else:
            starts.append(True)
    return torch.tensor(starts, dtype=torch.bool)


def wordpiece_loss(logits, ids, mask, starts):
    is_start = starts.to(ids.device)[ids]
    pairs = mask[:, 1:].bool() & mask[:, :-1].bool() & ~is_start[:, 1:]
    if not pairs.any():
        return logits.float().new_zeros(())
    delta = logits.float()[:, 1:] - logits.float()[:, :-1]
    return F.smooth_l1_loss(delta[pairs], torch.zeros_like(delta[pairs]))


def maxsim_target_loss(q, qmask, d, dmask, logits):
    # Only the first B documents are paired positives.
    batch = q.size(0)
    valid = dmask[:batch].bool()
    with torch.no_grad():
        sim = torch.einsum("bqd,bnd->bqn", q.float(), d[:batch].float())
        sim = sim.masked_fill(~qmask.bool().unsqueeze(-1), -float("inf"))
        sim = sim.masked_fill(~valid.unsqueeze(1), -float("inf"))
        salience = sim.amax(dim=1)
        labels = torch.zeros_like(logits[:batch], dtype=torch.float32)
        counts = keep_counts(valid)
        for row in range(batch):
            labels[row, salience[row].topk(int(counts[row].item())).indices] = 1
    return F.binary_cross_entropy_with_logits(logits[:batch].float()[valid], labels[valid])


def save_checkpoint(
    model: XTRModel,
    tokenizer,
    path: Path,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "xtrbert.pt")
    config = model.encoder.config.to_dict()
    config["projection_dim"] = DIM
    config["token_gate"] = model.token_gate is not None
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(path)


def load_xtr_checkpoint(model: XTRModel, path: Path) -> None:
    state = torch.load(path / "xtrbert.pt", map_location="cpu", weights_only=True)
    if any(key.startswith("token_gate") for key in state):
        model.add_gate()
    model.load_state_dict(state, strict=True)


def loader(tokenizer):
    return DataLoader(
        Triplets(), batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=0, collate_fn=Collator(tokenizer),
    )


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_xtr(output: Path, tokenizer, device: torch.device) -> Path:
    model = XTRModel()
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=XTR_LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, XTR_WARMUP, XTR_STEPS
    )
    data = loader(tokenizer)
    iterator = iter(data)
    out = output / "xtr96"

    for step in range(1, XTR_STEPS + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data)
            batch = next(iterator)
        batch = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        q = model.encode(batch["query_ids"], batch["query_mask"])
        d = model.encode(batch["doc_ids"], batch["doc_mask"])
        loss = contrastive_loss(q, d, batch["query_mask"], batch["doc_mask"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % LOG_EVERY == 0:
            print(
                f"xtr step={step}/{XTR_STEPS} loss={loss.item():.6f} "
                f"lr={scheduler.get_last_lr()[0]:.3e}",
                flush=True,
            )
        if step % SAVE_EVERY == 0 and step != XTR_STEPS:
            save_checkpoint(model, tokenizer, out / f"step_{step}")

    final = out / "final"
    save_checkpoint(model, tokenizer, final)
    return final


def train_gate(output: Path, tokenizer, device, checkpoint: Path) -> Path:
    model = XTRModel()
    load_xtr_checkpoint(model, checkpoint)
    model.add_gate()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.token_gate_norm.parameters():
        parameter.requires_grad = True
    for parameter in model.token_gate.parameters():
        parameter.requires_grad = True
    model.to(device).train()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=GATE_LR, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, GATE_WARMUP, GATE_SCHEDULE_STEPS
    )
    starts = word_start_flags(tokenizer)
    data = loader(tokenizer)
    iterator = iter(data)
    out = output / "gate30"

    for step in range(1, GATE_STEPS + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data)
            batch = next(iterator)
        batch = move(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            q_hidden = model.encoder(
                input_ids=batch["query_ids"], attention_mask=batch["query_mask"]
            ).last_hidden_state
            d_hidden = model.encoder(
                input_ids=batch["doc_ids"], attention_mask=batch["doc_mask"]
            ).last_hidden_state
            q = F.normalize(model.linear(q_hidden), dim=-1) * batch["query_mask"].unsqueeze(-1)
            d = F.normalize(model.linear(d_hidden), dim=-1) * batch["doc_mask"].unsqueeze(-1)
        logits = model.token_gate(model.token_gate_norm(d_hidden.detach())).squeeze(-1)
        logits = logits.masked_fill(~batch["doc_mask"].bool(), -float("inf"))

        progress = min(1.0, step / GATE_RAMP)
        base = contrastive_loss(q, d, batch["query_mask"], batch["doc_mask"], logits)
        nested = sum(
            weight * contrastive_loss(
                q, d, batch["query_mask"],
                prefix_mask(logits, batch["doc_mask"], factor), logits,
            )
            for factor, weight in zip(PREFIX_FACTORS, PREFIX_WEIGHTS)
        ) / sum(PREFIX_WEIGHTS)
        budget, keep = budget_loss(logits, batch["doc_mask"], progress)
        wp = wordpiece_loss(logits, batch["doc_ids"], batch["doc_mask"], starts)
        target = maxsim_target_loss(q, batch["query_mask"], d, batch["doc_mask"], logits)
        loss = (
            (1.0 - progress * 0.35) * base
            + progress * 0.35 * nested
            + progress * GATE_BUDGET_WEIGHT * budget
            + progress * GATE_WORDPIECE_WEIGHT * wp
            + progress * GATE_MAXSIM_WEIGHT * target
        )
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % LOG_EVERY == 0:
            print(
                f"gate step={step}/{GATE_STEPS} schedule={GATE_SCHEDULE_STEPS} "
                f"loss={loss.item():.6f} keep={keep.item():.3f} "
                f"budget={budget.item():.5f} wordpiece={wp.item():.5f} "
                f"maxsim={target.item():.5f} lr={scheduler.get_last_lr()[0]:.3e}",
                flush=True,
            )
        if step % SAVE_EVERY == 0 and step != GATE_STEPS:
            save_checkpoint(model, tokenizer, out / f"step_{step}")

    final = out / "final"
    save_checkpoint(model, tokenizer, final)
    return final


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} CHECKPOINT_DIR")
    output = Path(sys.argv[1])
    seed_everything(SEED)
    device = device_for_training()
    print(
        f"device={device} model={MODEL} dim={DIM} batch={BATCH_SIZE} "
        f"gate_stop={GATE_STEPS}"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    checkpoint = train_xtr(output, tokenizer, device)
    result = train_gate(output, tokenizer, device, checkpoint)
    print(f"finished: {result}")


if __name__ == "__main__":
    main()
