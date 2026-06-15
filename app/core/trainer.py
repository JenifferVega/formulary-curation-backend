"""Full-parameter retraining on HF data + curated examples.

Strategy: prevent catastrophic forgetting by mixing HF training data with
curated examples. All model parameters are updated. Training runs in a
background thread. Trained models are saved to TRAINED_MODELS_DIR/{key}/
and reloaded on startup.

Training improvements aligned with original training scripts:
- All models: warmup scheduler (warmup_ratio=0.1) + gradient clipping (1.0)
- Segmenter: sliding window stride=128, LR 5e-5 (matches original)
- FLAN-T5:   bf16 on CUDA (fp16 causes NaN gradients in T5)
- Matcher:   pairwise MarginRankingLoss on (query, pos, neg) triples,
             num_labels=1 regression, GRAD_ACCUM=4, AMP
"""
from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app import config
from app.core import store

# ---------------------------------------------------------------------------
# Config per model
# ---------------------------------------------------------------------------

TRAIN_META: dict[str, dict] = {
    "relevance": {
        "name": "1 · Pages",
        "kind": "sequence",
        "repo": config.RELEVANCE_MODEL,
        "dataset_dir": config.RELEVANCE_OUT,
        "num_labels": 2,
        "max_len": 512,
        "batch_size": 8,
        "lr": 2e-5,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
    },
    "segmenter": {
        "name": "2 · Drug entries",
        "kind": "token",
        "repo": config.SEGMENTER_MODEL,
        "dataset_dir": config.SEGMENTER_OUT,
        "max_len": 512,
        "stride": 128,           # sliding window stride (matches original)
        "batch_size": 4,
        "lr": 5e-5,              # original used 5e-5, not 2e-5
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
    },
    "span_classifier": {
        "name": "3 · Single/Multi/Broken",
        "kind": "sequence",
        "repo": config.SPAN_CLASSIFIER_MODEL,
        "dataset_dir": config.SPAN_CLASSIFIER_OUT,
        "num_labels": 3,
        "max_len": 256,
        "batch_size": 16,
        "lr": 2e-5,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
    },
    "cleaner": {
        "name": "4 · Clean / Normalize",
        "kind": "seq2seq",
        "repo": config.SEQ2SEQ_MODEL,
        "dataset_dir": config.CLEANER_OUT,
        "max_len": 128,
        "batch_size": 8,
        "lr": 3e-5,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
    },
    "matcher": {
        "name": "5 · RxNorm match",
        "kind": "sequence_pair",
        "repo": config.PAIRWISE_MODEL,
        "dataset_dir": config.MATCHER_OUT,
        "num_labels": 1,         # regression head (matches original BGE training)
        "max_len": 192,
        "batch_size": 8,
        "grad_accum": 4,         # effective batch = 32 triples
        "margin": 1.0,           # MarginRankingLoss margin
        "lr": 2e-5,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
    },
}

# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

@dataclass
class _TrainState:
    status: str = "idle"        # idle | running | done | error
    progress: int = 0           # 0-100
    message: str = ""
    error: str = ""
    trained_at: Optional[float] = None
    epochs_done: int = 0
    total_examples: int = 0
    conflicts: list = None      # list of {text, labels, sources} dicts
    push_status: str = "idle"   # idle | running | done | error
    push_message: str = ""
    push_error: str = ""

    def __post_init__(self):
        if self.conflicts is None:
            self.conflicts = []


_states: dict[str, _TrainState] = {k: _TrainState() for k in TRAIN_META}
_trained_models: dict[str, tuple] = {}   # key -> (tok, model, device)
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Curated data helpers
# ---------------------------------------------------------------------------

def _fingerprint(r: dict, key: str) -> str:
    """Stable hash of the input field(s) used to deduplicate across HF + curated."""
    import hashlib
    if key == "cleaner":
        text = r.get("input", "")
    elif key == "matcher":
        text = (str(r.get("query") or r.get("raw_text", ""))
                + "\x00"
                + str(r.get("candidate_text", "")))
    else:
        text = r.get("text", "")
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def curated_records(key: str) -> list[dict]:
    """All curated examples from local JSONL, tagged with _source_file."""
    meta = TRAIN_META[key]
    records: list[dict] = []
    for fp in sorted(Path(meta["dataset_dir"]).glob("*.jsonl")):
        for r in store.read_file(fp):
            if isinstance(r, dict):
                r["_source_file"] = fp.name
                records.append(r)
    return records


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def _label_for_conflict(r: dict, key: str) -> str:
    """Human-readable label string for a record, used in conflict reporting."""
    if key == "cleaner":
        return r.get("output", "")
    if key == "matcher":
        return r.get("bucket", "")
    if key == "segmenter":
        spans = r.get("spans", [])
        return f"{len(spans)} spans"
    lbl = r.get("label", "")
    # Normalize int labels for span_classifier
    if isinstance(lbl, int):
        return {0: "single", 1: "multi", 2: "broken"}.get(lbl, str(lbl))
    return str(lbl)


def detect_conflicts(key: str, curated: list[dict]) -> list[dict]:
    """Find records in curated that share the same input text but have different labels.

    Returns a list of conflict dicts:
        {
            "snippet":  first 120 chars of the input text,
            "labels":   list of distinct label values found,
            "sources":  list of _source_file values where conflicts appear,
            "count":    total number of conflicting records,
        }
    """
    # Group by fingerprint
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in curated:
        groups[_fingerprint(r, key)].append(r)

    conflicts = []
    for fp, recs in groups.items():
        if len(recs) < 2:
            continue
        labels = [_label_for_conflict(r, key) for r in recs]
        unique_labels = list(dict.fromkeys(labels))  # preserve order, dedupe
        if len(unique_labels) < 2:
            continue  # same label repeated — not a conflict

        # Build a readable snippet of the input
        if key == "cleaner":
            snippet = recs[0].get("input", "")[:120]
        elif key == "matcher":
            q = recs[0].get("query") or recs[0].get("raw_text", "")
            c = recs[0].get("candidate_text", "")
            snippet = f"{q[:60]} ↔ {c[:60]}"
        else:
            snippet = recs[0].get("text", "")[:120]

        sources = list(dict.fromkeys(
            r.get("_source_file", "unknown") for r in recs
        ))
        conflicts.append({
            "snippet": snippet,
            "labels": unique_labels,
            "sources": sources,
            "count": len(recs),
        })

    return conflicts


# ---------------------------------------------------------------------------
# Label normalizers
# ---------------------------------------------------------------------------

def _relevance_label(r: dict) -> int:
    lbl = r.get("label")
    if isinstance(lbl, int):
        return lbl
    s = str(lbl).upper()
    if s == "RELEVANT":
        return 1
    if s in ("NOT_RELEVANT", "NOT RELEVANT"):
        return 0
    return -1


def _span_label(r: dict) -> int:
    lbl = r.get("label")
    if isinstance(lbl, int):
        return lbl
    if str(lbl).isdigit():
        return int(lbl)
    return {"single": 0, "multi": 1, "broken": 2}.get(str(lbl).lower(), -1)


def _matcher_label(r: dict) -> int:
    return 1 if r.get("bucket") == "positive" else 0


# ---------------------------------------------------------------------------
# BIO alignment (segmenter)
# ---------------------------------------------------------------------------

def _align_bio(offsets: list, spans: list[tuple], label2id: dict) -> list[int]:
    """Map character spans → per-token BIO label IDs. -100 for special tokens."""
    o_id = label2id.get("O", 0)
    b_id = label2id.get("B-DRUG", 1)
    i_id = label2id.get("I-DRUG", 2)
    spans_sorted = sorted(spans, key=lambda s: s[0])
    labels: list[int] = []
    for tok_s, tok_e in offsets:
        if tok_s == tok_e:
            labels.append(-100)
            continue
        lbl = o_id
        for s, e in spans_sorted:
            if tok_e <= s:
                break
            if tok_s >= e:
                continue
            lbl = b_id if tok_s <= s else i_id
            break
        labels.append(lbl)
    return labels


# ---------------------------------------------------------------------------
# Shared scheduler helper
# ---------------------------------------------------------------------------

def _make_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    from transformers import get_linear_schedule_with_warmup
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


# ---------------------------------------------------------------------------
# Training implementations
# ---------------------------------------------------------------------------

def _train_sequence(key: str, records: list[dict], epochs: int, state: _TrainState) -> tuple:
    """Models 1 (Relevance) and 3 (SpanClassifier) — DistilBERT sequence classification."""
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta = TRAIN_META[key]
    label_fn = _relevance_label if key == "relevance" else _span_label
    device = config.resolve_device()

    state.message = f"Loading {meta['name']} from HuggingFace…"
    tok = AutoTokenizer.from_pretrained(meta["repo"], token=config.HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        meta["repo"], token=config.HF_TOKEN, num_labels=meta["num_labels"]
    ).to(device).train()

    valid = [(r.get("text", ""), label_fn(r)) for r in records]
    valid = [(t, l) for t, l in valid if t and l >= 0]
    if not valid:
        raise RuntimeError("No valid labeled records after filtering.")

    bs = meta["batch_size"]
    total_steps = epochs * max(1, (len(valid) + bs - 1) // bs)
    optimizer = AdamW(model.parameters(), lr=meta["lr"], weight_decay=0.01)
    scheduler = _make_scheduler(optimizer, total_steps, meta["warmup_ratio"])
    done_steps = 0

    for epoch in range(epochs):
        random.shuffle(valid)
        for i in range(0, len(valid), bs):
            batch = valid[i: i + bs]
            texts, labels = zip(*batch)
            enc = tok(list(texts), return_tensors="pt", padding=True,
                      truncation=True, max_length=meta["max_len"])
            enc = {k: v.to(device) for k, v in enc.items()}
            lbl_t = torch.tensor(list(labels), dtype=torch.long).to(device)
            loss = model(**enc, labels=lbl_t).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), meta["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            done_steps += 1
            state.progress = int(5 + 90 * done_steps / total_steps)
            state.message = (f"Epoch {epoch+1}/{epochs} · batch "
                             f"{done_steps} · loss {loss.item():.4f}")

    return tok, model, device


def _train_token(key: str, records: list[dict], epochs: int, state: _TrainState) -> tuple:
    """Model 2 (Segmenter) — XLM-RoBERTa BIO token classification with sliding window."""
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    meta = TRAIN_META[key]
    device = config.resolve_device()
    stride = meta.get("stride", 128)

    state.message = "Loading segmenter model…"
    tok = AutoTokenizer.from_pretrained(meta["repo"], token=config.HF_TOKEN)
    model = AutoModelForTokenClassification.from_pretrained(
        meta["repo"], token=config.HF_TOKEN
    ).to(device).train()

    label2id = {v: int(k) for k, v in model.config.id2label.items()}

    valid = [r for r in records if r.get("text") and isinstance(r.get("spans"), list)]
    if not valid:
        raise RuntimeError("No valid segmenter records (need {text, spans}).")

    # Pre-build all sliding windows so we know total_steps for the scheduler
    def _build_windows(r: dict) -> list[dict]:
        text = r["text"]
        raw_spans = r.get("spans") or []
        span_list: list[tuple] = []
        for sp in raw_spans:
            if isinstance(sp, (list, tuple)) and len(sp) >= 2:
                span_list.append((int(sp[0]), int(sp[1])))
            elif isinstance(sp, dict):
                span_list.append((int(sp.get("start", 0)), int(sp.get("end", 0))))

        enc = tok(
            text,
            max_length=meta["max_len"],
            truncation=True,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        windows = []
        for w in range(len(enc["input_ids"])):
            offsets = enc["offset_mapping"][w]
            bio = _align_bio(offsets, span_list, label2id)
            windows.append({
                "input_ids":      enc["input_ids"][w],
                "attention_mask": enc["attention_mask"][w],
                "bio":            bio,
            })
        return windows

    all_windows = []
    for r in valid:
        all_windows.extend(_build_windows(r))

    accum = meta["batch_size"]
    total_steps = epochs * max(1, len(all_windows) // accum)
    optimizer = AdamW(model.parameters(), lr=meta["lr"], weight_decay=0.01)
    scheduler = _make_scheduler(optimizer, total_steps, meta["warmup_ratio"])
    done = 0

    for epoch in range(epochs):
        random.shuffle(all_windows)
        optimizer.zero_grad()
        for idx, w in enumerate(all_windows):
            input_ids = torch.tensor([w["input_ids"]], dtype=torch.long).to(device)
            attention_mask = torch.tensor([w["attention_mask"]], dtype=torch.long).to(device)
            lbl_t = torch.tensor([w["bio"]], dtype=torch.long).to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask,
                         labels=lbl_t).loss / accum
            loss.backward()

            if (idx + 1) % accum == 0 or idx == len(all_windows) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), meta["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            done += 1
            state.progress = int(5 + 90 * done / (epochs * len(all_windows)))
            state.message = (f"Epoch {epoch+1}/{epochs} · window {idx+1}/{len(all_windows)} "
                             f"· loss {loss.item() * accum:.4f}")

    return tok, model, device


def _train_seq2seq(key: str, records: list[dict], epochs: int, state: _TrainState) -> tuple:
    """Model 4 (Cleaner) — FLAN-T5 seq2seq. Uses bf16 on CUDA (fp16 causes NaN in T5)."""
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    meta = TRAIN_META[key]
    device = config.resolve_device()

    # FLAN-T5 produces NaN gradients with fp16 — use bf16 on CUDA, fp32 on CPU
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())

    state.message = "Loading FLAN-T5 model…"
    tok = AutoTokenizer.from_pretrained(meta["repo"], token=config.HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        meta["repo"], token=config.HF_TOKEN,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    ).to(device).train()

    valid = [(r.get("input", ""), r.get("output", "")) for r in records]
    valid = [(i, o) for i, o in valid if i and o]
    if not valid:
        raise RuntimeError("No valid cleaner records (need {input, output}).")

    bs = meta["batch_size"]
    total_steps = epochs * max(1, (len(valid) + bs - 1) // bs)
    optimizer = AdamW(model.parameters(), lr=meta["lr"], weight_decay=0.01)
    scheduler = _make_scheduler(optimizer, total_steps, meta["warmup_ratio"])
    done_steps = 0

    for epoch in range(epochs):
        random.shuffle(valid)
        for i in range(0, len(valid), bs):
            batch = valid[i: i + bs]
            srcs = [f"{config.SEQ2SEQ_INSTRUCTION} {inp}" for inp, _ in batch]
            tgts = [out for _, out in batch]
            enc = tok(srcs, return_tensors="pt", padding=True,
                      truncation=True, max_length=meta["max_len"])
            tgt_enc = tok(tgts, return_tensors="pt", padding=True,
                          truncation=True, max_length=meta["max_len"])
            labels = tgt_enc["input_ids"].clone()
            labels[labels == tok.pad_token_id] = -100
            enc = {k: v.to(device) for k, v in enc.items()}
            loss = model(**enc, labels=labels.to(device)).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), meta["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            done_steps += 1
            state.progress = int(5 + 90 * done_steps / total_steps)
            state.message = (f"Epoch {epoch+1}/{epochs} · batch "
                             f"{done_steps} · loss {loss.item():.4f}")

    return tok, model, device


def _train_sequence_pair(key: str, records: list[dict], epochs: int, state: _TrainState) -> tuple:
    """Model 5 (Matcher) — BGE cross-encoder with pairwise MarginRankingLoss.

    Groups records by query, builds (query, positive, negative) triples each epoch,
    and trains with MarginRankingLoss so the model learns to rank positives above
    negatives — matching how the original cross_encoder_pairwise_bge was trained.
    Uses num_labels=1 regression (not binary classification).
    """
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta = TRAIN_META[key]
    device = config.resolve_device()
    use_amp = (device == "cuda")
    grad_accum = meta.get("grad_accum", 4)
    margin = meta.get("margin", 1.0)

    state.message = "Loading matcher model…"
    tok = AutoTokenizer.from_pretrained(meta["repo"], token=config.HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        meta["repo"], token=config.HF_TOKEN,
        num_labels=1,
        problem_type="regression",
    ).to(device).train()

    # Group records by query into {query, positives, negatives}
    groups: dict[str, dict] = defaultdict(lambda: {"query": "", "pos": [], "neg": []})
    for r in records:
        q = r.get("query") or r.get("raw_text", "")
        c = r.get("candidate_text", "")
        if not q or not c:
            continue
        g = groups[q]
        g["query"] = q
        if _matcher_label(r) == 1:
            g["pos"].append(c)
        else:
            g["neg"].append(c)

    usable = [g for g in groups.values() if g["pos"] and g["neg"]]
    if not usable:
        raise RuntimeError(
            "No usable matcher groups — need records with both positive and "
            "negative candidates for the same query (bucket='positive' and others)."
        )

    def _build_triples(seed: int) -> list[tuple]:
        rng = random.Random(seed)
        triples = []
        for g in usable:
            for pos in g["pos"]:
                neg = rng.choice(g["neg"])
                triples.append((g["query"], pos, neg))
        rng.shuffle(triples)
        return triples

    bs = meta["batch_size"]
    # Estimate total optimizer steps
    triples_per_epoch = sum(len(g["pos"]) for g in usable)
    steps_per_epoch = max(1, (triples_per_epoch + bs - 1) // bs // grad_accum)
    total_steps = epochs * steps_per_epoch

    # Split weight decay: no decay on bias and LayerNorm
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=meta["lr"])
    scheduler = _make_scheduler(optimizer, total_steps, meta["warmup_ratio"])
    rank_loss = torch.nn.MarginRankingLoss(margin=margin)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    done_steps = 0
    optimizer.zero_grad()

    for epoch in range(epochs):
        triples = _build_triples(epoch * 1000)
        step_in_epoch = 0

        for i in range(0, len(triples), bs):
            batch = triples[i: i + bs]
            queries = [t[0] for t in batch]
            positives = [t[1] for t in batch]
            negatives = [t[2] for t in batch]

            enc_pos = tok(queries, positives, return_tensors="pt", padding=True,
                          truncation=True, max_length=meta["max_len"])
            enc_neg = tok(queries, negatives, return_tensors="pt", padding=True,
                          truncation=True, max_length=meta["max_len"])
            enc_pos = {k: v.to(device) for k, v in enc_pos.items()}
            enc_neg = {k: v.to(device) for k, v in enc_neg.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                s_pos = model(**enc_pos).logits.squeeze(-1)
                s_neg = model(**enc_neg).logits.squeeze(-1)
                target = torch.ones_like(s_pos)
                loss = rank_loss(s_pos, s_neg, target) / grad_accum

            scaler.scale(loss).backward()
            step_in_epoch += 1

            if step_in_epoch % grad_accum == 0 or i + bs >= len(triples):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), meta["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                done_steps += 1

            pair_acc = (s_pos > s_neg).float().mean().item()
            state.progress = int(5 + 90 * (epoch * len(triples) + i) /
                                 (epochs * max(1, len(triples))))
            state.message = (f"Epoch {epoch+1}/{epochs} · "
                             f"triple {i+len(batch)}/{len(triples)} · "
                             f"loss {loss.item() * grad_accum:.4f} · "
                             f"pair_acc {pair_acc:.2f}")

    return tok, model, device


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _save_model(key: str, tok, model) -> None:
    save_dir = config.TRAINED_MODELS_DIR / key
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    tok.save_pretrained(str(save_dir))
    model.save_pretrained(str(save_dir))


def _run(key: str, records: list[dict], epochs: int) -> None:
    state = _states[key]
    try:
        state.status = "running"
        state.progress = 0
        state.error = ""
        state.total_examples = len(records)

        kind = TRAIN_META[key]["kind"]
        if kind == "sequence":
            tok, model, device = _train_sequence(key, records, epochs, state)
        elif kind == "token":
            tok, model, device = _train_token(key, records, epochs, state)
        elif kind == "seq2seq":
            tok, model, device = _train_seq2seq(key, records, epochs, state)
        elif kind == "sequence_pair":
            tok, model, device = _train_sequence_pair(key, records, epochs, state)
        else:
            raise RuntimeError(f"Unknown kind: {kind}")

        state.message = "Saving model to disk…"
        state.progress = 96
        _save_model(key, tok, model)

        with _lock:
            _trained_models[key] = (tok, model, device)

        state.status = "done"
        state.progress = 100
        state.trained_at = time.time()
        state.epochs_done = epochs
        state.message = (f"Trained · {len(records)} examples · "
                         f"{epochs} epochs · saved to disk")

    except Exception as exc:  # noqa: BLE001
        state.status = "error"
        state.error = str(exc)
        state.message = f"Training failed: {exc}"
        state.progress = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_training(key: str, indices: list[int] | None, epochs: int) -> None:
    """Launch background training. indices=None means use all curated records.
    Training set = selected curated + all HF records (to prevent catastrophic forgetting).
    """
    if key not in TRAIN_META:
        raise KeyError(f"Unknown model key: '{key}'")
    if _states[key].status == "running":
        raise RuntimeError(f"Training already running for '{key}'")

    curated = curated_records(key)
    if indices is not None:
        curated = [curated[i] for i in indices if 0 <= i < len(curated)]

    # Detect label conflicts across curated files before training
    _states[key].conflicts = detect_conflicts(key, curated)

    if len(curated) < config.MIN_TRAIN_EXAMPLES:
        raise ValueError(
            f"Need ≥{config.MIN_TRAIN_EXAMPLES} curated examples, got {len(curated)}"
        )

    # Mix in original HF data to prevent catastrophic forgetting
    hf_recs: list[dict] = []
    try:
        from app.core import hf_datasets
        data = hf_datasets.get(key)
        hf_recs = data["records"]
    except Exception:  # noqa: BLE001
        pass  # proceed with curated only if HF unavailable

    # Deduplicate across HF + curated: curated wins when the same input exists in both
    # (user may have curated a corrected label for an HF example)
    seen: dict[str, dict] = {}
    for r in hf_recs + curated:          # curated last → overwrites HF on collision
        seen[_fingerprint(r, key)] = r
    all_records = list(seen.values())

    _states[key].status = "idle"   # reset before thread starts
    threading.Thread(target=_run, args=(key, all_records, epochs), daemon=True).start()


def training_status(key: str) -> dict:
    if key not in _states:
        raise KeyError(key)
    s = _states[key]
    return {
        "key": key,
        "status": s.status,
        "progress": s.progress,
        "message": s.message,
        "error": s.error,
        "trained_at": s.trained_at,
        "epochs_done": s.epochs_done,
        "total_examples": s.total_examples,
        "conflicts": s.conflicts,
        "push_status": s.push_status,
        "push_message": s.push_message,
        "push_error": s.push_error,
    }


def get_trained_model(key: str) -> tuple | None:
    """Return (tok, model, device) for trained model, or None if not available."""
    with _lock:
        return _trained_models.get(key)


def discard_trained_model(key: str) -> None:
    """Remove trained model from memory and delete saved files from disk."""
    import shutil
    if key not in TRAIN_META:
        raise KeyError(f"Unknown model key: '{key}'")
    if _states[key].status == "running":
        raise RuntimeError("Cannot discard a model while training is running.")

    with _lock:
        _trained_models.pop(key, None)

    save_dir = config.TRAINED_MODELS_DIR / key
    if save_dir.exists():
        shutil.rmtree(save_dir)

    s = _states[key]
    s.status = "idle"
    s.progress = 0
    s.message = ""
    s.error = ""
    s.trained_at = None
    s.epochs_done = 0
    s.total_examples = 0
    s.conflicts = []
    s.push_status = "idle"
    s.push_message = ""
    s.push_error = ""


def push_to_hub(key: str) -> None:
    """Launch background push of trained model + merged dataset to HuggingFace Hub."""
    if key not in TRAIN_META:
        raise KeyError(f"Unknown model key: '{key}'")
    s = _states[key]
    if s.push_status == "running":
        raise RuntimeError("Push already in progress.")
    with _lock:
        entry = _trained_models.get(key)
    if entry is None:
        raise RuntimeError("No trained model to push. Train first.")
    if not config.HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set — cannot push to HuggingFace.")

    def _run_push():
        tok, model, _ = entry
        meta = TRAIN_META[key]
        model_repo = meta["repo"]
        dataset_repo = f"{model_repo}-dataset"
        s.push_status = "running"
        s.push_error = ""
        try:
            # ── 1. Build merged dataset (same logic as training) ──────────────
            s.push_message = "Building merged dataset…"
            curated = curated_records(key)
            hf_recs: list[dict] = []
            try:
                from app.core import hf_datasets
                data = hf_datasets.get(key)
                hf_recs = data["records"]
            except Exception:
                pass
            seen: dict[str, dict] = {}
            for r in hf_recs + curated:
                seen[_fingerprint(r, key)] = r
            all_records = list(seen.values())

            # Strip internal fields before pushing
            clean = []
            for r in all_records:
                clean.append({k: v for k, v in r.items() if not k.startswith("_")})

            # ── 2. Push dataset ───────────────────────────────────────────────
            s.push_message = f"Pushing dataset ({len(clean)} records) to {dataset_repo}…"
            from datasets import Dataset
            ds = Dataset.from_list(clean)
            ds.push_to_hub(dataset_repo, token=config.HF_TOKEN, private=True)

            # ── 3. Push model + tokenizer ─────────────────────────────────────
            s.push_message = f"Pushing model to {model_repo}…"
            tok.push_to_hub(model_repo, token=config.HF_TOKEN, private=True)
            model.push_to_hub(model_repo, token=config.HF_TOKEN, private=True)

            s.push_status = "done"
            s.push_message = (f"Pushed {len(clean)} records → {dataset_repo} "
                              f"and model → {model_repo}")
        except Exception as exc:  # noqa: BLE001
            s.push_status = "error"
            s.push_error = str(exc)
            s.push_message = f"Push failed: {exc}"

    threading.Thread(target=_run_push, daemon=True).start()


def status_all() -> list[dict]:
    result = []
    for key, meta in TRAIN_META.items():
        curated = curated_records(key)
        s = _states[key]
        has_saved = (config.TRAINED_MODELS_DIR / key).exists()
        result.append({
            "key": key,
            "name": meta["name"],
            "curated_count": len(curated),
            "has_trained": key in _trained_models,
            "has_saved": has_saved,
            "min_examples": config.MIN_TRAIN_EXAMPLES,
            **training_status(key),
        })
    return result


# ---------------------------------------------------------------------------
# Load saved models on startup
# ---------------------------------------------------------------------------

def _load_saved_models() -> None:
    """Check TRAINED_MODELS_DIR and load any previously saved trained models."""
    import transformers

    _KIND_CLS = {
        "sequence":      ("AutoTokenizer", "AutoModelForSequenceClassification"),
        "sequence_pair": ("AutoTokenizer", "AutoModelForSequenceClassification"),
        "token":         ("AutoTokenizer", "AutoModelForTokenClassification"),
        "seq2seq":       ("AutoTokenizer", "AutoModelForSeq2SeqLM"),
    }

    try:
        device = config.resolve_device()
    except Exception:  # noqa: BLE001
        device = "cpu"

    for key, meta in TRAIN_META.items():
        save_dir = config.TRAINED_MODELS_DIR / key
        if not save_dir.exists():
            continue
        try:
            tok_name, model_name = _KIND_CLS[meta["kind"]]
            tok_cls = getattr(transformers, tok_name)
            model_cls = getattr(transformers, model_name)
            tok = tok_cls.from_pretrained(str(save_dir))
            model = model_cls.from_pretrained(str(save_dir)).to(device).eval()
            _trained_models[key] = (tok, model, device)
            s = _states[key]
            s.status = "done"
            s.message = "Loaded from disk"
            s.trained_at = save_dir.stat().st_mtime
        except Exception as exc:  # noqa: BLE001
            _states[key].error = f"Failed to load saved model: {exc}"


_load_saved_models()
