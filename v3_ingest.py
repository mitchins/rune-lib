#!/usr/bin/env python3
"""
v3 deterministic ingest pipeline for RUNE NER.

Modes
-----
  ingest         Validate raw JSONL, convert to BIO labels via StoryPreprocessor,
                 bucket by tokenized length, write deterministic bucket_1024 dataset.
  preflight      Assert B-PERSON / B-LOCATION / B-AGENT all present, sample N batches,
                 print per-batch entity counts, abort if LOCATION or AGENT absent.
  overfit-probe  Fine-tune on 500 samples for a few epochs; abort if LOC or AGT F1
                 remain at zero after training.

Usage
-----
  python3 v3_ingest.py ingest \\
      --input ner_training_agent_rich_training_ultra_short_agent_rich_20260507_172953.jsonl \\
      --output v3_bucket_1024

  python3 v3_ingest.py preflight --dataset v3_bucket_1024

  python3 v3_ingest.py overfit-probe \\
      --dataset v3_bucket_1024 \\
      --model answerdotai/ModernBERT-base \\
      --output v3_overfit_probe
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional

from rune.ner.label_alignment import align_labels_pretokenized

# ---------------------------------------------------------------------------
# Label vocabulary (v3 uses simplified PERSON/LOCATION/AGENT)
# ---------------------------------------------------------------------------
LABEL_TO_ID: Dict[str, int] = {
    "O": 0,
    "B-PERSON": 1,
    "I-PERSON": 2,
    "B-LOCATION": 3,
    "I-LOCATION": 4,
    "B-AGENT": 5,
    "I-AGENT": 6,
}
ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}

REQUIRED_B_TAGS = {"B-PERSON", "B-LOCATION", "B-AGENT"}

# v3 locked training class weights: [O, B-PERSON, I-PERSON, B-LOCATION, I-LOCATION, B-AGENT, I-AGENT]
# These are the mission-critical weights that all v3 training runs MUST use.
# They are passed explicitly to avoid silent drift from auto-derived values.
V3_LOCKED_CLASS_WEIGHTS = [1, 5, 10, 25, 20, 15, 15]

# Role-label → simplified-label mapping (for StoryPreprocessor output).
# Covers both role-aware tags (B-PROTAGONIST etc.) and already-simplified tags
# (B-PERSON) since StoryPreprocessor with extended_entity_types=True emits B-PERSON.
ROLE_TO_SIMPLE: Dict[str, str] = {
    # Role-aware (legacy / when simplify_labels not used)
    "B-PROTAGONIST": "B-PERSON",
    "I-PROTAGONIST": "I-PERSON",
    "B-ANTAGONIST":  "B-PERSON",
    "I-ANTAGONIST":  "I-PERSON",
    "B-SUPPORTING":  "B-PERSON",
    "I-SUPPORTING":  "I-PERSON",
    # Already simplified (StoryPreprocessor with extended_entity_types=True)
    "B-PERSON":      "B-PERSON",
    "I-PERSON":      "I-PERSON",
    "B-LOCATION":    "B-LOCATION",
    "I-LOCATION":    "I-LOCATION",
    "B-AGENT":       "B-AGENT",
    "I-AGENT":       "I-AGENT",
    "O":             "O",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)


def _simplify_bio_tags(bio_tags: List[str]) -> List[str]:
    """Map role-aware BIO tags to simplified PERSON/LOCATION/AGENT tags."""
    result = []
    for tag in bio_tags:
        simple = ROLE_TO_SIMPLE.get(tag)
        if simple is None:
            raise ValueError(f"Unknown BIO tag encountered: {tag!r}")
        result.append(simple)
    return result


# ---------------------------------------------------------------------------
# INGEST
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """
    Process raw JSONL → prefiltered bucket_1024 dataset.

    "bucket_1024" means this dataset is intended for training at max_length=1024.
    All stories are included (tokenizer truncation during training handles windowing).
    Stories with spaCy word-count exceeding --hard-max-words are dropped.

    Hard aborts on:
      - malformed JSON line
      - missing 'text' or 'entities' fields
      - span/alignment failure (len(tokens) != len(bio_tags))
      - no B-LOCATION or no B-AGENT labels in the complete processed corpus
      - no B-PERSON labels in the complete processed corpus
    """
    input_path = args.input
    output_dir = Path(args.output)
    training_max_length = args.max_length    # intended training window (stored in summary)
    hard_max_words = args.hard_max_words     # drop pathologically long stories

    print("=" * 70)
    print("🚀 V3 INGEST PIPELINE")
    print("=" * 70)
    print(f"  Input          : {input_path}")
    print(f"  Output         : {output_dir}")
    print(f"  Training window: {training_max_length} tokens (stored in summary)")
    print(f"  Hard max filter: {hard_max_words} spaCy words (drop if longer)")
    print()

    # ------------------------------------------------------------------
    # Step 1: validate JSONL integrity
    # ------------------------------------------------------------------
    print("🔍 Step 1: Validating raw JSONL integrity …")
    total_lines = 0
    with open(input_path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                raise RuntimeError(
                    f"ABORT: blank line at line {lineno} in {input_path}"
                )
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"ABORT: malformed JSON at line {lineno}: {exc}"
                ) from exc
            if "text" not in obj:
                raise RuntimeError(
                    f"ABORT: missing 'text' field at line {lineno}"
                )
            if "entities" not in obj:
                raise RuntimeError(
                    f"ABORT: missing 'entities' field at line {lineno}"
                )
            total_lines += 1
    print(f"  ✅ {total_lines:,} lines — all valid (text + entities present)")
    print()

    # ------------------------------------------------------------------
    # Step 2: load preprocessor
    # ------------------------------------------------------------------
    print("🔄 Step 2: Loading StoryPreprocessor(extended_entity_types=True) …")
    from rune.data.story_preprocessor import StoryPreprocessor
    preprocessor = StoryPreprocessor(extended_entity_types=True)
    print("  ✅ Ready")
    print()

    # ------------------------------------------------------------------
    # Step 3: process stories → BIO labels → save all (hard cap on word count)
    # ------------------------------------------------------------------
    print(f"⚙️  Step 3: Processing {total_lines:,} stories …")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.jsonl"

    global_tag_counts: Counter = Counter()
    processed_count = 0
    skipped_long = 0
    start_time = time.time()

    with open(input_path, "r", encoding="utf-8") as fh_in, \
         open(dataset_path, "w", encoding="utf-8") as fh_out:

        for lineno, raw_line in enumerate(fh_in, 1):
            obj = json.loads(raw_line.strip())

            # --- preprocess (hard abort on failure) ---
            try:
                processed = preprocessor.process_story(obj)
            except Exception as exc:
                raise RuntimeError(
                    f"ABORT: StoryPreprocessor failed at line {lineno}: {exc}"
                ) from exc

            tokens = processed.get("tokens", [])
            bio_tags = processed.get("bio_tags", [])

            # --- span/alignment hard abort ---
            if len(tokens) != len(bio_tags):
                raise RuntimeError(
                    f"ABORT: span alignment failure at line {lineno} — "
                    f"tokens={len(tokens)} bio_tags={len(bio_tags)}"
                )
            if not tokens:
                raise RuntimeError(
                    f"ABORT: empty token sequence at line {lineno}"
                )

            # --- simplify role labels → PERSON/LOCATION/AGENT ---
            try:
                simple_tags = _simplify_bio_tags(bio_tags)
            except ValueError as exc:
                raise RuntimeError(
                    f"ABORT: label simplification failed at line {lineno}: {exc}"
                ) from exc

            # --- drop pathologically long stories (hard word-count cap) ---
            if len(tokens) > hard_max_words:
                skipped_long += 1
                continue

            # --- accumulate tag counts ---
            for tag in simple_tags:
                global_tag_counts[tag] += 1

            # --- write to dataset ---
            record = {
                "story_id": processed.get("story_id", f"line_{lineno}"),
                "tokens": tokens,
                "bio_tags": simple_tags,
            }
            fh_out.write(json.dumps(record) + "\n")
            processed_count += 1

            if processed_count % 5000 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                print(
                    f"  [{processed_count:6,} processed | {skipped_long:4,} skipped-long | "
                    f"{lineno:6,}/{total_lines:,} lines | {elapsed:.0f}s | {rate:.0f}/s]"
                )

    elapsed = time.time() - start_time
    print(f"\n  ✅ Done in {elapsed:.1f}s")
    print(f"     Processed into dataset : {processed_count:,}")
    print(f"     Skipped (too long)     : {skipped_long:,}")
    print()

    # ------------------------------------------------------------------
    # Step 4: label distribution check — hard abort if LOCATION/AGENT missing
    # ------------------------------------------------------------------
    print("📊 Step 4: Label distribution (full corpus):")
    for label, lid in sorted(LABEL_TO_ID.items(), key=lambda x: x[1]):
        count = global_tag_counts.get(label, 0)
        print(f"  {label:15s}: {count:>10,}")

    b_loc = global_tag_counts.get("B-LOCATION", 0)
    b_agt = global_tag_counts.get("B-AGENT", 0)
    b_per = global_tag_counts.get("B-PERSON", 0)

    if b_loc == 0:
        raise RuntimeError(
            "ABORT: No B-LOCATION labels found in bucket_1024 corpus — "
            "LOCATION entity conversion failed completely."
        )
    if b_agt == 0:
        raise RuntimeError(
            "ABORT: No B-AGENT labels found in bucket_1024 corpus — "
            "AGENT entity conversion failed completely."
        )
    if b_per == 0:
        raise RuntimeError(
            "ABORT: No B-PERSON labels found in bucket_1024 corpus — "
            "PERSON entity conversion failed completely."
        )
    print()
    print("  ✅ B-PERSON, B-LOCATION, B-AGENT all present — all > 0")
    print()

    # ------------------------------------------------------------------
    # Step 5: compute inverse-frequency class weights (capped at 15× O)
    # ------------------------------------------------------------------
    total_tokens_count = sum(global_tag_counts.values()) or 1
    num_labels = len(LABEL_TO_ID)
    raw_weights = {}
    for lid in range(num_labels):
        tag = ID_TO_LABEL[lid]
        count = global_tag_counts.get(tag, 1)
        raw_weights[tag] = total_tokens_count / (num_labels * count)

    o_weight = raw_weights["O"]
    max_weight_cap = max(o_weight * 15.0, 1.0)
    class_weights = {tag: min(w, max_weight_cap) for tag, w in raw_weights.items()}

    print("⚖️  Class weights (inv-freq, capped 15× O):")
    for lid in range(num_labels):
        tag = ID_TO_LABEL[lid]
        print(f"  id={lid}  {tag:15s}: {class_weights[tag]:.4f}")
    print()

    # ------------------------------------------------------------------
    # Step 6: write preprocessing_summary.json
    # ------------------------------------------------------------------
    summary = {
        "v3": True,
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "output": str(dataset_path),
        "training_max_length": training_max_length,
        "hard_max_words": hard_max_words,
        "total_input_lines": total_lines,
        "processed_count": processed_count,
        "skipped_long": skipped_long,
        "tag_counts": dict(global_tag_counts),
        "class_weights": class_weights,
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"💾 Saved: {dataset_path}")
    print(f"💾 Saved: {summary_path}")
    print()
    print("=" * 70)
    print(f"✅ V3 INGEST COMPLETE — {processed_count:,} examples in {output_dir}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# PREFLIGHT
# ---------------------------------------------------------------------------

def cmd_preflight(args: argparse.Namespace) -> None:
    """
    Preflight checks on an existing v3 bucket_1024 dataset.

    1. Assert B-PERSON / B-LOCATION / B-AGENT all present (global scan).
    2. Tokenize dataset with the configured tokenizer.
    3. Sample N batches; print per-batch entity counts.
    4. Abort if LOCATION or AGENT absent across sampled batches.
    """
    dataset_dir = Path(args.dataset)
    dataset_path = dataset_dir / "dataset.jsonl"
    summary_path = dataset_dir / "preprocessing_summary.json"
    batch_size = args.batch_size
    n_batches = args.n_batches
    model_name = args.model

    print("=" * 70)
    print("🔬 V3 PREFLIGHT CHECKS")
    print("=" * 70)
    print(f"  Dataset  : {dataset_path}")
    print(f"  Batches  : {n_batches} × {batch_size}")
    print(f"  Model    : {model_name}")
    print()

    if not dataset_path.exists():
        raise RuntimeError(f"ABORT: dataset not found at {dataset_path}")

    # ------------------------------------------------------------------
    # 1. Label presence assertion (global scan)
    # ------------------------------------------------------------------
    print("🔍 Check 1: global label presence …")
    global_counts: Counter = Counter()
    n_examples = 0
    with open(dataset_path) as fh:
        for line in fh:
            rec = json.loads(line)
            for tag in rec["bio_tags"]:
                global_counts[tag] += 1
            n_examples += 1

    print(f"  Total examples : {n_examples:,}")
    for label, lid in sorted(LABEL_TO_ID.items(), key=lambda x: x[1]):
        print(f"  {label:15s}: {global_counts.get(label, 0):>10,}")
    print()

    missing = [t for t in REQUIRED_B_TAGS if global_counts.get(t, 0) == 0]
    if missing:
        raise RuntimeError(
            f"ABORT: Missing required B-tags in dataset: {missing}"
        )
    print("  ✅ B-PERSON, B-LOCATION, B-AGENT all present globally")
    print()

    # ------------------------------------------------------------------
    # 2. Load summary for expected class weights (if available)
    # ------------------------------------------------------------------
    if summary_path.exists():
        with open(summary_path) as fh:
            summary = json.load(fh)
        print("📋 Preprocessing summary loaded:")
        print(f"  processed_count: {summary.get('processed_count', summary.get('bucket_count', 'n/a'))}")
        print(f"  input_sha256   : {summary.get('input_sha256', 'n/a')[:16]}…")
        print()

    # ------------------------------------------------------------------
    # 3. Load tokenizer and sample N batches
    # ------------------------------------------------------------------
    print(f"🔄 Check 2: sampling {n_batches} batches of {batch_size} …")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    # Load all examples into memory for sampling (they are already prefiltered)
    all_examples = []
    with open(dataset_path) as fh:
        for line in fh:
            all_examples.append(json.loads(line))

    # Deterministic: take first n_batches * batch_size examples
    needed = n_batches * batch_size
    sample = all_examples[:needed]
    if len(sample) < needed:
        print(
            f"  ⚠️  Dataset has only {len(all_examples)} examples; "
            f"sampling {len(sample)} (< requested {needed})"
        )

    batch_loc_totals = []
    batch_agt_totals = []

    for batch_idx in range(n_batches):
        batch = sample[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        if not batch:
            break

        b_counts: Counter = Counter()
        for rec in batch:
            tokens = rec["tokens"]
            bio_tags = rec["bio_tags"]

            # verify span alignment within each batch example
            if len(tokens) != len(bio_tags):
                raise RuntimeError(
                    f"ABORT: span alignment failure in batch {batch_idx}, "
                    f"story_id={rec.get('story_id')}"
                )

            for tag in bio_tags:
                b_counts[tag] += 1

        per = b_counts.get("B-PERSON", 0) + b_counts.get("I-PERSON", 0)
        loc = b_counts.get("B-LOCATION", 0) + b_counts.get("I-LOCATION", 0)
        agt = b_counts.get("B-AGENT", 0) + b_counts.get("I-AGENT", 0)

        print(
            f"  Batch {batch_idx+1:2d}/{n_batches} : "
            f"PERSON={per:6,}  LOCATION={loc:6,}  AGENT={agt:6,}"
        )
        batch_loc_totals.append(loc)
        batch_agt_totals.append(agt)

    print()
    total_loc = sum(batch_loc_totals)
    total_agt = sum(batch_agt_totals)

    if total_loc == 0:
        raise RuntimeError(
            "ABORT: LOCATION labels are absent across all sampled batches — "
            "LOCATION conversion appears broken."
        )
    if total_agt == 0:
        raise RuntimeError(
            "ABORT: AGENT labels are absent across all sampled batches — "
            "AGENT conversion appears broken."
        )

    print(f"  ✅ LOCATION present in sampled batches (total tokens: {total_loc:,})")
    print(f"  ✅ AGENT    present in sampled batches (total tokens: {total_agt:,})")
    print()
    print("=" * 70)
    print("✅ V3 PREFLIGHT PASSED")
    print("=" * 70)


# ---------------------------------------------------------------------------
# OVERFIT PROBE
# ---------------------------------------------------------------------------

def cmd_overfit_probe(args: argparse.Namespace) -> None:
    """
    Train on 500 samples for a few epochs; abort if LOC or AGT F1 stay at zero.

    This is a smoke-test that the model can learn LOCATION and AGENT labels at all.
    """
    dataset_dir = Path(args.dataset)
    dataset_path = dataset_dir / "dataset.jsonl"
    output_dir = Path(args.output)
    model_name = args.model
    n_samples = args.n_samples
    epochs = args.epochs
    batch_size = args.batch_size
    max_length = args.max_length

    print("=" * 70)
    print("🧪 V3 OVERFIT PROBE (500-sample)")
    print("=" * 70)
    print(f"  Dataset   : {dataset_path}")
    print(f"  Model     : {model_name}")
    print(f"  Samples   : {n_samples}")
    print(f"  Epochs    : {epochs}")
    print(f"  Batch     : {batch_size}")
    print(f"  MaxLen    : {max_length}")
    print(f"  Weights   : {args.class_weights} (locked v3 vector)")
    print()

    if not dataset_path.exists():
        raise RuntimeError(f"ABORT: dataset not found at {dataset_path}")

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        DataCollatorForTokenClassification,
    )
    from seqeval.metrics import f1_score, classification_report
    from torch.utils.data import Dataset

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    # Load n_samples from dataset
    examples = []
    with open(dataset_path) as fh:
        for line in fh:
            examples.append(json.loads(line))
            if len(examples) >= n_samples:
                break

    if len(examples) < n_samples:
        print(
            f"  ⚠️  Only {len(examples)} examples available "
            f"(requested {n_samples}); proceeding with {len(examples)}"
        )

    class OverfitDataset(Dataset):
        def __init__(self, recs, tok, max_len):
            self.recs = recs
            self.tok = tok
            self.max_len = max_len

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, idx):
            rec = self.recs[idx]
            tokens = rec["tokens"]
            bio_tags = rec["bio_tags"]

            if len(tokens) != len(bio_tags):
                raise RuntimeError(
                    f"ABORT: span alignment failure in probe example idx={idx}"
                )

            result = align_labels_pretokenized(
                tokens=tokens,
                bio_tags=bio_tags,
                tokenizer=self.tok,
                max_length=self.max_len,
                label_to_id=LABEL_TO_ID,
                simplify_labels=False,  # bucket_1024 already stores simplified tags
            )

            return {
                "input_ids": result["input_ids"],
                "attention_mask": result["attention_mask"],
                "labels": result["labels"],
            }

    train_ds = OverfitDataset(examples, tokenizer, max_length)
    # Use full dataset as eval to measure overfit capacity
    eval_ds = OverfitDataset(examples, tokenizer, max_length)

    label_list = [ID_TO_LABEL[i] for i in range(len(LABEL_TO_ID))]

    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        predictions = predictions.argmax(axis=-1)

        true_preds = [
            [label_list[p] for p, l in zip(pred_row, lbl_row) if l != -100]
            for pred_row, lbl_row in zip(predictions, label_ids)
        ]
        true_labels = [
            [label_list[l] for p, l in zip(pred_row, lbl_row) if l != -100]
            for pred_row, lbl_row in zip(predictions, label_ids)
        ]

        report = classification_report(
            true_labels, true_preds, output_dict=True, zero_division=0
        )
        loc_f1 = report.get("LOCATION", {}).get("f1-score", 0.0)
        agt_f1 = report.get("AGENT", {}).get("f1-score", 0.0)
        per_f1 = report.get("PERSON", {}).get("f1-score", 0.0)
        return {
            "f1": f1_score(true_labels, true_preds),
            "per_f1": per_f1,
            "loc_f1": loc_f1,
            "agt_f1": agt_f1,
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse the explicit class weights (locked v3 vector; do NOT load from summary
    # to avoid silent drift between probe and full training).
    raw_weights = [float(x) for x in args.class_weights.split(",")]
    if len(raw_weights) != len(LABEL_TO_ID):
        raise RuntimeError(
            f"ABORT: --class-weights has {len(raw_weights)} values; "
            f"expected {len(LABEL_TO_ID)} (one per label)."
        )
    class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float)
    print("⚖️  Class weights (locked v3 vector):")
    for i, w in enumerate(raw_weights):
        print(f"  {ID_TO_LABEL[i]:15s}: {w}")
    print()

    from train_modernbert_streaming import WeightedLossTrainer

    effective_batch = batch_size
    total_steps = (len(examples) // effective_batch) * epochs

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=total_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=max(1, total_steps // 20),
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt"
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    print(f"🏋️  Training overfit probe ({epochs} epochs, {total_steps} steps) …")
    trainer.train()

    print()
    print("📊 Final eval on training set (overfit capacity check):")
    eval_results = trainer.evaluate()
    loc_f1 = eval_results.get("eval_loc_f1", 0.0)
    agt_f1 = eval_results.get("eval_agt_f1", 0.0)
    per_f1 = eval_results.get("eval_per_f1", 0.0)

    print(f"  PERSON   F1: {per_f1:.4f}")
    print(f"  LOCATION F1: {loc_f1:.4f}")
    print(f"  AGENT    F1: {agt_f1:.4f}")
    print()

    if loc_f1 == 0.0:
        raise RuntimeError(
            "ABORT: LOCATION F1 remains at 0.0 after overfit probe — "
            "model cannot learn LOCATION labels. "
            "Check label distribution, class weights, and data quality."
        )
    if agt_f1 == 0.0:
        raise RuntimeError(
            "ABORT: AGENT F1 remains at 0.0 after overfit probe — "
            "model cannot learn AGENT labels. "
            "Check label distribution, class weights, and data quality."
        )

    # Save probe results
    probe_results = {
        "model": model_name,
        "n_samples": len(examples),
        "epochs": epochs,
        "class_weights": raw_weights,
        "per_f1": per_f1,
        "loc_f1": loc_f1,
        "agt_f1": agt_f1,
        "passed": True,
    }
    results_path = output_dir / "overfit_probe_results.json"
    with open(results_path, "w") as fh:
        json.dump(probe_results, fh, indent=2)

    print(f"💾 Probe results saved to {results_path}")
    print()
    print("=" * 70)
    print("✅ V3 OVERFIT PROBE PASSED — LOC and AGT F1 both > 0")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="v3 deterministic ingest/preflight/overfit-probe pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- ingest ----
    p_ingest = sub.add_parser("ingest", help="Validate + preprocess → bucket_1024")
    p_ingest.add_argument("--input", required=True, help="Raw JSONL input path")
    p_ingest.add_argument(
        "--output", default="v3_bucket_1024", help="Output directory (default: v3_bucket_1024)"
    )
    p_ingest.add_argument(
        "--max-length", type=int, default=1024,
        help="Intended training context window (recorded in summary; default: 1024)"
    )
    p_ingest.add_argument(
        "--hard-max-words", type=int, default=10000,
        help="Drop stories exceeding this spaCy word count (default: 10000)"
    )

    # ---- preflight ----
    p_pre = sub.add_parser("preflight", help="Label assertions + batch sampling")
    p_pre.add_argument("--dataset", required=True, help="Path to v3 bucket dir")
    p_pre.add_argument(
        "--model", default="answerdotai/ModernBERT-base",
        help="HuggingFace model name for tokenization"
    )
    p_pre.add_argument(
        "--n-batches", type=int, default=10,
        help="Number of batches to sample (default: 10)"
    )
    p_pre.add_argument(
        "--batch-size", type=int, default=8,
        help="Examples per batch (default: 8)"
    )

    # ---- overfit-probe ----
    p_probe = sub.add_parser(
        "overfit-probe", help="500-sample overfit probe (aborts if LOC/AGT F1=0)"
    )
    p_probe.add_argument("--dataset", required=True, help="Path to v3 bucket dir")
    p_probe.add_argument(
        "--model", default="answerdotai/ModernBERT-base",
        help="HuggingFace model name"
    )
    p_probe.add_argument(
        "--output", default="v3_overfit_probe",
        help="Output directory for probe checkpoints (default: v3_overfit_probe)"
    )
    p_probe.add_argument(
        "--n-samples", type=int, default=500, help="Number of training examples (default: 500)"
    )
    p_probe.add_argument(
        "--epochs", type=int, default=10, help="Training epochs (default: 10)"
    )
    p_probe.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: 4)"
    )
    p_probe.add_argument(
        "--max-length", type=int, default=1024, help="Max tokenized length (default: 1024)"
    )
    p_probe.add_argument(
        "--class-weights",
        default=",".join(str(w) for w in V3_LOCKED_CLASS_WEIGHTS),
        help=(
            "Comma-separated class weights for weighted loss "
            "(order: O,B-PERSON,I-PERSON,B-LOCATION,I-LOCATION,B-AGENT,I-AGENT). "
            f"Default: {V3_LOCKED_CLASS_WEIGHTS} (v3 locked vector)."
        ),
    )

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "preflight":
        cmd_preflight(args)
    elif args.command == "overfit-probe":
        cmd_overfit_probe(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
