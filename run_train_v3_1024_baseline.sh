#!/usr/bin/env bash
# v3 1024-token baseline — fresh training run, no resume, no curriculum.
#
# Locked invariants (do NOT change without updating v3_ingest.py and SQL todo):
#   --input           v3_bucket_1024/dataset.jsonl          (59 832 examples, sha256 in preprocessing_summary.json)
#   --eval-ood        ood_validation_ground_truth_extended.jsonl
#   --extended-entities                                      (enables LOCATION + AGENT)
#   --simplify-labels                                        (role → PERSON/LOC/AGT collapse)
#   --max-length      1024
#   --class-weights   "1,5,10,25,20,15,15"                  (V3_LOCKED_CLASS_WEIGHTS)
#   (no --resume-from-checkpoint)
#   (no --curriculum)

set -euo pipefail
cd "$(dirname "$0")"

DATASET="v3_bucket_1024/dataset.jsonl"
EVAL_OOD="ood_validation_ground_truth_extended.jsonl"
OUTPUT="story_ner_model_v3_1024_baseline"

if [[ ! -f "$DATASET" ]]; then
    echo "❌ Dataset not found: $DATASET"
    exit 1
fi
if [[ ! -f "$EVAL_OOD" ]]; then
    echo "❌ Eval file not found: $EVAL_OOD"
    exit 1
fi

echo "============================================================"
echo " v3 1024-token baseline"
echo " Dataset:  $DATASET  ($(wc -l < "$DATASET") examples)"
echo " Eval OOD: $EVAL_OOD"
echo " Output:   $OUTPUT"
echo "============================================================"
echo ""

WANDB_DISABLED=true python3 train_modernbert_streaming.py \
    --model answerdotai/ModernBERT-base \
    --input "$DATASET" \
    --eval-ood "$EVAL_OOD" \
    --extended-entities \
    --simplify-labels \
    --max-length 1024 \
    --class-weights "1,5,10,25,20,15,15" \
    --use-bf16 \
    --learning-rate 2e-5 \
    --output "$OUTPUT"

echo ""
echo "✅ v3 1024 baseline training complete → $OUTPUT"
