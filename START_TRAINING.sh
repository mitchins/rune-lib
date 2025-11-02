#!/bin/bash
# Canonical NER training command
# Trains ModernBERT NER model for 3 epochs with proper checkpoint management

python3 train_modernbert_streaming.py \
    --input ner_training_ready.jsonl \
    --output ./ner_model_3epochs \
    --max-length 8000 \
    --simplify-labels \
    --disable-early-stopping \
    --num-epochs 3 \
    --save-every-n-epochs 0.2

# Configuration:
# - 3 full epochs (~12,530 steps, ~6-8 hours)
# - Checkpoint every 0.2 epochs (~835 steps)
# - 15 total checkpoints, keep 10 best
# - Load best model at end
# - NO early stopping (wrestle with local maxima)
