#!/bin/bash
# Start curriculum-based NER training with COMPREHENSIVE dataset
# 60,224 stories (4.3x more than baseline!)
# 3-phase curriculum: scenes → mixed → stories

echo "========================================="
echo "CURRICULUM TRAINING - COMPREHENSIVE DATASET"
echo "========================================="
echo ""
echo "Dataset: 60,224 stories (4.3x baseline)"
echo "Unique names: 114,877"
echo ""
echo "Memory-efficient STREAMING:"
echo "  ✓ No dataset loaded into RAM"
echo "  ✓ Streams from disk on-the-fly"
echo "  ✓ Handles 305K+ scenes without OOM"
echo ""
echo "Optimized for 32GB VRAM with DYNAMIC PADDING:"
echo "  Phase 1: train_batch=12, eval_batch=1 (effective=12, max 1500 BERT tokens)"
echo "  Phase 2: train_batch=6, eval_batch=1, grad_accum=3 (effective=18, max 3000 BERT tokens)"
echo "  Phase 3: train_batch=2, eval_batch=1, grad_accum=8 (effective=16, max 8000 BERT tokens)"
echo ""
echo "Memory optimization: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""
echo "✓ Minimal eval batches prevent OOM during checkpointing"
echo ""
echo "========================================="
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train_curriculum.py \
    --scenes-file ner_training_comprehensive_scenes.jsonl \
    --stories-file ner_training_comprehensive_ready.jsonl \
    --output ./ner_model_curriculum \
    --max-length 8000 \
    --simplify-labels

# Output structure:
# ner_model_curriculum/
#   phase1_scenes/          - Phase 1 checkpoints
#   phase1_complete/        - Best Phase 1 model
#   phase2_mixed/           - Phase 2 checkpoints
#   phase2_complete/        - Best Phase 2 model
#   phase3_stories/         - Phase 3 checkpoints
#   phase3_complete/        - Best Phase 3 model
#   final/                  - Final curriculum-trained model

# To resume from a specific phase:
# python3 train_curriculum.py --start-phase 2 --resume-from ./ner_model_curriculum/phase1_complete [...]
