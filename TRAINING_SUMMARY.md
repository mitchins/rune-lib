# NER Training Pipeline

## Complete Training Pipeline (3 Steps)

### Step 1: Preprocess and Split Scenes
```bash
python3 split_scenes.py \
  --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
  --output diverse_subsets/scenes_split.jsonl \
  --min-tokens 100 \
  --max-tokens 4096 \
  --batch-size 100 \
  --n-process 12
```
**Output:** 547k preprocessed scenes from 115k stories (4.7x expansion)

### Step 2: Create Diverse Training Subset
```bash
python3 create_diverse_subset_v2.py \
  --input diverse_subsets/scenes_split.jsonl \
  --output diverse_subsets/final_training.jsonl \
  --phase1-size 50000 \
  --phase2-size 80000
```
**Output:** 130k diverse examples split into phase1 (short) and phase2 (long)

### Step 3: Train with Curriculum Learning
```bash
python3 train_modernbert_streaming.py \
  --model allenai/longformer-base-4096 \
  --curriculum \
  --curriculum-stage1-input diverse_subsets/final_training.jsonl/diverse_phase1.jsonl \
  --curriculum-stage2-input diverse_subsets/final_training.jsonl/diverse_phase2.jsonl \
  --curriculum-stage1-epochs 2 \
  --curriculum-stage2-epochs 3 \
  --output ./story_ner_model_curriculum \
  --gradient-accumulation 2 \
  --use-bf16 \
  --simplify-labels \
  --stage1-batch-size 20 \
  --stage2-batch-size 6
```
**Result:** Two-stage curriculum training (short examples → long examples)

---

## Legacy Training (Outdated)
python3 train_modernbert_streaming.py \
    --model answerdotai/ModernBERT-base \
    --input ner_training_latest_processed.jsonl \
    --output ./story_ner_model_modernbert \
    --simplify-labels \
    --num-epochs 3
```

### Training Script Features

The `train_modernbert_streaming.py` now supports:
- ✅ Flexible model selection (`--model`)
- ✅ Auto-detects max_length (4096 for Longformer, 8192 for ModernBERT)
- ✅ Manual override with `--max-length`
- ✅ Role simplification with `--simplify-labels` (all → B/I-PERSON)
- ✅ Streaming dataset (memory efficient for 115K stories)

### Why Longformer?

ModernBERT has 2x the context (8192 vs 4096) but is **harder to train**:
- More sensitive to hyperparameters
- Slower convergence
- Higher memory usage

Start with Longformer, switch to ModernBERT once you have a baseline.

### Model Comparison

| Model | Max Tokens | Training Ease | Context |
|-------|-----------|---------------|---------|
| **Longformer** | 4096 | ✅ Easier | Good for scenes |
| **ModernBERT** | 8192 | ⚠️ Harder | Good for chapters |

### Expected Training Time

With 115K stories:
- **3 epochs:** ~8-12 hours (GPU dependent)
- **Checkpoints:** Every 0.2 epochs (~10-15 checkpoints)
- **Early stopping:** Disabled (set `--enable-early-stopping` to enable)

### Scene-Based Data (Alternative)

If full stories are too long, use scene-based splits:
- `ner_training_scenes.jsonl` - 1.2GB
- `ner_training_comprehensive_scenes.jsonl` - 5.3GB

Both are pre-tokenized and ready to train.

---

## Troubleshooting

**Out of memory?**
- Reduce `--max-length` (e.g., `--max-length 2048`)
- Use smaller batch size in script (edit `per_device_train_batch_size`)

**Training diverging?**
- Enable early stopping: remove `--disable-early-stopping`
- Reduce learning rate in script

**Need raw data processed?**
```python
from rune.data.story_preprocessor import StoryPreprocessor
p = StoryPreprocessor(use_spacy=False)
p.process_jsonl_file('raw.jsonl', 'processed.jsonl')
```

---

## Next Steps After Training

1. **Evaluate on validation set:**
   ```bash
   python3 evaluate_single_model.py --model ./story_ner_model_longformer
   ```

2. **Test on real text:**
   ```python
   from rune.core.tagger import BioTagger
   tagger = BioTagger("./story_ner_model_longformer")
   entities = tagger.tag_and_extract("Your text here...")
   ```

3. **Add LOCATION support** (future work - currently only PERSON entities)

---

For more details, see:
- `README.md` - Full library documentation
- `METHODOLOGY.md` - Training approach and design decisions
- `START_TRAINING.sh` - Canonical training command
