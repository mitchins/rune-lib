# Dataset Corruption - Root Cause and Fix

## Problem Discovered

**92% of scene-split training data is corrupted** with catastrophic label errors:
- Commas labeled as PROTAGONIST: 341 occurrences
- "the" labeled as entities: 473 occurrences  
- Periods labeled as entities: 514 occurrences
- Actual character names (e.g., "Leif Thorsson") labeled as `O` (not entities)

## Root Cause

**File:** `experiments/split_stories_into_scenes.py` (Line 84)

```python
# BUG: Slicing bio_tags from story-level indices after re-tokenizing scene
scene_bio_tags = story_bio_tags[token_offset:token_offset + scene_token_count]
```

**The issue:**
1. Scene text is **re-tokenized separately** (line 80): `scene_tokens = tokenizer.split_words(scene_text)`
2. This produces different token boundaries than the original story tokenization
3. But the code tries to slice bio_tags using token counts from the NEW tokenization
4. Token misalignment causes tags to shift and attach to wrong tokens

**Example:**
- Story token 73: `"introspective"` → `O`
- Story token 74: `","` → `O`
- After scene re-tokenization:
  - Token indices shift
  - Tag for "introspective" accidentally lands on `","`
  - Result: `","` gets tagged as `B-PROTAGONIST`

## Solution

**New script:** `create_hybrid_dataset.py`

### Key Fixes:

1. **Use entity boundaries, not token slicing:**
   ```python
   # CORRECT: Reconstruct tags from entity start/end indices
   def reconstruct_bio_tags_from_entities(tokens, entities):
       bio_tags = ['O'] * len(tokens)
       for entity in entities:
           bio_tags[entity['start']] = f'B-{entity["role"]}'
           for i in range(entity['start'] + 1, entity['end']):
               bio_tags[i] = f'I-{entity["role"]}'
       return bio_tags
   ```

2. **Hybrid strategy:**
   - Stories <4096 tokens → **Keep whole** (no splitting)
   - Stories ≥4096 tokens → **Split by scenes** with proper alignment

3. **Built-in validation:**
   - Checks tokens/bio_tags length match
   - Flags suspicious B- tags on punctuation/articles
   - Generates validation report

4. **Two splitting modes:**
   - Scene markers (respects story structure)
   - Sliding windows (fallback for stories without markers)

## Reprocessing Pipeline

**Run:** `./REPROCESS_PIPELINE.sh`

**Steps:**
1. Create hybrid dataset from original data
2. Apply diversity sampling (removes duplicates, balances distribution)
3. Validate output quality
4. Generate training-ready datasets

**Output:**
- `hybrid_dataset/combined_hybrid.jsonl` - Full clean dataset
- `diverse_hybrid/diverse_phase1_20k.jsonl` - Quick POC (20k)
- `diverse_hybrid/diverse_combined_60k.jsonl` - Full training (60k)
- `hybrid_dataset/validation_report.txt` - Quality metrics

## Validation Results

**Original data:** 1.0% corruption (acceptable - mostly edge cases like "Li", "Yu")
**Scene-split data:** 92% corruption (catastrophic)
**Hybrid data:** Expected <2% corruption (validated during creation)

## Curriculum Learning Strategy

With hybrid dataset:
1. **Stage 1:** Short whole stories (512-1500 tokens)
2. **Stage 2:** Medium whole stories (1500-4096 tokens)
3. **Stage 3:** Long story scenes (scenes from 4096+ token stories)

This preserves context while fitting in transformer limits.

## Next Steps

1. **Run reprocessing:** `./REPROCESS_PIPELINE.sh` (~10-15 min)
2. **Validate visually:** `python3 view_story_labels.py --input diverse_hybrid/diverse_phase1_20k.jsonl --index 42`
3. **Train on clean data:** Use `diverse_hybrid/diverse_phase1_20k.jsonl`

## Files Created

- `create_hybrid_dataset.py` - New validated hybrid dataset creator
- `REPROCESS_PIPELINE.sh` - Complete end-to-end pipeline
- `view_story_labels.py` - Visual validation tool (already existed)
- `DATASET_FIX_SUMMARY.md` - This document
