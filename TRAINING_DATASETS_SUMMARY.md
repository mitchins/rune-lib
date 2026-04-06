# NER Training Datasets - All-Mentions Fix

## Overview
Fixed critical bugs in NER entity tagging that resulted in **6.9x more supervision signal** per story.

## Bugs Fixed

### Bug 1: Entity Index Off-By-One
- **Issue**: Entities stored with inclusive `end` index instead of exclusive
- **Impact**: Structured as `[31:32]` instead of `[31:33]`
- **Fix**: Changed `_extract_entities()` to use Python convention (exclusive end)

### Bug 2: First-Mention-Only Tagging
- **Issue**: Only first occurrence of each character name was tagged as B-/I-PERSON
- **Impact**: Subsequent mentions ("Angel", "Volkov") were tagged as O
- **Root Cause**: Conservative BIO tag generation logic
- **Fix**: Changed `_generate_bio_tags()` to tag ALL capitalized occurrences

## Improvements Achieved

### Supervision Signal (Story 39 Example)
```
OLD (first-mention-only):   13 B-tags
NEW (all-mentions):          90 B-tags
Improvement:                 6.9x
```

### Per-Entity Coverage (Story 39)
| Entity | Old | New | % Coverage |
|--------|-----|-----|-----------|
| Dmitri | 3/52 | 52/52 | 100% |
| Volkov | 3/6 | 6/6 | 100% |
| Angel | 2/21 | 21/21 | 100% |

### Source Data Statistics
- **Input**: `ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl`
- **Records**: 115,422 stories
- **Processing Time**: ~10 minutes
- **Output**: `ner_training_all_mentions.jsonl` (7.4GB)

## Processing Pipeline

### Step 1: Reprocess Source Data ✅
```bash
python3 reprocess_source_data.py \
    --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
    --output ner_training_all_mentions.jsonl \
    --validate
```
**Result**: 115,422 items reprocessed with fixed preprocessor

### Step 2: Create Hybrid Dataset ✅
```bash
python3 create_hybrid_dataset.py \
    --input ner_training_all_mentions.jsonl \
    --output hybrid_all_mentions/
```
**Result**: 
- Whole stories (< 4k tokens): 55,405 (48.0%)
- Split stories (≥ 4k tokens): 60,017 (52.0%)
- Total chunks: 473,714

### Step 3: Create Diverse Subsets ✅
```bash
python3 create_diverse_subset.py \
    --input hybrid_all_mentions/combined_hybrid.jsonl \
    --output diverse_subsets_all_mentions \
    --phase1-size 20000 \
    --phase2-size 40000
```

## Final Training Datasets

### Phase 1 (20k items)
- File: `diverse_subsets_all_mentions/diverse_phase1_20k.jsonl`
- Size: 366MB
- Avg tokens/story: 1,063
- Avg entity mentions: 21.4 per story
- **Use for**: Initial model training, quick iterations

### Phase 2 (40k items)
- File: `diverse_subsets_all_mentions/diverse_phase2_40k.jsonl`
- Size: 731MB
- Avg tokens/story: 1,111
- Avg entity mentions: 22.1 per story
- **Use for**: Extended training, fine-tuning

### Combined (60k items)
- File: `diverse_subsets_all_mentions/diverse_combined_60k.jsonl`
- Size: 1.1GB
- Avg tokens/story: 1,106
- Avg entity mentions: 21.4 per story
- **Use for**: Final model training

## Quality Metrics

### Tokenization
- Avg tokens per story: ~1,100
- Max tokens (soft limit): 4,096 (enforced via scene splitting)

### Entity Tagging
- Avg B-tags per story: 21.4 (vs. 3.2 in old data)
- Coverage: 100% of known character mentions
- Precision: High (only known named entities tagged)

### Data Integrity
- Validation errors: < 0.01%
- Exact duplicates removed: ~10,000
- All entities have valid BIO sequences

## Test Coverage

### Unit Tests ✅
- `tests/test_story_preprocessor.py`: 25 tests (all pass)
- `tests/test_entity_extraction.py`: 11 tests (all pass)
- `tests/test_classic_literature.py`: 17 tests (all pass)
  - Sherlock Holmes (Victorian names)
  - Pride and Prejudice (Formal names)

### Stress Tests
- Round-trip entity extraction validation
- Multi-word entity coverage (82% multi-word)
- Title prefix handling (Mr., Mrs., Sir, Lady)
- Repeated name mention tagging

## Known Limitations

### Nicknames
- Dataset uses canonical names (e.g., "Elizabeth Bennet" not "Lizzy")
- Nicknames must be added to character metadata explicitly

### Pronouns
- Pronouns (he, she, they) are NOT tagged
- Coreference resolution would require separate model layer

### Common Nouns
- "the king" where "king" is not a character → NOT tagged
- "Master Smith" → "Smith" tagged, "Master" NOT tagged

## Recommendations for Training

1. **Curriculum Learning**: Start with phase1, then phase2
2. **Batch Size**: ~32-64 (adjust based on available GPU memory)
3. **Learning Rate**: 2e-5 to 5e-5 for fine-tuning
4. **Epochs**: 3-5 recommended
5. **Evaluation Metric**: F1 score with entity-level granularity

## Files Generated

```
hybrid_all_mentions/
├── combined_hybrid.jsonl          (8.1GB - all 473,714 chunks)
├── whole_stories_under_4k.jsonl   (1.5GB - 55,405 stories)
└── split_stories_over_4k.jsonl    (6.7GB - 418,309 scenes)

diverse_subsets_all_mentions/
├── diverse_phase1_20k.jsonl       (366MB - 19,998 stories)
├── diverse_phase2_40k.jsonl       (731MB - 39,998 stories)
└── diverse_combined_60k.jsonl     (1.1GB - 59,996 stories)
```

## Training Command Example

```bash
python3 train_modernbert_streaming.py \
    --input diverse_subsets_all_mentions/diverse_phase1_20k.jsonl \
    --output story_ner_model_all_mentions \
    --epochs 3 \
    --batch-size 32
```

---

**Status**: Ready for training ✅
**Generated**: 2026-01-29
**Total Processing Time**: ~30 minutes (10min reprocess + 10min hybrid + 10min diversity)
