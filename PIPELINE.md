# Story NER Training Pipeline

**Status**: ✅ Verified and cleaned (2026-01-30)

## Data Files

### Canonical Source
- `ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl`
  - 115,422 stories with `### Scene N:` markers
  - 2.1GB, entities format: `{text, entities: [{text, type: "PERSON", role}], _meta}`

### Processed Data  
- `ner_training_scenes_person_only.jsonl` - Scene-split with PERSON-only tags (4.6GB, ~300k scenes)
- `validation_ground_truth.jsonl` - Validation set (106KB)

## Pipeline Scripts

### 1. Scene Splitting
```bash
python3 split_scenes.py \
  --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
  --output ner_training_scenes_person_only.jsonl \
  --min-tokens 100
```
- Splits stories at `### Scene N:` markers
- Regenerates tokens + BIO tags per scene (PERSON-only)
- Keeps long scenes (tokenizer truncates during training)

### 2. Diversity Bucketing
```bash
python3 create_diverse_subset.py \
  --input ner_training_scenes_person_only.jsonl \
  --output ./diverse_subsets \
  --phase1-size 30000 \
  --phase2-size 50000
```
- Deduplicates by context patterns (LSH)
- Buckets by token length for curriculum
- Greedy diversity sampling

### 3. Training
```bash
python3 train_modernbert_streaming.py \
  --model allenai/longformer-base-4096 \
  --curriculum \
  --curriculum-stage1-input diverse_subsets/diverse_phase1_30k.jsonl \
  --curriculum-stage2-input diverse_subsets/diverse_phase2_50k.jsonl \
  --output story_ner_model_curriculum3 \
  --curriculum-stage1-epochs 2 \
  --curriculum-stage2-epochs 3 \
  --use-bf16
```
- 3-label system: O, B-PERSON, I-PERSON
- Curriculum: short sequences first, then long
- Class weights: 50x for entity tokens

### 4. Evaluation
```bash
python3 evaluate_single_model.py \
  --model story_ner_model_curriculum3/stage2 \
  --input validation_ground_truth.jsonl \
  --output eval_results.json
```

### 5. CLI Viewer
```bash
python3 view_story.py 42  # View story at index 42
```

## Support Scripts

- `reprocess_source_data.py` - Convert between data formats
- `validate_alignment.py` - Verify label alignment correctness

## Key Design Decisions

### ✅ PERSON-Only Tags
- All entities tagged as PERSON (role field ignored)
- Simpler than role-based (protagonist/antagonist/etc.)
- Uses implicit structure: anything in `entities` list is a PERSON

### ✅ Scene Splitting
- Each scene processed independently (no tag slicing)
- Entity boundaries preserved per scene
- Long scenes kept (tokenizer handles truncation)

### ✅ Diversity via Context Patterns
- Extracts n-grams around entities: "said [E] quietly"
- Deduplicates similar contexts (LSH hashing)
- Preserves linguistic diversity, not just entity name variety

## Archived Data
All outdated/intermediate files moved to `archived/` (see archived/README.md for manifest)
