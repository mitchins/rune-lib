# Training Data Pipeline Verification

## Canonical Source
**File**: `ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl`
- 115,422 stories
- 2.1GB
- Has `### Scene N:` markers
- Format: `{text: "...", entities: [{text: "Alice", type: "PERSON", role: "protagonist"}], _meta: {}}`

## Pipeline Flow

### Step 1: split_scenes.py ✅ VERIFIED
**Input**: Canonical source with scene markers
**Output**: Scene-level JSONL (one scene per line)

**Process**:
1. Split text at `### Scene N:` regex pattern
2. For each scene:
   - Pass scene text + ALL entities to preprocessor
   - Preprocessor filters to entities that actually appear in scene
   - Preprocessor generates fresh tokens + BIO tags (PERSON-only)
3. Filter scenes by min token count (no max - tokenizer truncates)

**Verified**:
- ✅ Handles `entities` format correctly
- ✅ Passes role field through (ignored by preprocessor)
- ✅ Generates PERSON-only tags (B-PERSON, I-PERSON, O)
- ✅ Keeps long scenes (tokenizer truncates during training)

### Step 2: create_diverse_subset.py ✅ VERIFIED
**Input**: Scene-level JSONL
**Output**: Curriculum phase1 + phase2 JSONL files

**Process**:
1. Pass 1: Build metadata index (hash, length bucket, context patterns)
2. Exact dedupe by content hash
3. Bucket by token length (e.g., 1500-3000, 3000+)
4. Within each bucket:
   - Dedupe by context patterns using LSH (similar contexts)
   - Greedy diversity sampling
5. Allocate samples proportionally to buckets for phase1/phase2

**Verified**:
- ✅ Works with any BIO tags (just checks for `B-` prefix)
- ✅ No hardcoded role expectations
- ✅ Extracts n-gram context patterns around entities
- ✅ Creates bucketed curriculum stages

### Step 3: train_modernbert_streaming.py ✅ VERIFIED
**Input**: Phase1 + Phase2 JSONL
**Output**: Trained model

**Verified**:
- ✅ Uses 3-label system (O, B-PERSON, I-PERSON)
- ✅ Class weights correct for 3 labels
- ✅ Canonical label alignment

## Critical Points (Regression Prevention)

### ✅ VERIFIED: Role Simplification
- Preprocessor ignores `role` field
- All entities tagged as PERSON
- No B-PROTAGONIST, B-ANTAGONIST, etc.

### ✅ VERIFIED: Scene Splitting
- Splits at `### Scene N:` markers
- Each scene reprocessed independently (no tag slicing)
- Entity boundaries preserved per scene

### ⚠️ FIXED: Long Scene Handling
**Problem**: Scenes > 4096 tokens were being skipped (data loss)
**Solution**: Keep all scenes ≥ min_tokens, let tokenizer truncate at max_length during training
**Status**: ✅ FIXED in split_scenes.py

## Complete Pipeline Commands

```bash
# Step 1: Split scenes from canonical source (115k stories → ~300k+ scenes)
python3 split_scenes.py \
  --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
  --output ner_training_scenes_person_only.jsonl \
  --min-tokens 100

# Step 2: Create diverse curriculum subsets
python3 create_diverse_subset.py \
  --input ner_training_scenes_person_only.jsonl \
  --output ./diverse_subsets \
  --phase1-size 30000 \
  --phase2-size 50000

# Step 3: Train curriculum model
python3 train_modernbert_streaming.py \
  --model allenai/longformer-base-4096 \
  --curriculum \
  --curriculum-stage1-input diverse_subsets/diverse_phase1_30k.jsonl \
  --curriculum-stage2-input diverse_subsets/diverse_phase2_50k.jsonl \
  --output story_ner_model_curriculum3_person_only \
  --curriculum-stage1-epochs 2 \
  --curriculum-stage2-epochs 3 \
  --use-bf16
```
