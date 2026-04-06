# Running Diversity Pipeline on Latest Dataset

## Quick Start

### Latest Dataset
```
ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl
```

### Run Complete Pipeline (Recommended)

```bash
python create_diverse_subset_with_splitting.py \
  --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
  --output ./diverse_output \
  --max-tokens 4096 \
  --auto-stop
```

**What this does:**
1. ✅ Splits any story over 4096 tokens into scenes
2. ✅ Each scene regenerates proper BIO tags (safe)
3. ✅ Applies diversity sampling with balanced scoring
4. ✅ Auto-stops when novelty exhausted

**Expected output:**
- `diverse_output/diverse_combined.jsonl` (main training file)
- All scenes within 4096 token limit
- ~20-30% of original size with maximum diversity

---

## How Scene Splitting Works

### Existing Utility: `split_scenes.py`

**What it does:**
- Finds `### Scene N:` markers in text
- Splits long stories at scene boundaries
- **Regenerates BIO tags** for each scene (safe approach)
- Filters by token range (min=100, max=4096)

**Key feature:** Safe splitting
```python
# V1 (dangerous): Slice existing BIO tags → corrupts entity boundaries
bio_tags_scene1 = bio_tags[0:500]  # ❌ Can split B-I tags

# V2 (safe): Regenerate BIO tags per scene → correct boundaries
scene_text = text[scene_start:scene_end]
bio_tags_scene1 = preprocessor.tag(scene_text)  # ✅ Fresh tags
```

### Pipeline Integration

The combined script (`create_diverse_subset_with_splitting.py`) chains:

```
Input JSONL
    ↓
[1] Scene Splitting (split_scenes.py)
    → Splits stories > 4096 tokens
    → Regenerates BIO tags per scene
    ↓
Temp scenes file
    ↓
[2] Diversity Sampling (create_diverse_subset_v2.py)
    → Balanced novelty-redundancy scoring
    → Auto-stops when exhausted
    ↓
Output: diverse_combined.jsonl
```

---

## Usage Options

### 1. Auto-Stop Mode (Recommended)

```bash
python create_diverse_subset_with_splitting.py \
  --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
  --output ./diverse_output \
  --max-tokens 4096 \
  --auto-stop \
  --alpha 1.0 \
  --beta 0.5 \
  --epsilon 0.01
```

**Parameters:**
- `--max-tokens 4096`: Split scenes longer than this
- `--auto-stop`: Stop when novelty < ε
- `--alpha 1.0`: Novelty appetite (higher = more exploration)
- `--beta 0.5`: Redundancy aversion (higher = more compact)
- `--epsilon 0.01`: Stop threshold (lower = stricter)

### 2. Fixed Size Mode

```bash
python create_diverse_subset_with_splitting.py \
  --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
  --output ./diverse_output \
  --max-tokens 4096 \
  --phase1-size 20000 \
  --phase2-size 40000
```

**Use when:** You need exact output sizes for benchmarking.

### 3. Skip Splitting (If Already Split)

```bash
python create_diverse_subset_with_splitting.py \
  --input already_split_scenes.jsonl \
  --output ./diverse_output \
  --skip-splitting \
  --auto-stop
```

**Use when:** Data already has scenes split, just need diversity sampling.

### 4. Keep Intermediate Files

```bash
python create_diverse_subset_with_splitting.py \
  --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
  --output ./diverse_output \
  --max-tokens 4096 \
  --auto-stop \
  --keep-temp
```

**What it does:**
- Saves scene-split file: `diverse_output/scenes_split.jsonl`
- Useful for debugging or reusing split scenes

---

## Understanding the Token Limit

### Why 4096?

**Model limits:**
- ModernBERT: 8192 tokens max
- Most transformers: 512-4096 tokens
- Training efficiency: Shorter = faster

**Setting it:**
```bash
--max-tokens 4096  # Default, good for most models
--max-tokens 2048  # For older models (BERT-base)
--max-tokens 8192  # For long-context models (ModernBERT)
```

### What Happens to Long Stories?

**Before splitting:**
```
Story: "The Epic Tale" (8,500 tokens)
└─ Too long! Would be truncated in training → lose entities
```

**After splitting:**
```
Story: "The Epic Tale" (8,500 tokens)
├─ Scene 1: "Opening" (1,200 tokens) ✅
├─ Scene 2: "Middle" (2,800 tokens) ✅
├─ Scene 3: "Climax" (2,100 tokens) ✅
└─ Scene 4: "End" (2,400 tokens) ✅

Result: 4 training examples instead of 1 truncated one
```

---

## Checking Results

### Token Distribution

The pipeline automatically checks:
```
TOKEN DISTRIBUTION CHECK
  Total items: 18,453
  Token range: 102 - 4095
  Average: 1,847 tokens
  Over limit (4096): 0 items
  ✅ All items within 4096 token limit
```

### Manual Check

```python
import json

# Load output
with open('diverse_output/diverse_combined.jsonl') as f:
    items = [json.loads(line) for line in f]

# Check token lengths
lengths = [len(item['tokens']) for item in items]
print(f"Total: {len(items)}")
print(f"Max tokens: {max(lengths)}")
print(f"Avg tokens: {sum(lengths)/len(lengths):.0f}")
print(f"Over 4096: {sum(1 for l in lengths if l > 4096)}")
```

### Verify Scene Splitting

```python
# Check for scene IDs
scene_stories = [
    item for item in items
    if 'scene' in item.get('story_id', '').lower()
]
print(f"Stories split into scenes: {len(scene_stories)}")

# Check parent references
parents = set(
    item.get('metadata', {}).get('parent_story_id', '')
    for item in scene_stories
)
print(f"Unique parent stories: {len(parents)}")
```

---

## File Structure

```
/mnt/storage/rune-lib/
├── split_scenes.py                              ← Existing scene splitter
├── create_diverse_subset_v2.py                  ← V2 diversity sampler
├── create_diverse_subset_with_splitting.py      ← Combined pipeline (NEW)
└── RUN_DIVERSITY_PIPELINE.md                    ← This guide
```

---

## Common Issues

### "split_scenes.py not found"

**Solution:** Run from `/mnt/storage/rune-lib/` directory:
```bash
cd /mnt/storage/rune-lib
python create_diverse_subset_with_splitting.py --input ...
```

### "Context patterns disabled warning"

**Solution:** You're using V1, use the pipeline which calls V2:
```bash
python create_diverse_subset_with_splitting.py  # Uses V2 automatically
```

### "Some items over 4096 tokens"

**Causes:**
1. Scene splitting was skipped (`--skip-splitting`)
2. Stories don't have `### Scene N:` markers

**Solution:** Don't use `--skip-splitting`, or add scene markers to source data.

### Out of Memory

**Solution 1:** Process in chunks
```bash
# Split input into chunks
split -l 10000 input.jsonl input_chunk_

# Process each chunk
for chunk in input_chunk_*; do
  python create_diverse_subset_with_splitting.py \
    --input $chunk \
    --output ./output_$chunk \
    --auto-stop
done
```

**Solution 2:** Lower context window
```bash
--context-window 2  # Instead of 3 (default)
```

---

## Expected Runtime

**Dataset:** 2.1 GB (mixed training)
**Hardware:** Typical workstation

| Step | Time | Memory |
|------|------|--------|
| Scene splitting | 10-20 min | 2-4 GB |
| Diversity sampling | 30-60 min | 4-8 GB |
| **Total** | **40-80 min** | **4-8 GB** |

**Optimization tips:**
- Install optional deps: `pip install datasketch pybloom-live xxhash`
- Use SSD for I/O speed
- Close other applications

---

## Recommended Workflow

### 1. Quick Test (5 minutes)

```bash
# Test on small sample
head -5000 ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl > test_sample.jsonl

python create_diverse_subset_with_splitting.py \
  --input test_sample.jsonl \
  --output ./test_output \
  --max-tokens 4096 \
  --auto-stop
```

**Check:**
- Scene splitting works
- Token limits respected
- Output files created

### 2. Full Run (40-80 minutes)

```bash
python create_diverse_subset_with_splitting.py \
  --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
  --output ./diverse_output \
  --max-tokens 4096 \
  --auto-stop \
  --keep-temp
```

**Monitor:**
- Watch for "Unique contexts accumulated"
- Check "Avg score" stays above epsilon
- Verify auto-stop happens naturally

### 3. Validate Output

```bash
python << 'EOF'
import json

with open('./diverse_output/diverse_combined.jsonl') as f:
    items = [json.loads(line) for line in f]

lengths = [len(item['tokens']) for item in items]
print(f"✅ Total items: {len(items):,}")
print(f"✅ Max tokens: {max(lengths)}")
print(f"✅ Avg tokens: {sum(lengths)/len(lengths):.0f}")
print(f"✅ Items > 4096: {sum(1 for l in lengths if l > 4096)}")
EOF
```

### 4. Use for Training

```bash
# The output is ready to use
python train_modernbert_streaming.py \
  --train-file diverse_output/diverse_combined.jsonl \
  --val-file ood_validation_ground_truth_modern_heavy.jsonl \
  --max-length 4096
```

---

## Advanced: Custom Scene Boundaries

If your data doesn't have `### Scene N:` markers, add them:

```python
import json
import re

def add_scene_markers(text: str, chunk_size: int = 2000) -> str:
    """Add scene markers every N tokens."""
    words = text.split()
    scenes = []

    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        scenes.append(chunk)

    # Join with scene markers
    marked = '\n\n### Scene 1:\n\n' + \
             '\n\n'.join(f'### Scene {i+2}:\n\n{s}' for i, s in enumerate(scenes[1:]))

    return scenes[0] + marked

# Process data
with open('input.jsonl') as fin, open('output.jsonl', 'w') as fout:
    for line in fin:
        item = json.loads(line)
        item['text'] = add_scene_markers(item['text'])
        fout.write(json.dumps(item) + '\n')
```

---

## Summary Commands

**Most common usage:**
```bash
# Complete pipeline, auto-stop, 4096 tokens
python create_diverse_subset_with_splitting.py \
  --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
  --output ./diverse_output \
  --max-tokens 4096 \
  --auto-stop
```

**Fixed size (20k + 40k):**
```bash
python create_diverse_subset_with_splitting.py \
  --input ner_training_mixed_training_unusual_names_training_ambiguous_names_20260127_154404.jsonl \
  --output ./diverse_output \
  --max-tokens 4096 \
  --phase1-size 20000 \
  --phase2-size 40000
```

**Already split, just diversity:**
```bash
python create_diverse_subset_with_splitting.py \
  --input already_split.jsonl \
  --output ./diverse_output \
  --skip-splitting \
  --auto-stop
```

---

**Status:** Ready to run on latest dataset
**Next:** Test on small sample, then full run
