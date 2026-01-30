# Story NER Model v0.1 - Frozen Checkpoint

**Status**: ✅ FROZEN (2026-01-30 21:00 UTC)  
**Directory**: `./story_ner_model_curriculum3_person_only/`  
**Model Size**: 570 MB (safetensors)

---

## Model Identification

| Attribute | Value |
|-----------|-------|
| **Base Model** | allenai/longformer-base-4096 |
| **Training Mode** | Streaming (no fixed batch accumulation) |
| **Max Token Length** | 4,096 tokens |
| **Curriculum Stage** | 3 (curriculum learning with diverse scenes) |
| **Label Scope** | PERSON entities only (BIO tags) |
| **Training Stories** | 79,995 narratives |

---

## Training Regime

### Pipeline & Data

**Training Data Stack** (in order):
1. **Stage 0** (initial warmup): Core 15k canonical stories from classic literature
2. **Stage 1** (diversification): 40k scenes from diverse literary corpus (curriculum2 dataset)
3. **Stage 2** (final): ~80k diverse scenes with curriculum3 selection

**Scene Processing**:
- Scene splitting: Large stories split into <4096-token chunks
- Diversity sampling: LSH-based novelty selection (auto-stop disabled, all scenes included)
- Deduplication: Exact hash + MinHash (0.7 threshold for context similarity)

**Preprocessing**:
- spaCy-based surname licensing (dependency parsing for subject/possessive anchoring)
- Multi-token title handling (Mr., Mrs., Dr., etc. as single tokens post-spaCy)
- Naming-verb suppression (block surnames after "named", "called", "dubbed", "christened")
- Full BIO tag alignment (O/B-PERSON/I-PERSON)

### Training Parameters

```
Model:           allenai/longformer-base-4096 (base 12L)
Optimizer:       AdamW (β₁=0.9, β₂=0.999, eps=1e-8)
Learning rate:   2e-4 (initial)
Warmup steps:    500
Gradient accum:  2
Max epochs:      3
Batch size:      16 (per device, streaming mode)
Loss function:   Cross-entropy (unweighted, no label smoothing)
```

### Hardware
- GPU: NVIDIA A100 (40GB)
- Framework: PyTorch + Hugging Face Transformers
- Precision: fp32 (full precision)

---

## Performance Metrics

### Aggregate on OOD Validation Set

**Dataset**: `ood_validation_ground_truth_modern_heavy.jsonl` (21 contemporary + classic stories)

```
Precision:  82.21%  (TP / (TP + FP))
Recall:     86.00%  (TP / (TP + FN))
F1-Score:   84.06%  (harmonic mean)
```

### Per-Story Performance

**Top Performers** (F1 > 90%):
- pride_prejudice_sample: 91%+ F1
- sherlock_holmes_scandal_bohemia: 90%+ F1
- jane_eyre_scene: 88%+ F1

**Challenge Areas** (F1 < 75%):
- Complex fantasy narratives with non-standard name conventions
- Stories with heavy use of titles without character names
- Mixed-language or archaic texts

### Known Behaviors

**Strengths**:
- ✅ Subject-verb anchoring: "Holmes rushed" correctly tagged
- ✅ Possessives: "Holmes's cane" correctly tagged
- ✅ Dialogue contexts: "said Mr. Bennet" handles title+name
- ✅ Canonical name variants: "Sherlock Holmes" vs "Holmes" vs "Mr. Holmes" unified

**Limitations**:
- ⚠️ Weak metadata references: "named Norton" may suppress orphan surnames
- ⚠️ Same-story variations: Multiple scenes from same story slightly dilute diversity
- ⚠️ Tokenizer mismatch: spaCy preprocessing tokens vs Longformer tokenizer (not yet aligned)
- ⚠️ Title-only references: "the Baker" incorrectly tagged in some contexts

---

## Data Characteristics

### Training Corpus

**Size**: ~80k scenes, ~62 million tokens total

**Composition**:
- 60% Classic literature (English canon: Austen, Brontë, Dickens, Conan Doyle, etc.)
- 25% 19th-century narratives (varied genres, historical fiction)
- 15% Contemporary literary works (modern classics, recent publications)

**Entity Density**:
- Average entities per scene: 3-5 PERSON tags
- Entity name frequency: Power-law distribution (few common names, long tail of rare names)
- Scene length: 100-4096 tokens (bell curve centered ~800)

### Preprocessing Artifacts

**Applied Transformations**:
1. spaCy dependency parsing for surname licensing
2. Multi-token title contraction (Mr. = single token in output)
3. Exact dedup on serialized token sequences
4. MinHash dedup for context patterns (LSH, τ=0.7)

**Known Gaps**:
- Tokenizer alignment: spaCy vs Longformer (position mapping not yet implemented)
- Fallback behavior: No graceful fallback if spaCy missing (fail-fast by design)

---

## Label Policy & Justification

### Official NER Tagging Rules

**Tag B-PERSON if and only if**:
1. Token is capitalized, title-cased, or proper noun
2. **AND** one of:
   - Preceded by title marker (Mr., Mrs., Dr., Sir, Lady, etc.)
   - Subject of verb (nsubj/nsubjpass dependency)
   - Head of possessive phrase ("token's")
   - Direct address (appos, vocative)
   - Part of multi-token name phrase

**Tag O (suppress) if**:
1. Orphan surname without verb/title anchor
2. Preceded by naming verb (named, called, dubbed) in attribute position
3. Role/title masquerading as name ("the Baker", "the Cook")
4. Location or non-person entity incidentally capitalized

### Rationale

The "X of Y" distinction disambiguates **entity** (person with agency) from **role** (description). Example:
- ✅ "Catherine of Aragon" → PERSON (historical entity)
- ❌ "Archmage of Ymitury" → O (role descriptor)

Test: Can you prepend "the"? If yes (e.g., "the Archmage"), suppress. If no (e.g., "the Catherine"), tag.

---

## Files & Artifacts

### Model Files
```
story_ner_model_curriculum3_person_only/
├── model.safetensors             # 565 MB model weights
├── config.json                   # Longformer base config
├── tokenizer.json                # Byte-pair encoding tokenizer
├── vocab.json                    # BPE vocabulary (50k tokens)
├── merges.txt                    # BPE merge rules
├── special_tokens_map.json       # [CLS], [SEP], [PAD] mappings
├── tokenizer_config.json         # add_prefix_space, etc.
├── training_metadata.json        # Dataset & regime summary
├── training_args.bin             # Full HF Trainer args (serialized)
├── checkpoint-674, 1348, 2022, 2696  # Intermediate checkpoints (archived)
├── stage1/                       # Stage 1 pretrained weights
└── logs/                         # Training curves & loss logs
```

### Evaluation Results
```
eval_preview_curriculum3-results-fixed.json    # Full per-story metrics
```

### Documentation
```
CHECKPOINT_V0.1_FROZEN.md        # This file (regime + performance)
METHODOLOGY.md                   # Data processing pipeline
README.md                        # High-level overview
```

---

## How to Use v0.1

### Basic Inference
```python
from transformers import pipeline

nlp = pipeline(
    "token-classification",
    model="./story_ner_model_curriculum3_person_only",
    device=0
)

text = "Mr. Bennet replied that he had not heard."
entities = nlp(text)
```

### Batch Processing
```python
from datasets import Dataset

texts = ["...", "...", "..."]
dataset = Dataset.from_dict({"text": texts})
predictions = nlp(dataset, batch_size=32)
```

### Advanced: Fine-tuning
```python
from transformers import Trainer, TrainingArguments

# Load pretrained v0.1
model = AutoModelForTokenClassification.from_pretrained(
    "./story_ner_model_curriculum3_person_only"
)

# Fine-tune on new domain
trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=new_data,
)
trainer.train()
```

---

## Checkpoints Retained vs Archived

### Kept (v0.1)
- ✅ Final frozen model + all weights
- ✅ Training metadata & regime docs
- ✅ Evaluation results on OOD validation set

### Archived (in `/archived/`)
- Curriculum 2 models (earlier, lower performance)
- Individual checkpoint directories (C1348, C2022, etc.)
- Intermediate evaluations and logs
- Alternative architectures (not selected)

---

## Next Steps & Backlog

### Immediate (v0.2)
- [ ] Implement tokenizer alignment (spaCy ↔ Longformer position mapping)
- [ ] Re-evaluate using aligned tokenizer metrics
- [ ] Retrain with smarter diversity sampling (auto-stop with story penalty)

### Medium-term (v0.3)
- [ ] Extend to multi-label (PERSON + PLACE + ORGANIZATION)
- [ ] Add confidence scores per entity
- [ ] Active learning loop for hard examples

### Long-term
- [ ] Domain adaptation for non-English narratives
- [ ] Streaming prediction API
- [ ] Integration with downstream NLP tasks (coreference, relation extraction)

---

## Metadata Summary

```json
{
  "version": "v0.1",
  "frozen_date": "2026-01-30T21:00:00Z",
  "model_path": "./story_ner_model_curriculum3_person_only",
  "base_model": "allenai/longformer-base-4096",
  "training_regime": "curriculum3_diverse_scenes",
  "label_scope": "PERSON_only",
  "performance": {
    "precision": 0.8221,
    "recall": 0.8600,
    "f1": 0.8406
  },
  "validation_dataset": "ood_validation_ground_truth_modern_heavy.jsonl",
  "data_stories_count": 79995,
  "max_seq_length": 4096
}
```

---

**Status**: This checkpoint is now frozen and will not be modified. Future improvements will be tracked as v0.2, v0.3, etc.
