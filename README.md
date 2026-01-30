# Rune - Named Entity Recognition Library

A minimalistic Python library for BIO-tagging NER with BERT-based transformers for literary/narrative texts.

---

## ⚠️ NER Labeling Policy (CRITICAL)

### Tag Names, Not Roles

**Rule:** Tag what someone is **CALLED**, not what they **ARE**.

#### ✅ DO Tag (B-PERSON):
- **Proper names:** `Catherine of Aragon`, `Joan of Arc`, `Leonardo da Vinci`
- **Character names used as identifiers:** `Rincewind`, `Elizabeth Bennet`, `The Weasel` (when used as a name)

#### ❌ DON'T Tag (O):
- **Titles + prepositional phrases:** `the Archmage of Ymitury`, `the King of England`, `a knight of the realm`
- **Test:** If you can add "the" and it still makes sense → it's a role/description → O-tag

#### Decision Heuristic:

| Type | Example | Tag? | Why |
|------|---------|------|-----|
| Proper name | `Catherine of Aragon` | ✅ B-PERSON | Canonical identifier, functions as fixed surface form |
| Honorific epithet | `Joan of Arc` | ✅ B-PERSON | Stable referent, "of Arc" is part of name's identity |
| Title + domain | `the Archmage of Ymitury` | ❌ O | Article "the" → descriptive noun phrase, not personal name |
| Generic role | `a knight of the realm` | ❌ O | No referential uniqueness, indefinite article |

**Quick tests:**
1. **Article check:** Starts with `the`, `a`, `an` → likely O-tag
2. **Determiner check:** `my`/`our`/`his` + "of" phrase → O-tag (e.g., "my lord of Daruun")
3. **Can you add "the"?** If yes → O-tag (role/title)

---

### Why This Matters (Strategic Rationale)

**Current phase:** PERSON-only NER (no LOCATION/ORG labels yet)

We intentionally **don't** tag location names in constructions like "Archmage of Ymitury" because:

1. **Reduces class confusion:** Mixing roles/titles into B-PERSON creates conflicting gradients
   - Model sees: `Catherine of Aragon` → PERSON ✅
   - Model sees: `Archmage of Ymitury` → PERSON ❌
   - Result: "of [CapitalWord]" becomes ambiguous noise

2. **Prepares for multi-label future:** When we add B-LOCATION in the next phase:
   ```
   the Archmage of Ymitury
   O   O        O  B-LOCATION
   ```
   - Model learns: "of + B-LOCATION" ≠ person entity
   - B-PERSON precision improves (cleaner boundaries)
   - **Entropy removal:** Splits messy mixed cluster into orthogonal, learnable patterns

3. **Cognitive load reduction:** Not forcing the model to resolve title vs. name ambiguity at this stage
   - Let B-PERSON learn **true proper names** cleanly
   - Save role/place disambiguation for B-LOCATION/B-ORG phase
   - Result: Sharper gradients, better F1, easier coreference resolution later

**Expected outcome when B-LOCATION is added:**
- Higher F1 for person detection (cleaner boundary signal)
- Emergent zero-shot generalization for place mentions
- Easier relation extraction (clearer "who vs where" demarcations)

**Policy in one sentence:**  
*Adding B-LOCATION later isn't extra weight — it's entropy removal. We're not burdening the model; we're partitioning ambiguity.*

---

### Name Variant Expansion: First vs. Last Names + spaCy Integration

**Current State (V2 Preprocessor):** Expands all name parts equally, then uses **mandatory spaCy** dependency parsing to gate surname-only mentions at BIO-tagging time.

#### spaCy-Based Surname Licensing (MANDATORY) 🔧

The preprocessor now **requires spaCy** (en_core_web_sm) for POS tagging and dependency parsing. There is NO fallback mode.

**Why mandatory:**
- ✅ Eliminates duplicate code paths (no debugging two systems)
- ✅ POS tags automatically handle verb conjugations ("said"/"says"/"remarked")
- ✅ Dependency parsing provides true syntactic structure (not surface patterns)
- ✅ Cleaner, more maintainable code

**Surname gating logic** (from `_check_surname_licensing`):
1. ✅ **Subject of verb** (nsubj/nsubjpass dependency): "Holmes rushed"
2. ✅ **Has possessive marker** (poss dependency): "Holmes's coat"
3. ✅ **Vocative/appositive** (direct address): "Holmes, come here!"
4. ✅ **Object of speech verb** (dobj after dialogue verb): "remarked Holmes"
5. ✅ **Capitalized non-article predecessor** (excluding non-speech verbs): "Mr. Bennet", "Elizabeth Bennet"

**Additional global filter** (applies to ALL single-token names):
- ❌ **Object of naming verb**: "named Norton", "called John" (weak introductory metadata, not functional entity mention)

**Blocks:**
- ❌ Article-prefixed: "the Bennet", "The Bennet"
- ❌ Preposition-prefixed: "of Aragon", "in Havish"
- ❌ Metalinguistic use: "'Seagrass' was memorable"
- ❌ Naming-verb object: "lawyer named Norton" (oprd/attr/dobj after naming verbs)

#### First Names (Liberal Expansion) ✅
- **Behavior:** Aggressively expand standalone usage
- **Example:** `"Mahit Dzmare"` → auto-generates variant `"Mahit"`
- **Why safe:** First names are contextually unambiguous in most narratives
- **Tagging confidence:** High - tag if capitalized + (dialogue context OR within 3 sentences of canonical form)

#### Last Names (Conservative Expansion) ⚠️
- **Behavior:** Requires contextual anchors (enforced via spaCy at BIO-tagging time)
- **Example:** `"Three Seagrass"` → auto-generates variant `"Seagrass"`
- **Risk:** High false-positive potential (common nouns, metalinguistic uses, generic references)
- **Required anchors for tagging:** See spaCy-based gating logic above

#### V2 Behavior vs. Model Reality

| Aspect | Preprocessor (V2) | Trained Model | Impact |
|--------|-------------------|---------------|--------|
| Last name expansion | Unconditional | Learns contextual gating | ✅ Model generalizes correctly |
| Metalinguistic use | Tags if in GT | Often skips (correct!) | Minor recall drop, good precision |
| Common noun overlap | Over-generates variants | Context disambiguates | Acceptable noise |

**Key insight:** Current model **already learned** to discriminate despite noisy training signal. Evidence that transformer context is sufficient, but preprocessing improvements will sharpen boundaries.

#### V3 Roadmap (Future)
Implement contextual gating at BIO-tag generation time:
```python
if token in lastname_variants:
    if not has_title_prefix(prev_token) and not near_canonical_form(context):
        skip_tagging()  # O-tag instead of B-PERSON
```

**Why not implement now:** Model demonstrates good generalization from existing data. Save sophisticated gating for next full training cycle to measure impact systematically.

---

## Installation

```bash
git clone <repository-url>
cd rune-lib
pip install -r requirements.txt
```

## Usage

```python
from rune.core.tagger import BioTagger

# Load trained model
tagger = BioTagger("./story_ner_model_longformer")

# Extract entities from text
text = "Sarah walked into the room where Marcus was waiting."
entities = tagger.predict(text)

for entity in entities:
    print(f"{entity['text']} → {entity['label']}")
```

## Library Structure

```
rune/
├── core/
│   ├── tagger.py              # Entity extraction
│   └── schema.py              # BIO tag schemas
├── data/
│   ├── story_preprocessor.py  # Data preprocessing with contextual labeling
│   └── improved_tokenizer.py  # Word tokenization for narrative text
├── models/
│   ├── bert.py                # BERT-based NER model
│   └── longformer.py          # Longformer for long sequences
└── training/
    └── trainer.py             # Training utilities
```

## Training

### Dataset Preparation

**Source Data:** `ner_training_latest_processed.jsonl` (7.4GB, 115,422 stories)

#### Step 1: Create Hybrid Dataset (Fixes Entity Index Bug)

The source data has an entity index bug where `end` indices are inclusive instead of exclusive,
causing only first names to be tagged (e.g., "Dmitri" instead of "Dmitri Volkov"). The hybrid
creator fixes this by reconstructing entities from BIO tags:

```bash
python3 create_hybrid_dataset.py \
    --input ner_training_latest_processed.jsonl \
    --output ./hybrid_dataset_fixed \
    --split-threshold 4096
```

**Strategy:**
- Reconstructs entity indices from BIO tags (fixes off-by-one bug)
- Stories <4096 tokens → Kept whole (preserves full context)
- Stories ≥4096 tokens → Split by scene markers with proper alignment
- Output: `hybrid_dataset_fixed/combined_hybrid.jsonl` (validated, clean labels)

#### Step 2: Apply Diversity Sampling

Removes pattern duplicates and balances entity context diversity:

```bash
python3 create_diverse_subset.py \
    --input hybrid_dataset_fixed/combined_hybrid.jsonl \
    --output ./diverse_hybrid_fixed \
    --phase1-size 20000 \
    --phase2-size 40000
```

**Output:**
- `diverse_hybrid_fixed/diverse_phase1_20k.jsonl` - Quick POC (20k, ~2 hours to train)
- `diverse_hybrid_fixed/diverse_combined_60k.jsonl` - Full dataset (60k, ~5 hours to train)

#### One-Command Pipeline

Run both steps with automatic validation:

```bash
bash REPROCESS_PIPELINE.sh
```

### Validate Your Data

Before training, inspect labels to catch corruption:

```bash
python3 view_story_labels.py \
    --input diverse_hybrid_fixed/diverse_phase1_20k.jsonl \
    --index 0
```

**Output:** Color-coded BIO tags with pre/post-tokenizer alignment. Look for:
- ✅ Full names tagged (e.g., "Dmitri Volkov" not just "Dmitri")
- ✅ Multi-word ratio >50% (check with viewer)
- ❌ Punctuation/articles tagged as entities (= corruption)

### Quick Training

```bash
# Using 20k subset (recommended for quick iteration)
python3 train_modernbert_streaming.py \
    --model allenai/longformer-base-4096 \
    --input diverse_hybrid_fixed/diverse_phase1_20k.jsonl \
    --output ./story_ner_model_clean \
    --simplify-labels \
    --num-epochs 3 \
    --batch-size 12 \
    --use-bf16

# Using full 60k subset (better accuracy, longer training)
python3 train_modernbert_streaming.py \
    --model allenai/longformer-base-4096 \
    --input diverse_hybrid_fixed/diverse_combined_60k.jsonl \
    --output ./story_ner_model_production \
    --simplify-labels \
    --num-epochs 3 \
    --batch-size 8 \
    --use-bf16
```

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | ModernBERT-base | Model architecture (`allenai/longformer-base-4096`, `answerdotai/ModernBERT-base`) |
| `--batch-size` | 8 | Batch size (use 12-16 for 16GB VRAM, 8 for 12GB) |
| `--use-bf16` | False | Use BF16 precision (saves ~30% memory) |
| `--simplify-labels` | False | Collapse all roles to B/I-PERSON (simpler, recommended) |
| `--num-epochs` | 3 | Training epochs |
| `--disable-early-stopping` | False | Disable early stopping |
| `--save-every-n-epochs` | 0.2 | Checkpoint frequency |

### Processing Raw Data (Advanced)

If you have raw data without `tokens`/`bio_tags`:

```python
from rune.data.story_preprocessor import StoryPreprocessor

preprocessor = StoryPreprocessor(use_spacy=False)
preprocessor.process_jsonl_file(
    'raw_stories.jsonl',
    'processed_stories.jsonl'
)
```

## Design Notes

### Title Handling
Titles (`Dr.`, `Mr.`, `Captain`) are tagged as `O` (outside entity) and serve as context cues. This prevents the model from learning titles as part of person names:
- `"Dr. Anderson"` → `['Dr', '.', 'Anderson']` → `['O', 'O', 'B-PERSON']`

### Contextual Gating
Prevents false positives from ambiguous words:
- `"the lamp"` → not tagged (lowercase, no context)
- `"Helena Lamp"` → tagged (full name)
- `"Dr. Lamp"` → tagged (title provides context)

## Testing

```bash
pytest tests/
```

## Documentation

- `START_TRAINING.sh` - Canonical training command
- `METHODOLOGY.md` - Training methodology and approach

## License

MIT License
