# NER Training Methodology

## Data Preprocessing

### Contextual Labeling Strategy

**Implemented in:** `rune/data/story_preprocessor.py`

**Rules:**
1. Titles (Mr., Dr., Lady, etc.) → labeled as O (outside entity spans)
2. Surnames → only labeled when:
   - Preceded by title (e.g., "Mr. Cook" → O O B-PERSON)
   - Part of multi-word name (e.g., "James Cook" → B-PERSON I-PERSON)
3. Common noun surnames (Cook, Baker, Stone, etc.) → require contextual cues:
   - "the cook" → O O (no label)
   - "Mr. Cook" → O O B-PERSON
   - "James Cook" → B-PERSON I-PERSON

**Rationale:** Prevents false positive labels (~2.8% of training samples). Old greedy matching would label "the cook" as B-PERSON when character named "James Cook" exists in story.

### Name Variant Expansion

**Implemented in:** `rune/data/story_preprocessor.py::_expand_name_variants()`

Canonical name "James Cook" generates variants:
- First name: "James"
- Last name: "Cook"
- Full name: "James Cook"
- Case variants: lowercase, uppercase

Title handling for "Dr. Alessandro Sanna":
- "Dr. Alessandro"
- "Dr. Sanna"
- "Alessandro"
- "Sanna"
- "Alessandro Sanna"

### Data Filtering

**Name frequency cap:** `--max-stories-per-name 15`
- Prevents overfitting to frequent character names
- Script: `diversify_by_name_frequency.py`

**Token length filter:** `--max-tokens 8000`
- Matches ModernBERT 8k context window
- Script: `prefilter_by_length_8k.py`

## Training Configuration

### Model
- **Architecture:** ModernBERT-base
- **Context window:** 8192 tokens (cased)
- **Label scheme:** B/I-PERSON (simplified, role-agnostic)

### Hyperparameters
- **Epochs:** 3
- **Batch size:** 3 (effective: 15 with gradient accumulation)
- **Checkpoints:** Every 0.2 epochs (15 total)
- **Early stopping:** Disabled (train through local maxima)
- **Checkpoint retention:** Keep 10 best by validation F1

### Data Characteristics
- **Training size:** ~14k stories (after diversification + filtering)
- **Data source:** 100% synthetic narrative fiction
- **Domain:** Character-driven fiction with dialogue

## Label Schema

### Current: Simplified B/I-PERSON
- B-PERSON: Beginning of person entity
- I-PERSON: Inside person entity (continuation)
- O: Outside entity

**Rationale:** Role information (protagonist/antagonist/supporting) separated into entity metadata. Simpler schema reduces label space complexity.

### Previous: Role-Specific Labels
- B-protagonist, I-protagonist
- B-antagonist, I-antagonist
- B-supporting, I-supporting

**Issue:** 7-way classification harder to learn. Role classification decoupled to separate task.

## Validation Strategy

### Strict Matching
Entity boundaries and text must match exactly.

### Partial Matching
Allows partial overlap (e.g., "Mr. Bennet" predicted when "Bennet" is ground truth).

**Use case:** Better reflects real-world utility where partial detection is useful.

## Known Limitations

1. **Domain shift:** Model trained on 100% synthetic data, may underperform on classic literature
2. **Title handling:** Titles treated as context cues only (not part of entity spans)
3. **Pronoun resolution:** Not handled (out of scope for current NER task)
4. **Nested entities:** Not supported (e.g., "Captain America" as single entity vs "America" as location)
