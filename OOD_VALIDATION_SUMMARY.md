# OOD Validation Set Summary

## Overview

Created an Out-of-Distribution validation dataset for narrative-based Named Entity Recognition (NER) classification, extending the existing validation set with high-quality classic literature scenes.

**Location:** `/mnt/storage/rune-lib/ood_validation_ground_truth.jsonl`

## Dataset Composition

### Original Records (5)
- Pride and Prejudice sample
- Colour of Magic extract (fantasy)
- The Ships of Merior extract (fantasy)
- The Ghost in the Doll extract
- A Memory Called Empire extract (modern SF)

### New Classic Scenes (5)

| Book | Scene | Words | Characters | Density | Key Features |
|------|-------|-------|------------|---------|--------------|
| Call of the Wild | The Thousand-Pound Wager | 1,087 | 6 | 0.0370 | High-stakes dialogue, action narrative |
| Jane Eyre | Charades at Thornfield Hall | 727 | 10 | 0.0271 | Aristocratic gathering, theatrical performance |
| Jane Eyre | The Governess Discussion | 1,096 | 10 | 0.0187 | Multi-speaker dialogue, social commentary |
| Moby Dick | Meeting Queequeg | 361 | 3 | 0.0258 | Cultural diversity, character-driven dialogue |
| Sherlock Holmes | A Scandal in Bohemia - Resolution | 386 | 5 | 0.0154 | Mystery plot, investigative dialogue |

## Quality Metrics

- **Total Records:** 10
- **Total Words:** 20,658
- **Average Scene Length:** 2,066 words
- **Average Characters per Scene:** 8.8
- **Entity Density Range:** 0.0070 - 0.0377
- **Average Entity Density:** 0.0186

### Entity Density Analysis

The new classic scenes were selected for higher entity density:
- **Highest:** Call of the Wild (0.0377) - 70% above original P&P baseline
- **Target Range:** 0.015+ (matching or exceeding the validation set baseline)
- All new scenes meet or exceed the 0.015 threshold

## Data Sources

### Texts Downloaded from Project Gutenberg
1. **The Adventures of Sherlock Holmes** (ID: 1661)
   - Selected story: "A Scandal in Bohemia"

2. **Call of the Wild** by Jack London (ID: 215)
   - Selected chapter: VI - The Thousand-Pound Wager

3. **Jane Eyre** by Charlotte Brontë (ID: 1260)
   - Selected scenes: Charades at Thornfield, Governess Discussion

4. **Moby Dick** by Herman Melville (ID: 2701)
   - Selected scene: Chapter 3 - First Meeting with Queequeg

5. **Great Expectations** by Charles Dickens (ID: 1083)
   - Note: Initially scanned but no scenes met entity density requirements

## Format Specification

Each record follows this JSONL structure:

```json
{
  "story_id": "unique_identifier",
  "text": "scene text content",
  "characters": [
    {"name": "Character Name", "role": "PROTAGONIST|SUPPORTING|ANTAGONIST"},
    ...
  ],
  "metadata": {
    "genre": "classic",
    "source": "validation",
    "book": "Book Title (optional)",
    "entity_density": 0.XXXX (optional)
  }
}
```

## Key Characteristics

### Narrative Diversity
- Mix of dialogue-heavy scenes (Governess Discussion, Sherlock)
- Action-narrative scenes (Call of the Wild Wager)
- Theatrical/structured scenes (Jane Eyre Charades)
- Cross-cultural dialogue (Moby Dick)

### Entity Recognition Challenges
- **Name Variations:** Theodore/Tedo, formal titles (Mr., Miss, Colonel, Sir)
- **Family Name Sharing:** Multiple Ingrams, Eshtons in Jane Eyre
- **Character Roles:** Clear protagonist/supporting distinction
- **Context Sensitivity:** Names embedded in various narrative contexts

### Use Cases
✓ Out-of-distribution validation for literary NER models
✓ Benchmark for entity density and character interaction detection
✓ Evaluation of model generalization to classic literature
✓ Testing of formal/archaic language processing

## Backup

Original validation set backed up to:
`/mnt/storage/rune-lib/validation_ground_truth.jsonl.backup`

## Next Steps for Annotation

The dataset is ready for:
1. **Entity Tag Annotation** - Mark entities with IOB tags if needed
2. **Role Refinement** - Verify and refine PROTAGONIST/SUPPORTING assignments
3. **Coreference Resolution** - Track character mentions across scenes
4. **Domain-Specific Testing** - Evaluate model performance on classic vs. modern text

## Files Generated

- `ood_validation_ground_truth.jsonl` - Final validation dataset (10 records)
- `validation_ground_truth.jsonl.backup` - Original validation set backup
- Source texts in `/tmp/claude/-mnt-storage-rune-lib/9b206c55-560e-44ef-99a8-f50642df3d90/scratchpad/texts/`
  - `sherlock_holmes.txt` (594 KB)
  - `call_of_the_wild.txt` (196 KB)
  - `jane_eyre.txt` (1.1 MB)
  - `moby_dick.txt` (1.3 MB)
  - `great_expectations.txt` (618 KB)
