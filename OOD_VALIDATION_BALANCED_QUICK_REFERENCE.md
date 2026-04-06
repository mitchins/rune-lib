# Balanced OOD Validation Set - Quick Reference

## TL;DR

✅ **16-record balanced OOD validation dataset created**
- 10 classical records (literary + fantasy baseline)
- 6 modern records (contemporary fantasy/sci-fi)
- **Location:** `/mnt/storage/rune-lib/ood_validation_ground_truth_balanced.jsonl`

## What You Get

```
📊 Dataset Stats
├─ Total Records: 16
├─ Total Words: 38,801
├─ Avg Scene Length: 2,425 words
├─ Temporal Span: 1851-2014 (163 years)
└─ Total Unique Characters: 108

🏛️ Classical Records (10)
├─ Original 5 (Pride & Prejudice + 4 fantasy baseline)
└─ New 5 (Call of the Wild, Jane Eyre x2, Moby Dick, Sherlock Holmes)
   └─ Entity Density: 0.0217 avg

🚀 Modern Records (6)
├─ Mistborn (0.0487) ⭐ highest
├─ Assassin's Apprentice (0.0430)
├─ Magician (0.0505) ⭐ highest
├─ Red Rising (0.0362)
├─ Dune (0.0465)
└─ The Lies of Locke Lamora (0.0422)
   └─ Entity Density: 0.0445 avg
```

## File Locations

**Main validation set (use this):**
```
/mnt/storage/rune-lib/ood_validation_ground_truth_balanced.jsonl
```

**Supporting files:**
```
/mnt/storage/rune-lib/
├── ood_validation_ground_truth.jsonl           (classical only, 10 records)
├── modern_validation_scenes.jsonl              (modern only, 6 records)
├── validation_ground_truth.jsonl.backup        (original backup)
└── OOD_VALIDATION_BALANCED_SUMMARY.md          (full analysis)
```

## Record Breakdown

### Records 1-5: Original (Baseline)
- Pride & Prejudice
- Colour of Magic
- The Ships of Merior
- The Ghost in the Doll
- A Memory Called Empire

### Records 6-10: New Classical Literature
- #6: Call of the Wild (Jack London, 1903)
- #7-9: Jane Eyre scenes (Charlotte Brontë, 1847)
- #8: Moby Dick (Herman Melville, 1851)
- #10: Sherlock Holmes (Arthur Conan Doyle, 1892)

### Records 11-16: Modern Fantasy/Sci-Fi
- #11: Mistborn (2006)
- #12: Assassin's Apprentice (1995)
- #13: Magician (1986)
- #14: Red Rising (2014)
- #15: Dune (1965)
- #16: The Lies of Locke Lamora (2006)

## Key Characteristics

### Entity Density Distribution
```
Classical: 0.0217 (moderate - more challenging)
Modern:    0.0445 (high - good training signal)
Overall:   0.0318 (balanced)

Best for training: Mistborn, Magician, Dune
Best for challenge: Moby Dick, Jane Eyre, Red Rising
```

### Narrative Styles
- **Dialogue-heavy:** 10 scenes (best for NER)
- **Mixed:** 4 scenes
- **Descriptive:** 2 scenes

### Character Types
- **Protagonists:** All records (1 per scene)
- **Supporting:** 2-18 per scene
- **Antagonists:** Some records
- **Minor:** Background characters

## How to Use

### Load the Dataset
```python
import json

with open('/mnt/storage/rune-lib/ood_validation_ground_truth_balanced.jsonl') as f:
    records = [json.loads(line) for line in f]

print(f"Loaded {len(records)} records")
```

### Evaluate a Model
```python
from sklearn.metrics import classification_report

# Get all records
all_records = [json.loads(line) for line in open(...)]

# Classical vs Modern comparison
classical = [r for r in all_records if r['metadata']['genre'] == 'classic']
modern = [r for r in all_records if r['metadata']['genre'] == 'modern']

# Compare performance
classical_acc = evaluate(model, classical)
modern_acc = evaluate(model, modern)
gap = abs(classical_acc - modern_acc)
```

### Filter by Criteria
```python
# High entity density scenes
high_density = [r for r in records if r['metadata'].get('entity_density', 0) > 0.04]

# Large character ensemble
large_cast = [r for r in records if len(r['characters']) >= 8]

# By era
before_1900 = [r for r in records if 'classical' in r['metadata'].get('genre', '')]
```

## Format Reference

**Each record includes:**
```json
{
  "story_id": "book_title_scene_number",
  "text": "full scene text (300-6287 words)",
  "characters": [
    {"name": "Character Name", "role": "PROTAGONIST|SUPPORTING|ANTAGONIST|MINOR"},
    ...
  ],
  "metadata": {
    "genre": "classic|modern",
    "source": "validation",
    "entity_density": 0.XXXX
  }
}
```

## Statistics Summary

| Aspect | Value |
|--------|-------|
| Total Records | 16 |
| Classical Records | 10 (62.5%) |
| Modern Records | 6 (37.5%) |
| Total Words | 38,801 |
| Min Scene Length | 361 words (Moby Dick) |
| Max Scene Length | 6,287 words (A Memory Called Empire) |
| Avg Scene Length | 2,425 words |
| Total Characters | 108 |
| Avg Characters/Scene | 6.75 |
| Min Characters | 3 (The Ships of Merior, Moby Dick) |
| Max Characters | 18 (The Ghost in the Doll) |
| Avg Entity Density | 0.0318 |
| Min Density | 0.0187 (Jane Eyre #2) |
| Max Density | 0.0505 (Magician) |
| Temporal Span | 1851-2014 |

## Quality Checks

✅ All records have valid JSONL format
✅ All records have unique story_ids
✅ All records have complete text
✅ All records have character lists
✅ All records have metadata
✅ Character roles are consistent
✅ Entity densities calculated
✅ No duplicate records
✅ UTF-8 encoding verified

## Next Steps

1. **Load dataset** into your NER evaluation framework
2. **Compare performance** on classical vs. modern text
3. **Identify weak areas** (which genres/styles cause errors?)
4. **Analyze entity density** impact on model accuracy
5. **Test generalization** across temporal boundaries

## Advanced Features

The balanced set enables:

### Domain Adaptation Testing
- Classical text (19th century narrative style)
- Modern text (contemporary prose)
- Genre shift (literary → sci-fi/fantasy)

### Entity Density Analysis
- Low-density scenes challenge co-reference resolution
- High-density scenes test mention frequency handling
- Mixed approach tests adaptive strategies

### Character Complexity Testing
- Small ensembles (3-4 chars) → isolation testing
- Large ensembles (10-18 chars) → disambiguation testing
- Named vs. referenced entities

---

**File:** `/mnt/storage/rune-lib/ood_validation_ground_truth_balanced.jsonl`
**Records:** 16
**Status:** Ready for evaluation
**Created:** 2026-01-30
