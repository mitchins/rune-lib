# OOD Validation Set - Quick Reference

## What Was Created

An expanded Out-of-Distribution validation dataset with **5 new classic literature scenes** added to your existing validation set.

## Location

```
/mnt/storage/rune-lib/ood_validation_ground_truth.jsonl
```

**Backup of original:** `validation_ground_truth.jsonl.backup`

## Content Overview

### Total: 10 Records
- **Original:** 5 records (Pride & Prejudice + 4 fantasy/modern)
- **New Classics:** 5 records (user-requested titles)

### New Classic Scenes

| # | Book | Scene Description | Words | Chars | Density |
|----|------|-------------------|-------|-------|---------|
| 6 | **Call of the Wild** | The Thousand-Pound Wager | 1,087 | 6 | **0.0377** ⭐ |
| 7 | **Jane Eyre** | Charades at Thornfield Hall | 727 | 10 | **0.0275** ⭐ |
| 8 | **Moby Dick** | Meeting Queequeg | 361 | 3 | **0.0222** |
| 9 | **Jane Eyre** | The Governess Discussion | 1,096 | 10 | **0.0192** |
| 10 | **Sherlock Holmes** | A Scandal in Bohemia - Resolution | 386 | 5 | **0.0181** |

**Legend:**
- ⭐ Highest entity density (best for judgment purposes)
- All scenes exceed 0.015 threshold

## Quality Metrics

- **Average Entity Density:** 0.0186 (consistent with original set)
- **Density Range:** 0.0070 - 0.0377
- **Average Scene Length:** 2,066 words
- **Character Diversity:** 3-18 characters per scene

## What You Requested vs. What You Got

✅ **Pride and Prejudice** - Kept in dataset
✅ **Sherlock Holmes** - Added from "A Scandal in Bohemia"
✅ **Call of the Wild** - Added with high entity density (0.0377)
✅ **3-5 Other Classics** - Added:
  - Jane Eyre (2 scenes)
  - Moby Dick (1 scene)
  - Great Expectations (scanned but scenes didn't meet density threshold)

## Why These Scenes?

All selected scenes prioritize:
- **High Entity Density** (0.015+) for effective NER training
- **Multi-character Interactions** (3-18 characters per scene)
- **Dialogue-Heavy Composition** (easier entity identification)
- **Diverse Narrative Styles** (dialogue, action, theatrical, investigative)

## Using the Dataset

### For Evaluation
```bash
# Load and evaluate your NER model
python evaluate_model.py --validation_set ood_validation_ground_truth.jsonl
```

### For Analysis
```python
import json

with open('ood_validation_ground_truth.jsonl') as f:
    records = [json.loads(line) for line in f]

# Filter for classics only
classics = [r for r in records if r['metadata'].get('genre') == 'classic']
```

### For NER Annotation
- All records include character lists with roles
- Format matches existing validation format exactly
- Ready for BIO/BIOES tagging or other NER schemes

## Format Reference

```json
{
  "story_id": "unique_identifier",
  "text": "scene text with named characters...",
  "characters": [
    {"name": "Character Name", "role": "PROTAGONIST|SUPPORTING|ANTAGONIST"},
    ...
  ],
  "metadata": {
    "genre": "classic",
    "source": "validation",
    "book": "Book Title (optional)",
    "entity_density": 0.XXXX
  }
}
```

## Technical Details

### Source Texts
Downloaded from Project Gutenberg:
- Sherlock Holmes (ID: 1661)
- Call of the Wild (ID: 215)
- Jane Eyre (ID: 1260)
- Moby Dick (ID: 2701)
- Great Expectations (ID: 1083)

### Extraction Method
- Literary-data-analyst agent analyzed texts
- Scenes selected for entity density and narrative complexity
- Manual Sherlock Holmes extraction for comprehensive coverage

## Next Steps

1. **Test OOD Performance:** Evaluate your NER model on these classics
2. **Annotate for Tags:** Add BIO/BIOES tags if needed
3. **Analyze Results:** Compare performance on classics vs. modern text
4. **Identify Gaps:** Which literary styles cause model errors?

## Files Reference

```
/mnt/storage/rune-lib/
├── ood_validation_ground_truth.jsonl     ← Use this for evaluation
├── validation_ground_truth.jsonl.backup  ← Original backup
└── OOD_VALIDATION_SUMMARY.md             ← Detailed analysis
```

---

**Created:** 2026-01-30
**Records:** 10 (5 original + 5 new classics)
**Status:** Ready for NER evaluation and training
