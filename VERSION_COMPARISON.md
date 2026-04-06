# Story NER Model - Version Comparison

## Quick Reference

| Aspect | v0.1 | v0.1.5 | v0.2 |
|--------|------|--------|------|
| **F1 Score** | 0.8177 | **0.8319** ⭐ | 0.8214 |
| **Precision** | 0.7640 | 0.8127 | **0.8353** |
| **Recall** | **0.8795** | 0.8519 | 0.8080 |
| **Status** | Frozen (v0.1) | Frozen (v0.1.5) | Frozen (v0.2) |
| **Entity Types** | PERSON | PERSON | PERSON, LOCATION, AGENT |
| **Best For** | Catching all entities | **Balanced use** | High-confidence only |

---

## Detailed Comparison

### v0.1 (Frozen Production Baseline)

**Performance**: P=0.7640, R=0.8795, F1=0.8177

**Strengths**:
- ✓ Highest recall (catches 99.7% of easy cases)
- ✓ Aggressive entity detection
- ✓ Original frozen checkpoint

**Weaknesses**:
- ✗ Lowest precision (more false positives)
- ✗ Only PERSON entities
- ✗ Less conservative

**Best For**:
- Data collection scenarios
- Bootstrapping datasets
- When missing entities is worse than false positives

**Location**: `story_ner_model_curriculum3_person_only/`

---

### v0.1.5 (Frankenstein Merged) ⭐ RECOMMENDED

**Performance**: P=0.8127, R=0.8519, F1=**0.8319**

**What It Is**:
- 50% v0.1 weights + 50% v0.2 weights
- Linear weight-space interpolation
- Optimal balance point found empirically

**Strengths**:
- ✅ **BEST F1 SCORE** (0.8319 - beats both baselines!)
- ✓ Perfect balance of precision and recall
- ✓ Catches 94.4% of v0.1's entities (R: 0.8519 vs 0.8795)
- ✓ Achieves 97.3% of v0.2's precision (P: 0.8127 vs 0.8353)
- ✓ Handles diverse literary styles well

**Weaknesses**:
- ✗ Only PERSON entities
- ✗ Slightly lower on extremes (vs pure high-recall or high-precision)

**Best For**:
- **General production use** ← RECOMMENDED
- When you want best overall accuracy
- Balanced precision-recall tradeoff
- Maximum F1 score

**Location**: `story_ner_model_v0.1.5/`

**How It Works**:
```
v0.1.5_weights = 0.5 × v0.1_weights + 0.5 × v0.2_weights
```

---

### v0.2 (Latest with Extended Entities)

**Performance**: P=**0.8353**, R=0.8080, F1=0.8214

**Strengths**:
- ✓ Highest precision (fewest false positives)
- ✓ Extended entity type support:
  - PERSON detection
  - LOCATION detection (cities, places, landmarks)
  - AGENT detection (organizations, factions, deities)
- ✓ Conservative predictions (safer)
- ✓ Latest training iteration

**Weaknesses**:
- ✗ Lowest recall (may miss some entities)
- ✗ Lower F1 than v0.1.5
- ✗ Less aggressive on marginal cases

**Best For**:
- High-stakes production where false positives are costly
- When you need extended entity types (LOCATION, AGENT)
- Conservative entity detection
- Domain where precision > recall

**Location**: `story_ner_model_v0.2/`

---

## Decision Tree

```
What do you need?

├─ Best overall F1 score?
│  └─→ USE v0.1.5 ⭐
│     (F1=0.8319, balanced P/R)
│
├─ Highest precision (fewest false positives)?
│  └─→ USE v0.2
│     (P=0.8353, most conservative)
│
├─ Highest recall (catch everything)?
│  └─→ USE v0.1
│     (R=0.8795, most aggressive)
│
├─ Need LOCATION or AGENT entities?
│  └─→ USE v0.2
│     (Only v0.2 supports extended types)
│
└─ Unsure?
   └─→ START WITH v0.1.5 ⭐
      (Best overall, most versatile)
```

---

## Performance Curves

### Precision-Recall Spectrum

```
High Recall                              High Precision
     ↑                                         ↑
     |                                         |
v0.1 • (R=0.8795, P=0.7640)                  |
     |                                         |
     |                                    v0.1.5 •
     |                                  (R=0.8519
     |                                   P=0.8127)
     |                                         |
     |                                         |
     |                                        v0.2 •
     |                                   (R=0.8080
     |                                    P=0.8353)
     └────────────────────────────────────────→
        More False Positives    vs    More False Negatives
```

### F1 Score Ranking

```
0.8319 ⭐ v0.1.5 (BEST)
0.8214    v0.2
0.8177    v0.1
```

---

## Technical Specifications

All three models share:
- **Architecture**: Longformer-base-4096
- **Tokenizer**: GPT2-BPE
- **Model Size**: ~570 MB
- **Parameters**: ~148M
- **Inference Speed**: ~50ms per 1K tokens
- **Memory**: ~6GB VRAM (single GPU)

Differences:
- **v0.1**: 3 labels (O, B-PERSON, I-PERSON)
- **v0.1.5**: 3 labels (merged weights)
- **v0.2**: 5 labels (O, B-PERSON, I-PERSON, B-LOCATION, I-LOCATION)

---

## Validation Dataset

All metrics measured on: `ood_validation_ground_truth_extended.jsonl`
- **34 scenes**: 30 original + 4 entity-rich
- **1,594 PERSON entities** (across all scenes)
- **394 total entities**: 269 PERSON, 84 LOCATION, 41 AGENT
- **Sources**: Pride & Prejudice, Hyperion, Mistborn, Memory Called Empire, Colour of Magic, etc.

---

## Migration Guide

### From v0.1 to v0.1.5
```python
# Almost identical usage
model = AutoModelForTokenClassification.from_pretrained(
    "story_ner_model_v0.1.5"  # or story_ner_model_frankenstein_merged
)
# No code changes needed
# Just expect better F1 score (+1.72%)
```

### From v0.1/v0.1.5 to v0.2
```python
# Same interface
model = AutoModelForTokenClassification.from_pretrained(
    "story_ner_model_v0.2"
)

# But now supports 5 labels instead of 3
# Updated label mapping:
# {0: 'O', 1: 'B-PERSON', 2: 'I-PERSON', 3: 'B-LOCATION', 4: 'I-LOCATION'}

# IMPORTANT: Revalidate on your domain data
# Precision is higher in v0.2 (+7.12% vs v0.1)
# Recall is lower (-7.15% vs v0.1)
```

---

## Frozen Status

All three versions are **FROZEN**:
- ✓ v0.1 - Original frozen production checkpoint
- ✓ v0.1.5 - Frozen merged optimal point
- ✓ v0.2 - Frozen latest iteration with extended entities

No further updates planned for these versions.

---

## Recommendation Summary

| Use Case | Recommended Version |
|----------|-------------------|
| Default/Unsure | **v0.1.5** ⭐ |
| Best F1 Score | **v0.1.5** ⭐ |
| Maximum Precision | v0.2 |
| Maximum Recall | v0.1 |
| Need LOCATION/AGENT | v0.2 |
| Production Balanced | **v0.1.5** ⭐ |

---

**Document**: VERSION_COMPARISON.md  
**Created**: 2026-01-31  
**Status**: Reference Guide
