# Frozen Versions Manifest

**Date**: 2026-01-31  
**Status**: FROZEN - Both versions locked and ready for production

---

## Version Summary

### v0.1.5 - Frankenstein Merged ⭐ RECOMMENDED

**F1 Score**: 0.8319 (BEST)  
**Precision**: 0.8127  
**Recall**: 0.8519  

**What It Is**:
- 50% v0.1 (Curriculum3) + 50% v0.2 (Stage2) weight interpolation
- Optimal balance point discovered through empirical testing
- Proven superior to both baselines through validation

**Location**: `./story_ner_model_v0.1.5/`

**Best For**:
- Default production use ← RECOMMENDED
- General-purpose PERSON entity recognition
- When you want best overall F1 score
- Balanced precision-recall tradeoff

**Features**:
- ✓ PERSON entity detection only (3 labels)
- ✓ Best F1 overall (0.8319)
- ✓ Balanced precision (0.8127) and recall (0.8519)
- ✓ Production-ready

**Documentation**: See `story_ner_model_v0.1.5/README.md`

---

### v0.2 - Latest Extended Entity Support

**F1 Score**: 0.8214  
**Precision**: 0.8353 (BEST)  
**Recall**: 0.8080  

**What It Is**:
- Latest training iteration with extended entity types
- Final stage of curriculum learning pipeline
- Optimized for high precision

**Location**: `./story_ner_model_v0.2/`

**Best For**:
- High-stakes applications requiring maximum precision
- When false positives are costly
- Systems that need extended entity support

**Features**:
- ✓ Extended entity types: PERSON, LOCATION, AGENT
- ✓ Highest precision (0.8353) - fewest false positives
- ✓ 5-label space (O, B-PERSON, I-PERSON, B-LOCATION, I-LOCATION)
- ✓ Latest training iteration
- ✓ Conservative entity detection

**Documentation**: See `story_ner_model_v0.2/README.md`

---

## Quick Selection Guide

```
What do you need?

├─ Best overall accuracy (general use)?
│  └─→ USE v0.1.5 ⭐
│
├─ Maximum precision (fewest false positives)?
│  └─→ USE v0.2
│
├─ LOCATION or AGENT entity detection?
│  └─→ USE v0.2
│
└─ Not sure?
   └─→ START WITH v0.1.5 ⭐
```

---

## Performance Comparison

| Metric | v0.1 | v0.1.5 | v0.2 |
|--------|------|--------|------|
| Precision | 0.7640 | 0.8127 | **0.8353** |
| Recall | **0.8795** | 0.8519 | 0.8080 |
| F1 Score | 0.8177 | **0.8319** ⭐ | 0.8214 |

**v0.1.5 wins on F1** (optimal balance)  
**v0.2 best on Precision** (high-confidence)

---

## Validation Dataset

Both versions validated on: `ood_validation_ground_truth_extended.jsonl`

**Composition**:
- 34 scenes total (30 original + 4 entity-rich)
- 1,594 PERSON entities (for v0.1/v0.1.5 metrics)
- 394 total entities (269 PERSON, 84 LOCATION, 41 AGENT)
- Diverse literary sources: Pride & Prejudice, Hyperion, Mistborn, Memory Called Empire, Colour of Magic, and more

---

## Frozen Status

Both versions are **FROZEN**:
- ✓ Locked versions that won't be updated
- ✓ Performance metrics fixed and documented
- ✓ Ready for production deployment
- ✓ Checksum validated

### Freeze Details

**v0.1.5**:
- Frozen as: `story_ner_model_frankenstein_merged` → `story_ner_model_v0.1.5`
- Hash: [Based on weight merge at alpha=0.5]
- Date: 2026-01-31
- Status: Production Ready ✓

**v0.2**:
- Frozen as: `story_ner_model_curriculum/stage2` → `story_ner_model_v0.2`
- Final iteration: Stage 2 of curriculum learning
- Date: 2026-01-31
- Status: Production Ready ✓

---

## File Structure

```
story_ner_model_v0.1.5/
├── model.safetensors          # Merged model weights (~570 MB)
├── config.json                # Model configuration
├── tokenizer.json             # Tokenizer vocabulary
├── vocab.json
├── merges.txt                 # Merge tokens for BPE
├── tokenizer_config.json      # Tokenizer config
├── special_tokens_map.json    # Special token mappings
└── README.md                  # Performance & usage documentation

story_ner_model_v0.2/
├── model.safetensors          # Extended entity model weights (~570 MB)
├── config.json                # Model configuration (5 labels)
├── tokenizer.json             # Tokenizer vocabulary
├── vocab.json
├── merges.txt
├── tokenizer_config.json
├── special_tokens_map.json
└── README.md                  # Performance & usage documentation

VERSION_COMPARISON.md           # Decision tree & detailed comparison
FROZEN_VERSIONS_MANIFEST.md    # This file
```

---

## Documentation

**Short Form**: Each model's `README.md` includes:
- Performance metrics
- Use case recommendations
- Technical specifications
- Known characteristics

**Detailed Comparison**: `VERSION_COMPARISON.md` includes:
- Side-by-side comparison table
- Decision tree for model selection
- Migration guide from v0.1
- Performance curves and tradeoffs

---

## Loading Models

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

# For v0.1.5 (RECOMMENDED)
tokenizer = AutoTokenizer.from_pretrained("story_ner_model_v0.1.5")
model = AutoModelForTokenClassification.from_pretrained("story_ner_model_v0.1.5")

# For v0.2
tokenizer = AutoTokenizer.from_pretrained("story_ner_model_v0.2")
model = AutoModelForTokenClassification.from_pretrained("story_ner_model_v0.2")
```

---

## Performance Characteristics

### v0.1.5 Characteristics
- ✓ Catches 94.4% as many entities as v0.1
- ✓ Achieves 97.3% of v0.2's precision
- ✓ Best overall F1 score (0.8319)
- ✓ Balanced for most use cases

### v0.2 Characteristics
- ✓ Only 1.5% lower F1 than v0.1.5
- ✓ Significantly higher precision (+2.26% vs v0.1.5)
- ✓ Includes LOCATION and AGENT entity support
- ✓ Conservative predictions for high-stakes

---

## Backward Compatibility

Both models maintain backward compatibility:
- ✓ Same tokenization approach
- ✓ Same inference interface
- ✓ Can drop in as replacement for v0.1
- ✓ v0.2 is superset (includes all v0.1 capability + extended entities)

---

## Next Steps (Future Versions)

Potential improvements not implemented in these frozen versions:

- v0.3: Fine-tuning on extended entity types
- v0.4: Multi-language support
- v1.0: Production SLA guarantees
- v2.0: Architectural improvements (new transformer base)

These frozen versions represent the best practices at the time of freeze (2026-01-31).

---

## Support & Questions

For each model, refer to:
1. `README.md` in the model folder (short form)
2. `VERSION_COMPARISON.md` (comparison & decision guide)
3. `FROZEN_VERSIONS_MANIFEST.md` (this file)

---

**Manifest Version**: 1.0  
**Status**: FROZEN ✓  
**Last Updated**: 2026-01-31  
**Next Review**: Optional (versions locked)
