# Rune - Really Uncomplicated Named Entity Library

A modern, minimalistic library for bio-tagging with BERT and Hugging Face transformers. Designed to be like SparkNLP, Flair, or AllenNLP but with a focus on simplicity, modern practices, and client-first design.

## ✅ Project Status: Cast Detection Validated

**Latest Update**: Successfully trained and validated story NER model for cast detection (PROTAGONIST, ANTAGONIST, SUPPORTING characters).

### Key Results
- **Entity Detection F1**: 94.6% - Excellent character boundary detection
- **Fine-grained F1**: 70.6% - Good role classification
- **Training Data**: 8,255 synthetic stories (31.4M tokens)
- **Framework**: Custom Longformer + rune-lib NER pipeline

## Features

- **🎭 Story NER**: Specialized for character cast detection in narratives
- **🎯 Bio-tagging focused**: Built specifically for IOB2 format named entity recognition
- **🤗 HuggingFace first**: Seamless integration with transformers and tokenizers
- **⚡ Modern & Fast**: Python 3.10+ with type hints and efficient processing
- **🔧 Client-injectable**: Easily customizable tag schemas for any use case
- **📊 Comprehensive metrics**: Built-in evaluation with seqeval
- **🎛️ Training utilities**: End-to-end training pipeline included

## Quick Start

### Story Cast Detection

```python
from rune import BioTagger
import json

# Load trained cast detection model
with open("./story_ner_model_stable/tag_schema.json") as f:
    tag_schema = json.load(f)

tagger = BioTagger("./story_ner_model_stable", tag_schema)

# Detect characters in a story
story = "Sarah walked into the room where Marcus was waiting. The villain Dr. Evil lurked in the shadows."
entities = tagger.predict(story)

for entity in entities:
    print(f"{entity['text']} → {entity['label']}")
# Output:
# Sarah → PROTAGONIST
# Marcus → SUPPORTING
# Dr. Evil → ANTAGONIST
```

### Training Your Own Model

```bash
# Improved training script with proper evaluation
python train_story_ner_improved.py \
    --data_path ./processed_story_training_data.jsonl \
    --num_epochs 3 \
    --eval_steps 500 \
    --early_stopping_patience 3 \
    --use_wandb
```

## Project Structure

```
rune-lib/
├── rune/                          # Main library
│   ├── core/                      # Core NER functionality
│   ├── models/                    # Model implementations (Longformer, BERT)
│   └── data/                      # Dataset utilities
├── train_story_ner_improved.py    # Main training script (✅ recommended)
├── create_training_data.py        # Generate synthetic story data
├── test_story_inference.py        # Test model inference
├── processed_story_training_data.jsonl  # Training data (625MB)
├── story_ner_model_stable/        # Trained model
├── archive/                       # Old experimental scripts
└── TRAINING_ISSUES_ANALYSIS.md    # Training analysis & fixes
```

## Training Pipeline

```bash
# 1. Combine raw datasets
cat dataset1.jsonl dataset2.jsonl > combined_raw.jsonl

# 2. Reshape: characters: [{name, role}]
python3 reshape_synthetic_data.py combined_raw.jsonl combined_reshaped.jsonl

# 3. Diversify: max N stories per character name
python3 diversify_by_name_frequency.py --input combined_reshaped.jsonl --output combined_diversified.jsonl --max-stories-per-name 10

# 4. Preprocess & filter: tokenize + BIO tag + length filter
python3 prefilter_by_length_8k.py --input combined_diversified.jsonl --output final_training_data.jsonl

# 5. Train: ModernBERT streaming (8192 tokens)
python3 train_modernbert_streaming.py --input final_training_data.jsonl --output ./model_output
```

## Training Data

Successfully generated and validated high-quality training data:
- **8,255 synthetic stories** across genres (romance, thriller, fantasy, sci-fi, etc.)
- **31.4M tokens** with character-level annotations
- **Balanced cast roles**: 476K supporting, 197K protagonists, 28K antagonists
- **Clean BIO tagging**: 97.7% O-tags, 2.3% entity tags

## Model Performance

### Strengths ✅
- Excellent entity boundary detection (94.6% F1)
- Works well on synthetic story styles
- Robust framework and training pipeline
- Proper confidence scoring and evaluation

### Areas for Improvement ⚠️
- Domain adaptation to classic literature styles
- Role classification accuracy (70.6% F1)
- Confidence calibration (currently low ~25%)

## Training Issues Fixed

🔧 **Identified and resolved key training issues**:
1. **Overfitting**: Reduced epochs from 10 → 3
2. **Missing evaluation**: Added step-by-step monitoring
3. **Misleading accuracy**: 99.7% due to class imbalance, actual improvement only 2%
4. **No early stopping**: Added patience-based stopping

See `TRAINING_ISSUES_ANALYSIS.md` for detailed analysis.

## Installation

```bash
# Clone repository
git clone <repository-url>
cd rune-lib

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Contributing

1. Check `archive/` for experimental scripts and ideas
2. Use `train_story_ner_improved.py` for new training runs
3. Follow the tag schema format in `story_ner_model_stable/tag_schema.json`
4. Add evaluation metrics for new entity types

## License

MIT License

---

**🎭 Ready for cast detection in your stories!** The framework is validated and working - just needs domain-specific fine-tuning for your use case.