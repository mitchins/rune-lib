# Rune - Named Entity Recognition Library

A minimalistic Python library for BIO-tagging NER with BERT-based transformers.

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
tagger = BioTagger("./ner_model_3epochs")

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
│   └── tagger.py              # Entity extraction
├── data/
│   ├── story_preprocessor.py  # Data preprocessing with contextual labeling
│   └── improved_tokenizer.py  # Word tokenization for narrative text
└── utils/
    └── entity_normalization.py # Entity normalization utilities
```

## Training

For training models, see:
- `START_TRAINING.sh` - Canonical training command
- `METHODOLOGY.md` - Training methodology and approach

## Testing

```bash
pytest tests/
```

## License

MIT License
