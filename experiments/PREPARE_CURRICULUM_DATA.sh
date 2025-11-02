#!/bin/bash
# Complete data preparation pipeline for curriculum training
# Takes raw comprehensive dataset and produces curriculum-ready scene and story data

set -e  # Exit on error

INPUT_RAW="$1"
if [ -z "$INPUT_RAW" ]; then
    echo "Usage: $0 <raw_input_file.jsonl>"
    echo "Example: $0 ner_training_clean_training_mixed_training_titles_20251031_142457.jsonl"
    exit 1
fi

echo "="
echo "= CURRICULUM DATA PREPARATION PIPELINE"
echo "="
echo ""
echo "Input: $INPUT_RAW"
echo ""

# Step 1: Diversify by name frequency (limit 15x per name)
echo "[1/4] Diversifying by name frequency (15x limit)..."
python3 diversify_by_name_frequency.py \
    --input "$INPUT_RAW" \
    --output ner_training_comprehensive_diversified_15.jsonl \
    --max-stories-per-name 15

# Step 2: Preprocess (add tokens and bio_tags)
echo ""
echo "[2/4] Preprocessing stories (adding tokens and BIO tags)..."
python3 preprocess_raw_stories.py \
    --input ner_training_comprehensive_diversified_15.jsonl \
    --output ner_training_comprehensive_processed.jsonl

# Step 3: Filter by 8k token length
echo ""
echo "[3/4] Filtering by token length (<=8000)..."
python3 prefilter_by_length_8k.py \
    --input ner_training_comprehensive_processed.jsonl \
    --output ner_training_ready.jsonl \
    --max-tokens 8000

# Step 4: Split into scenes
echo ""
echo "[4/4] Splitting stories into scenes..."
python3 split_stories_into_scenes.py \
    --input ner_training_ready.jsonl \
    --output ner_training_scenes.jsonl \
    --min-scene-tokens 100 \
    --max-scene-tokens 2000

echo ""
echo "="
echo "= PIPELINE COMPLETE!"
echo "="
echo ""
echo "Output files:"
echo "  Story-level: ner_training_ready.jsonl"
echo "  Scene-level: ner_training_scenes.jsonl"
echo ""
echo "Ready for curriculum training with:"
echo "  ./START_CURRICULUM_TRAINING.sh"
