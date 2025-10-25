#!/usr/bin/env python3
"""
Combine multiple raw story datasets and intelligently reduce name duplication.

Strategy:
1. Combine all raw datasets
2. Reshape to expected format
3. Preprocess all stories (tokenize + BIO tagging)
4. Apply name frequency reduction (max N per name)
   - When choosing which stories to keep, PREFER LONGER STORIES
5. Pre-filter for token length
6. Shuffle and output final dataset

This script is reusable - just update the input_files list.
"""

import json
import random
import argparse
from collections import defaultdict
from transformers import AutoTokenizer
from rune.data.story_preprocessor import StoryPreprocessor


def reshape_story(raw_story):
    """Reshape story from various formats to expected format."""
    # Map roles to standard categories
    role_mapping = {
        'protagonist': 'PROTAGONIST',
        'antagonist': 'ANTAGONIST',
        'antagonist/rival': 'ANTAGONIST',
        'catalyst': 'SUPPORTING',
        'secondary_character': 'SUPPORTING',
        'supporting': 'SUPPORTING',
    }

    characters = []

    # Handle character_details format
    if 'character_details' in raw_story:
        for char_detail in raw_story.get('character_details', []):
            name = char_detail['name']
            raw_role = char_detail.get('role', 'supporting').lower()
            role = role_mapping.get(raw_role, 'SUPPORTING')
            characters.append({"name": name, "role": role})

    # Fallback: use characters list if it's already in correct format
    elif 'characters' in raw_story and isinstance(raw_story['characters'], list):
        chars = raw_story['characters']
        if chars and isinstance(chars[0], dict) and 'name' in chars[0]:
            characters = chars
        else:
            # Simple list of names, default to SUPPORTING
            characters = [{"name": name, "role": "SUPPORTING"} for name in chars]

    return {
        "text": raw_story['text'],
        "characters": characters,
        "metadata": raw_story.get('_meta', raw_story.get('metadata', {}))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process and combine all story datasets with intelligent name reduction"
    )
    parser.add_argument('--inputs', nargs='+', required=True,
                       help="Input JSONL files to combine")
    parser.add_argument('--output', default="ner_training_combined_final.jsonl",
                       help="Output JSONL file")
    parser.add_argument('--max-per-name', type=int, default=10,
                       help="Maximum stories per name")
    parser.add_argument('--max-tokens', type=int, default=8192,
                       help="Maximum token length")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for shuffling")
    args = parser.parse_args()

    random.seed(args.seed)

    print("🚀 PROCESSING ALL DATASETS WITH INTELLIGENT NAME REDUCTION")
    print("=" * 70)
    print()

    # Load tokenizer for token counting
    print("🔄 Loading ModernBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    print("✅ Tokenizer loaded")
    print()

    # Initialize preprocessor
    print("🔄 Initializing preprocessor...")
    preprocessor = StoryPreprocessor(use_spacy=False)
    print("✅ Preprocessor ready")
    print()

    # Phase 1: Load and combine all raw datasets
    print("📊 Phase 1: Loading and combining raw datasets...")
    print(f"   Input files ({len(args.inputs)}):")
    for input_file in args.inputs:
        print(f"      - {input_file}")
    print()

    all_raw_stories = []
    for input_file in args.inputs:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                file_stories = [json.loads(line.strip()) for line in f]
                print(f"   ✅ {input_file}: {len(file_stories):,} stories")
                all_raw_stories.extend(file_stories)
        except Exception as e:
            print(f"   ⚠️  Error loading {input_file}: {e}")
            continue

    print(f"\n   Total raw stories: {len(all_raw_stories):,}")
    print()

    # Phase 2: Reshape and preprocess all stories
    print("📊 Phase 2: Reshaping and preprocessing all stories...")

    processed_stories = []
    errors = 0

    for i, raw_story in enumerate(all_raw_stories):
        try:
            # Reshape to expected format
            reshaped = reshape_story(raw_story)

            # Preprocess to get tokens and BIO tags
            processed = preprocessor.process_story(reshaped)

            processed_stories.append(processed)

            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1:,} / {len(all_raw_stories):,} stories...")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"⚠️  Error processing story {i}: {e}")
            continue

    print(f"✅ Preprocessed {len(processed_stories):,} stories")
    if errors > 0:
        print(f"   Errors: {errors}")
    print()

    # Phase 3: Analyze name frequency and story lengths
    print("📊 Phase 3: Analyzing name frequency and story lengths...")

    # Build index: name -> list of (story_idx, story_length)
    name_to_stories = defaultdict(list)

    for idx, story in enumerate(processed_stories):
        story_length = len(story['tokens'])

        # Get unique names in this story
        names_in_story = set()
        for entity in story.get('entities', []):
            names_in_story.add(entity['text'])

        # Add to index
        for name in names_in_story:
            name_to_stories[name].append((idx, story_length))

    # Sort by frequency
    sorted_names = sorted(name_to_stories.items(),
                         key=lambda x: len(x[1]), reverse=True)

    print(f"   Total unique names: {len(name_to_stories):,}")
    print(f"   Top 10 most frequent names:")
    for name, story_list in sorted_names[:10]:
        avg_length = sum(length for _, length in story_list) / len(story_list)
        print(f"      '{name}': {len(story_list)} stories (avg {avg_length:.0f} tokens)")
    print()

    # Phase 4: Intelligent name frequency reduction
    print(f"📊 Phase 4: Reducing duplicates (max {args.max_per_name} per name)...")
    print(f"   Strategy: Prefer LONGER stories when culling duplicates")
    print()

    # For each name, sort its stories by length (descending) and keep top N
    stories_to_keep_set = set()

    for name, story_list in name_to_stories.items():
        # Sort by story length (descending) - keep longest stories
        sorted_by_length = sorted(story_list, key=lambda x: x[1], reverse=True)

        # Keep up to max_per_name stories
        for story_idx, story_length in sorted_by_length[:args.max_per_name]:
            stories_to_keep_set.add(story_idx)

    # Filter stories based on kept indices
    diversified_stories = [processed_stories[idx] for idx in sorted(stories_to_keep_set)]

    print(f"✅ Kept {len(diversified_stories):,} stories after name reduction")
    print(f"   Filtered out: {len(processed_stories) - len(diversified_stories):,} stories")
    print()

    # Phase 5: Pre-filter by token length
    print(f"📊 Phase 5: Pre-filtering for {args.max_tokens} token limit...")

    final_stories = []
    filtered_count = 0

    for story in diversified_stories:
        tokens = story["tokens"]

        # Check token length with actual tokenizer
        tokenized = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=False,
            add_special_tokens=True
        )
        token_length = len(tokenized["input_ids"])

        if token_length <= args.max_tokens:
            final_stories.append(story)
        else:
            filtered_count += 1

    print(f"✅ After token filtering: {len(final_stories):,} stories")
    print(f"   Filtered (>{args.max_tokens} tokens): {filtered_count:,} stories")
    print()

    # Phase 6: Shuffle and save
    print(f"🔀 Phase 6: Shuffling (seed={args.seed}) and saving...")
    random.shuffle(final_stories)

    with open(args.output, 'w', encoding='utf-8') as outfile:
        for story in final_stories:
            outfile.write(json.dumps(story) + '\n')

    print(f"✅ Saved {len(final_stories):,} stories to {args.output}")
    print()

    # Phase 7: Final statistics
    print("=" * 70)
    print("📊 FINAL STATISTICS")
    print("=" * 70)
    print(f"   Total input stories: {len(all_raw_stories):,}")
    print(f"   After preprocessing: {len(processed_stories):,}")
    print(f"   After name reduction (max {args.max_per_name}): {len(diversified_stories):,}")
    print(f"   After token filtering (<={args.max_tokens}): {len(final_stories):,}")
    print()

    # Final name diversity check
    final_name_to_stories = defaultdict(set)
    for story in final_stories:
        story_id = story.get('story_id', id(story))
        for entity in story.get('entities', []):
            final_name_to_stories[entity['text']].add(story_id)

    sorted_final = sorted(final_name_to_stories.items(),
                         key=lambda x: len(x[1]), reverse=True)
    max_freq_name, max_freq_stories = sorted_final[0]

    print(f"   Final unique names: {len(final_name_to_stories):,}")
    print(f"   Max name frequency: '{max_freq_name}' in {len(max_freq_stories)} stories")
    print(f"   Top 10 most frequent names:")
    for name, story_set in sorted_final[:10]:
        print(f"      '{name}': {len(story_set)} stories")
    print()

    print(f"📁 Output: {args.output}")
    print("🎉 Processing complete!")
    print()


if __name__ == "__main__":
    exit(main())
