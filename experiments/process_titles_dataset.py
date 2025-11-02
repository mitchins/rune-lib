#!/usr/bin/env python3
"""
Process the titles/unusual names dataset:
1. Reshape to match expected format
2. Preprocess with StoryPreprocessor
3. Apply name frequency reduction (max 10 per name)
4. Pre-filter for 8k token limit
5. Output ready-to-train dataset
"""

import json
import argparse
from collections import defaultdict
from transformers import AutoTokenizer
from rune.data.story_preprocessor import StoryPreprocessor


def reshape_story(raw_story):
    """Reshape story from old format to expected format."""
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
    for char_detail in raw_story.get('character_details', []):
        name = char_detail['name']
        raw_role = char_detail.get('role', 'supporting').lower()

        # Map to standard role
        role = role_mapping.get(raw_role, 'SUPPORTING')

        characters.append({
            "name": name,
            "role": role
        })

    return {
        "text": raw_story['text'],
        "characters": characters,
        "metadata": raw_story.get('_meta', {})
    }


def main():
    parser = argparse.ArgumentParser(description="Process titles/unusual names dataset")
    parser.add_argument('--input',
                       default="ner_training_titles_training_unusual_names_20251024_133216.jsonl",
                       help="Input JSONL file")
    parser.add_argument('--output',
                       default="ner_training_titles_processed.jsonl",
                       help="Output JSONL file")
    parser.add_argument('--max-per-name', type=int, default=10,
                       help="Maximum stories per name")
    parser.add_argument('--max-tokens', type=int, default=8192,
                       help="Maximum token length")
    args = parser.parse_args()

    print("🔄 PROCESSING TITLES/UNUSUAL NAMES DATASET")
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

    # Phase 1: Reshape and preprocess all stories
    print("📊 Phase 1: Reshaping and preprocessing stories...")
    print(f"   Input: {args.input}")
    print()

    temp_processed = []
    reshape_errors = 0
    preprocess_errors = 0

    with open(args.input, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            try:
                raw_story = json.loads(line.strip())

                # Reshape to expected format
                reshaped = reshape_story(raw_story)

                # Preprocess to get tokens and BIO tags
                processed = preprocessor.process_story(reshaped)

                temp_processed.append(processed)

                if (i + 1) % 500 == 0:
                    print(f"   Processed {i + 1:,} stories...")

            except Exception as e:
                if 'reshape' in str(type(e).__name__).lower():
                    reshape_errors += 1
                else:
                    preprocess_errors += 1
                if (reshape_errors + preprocess_errors) <= 5:
                    print(f"⚠️  Error processing story {i}: {e}")
                continue

    print(f"✅ Reshaped and preprocessed {len(temp_processed):,} stories")
    if reshape_errors > 0:
        print(f"   Reshape errors: {reshape_errors}")
    if preprocess_errors > 0:
        print(f"   Preprocessing errors: {preprocess_errors}")
    print()

    # Phase 2: Analyze name frequency
    print("📊 Phase 2: Analyzing name frequency...")
    name_frequency = defaultdict(list)

    for idx, story in enumerate(temp_processed):
        for entity in story.get('entities', []):
            name = entity['text']
            name_frequency[name].append(idx)

    # Sort by frequency
    sorted_names = sorted(name_frequency.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"   Total unique names: {len(name_frequency):,}")
    print(f"   Top 10 most frequent names:")
    for name, story_indices in sorted_names[:10]:
        print(f"      '{name}': {len(story_indices)} stories")
    print()

    # Phase 3: Apply name frequency reduction
    print(f"📊 Phase 3: Reducing duplicates (max {args.max_per_name} per name)...")

    # Track which stories to keep
    stories_to_keep_set = set()
    name_counts = defaultdict(int)

    # Greedy algorithm: iterate through stories, keep if name counts allow
    for idx, story in enumerate(temp_processed):
        entity_names = [entity['text'] for entity in story.get('entities', [])]

        # Check if we can keep this story
        can_keep = True
        for name in entity_names:
            if name_counts[name] >= args.max_per_name:
                can_keep = False
                break

        if can_keep:
            stories_to_keep_set.add(idx)
            for name in entity_names:
                name_counts[name] += 1

    # Filter stories
    diversified_stories = [temp_processed[idx] for idx in sorted(stories_to_keep_set)]

    print(f"✅ Kept {len(diversified_stories):,} stories after name reduction")
    print(f"   Filtered out: {len(temp_processed) - len(diversified_stories):,} stories")
    print()

    # Phase 4: Pre-filter by token length
    print(f"📊 Phase 4: Pre-filtering for {args.max_tokens} token limit...")

    final_stories = []
    filtered_count = 0

    for story in diversified_stories:
        tokens = story["tokens"]

        # Check token length
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

    print(f"✅ Final dataset: {len(final_stories):,} stories")
    print(f"   Filtered (>{args.max_tokens} tokens): {filtered_count:,} stories")
    print()

    # Phase 5: Save processed dataset
    print(f"💾 Phase 5: Saving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as outfile:
        for story in final_stories:
            outfile.write(json.dumps(story) + '\n')

    print(f"✅ Saved {len(final_stories):,} processed stories")
    print()

    # Summary statistics
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"   Input stories: {i + 1:,}")
    print(f"   After reshape/preprocess: {len(temp_processed):,}")
    print(f"   After name reduction (max {args.max_per_name}): {len(diversified_stories):,}")
    print(f"   After token filtering (<={args.max_tokens}): {len(final_stories):,}")
    print(f"   Final dataset: {args.output}")
    print()

    # Name diversity check
    final_name_freq = defaultdict(int)
    for story in final_stories:
        for entity in story.get('entities', []):
            final_name_freq[entity['text']] += 1

    max_freq_name = max(final_name_freq.items(), key=lambda x: x[1])
    print(f"   Max name frequency: '{max_freq_name[0]}' appears {max_freq_name[1]} times")
    print(f"   Total unique names: {len(final_name_freq):,}")
    print()
    print("🎉 Processing complete!")


if __name__ == "__main__":
    exit(main())
