#!/usr/bin/env python3
"""
Diversify training data by limiting name repetition.
Select maximum stories where no name appears in more than N different stories.
"""

import json
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import argparse


def analyze_name_distribution(jsonl_path: str) -> Tuple[Dict[str, Set[int]], int]:
    """
    Analyze which names appear in which stories.

    Returns:
        name_to_story_ids: Mapping from name to set of story indices
        total_stories: Total number of stories
    """
    name_to_story_ids = defaultdict(set)
    total_stories = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for story_idx, line in enumerate(f):
            story = json.loads(line.strip())
            total_stories += 1

            # Extract all character names
            for char in story.get('characters', []):
                name = char['name']
                name_to_story_ids[name].add(story_idx)

    return dict(name_to_story_ids), total_stories


def select_diverse_stories(
    name_to_story_ids: Dict[str, Set[int]],
    max_stories_per_name: int,
    total_stories: int
) -> Set[int]:
    """
    Greedy selection of stories to maximize diversity.

    Strategy: Process stories in order, keeping track of name counts.
    Accept a story only if all its names are under the limit.
    """
    # Build reverse mapping: story_id -> set of names
    story_to_names = defaultdict(set)
    for name, story_ids in name_to_story_ids.items():
        for story_id in story_ids:
            story_to_names[story_id].add(name)

    # Track how many stories each name appears in (in our selection)
    name_counts = defaultdict(int)
    selected_stories = set()

    # Process stories in order (maintains some sequential diversity)
    for story_id in range(total_stories):
        if story_id not in story_to_names:
            # Story has no characters? Skip
            continue

        names_in_story = story_to_names[story_id]

        # Check if all names are under the limit
        can_include = all(
            name_counts[name] < max_stories_per_name
            for name in names_in_story
        )

        if can_include:
            # Include this story
            selected_stories.add(story_id)
            for name in names_in_story:
                name_counts[name] += 1

    return selected_stories


def write_filtered_dataset(
    input_path: str,
    output_path: str,
    selected_story_ids: Set[int]
):
    """Write filtered dataset containing only selected stories."""
    written = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for story_idx, line in enumerate(infile):
            if story_idx in selected_story_ids:
                outfile.write(line)
                written += 1

                if written % 1000 == 0:
                    print(f"   Written {written:,} stories...")

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Diversify training data by limiting name repetition"
    )
    parser.add_argument(
        '--input',
        default='ner_training_filtered_8k.jsonl',
        help='Input JSONL file'
    )
    parser.add_argument(
        '--output',
        default='ner_training_diversified.jsonl',
        help='Output JSONL file'
    )
    parser.add_argument(
        '--max-stories-per-name',
        type=int,
        default=5,
        help='Maximum number of stories each name can appear in (default: 5)'
    )

    args = parser.parse_args()

    print("🔍 DIVERSIFYING TRAINING DATA BY NAME FREQUENCY")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max stories per name: {args.max_stories_per_name}")
    print("=" * 70)
    print()

    # Step 1: Analyze name distribution
    print("📊 Step 1: Analyzing name distribution...")
    name_to_story_ids, total_stories = analyze_name_distribution(args.input)

    print(f"   Total stories: {total_stories:,}")
    print(f"   Unique names: {len(name_to_story_ids):,}")
    print()

    # Show distribution of name frequencies
    name_freq_distribution = defaultdict(int)
    for name, story_ids in name_to_story_ids.items():
        freq = len(story_ids)
        name_freq_distribution[freq] += 1

    print("📈 Name frequency distribution:")
    print(f"   Names appearing in 1 story: {name_freq_distribution[1]:,}")
    print(f"   Names appearing in 2-5 stories: {sum(name_freq_distribution[i] for i in range(2, 6)):,}")
    print(f"   Names appearing in 6-10 stories: {sum(name_freq_distribution[i] for i in range(6, 11)):,}")
    print(f"   Names appearing in 11-20 stories: {sum(name_freq_distribution[i] for i in range(11, 21)):,}")
    print(f"   Names appearing in 21+ stories: {sum(name_freq_distribution[i] for i in range(21, 1000)):,}")

    # Find most common names
    most_common = sorted(
        name_to_story_ids.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]

    print()
    print("🔝 Top 10 most frequent names:")
    for name, story_ids in most_common:
        print(f"   '{name}': {len(story_ids)} stories")
    print()

    # Step 2: Select diverse subset
    print(f"🎯 Step 2: Selecting stories (max {args.max_stories_per_name} per name)...")
    selected_story_ids = select_diverse_stories(
        name_to_story_ids,
        args.max_stories_per_name,
        total_stories
    )

    kept_stories = len(selected_story_ids)
    filtered_stories = total_stories - kept_stories
    retention_rate = (kept_stories / total_stories) * 100

    print(f"   Stories kept: {kept_stories:,} ({retention_rate:.1f}%)")
    print(f"   Stories filtered: {filtered_stories:,} ({100-retention_rate:.1f}%)")
    print()

    # Step 3: Write filtered dataset
    print(f"💾 Step 3: Writing filtered dataset to {args.output}...")
    written = write_filtered_dataset(
        args.input,
        args.output,
        selected_story_ids
    )

    print(f"   ✅ Written {written:,} stories")
    print()

    # Summary
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Original dataset: {total_stories:,} stories")
    print(f"Diversified dataset: {kept_stories:,} stories ({retention_rate:.1f}%)")
    print(f"Unique names: {len(name_to_story_ids):,}")
    print(f"Max appearances per name: {args.max_stories_per_name}")
    print()
    print(f"💡 Impact:")
    print(f"   - Model will see {kept_stories:,} diverse training examples")
    print(f"   - Each name appears in at most {args.max_stories_per_name} stories")
    print(f"   - Forces learning from context rather than memorizing names")
    print(f"   - Should improve generalization to unseen names")
    print()

    # Show what would happen with different thresholds
    print("🔄 Alternative thresholds:")
    for threshold in [3, 4, 5, 6, 7, 8, 10]:
        alt_selected = select_diverse_stories(
            name_to_story_ids,
            threshold,
            total_stories
        )
        alt_kept = len(alt_selected)
        alt_rate = (alt_kept / total_stories) * 100
        marker = " ← current" if threshold == args.max_stories_per_name else ""
        print(f"   Max {threshold:2} per name: {alt_kept:,} stories ({alt_rate:.1f}%){marker}")

    print()
    print(f"✅ Diversified dataset ready: {args.output}")


if __name__ == "__main__":
    main()
