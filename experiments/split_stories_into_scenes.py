#!/usr/bin/env python3
"""
Split story-level training data into scene-level training data.

Takes stories with ### Scene N: markers and creates individual scene samples,
filtering entities to only those that appear in each scene.
"""

import json
import argparse
import re
from typing import List, Dict, Any
from pathlib import Path
from rune.data.improved_tokenizer import ImprovedTokenizer


def find_entity_tokens(entity_text: str, tokens: List[str], start_search_idx: int = 0) -> List[int]:
    """
    Find all token positions where this entity appears.

    Args:
        entity_text: The entity text to find (e.g., "Chen Li")
        tokens: List of all tokens
        start_search_idx: Where to start searching in token list

    Returns:
        List of starting token indices where entity appears
    """
    entity_tokens = entity_text.lower().split()
    positions = []

    for i in range(start_search_idx, len(tokens) - len(entity_tokens) + 1):
        # Check if tokens match at this position
        match = True
        for j, ent_tok in enumerate(entity_tokens):
            if tokens[i + j].lower() != ent_tok:
                match = False
                break

        if match:
            positions.append(i)

    return positions


def split_story_into_scenes(story_data: Dict[str, Any], tokenizer: ImprovedTokenizer) -> List[Dict[str, Any]]:
    """
    Split a story into individual scenes based on ### Scene N: markers.

    Args:
        story_data: Story dict with text, tokens, bio_tags, entities
        tokenizer: Tokenizer for word splitting

    Returns:
        List of scene dicts, each with scene-specific entities and tags
    """
    text = story_data['text']
    story_tokens = story_data['tokens']
    story_bio_tags = story_data['bio_tags']
    story_entities = story_data.get('entities', [])

    # Split text by scene markers
    scene_pattern = r'###\s*Scene\s*\d+:[^\n]*\n+'
    scene_splits = re.split(scene_pattern, text)
    scene_markers = re.findall(scene_pattern, text)

    # Remove empty first section if exists
    if scene_splits and len(scene_splits[0].strip()) == 0:
        scene_splits = scene_splits[1:]

    if len(scene_splits) == 0:
        # No scene markers - treat entire story as one scene
        return [story_data]

    scenes = []
    token_offset = 0

    for scene_idx, scene_text in enumerate(scene_splits):
        # Tokenize this scene
        scene_tokens = tokenizer.split_words(scene_text)
        scene_token_count = len(scene_tokens)

        # Extract BIO tags for this scene's token range
        scene_bio_tags = story_bio_tags[token_offset:token_offset + scene_token_count]

        # Find which entities appear in this scene
        scene_entities = []
        seen_entity_texts = set()  # Deduplicate entities

        for entity in story_entities:
            entity_text = entity.get('text', '')
            entity_role = entity.get('role', 'supporting')

            # Skip if we've already added this entity
            entity_key = (entity_text.lower(), entity_role)
            if entity_key in seen_entity_texts:
                continue

            # Check if entity appears in this scene's tokens
            positions = find_entity_tokens(entity_text, scene_tokens)

            if positions:
                # Entity appears in this scene - add it
                scene_entities.append({
                    'text': entity_text,
                    'label': 'PERSON',
                    'role': entity_role,
                    'positions': positions  # Store all positions where it appears
                })
                seen_entity_texts.add(entity_key)

        # Create scene dict
        scene_dict = {
            'story_id': f"{story_data.get('story_id', 'unknown')}_scene{scene_idx + 1}",
            'parent_story_id': story_data.get('story_id', 'unknown'),
            'scene_number': scene_idx + 1,
            'genre': story_data.get('genre', 'unknown'),
            'text': scene_text,
            'tokens': scene_tokens,
            'bio_tags': scene_bio_tags,
            'entities': scene_entities
        }

        scenes.append(scene_dict)
        token_offset += scene_token_count

    return scenes


def main():
    parser = argparse.ArgumentParser(description="Split stories into scenes for curriculum learning")
    parser.add_argument('--input', required=True, help="Input story-level JSONL file")
    parser.add_argument('--output', required=True, help="Output scene-level JSONL file")
    parser.add_argument('--min-scene-tokens', type=int, default=100, help="Minimum tokens per scene (filter out very short scenes)")
    parser.add_argument('--max-scene-tokens', type=int, default=2000, help="Maximum tokens per scene (split long scenes)")

    args = parser.parse_args()

    tokenizer = ImprovedTokenizer()

    # Stream processing: read stories one at a time
    print(f"Streaming stories from {args.input}...")
    print("Splitting stories into scenes...")

    stories_processed = 0
    stories_with_scenes = 0
    total_scenes = 0
    filtered_scenes = 0
    scenes_written = 0

    # Open output file for streaming writes
    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        for line in fin:
            story = json.loads(line.strip())
            stories_processed += 1

            # Split this story into scenes
            scenes = split_story_into_scenes(story, tokenizer)

            if len(scenes) > 1:
                stories_with_scenes += 1

            # Filter scenes by token length and write immediately
            for scene in scenes:
                total_scenes += 1
                scene_len = len(scene['tokens'])

                if scene_len < args.min_scene_tokens:
                    filtered_scenes += 1
                    continue

                # If scene is too long, we still keep it (will be truncated during training)
                # but log it
                if scene_len > args.max_scene_tokens:
                    print(f"  Warning: Scene {scene['story_id']} has {scene_len} tokens (>{args.max_scene_tokens})")

                # Write scene immediately (streaming)
                fout.write(json.dumps(scene) + '\n')
                scenes_written += 1

            # Progress update
            if stories_processed % 1000 == 0:
                print(f"  Processed {stories_processed:,} stories, written {scenes_written:,} scenes...")

    print(f"\nScene splitting statistics:")
    print(f"  Stories processed: {stories_processed:,}")
    print(f"  Stories with scene markers: {stories_with_scenes:,} ({stories_with_scenes/stories_processed*100:.1f}%)")
    print(f"  Total scenes created: {total_scenes:,}")
    print(f"  Scenes written: {scenes_written:,}")
    print(f"  Filtered out (too short): {filtered_scenes:,}")
    print(f"\n✅ Done! Created {scenes_written:,} scene-level training samples")


if __name__ == '__main__':
    main()
