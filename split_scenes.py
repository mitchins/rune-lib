#!/usr/bin/env python3
"""
Split stories at scene boundaries for curriculum learning.

CRITICAL: Scene splitting must NOT corrupt entity boundaries!

Strategy:
1. Find ### Scene N: markers in raw text
2. Split text at those boundaries
3. For EACH scene, run preprocessor to get clean tokens + BIO tags
4. This ensures entity boundaries are correct per scene

Input: Processed JSONL with full stories
Output: Scene-split JSONL with more training examples
"""

import json
import argparse
import re
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from rune.data.story_preprocessor import StoryPreprocessor


def split_story_at_scenes(
    story_data: Dict[str, Any],
    preprocessor: StoryPreprocessor,
    min_tokens: int = 100,
    max_tokens: int = 4096
) -> List[Dict[str, Any]]:
    """
    Split a story into scenes.
    
    FAST PATH: If tokens + bio_tags already exist, split them directly (no reprocessing).
    FALLBACK: If raw text only, run preprocessor (slow but correct).
    
    Args:
        story_data: Original story with text + entities (may have tokens/bio_tags)
        preprocessor: StoryPreprocessor instance (only used if tokens missing)
        min_tokens: Minimum tokens per scene (skip shorter)
        max_tokens: Maximum tokens per scene (for filtering)
    
    Returns:
        List of scene dicts, each properly tagged
    """
    # FAST PATH: If already preprocessed (tokens + bio_tags exist), split directly
    if 'tokens' in story_data and 'bio_tags' in story_data:
        return split_preprocessed_story(story_data, min_tokens, max_tokens)
    
    # FALLBACK: Raw text, need to preprocess
    text = story_data.get('text', '')
    
    # Handle both data formats
    if 'entities' in story_data:
        entities = story_data['entities']
    elif 'characters' in story_data:
        # Convert characters format to entities
        entities = []
        for char in story_data['characters']:
            if isinstance(char, dict):
                entities.append({
                    'text': char.get('name', ''),
                    'type': 'PERSON',
                    'role': char.get('role', 'supporting')
                })
            else:
                entities.append({
                    'text': char,
                    'type': 'PERSON',
                    'role': 'supporting'
                })
    else:
        entities = []
    
    # Find scene boundaries
    scene_pattern = r'###\s*Scene\s*\d+:[^\n]*\n+'
    scene_splits = re.split(scene_pattern, text)
    
    # Remove empty first section
    if scene_splits and len(scene_splits[0].strip()) == 0:
        scene_splits = scene_splits[1:]
    
    # If no scenes, process as single story
    if len(scene_splits) <= 1:
        single_story = {
            'story_id': story_data.get('story_id', 'unknown'),
            'text': text,
            'entities': entities,
            'metadata': story_data.get('metadata', {})
        }
        processed = preprocessor.process_story(single_story)
        
        # Filter by MIN token count only
        # Let tokenizer handle truncation at max_length during training
        # (Don't lose data by skipping long scenes)
        token_count = len(processed['tokens'])
        if token_count < min_tokens:
            return []
        
        return [processed]
    
    scenes = []
    
    for scene_idx, scene_text in enumerate(scene_splits):
        scene_text = scene_text.strip()
        if not scene_text:
            continue
        
        # Build mini-story for this scene
        # Include ALL entities (preprocessor will filter to what appears)
        scene_story = {
            'story_id': f"{story_data.get('story_id', 'unknown')}_scene{scene_idx + 1}",
            'text': scene_text,
            'entities': entities,  # Pass all entities, preprocessor filters
            'metadata': {
                'genre': story_data.get('genre', 'unknown'),
                'parent_story_id': story_data.get('story_id', 'unknown'),
                'scene_number': scene_idx + 1,
                'total_scenes': len(scene_splits)
            }
        }
        
        # Process scene with preprocessor - generates fresh tokens + BIO tags
        processed = preprocessor.process_story(scene_story)
        
        # Filter by MIN token count only
        # Let tokenizer handle truncation at max_length during training
        token_count = len(processed['tokens'])
        if token_count < min_tokens:
            continue
        
        scenes.append(processed)
    
    return scenes


def split_preprocessed_story(
    story_data: Dict[str, Any],
    min_tokens: int = 100,
    max_tokens: int = 4096
) -> List[Dict[str, Any]]:
    """
    FAST PATH: Split already-preprocessed story by finding scene markers in tokens.
    
    This avoids re-running spaCy on data that's already been processed.
    """
    tokens = story_data['tokens']
    bio_tags = story_data['bio_tags']
    
    # Quick check: if already within limits, return as-is
    if len(tokens) < max_tokens:
        if len(tokens) >= min_tokens:
            return [story_data]
        else:
            return []  # Too short
    
    # Find scene markers (### Scene N:)
    scene_indices = []
    for i in range(len(tokens) - 2):
        if (tokens[i] == '###' and 
            tokens[i+1].lower() == 'scene' and 
            (tokens[i+2].replace(':', '').isdigit() or tokens[i+2] == ':')):
            scene_indices.append(i)
    
    # If no scene markers, just chunk by max_tokens
    if not scene_indices:
        return chunk_by_tokens(story_data, min_tokens, max_tokens)
    
    # Split at scene boundaries
    scenes = []
    story_id = story_data.get('story_id', 'unknown')
    
    for idx, start in enumerate(scene_indices):
        # Determine end (next scene or EOF)
        end = scene_indices[idx + 1] if idx + 1 < len(scene_indices) else len(tokens)
        
        scene_tokens = tokens[start:end]
        scene_bio = bio_tags[start:end]
        
        # Skip if too short
        if len(scene_tokens) < min_tokens:
            continue
        
        # Chunk if too long
        if len(scene_tokens) > max_tokens:
            for chunk_idx, (chunk_tokens, chunk_bio) in enumerate(
                chunk_tokens_and_tags(scene_tokens, scene_bio, max_tokens)
            ):
                if len(chunk_tokens) >= min_tokens:
                    scenes.append({
                        'story_id': f"{story_id}_scene{idx+1}_chunk{chunk_idx+1}",
                        'tokens': chunk_tokens,
                        'bio_tags': chunk_bio
                    })
        else:
            scenes.append({
                'story_id': f"{story_id}_scene{idx+1}",
                'tokens': scene_tokens,
                'bio_tags': scene_bio
            })
    
    return scenes


def chunk_by_tokens(
    story_data: Dict[str, Any],
    min_tokens: int,
    max_tokens: int
) -> List[Dict[str, Any]]:
    """Chunk a story into fixed-size pieces."""
    tokens = story_data['tokens']
    bio_tags = story_data['bio_tags']
    story_id = story_data.get('story_id', 'unknown')
    
    chunks = []
    for i, (chunk_tokens, chunk_bio) in enumerate(
        chunk_tokens_and_tags(tokens, bio_tags, max_tokens)
    ):
        if len(chunk_tokens) >= min_tokens:
            chunks.append({
                'story_id': f"{story_id}_chunk{i+1}",
                'tokens': chunk_tokens,
                'bio_tags': chunk_bio
            })
    
    return chunks


def chunk_tokens_and_tags(tokens, bio_tags, chunk_size):
    """Generator to yield token/bio_tag chunks."""
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i+chunk_size], bio_tags[i:i+chunk_size]


def main():
    parser = argparse.ArgumentParser(
        description="Split stories at scene boundaries (safe - regenerates tags)"
    )
    parser.add_argument('--input', '-i', required=True, help="Input story JSONL")
    parser.add_argument('--output', '-o', required=True, help="Output scene JSONL")
    parser.add_argument('--min-tokens', type=int, default=100, help="Min tokens per scene")
    parser.add_argument('--max-tokens', type=int, default=4096, help="Max tokens per scene")
    
    args = parser.parse_args()
    
    print(f"📂 Splitting scenes from {args.input}...")
    print(f"   Token range: {args.min_tokens} - {args.max_tokens}")
    print()
    
    # Count lines and check if data is preprocessed
    with open(args.input, 'r') as f:
        first_line = f.readline()
        total_stories = 1 + sum(1 for _ in f)
    
    # Check if preprocessing needed
    first_story = json.loads(first_line)
    needs_preprocessing = 'tokens' not in first_story or 'bio_tags' not in first_story
    
    if needs_preprocessing:
        print(f"⚠️  Input not preprocessed - will run spaCy (SLOW)")
        print(f"🔧 Initializing preprocessor...")
        preprocessor = StoryPreprocessor()
    else:
        print(f"✅ Input already preprocessed - using FAST PATH (no spaCy)")
        preprocessor = None  # Won't be used
    
    print(f"🔄 Processing {total_stories:,} stories...")
    
    stories_with_scenes = 0
    total_scenes_written = 0
    stories_no_scenes = 0
    
    with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
        # Process first story
        scenes = split_story_at_scenes(
            first_story,
            preprocessor,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens
        )
        
        if len(scenes) > 1:
            stories_with_scenes += 1
        else:
            stories_no_scenes += 1
        
        for scene in scenes:
            outfile.write(json.dumps(scene) + '\n')
            total_scenes_written += 1
        
        # Process remaining stories
        for line in tqdm(infile, total=total_stories-1, desc="   Splitting"):
            try:
                story = json.loads(line)
                scenes = split_story_at_scenes(
                    story,
                    preprocessor,
                    min_tokens=args.min_tokens,
                    max_tokens=args.max_tokens
                )
                
                if len(scenes) > 1:
                    stories_with_scenes += 1
                else:
                    stories_no_scenes += 1
                
                for scene in scenes:
                    outfile.write(json.dumps(scene) + '\n')
                    total_scenes_written += 1
                    
            except Exception as e:
                print(f"\n   ⚠️  Error processing story: {e}")
                continue
    
    print()
    print(f"✅ Done!")
    print(f"   Stories with scenes: {stories_with_scenes:,}")
    print(f"   Stories without scenes: {stories_no_scenes:,}")
    print(f"   Total scenes written: {total_scenes_written:,}")
    print(f"   Expansion: {total_scenes_written / total_stories:.1f}x")
    print(f"💾 Output: {args.output}")


if __name__ == '__main__':
    main()
