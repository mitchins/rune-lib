#!/usr/bin/env python3
"""
Reshape synthetic training data to match RUNE's expected format.

Converts:
  - character_details → characters (with name and role)
  - Normalizes role names to uppercase
  - Adds story_id and metadata fields
"""

import json
from pathlib import Path


def normalize_role(role: str) -> str:
    """Normalize role name to expected format."""
    role = role.upper()

    # Map common role names to standard ones
    role_mapping = {
        "PROTAGONIST": "PROTAGONIST",
        "MAIN": "PROTAGONIST",
        "HERO": "PROTAGONIST",
        "ANTAGONIST": "ANTAGONIST",
        "VILLAIN": "ANTAGONIST",
        "SUPPORTING": "SUPPORTING",
        "SUSPECT": "SUPPORTING",  # Map suspect to supporting
        "VICTIM": "SUPPORTING",  # Map victim to supporting
        "SECONDARY": "SUPPORTING",
    }

    return role_mapping.get(role, "SUPPORTING")


def reshape_story(story: dict, story_idx: int) -> dict:
    """Reshape a single story to match expected format."""

    # Extract character details and reshape
    character_details = story.get("character_details", [])

    characters = []
    for char in character_details:
        characters.append({
            "name": char["name"],
            "role": normalize_role(char["role"])
        })

    # Get metadata
    meta = story.get("_meta", {})

    # Create reshaped story
    reshaped = {
        "text": story["text"],
        "characters": characters,
        "story_id": meta.get("story_id", f"story_{story_idx:05d}"),
        "metadata": {
            "genre": meta.get("genre", "unknown"),
            "word_count": meta.get("word_count", 0),
            "num_characters": len(characters),
        }
    }

    return reshaped


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reshape synthetic training data")
    parser.add_argument('input', help="Input JSONL file")
    parser.add_argument('output', help="Output JSONL file")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1

    print(f"📚 Reshaping synthetic data...")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    print()

    processed = 0
    role_counts = {}

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for idx, line in enumerate(infile):
            try:
                story = json.loads(line.strip())
                reshaped = reshape_story(story, idx)

                # Track role distribution
                for char in reshaped["characters"]:
                    role = char["role"]
                    role_counts[role] = role_counts.get(role, 0) + 1

                outfile.write(json.dumps(reshaped) + "\n")
                processed += 1

                if processed % 1000 == 0:
                    print(f"   Processed {processed} stories...")

            except Exception as e:
                print(f"⚠️  Error processing line {idx}: {e}")
                continue

    print(f"\n✅ Reshaped {processed} stories")
    print(f"\n📊 Character Role Distribution:")
    for role, count in sorted(role_counts.items()):
        print(f"   {role}: {count} characters")

    print(f"\n💾 Output saved to: {output_file}")
    print(f"\n🎯 Next step: Train with RUNE using this file!")

    return 0


if __name__ == "__main__":
    exit(main())
