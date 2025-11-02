#!/usr/bin/env python3
"""
Migrate ALL training datasets to use entities format (standard going forward).

Converts:
  characters: [{name: "X", role: "Y"}]
To:
  entities: [{text: "X", type: "PERSON", role: "Y"}]
"""

import json
import argparse
from pathlib import Path


def migrate_story(story):
    """Convert a single story to entities format."""
    # If already has entities, keep it
    if 'entities' in story:
        return story

    # Convert characters to entities
    if 'characters' in story:
        entities = []
        for char in story['characters']:
            entities.append({
                'text': char['name'],
                'type': 'PERSON',
                'role': char.get('role', 'PERSON')
            })

        # Replace characters with entities
        story['entities'] = entities
        del story['characters']

    return story


def migrate_file(input_path, output_path):
    """Migrate an entire JSONL file."""
    print(f"Migrating: {input_path}")
    print(f"Output: {output_path}")

    migrated = 0
    skipped = 0

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for i, line in enumerate(fin):
            story = json.loads(line.strip())

            had_entities = 'entities' in story
            migrated_story = migrate_story(story)

            if not had_entities and 'entities' in migrated_story:
                migrated += 1
            else:
                skipped += 1

            fout.write(json.dumps(migrated_story) + '\n')

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1} stories...")

    print(f"\nComplete!")
    print(f"  Migrated: {migrated}")
    print(f"  Already entities format: {skipped}")
    print(f"  Total: {migrated + skipped}")


def main():
    parser = argparse.ArgumentParser(description="Migrate datasets to entities format")
    parser.add_argument('--input', required=True, help="Input JSONL file")
    parser.add_argument('--output', required=True, help="Output JSONL file")

    args = parser.parse_args()

    migrate_file(args.input, args.output)


if __name__ == '__main__':
    main()
