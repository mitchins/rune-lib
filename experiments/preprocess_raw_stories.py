#!/usr/bin/env python3
"""
Preprocess raw story data by adding tokens and bio_tags.
Uses StoryPreprocessor to convert entities into BIO-tagged format.
"""

import json
import argparse
from rune.data.story_preprocessor import StoryPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw stories with StoryPreprocessor")
    parser.add_argument('--input', required=True, help="Input JSONL file (raw stories)")
    parser.add_argument('--output', required=True, help="Output JSONL file (processed stories)")

    args = parser.parse_args()

    print(f"Initializing StoryPreprocessor...")
    preprocessor = StoryPreprocessor(use_spacy=False)  # Use improved tokenizer

    print(f"Processing stories from {args.input}...")
    processed_count = 0

    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                story = json.loads(line.strip())

                # Process story to add tokens and bio_tags
                processed = preprocessor.process_story(story)

                # Write processed story
                fout.write(json.dumps(processed) + '\n')
                processed_count += 1

                if processed_count % 1000 == 0:
                    print(f"  Processed {processed_count} stories...")

            except Exception as e:
                print(f"  Error processing story {line_num}: {e}")
                continue

    print(f"\n✅ Preprocessing complete!")
    print(f"   Processed: {processed_count} stories")
    print(f"   Output: {args.output}")


if __name__ == '__main__':
    main()
