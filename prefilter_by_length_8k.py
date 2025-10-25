#!/usr/bin/env python3
"""
Pre-filter dataset for ModernBERT (8192 token limit).
Creates a clean filtered dataset with pre-processed stories.
"""

import json
import argparse
from transformers import AutoTokenizer
from rune.data.story_preprocessor import StoryPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Pre-filter dataset for ModernBERT")
    parser.add_argument('--input', default="ner_training_reshaped.jsonl", help="Input JSONL file")
    parser.add_argument('--output', default="ner_training_filtered_8k.jsonl", help="Output JSONL file")
    parser.add_argument('--max-tokens', type=int, default=8192, help="Maximum token length")
    args = parser.parse_args()

    print("🔍 PRE-FILTERING DATASET FOR MODERNBERT (8K TOKENS)")
    print("=" * 70)
    print()

    # Configuration
    input_file = args.input
    output_file = args.output
    max_tokens = args.max_tokens

    # Load ModernBERT tokenizer
    print(f"🔄 Loading ModernBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    print("✅ Tokenizer loaded")
    print()

    # Initialize preprocessor
    preprocessor = StoryPreprocessor(use_spacy=False)

    # Process and filter
    print(f"📊 Processing stories from {input_file}...")
    print(f"   Max token length: {max_tokens}")
    print()

    total_stories = 0
    kept_stories = 0
    filtered_stories = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for i, line in enumerate(infile):
            total_stories += 1

            try:
                # Load raw story
                raw_story = json.loads(line.strip())

                # Preprocess to get tokens
                processed = preprocessor.process_story(raw_story)
                tokens = processed["tokens"]

                # Check token length
                tokenized = tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=False,
                    add_special_tokens=True
                )
                token_length = len(tokenized["input_ids"])

                # Keep or filter
                if token_length <= max_tokens:
                    # Write processed story to output
                    outfile.write(json.dumps(processed) + "\n")
                    kept_stories += 1
                else:
                    filtered_stories += 1

                # Progress reporting
                if (i + 1) % 1000 == 0:
                    keep_rate = (kept_stories / total_stories) * 100
                    print(f"   Processed {i + 1:,} stories | "
                          f"Kept: {kept_stories:,} ({keep_rate:.1f}%) | "
                          f"Filtered: {filtered_stories:,}")

            except Exception as e:
                print(f"⚠️  Error processing story {i}: {e}")
                continue

    print()
    print("=" * 70)
    print(f"✅ PRE-FILTERING COMPLETE")
    print("=" * 70)
    print()
    print(f"📊 Statistics:")
    print(f"   Total stories processed: {total_stories:,}")
    print(f"   Stories kept (<= {max_tokens} tokens): {kept_stories:,} ({kept_stories/total_stories*100:.1f}%)")
    print(f"   Stories filtered (> {max_tokens} tokens): {filtered_stories:,} ({filtered_stories/total_stories*100:.1f}%)")
    print()
    print(f"💾 Filtered dataset saved to: {output_file}")
    print(f"📝 Format: Pre-processed with tokens and BIO tags ready for training")
    print()

    # Comparison with Longformer
    print(f"📈 Comparison:")
    print(f"   Longformer (4096): 5,028 stories (11.1%)")
    print(f"   ModernBERT (8192): {kept_stories:,} stories ({kept_stories/total_stories*100:.1f}%)")
    print(f"   Gain: {kept_stories - 5028:,} more stories ({(kept_stories/5028):.1f}x more data!)")
    print()
    print("🚀 Ready to train ModernBERT efficiently!")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
