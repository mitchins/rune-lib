#!/usr/bin/env python3
"""
Preprocess raw NER training data with extended entity support (PERSON + LOCATION + AGENT).

Reads raw JSONL (with text + entities), tokenizes, generates BIO tags, and writes
prefiltered JSONL ready for training. Splits by token length for curriculum learning.
Stories exceeding max_length are scene-split (or chunked if no scene markers).

Usage:
    python3 preprocess_extended_training.py \
        --input ner_training_ultra_short_agent_training_agent_20260406_193605.jsonl \
        --output-dir ./prefiltered_extended \
        --split-threshold 1024 \
        --max-length 8192

Minority-signal filter (--min-minority-ratio / filter C):
    Drops any output chunk that either:
      (a) has zero LOCATION tokens, or zero AGENT tokens  [filter A]
      (b) has (LOCATION+AGENT) < threshold * PERSON tokens  [tail removal]
    Default threshold: 0.30.  Set to 0.0 to apply only (a).
    Disable entirely with --min-minority-ratio -1.
"""

import json
import argparse
import time
from pathlib import Path
from collections import Counter
from rune.data.story_preprocessor import StoryPreprocessor
from split_scenes import split_preprocessed_story


def _passes_minority_filter(bio_tags, min_ratio):
    """Filter C: require ≥1 LOCATION token, ≥1 AGENT token, and
    (LOCATION+AGENT) / PERSON >= min_ratio.

    Args:
        bio_tags:  list of BIO tag strings for a single chunk
        min_ratio: float threshold (0.0 = only presence check, -1 = disabled)
    Returns:
        True if the chunk should be kept.
    """
    if min_ratio < 0:
        return True
    counts = Counter(bio_tags)
    loc = counts.get("B-LOCATION", 0) + counts.get("I-LOCATION", 0)
    agt = counts.get("B-AGENT", 0)    + counts.get("I-AGENT", 0)
    if loc == 0 or agt == 0:
        return False
    if min_ratio == 0.0:
        return True
    per = counts.get("B-PERSON", 0) + counts.get("I-PERSON", 0)
    return per == 0 or (loc + agt) / per >= min_ratio


def preprocess_and_split(input_path, output_dir, split_threshold=1024, max_length=8192,
                         scene_max_tokens=4096, limit=None, min_minority_ratio=0.30):
    """Preprocess raw data and split into short/long files.
    
    Stories exceeding max_length are scene-split into chunks ≤ scene_max_tokens.

    Args:
        min_minority_ratio: Filter C threshold. Chunks must have ≥1 LOCATION token,
            ≥1 AGENT token, and (LOC+AGT)/PERSON >= this value. Use 0.0 for presence-
            only check (filter A), -1 to disable entirely.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = StoryPreprocessor(extended_entity_types=True)

    short_path = output_dir / "stage1_short.jsonl"
    long_path = output_dir / "stage2_long.jsonl"

    tag_counts = Counter()
    entity_counts = Counter()
    short_count = 0
    long_count = 0
    scene_split_count = 0
    scenes_from_splits = 0
    error_count = 0
    filtered_count = 0
    total = 0

    start_time = time.time()

    with (
        open(input_path) as fin,
        open(short_path, "w") as f_short,
        open(long_path, "w") as f_long,
    ):
        for line_num, line in enumerate(fin):
            if limit and total >= limit:
                break

            try:
                raw = json.loads(line.strip())
                if not raw.get("text"):
                    continue

                processed = preprocessor.process_story(raw)

                if not processed or "tokens" not in processed:
                    error_count += 1
                    continue

                n_tokens = len(processed["tokens"])

                def _count_and_write(item, outfile):
                    """Apply minority filter, count tags/entities, write if kept."""
                    if not _passes_minority_filter(item["bio_tags"], min_minority_ratio):
                        nonlocal filtered_count
                        filtered_count += 1
                        return False
                    for tag in item["bio_tags"]:
                        tag_counts[tag] += 1
                    for ent in item.get("entities", []):
                        entity_counts[ent["label"]] += 1
                    outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                    return True

                if n_tokens > max_length:
                    # Scene-split overlong stories
                    scenes = split_preprocessed_story(processed, min_tokens=100, max_tokens=scene_max_tokens)
                    scene_split_count += 1
                    for scene in scenes:
                        scene_len = len(scene["tokens"])
                        if scene_len <= split_threshold:
                            if _count_and_write(scene, f_short):
                                short_count += 1
                        else:
                            if _count_and_write(scene, f_long):
                                long_count += 1
                        scenes_from_splits += 1
                elif n_tokens <= split_threshold:
                    if _count_and_write(processed, f_short):
                        short_count += 1
                else:
                    if _count_and_write(processed, f_long):
                        long_count += 1

                total += 1

                if total % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = total / elapsed
                    print(f"  Processed {total:,} stories ({rate:.0f}/s) | "
                          f"short={short_count:,} long={long_count:,} "
                          f"scene-split={scene_split_count:,}")

            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"  ⚠️  Error on line {line_num}: {e}")

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print(f"✅ Preprocessing complete in {elapsed:.1f}s ({total/elapsed:.0f} stories/s)")
    print()
    print(f"📊 Split results:")
    print(f"   Short (≤{split_threshold} tokens): {short_count:,} → {short_path}")
    print(f"   Long  (>{split_threshold} tokens):  {long_count:,} → {long_path}")
    print(f"   Scene-split stories:                {scene_split_count:,} → {scenes_from_splits:,} scenes")
    print(f"   Filtered (minority signal):         {filtered_count:,}"
          + (f"  [ratio≥{min_minority_ratio}]" if min_minority_ratio >= 0 else "  [disabled]"))
    print(f"   Errors:                             {error_count:,}")
    print()
    print(f"🏷️  BIO tag distribution:")
    for tag in sorted(tag_counts.keys()):
        print(f"   {tag:15s}: {tag_counts[tag]:>10,}")
    print()
    print(f"📍 Entity counts (detected in text):")
    for etype in sorted(entity_counts.keys()):
        print(f"   {etype:15s}: {entity_counts[etype]:>10,}")
    print()

    # Write summary
    summary = {
        "input": str(input_path),
        "total_processed": total,
        "short_count": short_count,
        "long_count": long_count,
        "scene_split_stories": scene_split_count,
        "scenes_from_splits": scenes_from_splits,
        "filtered_minority": filtered_count,
        "min_minority_ratio": min_minority_ratio,
        "error_count": error_count,
        "split_threshold": split_threshold,
        "max_length": max_length,
        "scene_max_tokens": scene_max_tokens,
        "tag_counts": dict(tag_counts),
        "entity_counts": dict(entity_counts),
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Summary written to: {summary_path}")

    return short_count, long_count, scene_split_count, filtered_count


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw NER data with extended entities")
    parser.add_argument("--input", required=True, help="Input raw JSONL file")
    parser.add_argument("--output-dir", default="./prefiltered_extended", help="Output directory")
    parser.add_argument("--split-threshold", type=int, default=1024,
                        help="Token count threshold for short/long split")
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Max token length before scene-splitting")
    parser.add_argument("--scene-max-tokens", type=int, default=4096,
                        help="Max tokens per scene when splitting overlong stories")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of stories to process")
    parser.add_argument(
        "--min-minority-ratio", type=float, default=0.30,
        help=(
            "Filter C: drop chunks where LOCATION or AGENT tokens are absent, or where "
            "(LOCATION+AGENT)/PERSON < this threshold. "
            "0.0 = presence-only check (filter A). -1 = disabled (keep all). "
            "Default: 0.30"
        ),
    )
    args = parser.parse_args()

    print("🔧 PREPROCESSING RAW NER DATA")
    print("=" * 70)
    print(f"📄 Input: {args.input}")
    print(f"📁 Output: {args.output_dir}")
    print(f"📏 Short/Long split at: {args.split_threshold} tokens")
    print(f"📏 Scene-split threshold: {args.max_length} tokens")
    print(f"📏 Scene max tokens: {args.scene_max_tokens} tokens")
    if args.min_minority_ratio < 0:
        print(f"🔽 Minority filter: disabled")
    elif args.min_minority_ratio == 0.0:
        print(f"🔽 Minority filter: presence-only (≥1 LOC and ≥1 AGT token)")
    else:
        print(f"🔽 Minority filter C: ≥1 LOC, ≥1 AGT, (LOC+AGT)/PER ≥ {args.min_minority_ratio}")
    if args.limit:
        print(f"🔢 Limit: {args.limit} stories")
    print("=" * 70)
    print()

    preprocess_and_split(
        args.input,
        args.output_dir,
        split_threshold=args.split_threshold,
        max_length=args.max_length,
        scene_max_tokens=args.scene_max_tokens,
        limit=args.limit,
        min_minority_ratio=args.min_minority_ratio,
    )


if __name__ == "__main__":
    main()
