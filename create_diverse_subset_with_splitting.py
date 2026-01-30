#!/usr/bin/env python3
"""
Complete pipeline: Scene splitting + Diversity sampling V2

Handles stories over 4096 tokens by splitting them into scenes first,
then applies novelty-redundancy balanced diversity sampling.

Usage:
  python create_diverse_subset_with_splitting.py \
    --input ner_training_latest.jsonl \
    --output ./diverse_output \
    --max-tokens 4096 \
    --auto-stop
"""

import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import sys


def run_scene_splitting(
    input_path: str,
    output_path: str,
    min_tokens: int = 100,
    max_tokens: int = 4096,
    batch_size: int = 100,
    n_process: int = 12
) -> bool:
    """
    Step 1: Split stories over max_tokens into scenes.

    Uses existing split_scenes.py utility with batching.
    """
    print(f"\n{'='*80}")
    print("STEP 1: SCENE SPLITTING")
    print(f"{'='*80}")
    print(f"Input: {input_path}")
    print(f"Token limit: {max_tokens}")
    print(f"Min tokens: {min_tokens}")
    print(f"Batch size: {batch_size}")
    print(f"Parallel processes: {n_process}")

    # Check if split_scenes.py exists
    split_script = Path(__file__).parent / "split_scenes.py"
    if not split_script.exists():
        print(f"❌ Error: split_scenes.py not found at {split_script}")
        return False

    cmd = [
        sys.executable,  # Use same Python interpreter
        str(split_script),
        '--input', input_path,
        '--output', output_path,
        '--min-tokens', str(min_tokens),
        '--max-tokens', str(max_tokens),
        '--batch-size', str(batch_size),
        '--n-process', str(n_process),
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ Scene splitting failed with code {result.returncode}")
        return False

    # Check output exists
    if not Path(output_path).exists():
        print(f"❌ Output file not created: {output_path}")
        return False

    # Count output scenes
    with open(output_path) as f:
        scene_count = sum(1 for _ in f)

    print(f"\n✅ Scene splitting complete: {scene_count:,} scenes")
    return True


def run_diversity_sampling(
    input_path: str,
    output_dir: str,
    auto_stop: bool = True,
    phase1_size: Optional[int] = None,
    phase2_size: Optional[int] = None,
    phase_split_tokens: int = 1024,
    alpha: float = 1.0,
    beta: float = 0.5,
    epsilon: float = 0.05,
    max_stagnant: int = 50,
    story_dup_penalty: float = 0.3,
    max_docs_per_name: int = 50,
    context_window: int = 3,
    title_context_boost: float = 1.3
) -> bool:
    """
    Step 2: Diversity sampling with balanced scoring.

    Uses create_diverse_subset_v2.py.
    """
    print(f"\n{'='*80}")
    print("STEP 2: DIVERSITY SAMPLING")
    print(f"{'='*80}")
    print(f"Input: {input_path}")
    print(f"Mode: {'Auto-stop' if auto_stop else 'Fixed size'}")
    if auto_stop:
        print(f"Parameters: α={alpha}, β={beta}, ε={epsilon}, stagnant={max_stagnant}, story_penalty={story_dup_penalty}")
        print(f"Phase split: <{phase_split_tokens} tokens → Phase 1, ≥{phase_split_tokens} → Phase 2")
    else:
        print(f"Sizes: phase1={phase1_size}, phase2={phase2_size}")

    # Check if V2 script exists
    diversity_script = Path(__file__).parent / "create_diverse_subset_v2.py"
    if not diversity_script.exists():
        print(f"❌ Error: create_diverse_subset_v2.py not found at {diversity_script}")
        return False

    cmd = [
        sys.executable,
        str(diversity_script),
        '--input', input_path,
        '--output', output_dir,
        '--alpha', str(alpha),
        '--beta', str(beta),
        '--max-docs-per-name', str(max_docs_per_name),
        '--context-window', str(context_window),
        '--title-context-boost', str(title_context_boost),
        '--phase-split-tokens', str(phase_split_tokens),
    ]

    if auto_stop:
        cmd.extend([
            '--auto-stop',
            '--epsilon', str(epsilon),
            '--max-stagnant', str(max_stagnant),
            '--story-dup-penalty', str(story_dup_penalty)
        ])
    else:
        if phase1_size:
            cmd.extend(['--phase1-size', str(phase1_size)])
        if phase2_size:
            cmd.extend(['--phase2-size', str(phase2_size)])

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ Diversity sampling failed with code {result.returncode}")
        return False

    print(f"\n✅ Diversity sampling complete")
    return True


def verify_output(output_dir: str) -> None:
    """Display summary of output files."""
    print(f"\n{'='*80}")
    print("OUTPUT SUMMARY")
    print(f"{'='*80}")

    output_path = Path(output_dir)
    files = [
        ('Phase 1', 'diverse_phase1.jsonl'),
        ('Phase 2', 'diverse_phase2.jsonl'),
        ('Combined', 'diverse_combined.jsonl'),
    ]

    for label, filename in files:
        filepath = output_path / filename
        if filepath.exists():
            with open(filepath) as f:
                count = sum(1 for _ in f)
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"  {label:10s}: {filepath}")
            print(f"              {count:,} items, {size_mb:.1f} MB")
        else:
            print(f"  {label:10s}: Not created")


def check_token_distribution(jsonl_path: str, max_tokens: int) -> None:
    """Check if any scenes exceed max_tokens."""
    print(f"\n{'='*80}")
    print("TOKEN DISTRIBUTION CHECK")
    print(f"{'='*80}")

    over_limit = []
    token_counts = []

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            tokens = item.get('tokens', [])
            token_count = len(tokens)
            token_counts.append(token_count)

            if token_count > max_tokens:
                over_limit.append((i, token_count))

    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        max_found = max(token_counts)
        min_found = min(token_counts)

        print(f"  Total items: {len(token_counts):,}")
        print(f"  Token range: {min_found} - {max_found}")
        print(f"  Average: {avg_tokens:.0f} tokens")
        print(f"  Over limit ({max_tokens}): {len(over_limit)} items")

        if over_limit:
            print(f"\n  ⚠️  WARNING: {len(over_limit)} items exceed {max_tokens} tokens!")
            print(f"     These will be truncated during training.")
            print(f"     Consider re-running with scene splitting on these items.")
            for idx, count in over_limit[:5]:
                print(f"       Item {idx}: {count} tokens")
            if len(over_limit) > 5:
                print(f"       ... and {len(over_limit) - 5} more")
        else:
            print(f"  ✅ All items within {max_tokens} token limit")


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: Scene splitting + Diversity sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-stop mode (recommended)
  python create_diverse_subset_with_splitting.py \\
    --input ner_training_latest.jsonl \\
    --output ./diverse_output \\
    --max-tokens 4096 \\
    --auto-stop

  # Fixed size mode
  python create_diverse_subset_with_splitting.py \\
    --input ner_training_latest.jsonl \\
    --output ./diverse_output \\
    --max-tokens 4096 \\
    --phase1-size 20000 \\
    --phase2-size 40000

  # Skip scene splitting (if already split)
  python create_diverse_subset_with_splitting.py \\
    --input already_split.jsonl \\
    --output ./diverse_output \\
    --skip-splitting \\
    --auto-stop
        """
    )

    # Input/Output
    parser.add_argument('--input', required=True, help="Input JSONL file")
    parser.add_argument('--output', required=True, help="Output directory")

    # Scene splitting options
    parser.add_argument('--skip-splitting', action='store_true',
                       help="Skip scene splitting step (if data already split)")
    parser.add_argument('--min-tokens', type=int, default=100,
                       help="Min tokens per scene (default: 100)")
    parser.add_argument('--max-tokens', type=int, default=4096,
                       help="Max tokens per scene (default: 4096)")
    parser.add_argument('--batch-size', type=int, default=100,
                       help="Batch size for scene processing (default: 100)")
    parser.add_argument('--n-process', type=int, default=12,
                       help="Number of parallel processes (default: 12)")

    # Diversity sampling mode
    parser.add_argument('--auto-stop', action='store_true',
                       help="Auto-stop when novelty exhausted (recommended)")
    parser.add_argument('--phase1-size', type=int, default=None,
                       help="Phase 1 size (if not auto-stop)")
    parser.add_argument('--phase2-size', type=int, default=None,
                       help="Phase 2 size (if not auto-stop)")
    parser.add_argument('--phase-split-tokens', type=int, default=1024,
                       help="Token threshold for phase routing in auto-stop mode (default: 1024)")

    # Diversity sampling parameters
    parser.add_argument('--alpha', type=float, default=1.0,
                       help="Novelty appetite (default: 1.0)")
    parser.add_argument('--beta', type=float, default=0.5,
                       help="Redundancy aversion (default: 0.5)")
    parser.add_argument('--epsilon', type=float, default=0.05,
                       help="Auto-stop threshold (default: 0.05)")
    parser.add_argument('--max-stagnant', type=int, default=50,
                       help="Max consecutive low-score items before stopping (default: 50)")
    parser.add_argument('--story-dup-penalty', type=float, default=0.3,
                       help="Score penalty for multiple scenes from same story (default: 0.3)")
    parser.add_argument('--max-docs-per-name', type=int, default=50,
                       help="Max docs per entity name (default: 50)")
    parser.add_argument('--context-window', type=int, default=3,
                       help="Context window around entities (default: 3)")
    parser.add_argument('--title-context-boost', type=float, default=1.3,
                       help="Quality boost for title/dialogue patterns (1.0=off, default: 1.3)")

    # Temp storage
    parser.add_argument('--keep-temp', action='store_true',
                       help="Keep temporary scene-split file")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("DIVERSITY SAMPLING WITH SCENE SPLITTING")
    print(f"{'='*80}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Scene splitting: {'Skipped' if args.skip_splitting else 'Enabled'}")
    print(f"Diversity mode: {'Auto-stop' if args.auto_stop else 'Fixed size'}")

    # Step 1: Scene splitting (if needed)
    if args.skip_splitting:
        print("\n⏭️  Skipping scene splitting (using input as-is)")
        split_output = args.input
    else:
        # Create temp file or use permanent location
        if args.keep_temp:
            split_output = str(output_path / "scenes_split.jsonl")
            print(f"\nScene split output will be saved to: {split_output}")
        else:
            # Use temporary file
            temp_dir = tempfile.mkdtemp(prefix="diverse_scenes_")
            split_output = str(Path(temp_dir) / "scenes_split.jsonl")
            print(f"\nUsing temporary directory: {temp_dir}")

        success = run_scene_splitting(
            input_path=args.input,
            output_path=split_output,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            n_process=args.n_process
        )

        if not success:
            print("\n❌ Pipeline failed at scene splitting stage")
            return 1

    # Step 2: Diversity sampling
    success = run_diversity_sampling(
        input_path=split_output,
        output_dir=args.output,
        auto_stop=args.auto_stop,
        phase1_size=args.phase1_size,
        phase2_size=args.phase2_size,
        phase_split_tokens=args.phase_split_tokens,
        alpha=args.alpha,
        beta=args.beta,
        epsilon=args.epsilon,
        max_stagnant=args.max_stagnant,
        story_dup_penalty=args.story_dup_penalty,
        max_docs_per_name=args.max_docs_per_name,
        context_window=args.context_window,
        title_context_boost=args.title_context_boost
    )

    if not success:
        print("\n❌ Pipeline failed at diversity sampling stage")
        return 1

    # Verify outputs
    verify_output(args.output)

    # Check final token distribution
    combined_path = output_path / "diverse_combined.jsonl"
    if combined_path.exists():
        check_token_distribution(str(combined_path), args.max_tokens)

    # Cleanup temp files (if not keeping)
    if not args.skip_splitting and not args.keep_temp:
        temp_dir = Path(split_output).parent
        print(f"\n🗑️  Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput directory: {args.output}")
    print(f"Use diverse_combined.jsonl for training")

    return 0


if __name__ == "__main__":
    sys.exit(main())
