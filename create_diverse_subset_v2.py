#!/usr/bin/env python3
"""
Create diverse training subsets for NER with novelty-redundancy balanced scoring.

Key improvements:
1. Balanced scoring: score = (Δ_unique/total)^α * (1-overlap)^β
2. Auto-throttling: stops when marginal novelty < threshold
3. Efficient context pattern tracking with Bloom filters
4. Optional size limits (None = auto-stop based on novelty)

Based on advice: maximize novelty per redundancy, not just unique patterns.
"""

import json
import hashlib
import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Iterator
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import math

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None, **kwargs):
        return iterable

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    print("⚠️  datasketch not installed - using simple dedup")

try:
    from pybloom_live import BloomFilter
    HAS_BLOOM = True
except ImportError:
    HAS_BLOOM = False
    print("⚠️  pybloom-live not installed - using set (slower)")

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DiversityConfig:
    """Configuration for novelty-redundancy balanced sampling."""

    # Length buckets (by token count)
    length_buckets: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 500),       # Short
        (500, 1500),    # Medium
        (1500, 3000),   # Long
        (3000, 100000), # Very long
    ])

    # Context pattern settings
    context_window: int = 3  # Words around entity

    # Balanced scoring parameters
    alpha: float = 1.0  # Novelty appetite (higher = more novelty-hungry)
    beta: float = 0.5   # Redundancy aversion (higher = more redundancy-averse)

    # Auto-stopping threshold (None = use fixed sizes)
    # Stop when marginal_score < epsilon for max_stagnant consecutive items
    epsilon: Optional[float] = 0.05  # Marginal novelty threshold (raised from 0.01)
    max_stagnant: int = 50  # Consecutive low-score items before stopping (reduced from 100)
    
    # Story duplication penalty (for scene-split corpora)
    # Multiply score by (1 - penalty) when selecting multiple scenes from same story
    story_dup_penalty: float = 0.3  # 0.0 = no penalty, 1.0 = completely block duplicates

    # Surface form capping
    max_docs_per_name: int = 50

    # Pattern quality boosting (favor title/dialogue contexts)
    title_context_boost: float = 1.3  # Multiplier for high-value patterns (0 = off)

    # Output sizes (None = auto-stop based on epsilon)
    phase1_size: Optional[int] = 20000
    phase2_size: Optional[int] = 40000
    
    # Phase routing by token length (for auto-stop mode)
    # Phase 1: items < phase_split_tokens (short, efficient batching)
    # Phase 2: items >= phase_split_tokens (long context)
    phase_split_tokens: int = 1024

    # MinHash for context dedup
    minhash_num_perm: int = 64
    minhash_threshold: float = 0.7

    # Bloom filter capacity (if using)
    bloom_capacity: int = 1000000
    bloom_error_rate: float = 0.001


# =============================================================================
# Efficient Context Storage
# =============================================================================

class ContextIndex:
    """Efficient storage for seen context patterns using Bloom filter or set."""

    def __init__(self, config: DiversityConfig):
        self.config = config
        if HAS_BLOOM:
            self.storage = BloomFilter(
                capacity=config.bloom_capacity,
                error_rate=config.bloom_error_rate
            )
            self.use_bloom = True
        else:
            self.storage = set()
            self.use_bloom = False

        self.exact_count = 0  # Track exact count even with Bloom

    def add(self, pattern: str):
        """Add pattern to index."""
        if self.use_bloom:
            self.storage.add(pattern)
        else:
            self.storage.add(pattern)
        self.exact_count += 1

    def contains(self, pattern: str) -> bool:
        """Check if pattern exists (may have false positives with Bloom)."""
        return pattern in self.storage

    def __len__(self) -> int:
        return self.exact_count


# =============================================================================
# Lightweight Metadata
# =============================================================================

@dataclass
class ItemMeta:
    """Lightweight metadata for an item."""
    line_number: int
    text_hash: str
    token_count: int
    length_bucket: int
    entity_names: Set[str]
    context_patterns: Set[str]
    story_id: Optional[str] = None  # Track source story for duplicate penalty

    def compute_novelty_score(
        self,
        global_context_index: ContextIndex,
        name_counts: Counter,
        config: DiversityConfig
    ) -> Tuple[float, int, int]:
        """
        Compute balanced novelty-redundancy score.

        Returns: (score, delta_unique, overlap_count)

        score = (Δ_unique / total)^α * (1 - overlap_ratio)^β * quality_bonus
        """
        if not self.context_patterns:
            return 0.0, 0, 0

        total_contexts = len(self.context_patterns)

        # Count new unique contexts (Δ_unique)
        delta_unique = sum(
            1 for p in self.context_patterns
            if not global_context_index.contains(p)
        )

        # Count overlapping contexts
        overlap_count = total_contexts - delta_unique
        overlap_ratio = overlap_count / total_contexts if total_contexts > 0 else 0

        # Name penalty (soft cap on overrepresented entities)
        name_penalty = sum(
            1 for n in self.entity_names
            if name_counts[n] >= config.max_docs_per_name
        )

        # Balanced score
        if delta_unique == 0:
            return 0.0, delta_unique, overlap_count

        novelty_term = (delta_unique / total_contexts) ** config.alpha
        redundancy_term = (1 - overlap_ratio) ** config.beta

        base_score = novelty_term * redundancy_term

        # Apply name penalty (exponential decay)
        if name_penalty > 0:
            base_score *= math.exp(-0.1 * name_penalty)

        # Pattern quality bonus (favor title/dialogue contexts)
        if config.title_context_boost > 0:
            quality_multiplier = _compute_pattern_quality_bonus(
                self.context_patterns, 
                config.title_context_boost
            )
            base_score *= quality_multiplier

        return base_score, delta_unique, overlap_count


def _compute_pattern_quality_bonus(patterns: Set[str], boost_factor: float) -> float:
    """
    Compute quality multiplier based on high-value pattern presence.
    
    High-value patterns include:
    - Titles: "mr [E]", "mrs [E]", "sir [E]", etc.
    - Speech verbs: "said [E]", "[E] replied", "[E] asked", etc.
    - Action verbs: "[E] walked", "[E] ran", "[E] looked", etc.
    
    Returns multiplier in range [1.0, boost_factor]
    """
    if not patterns or boost_factor <= 1.0:
        return 1.0
    
    # High-value contextual indicators (lowercased)
    TITLES = {'mr', 'mrs', 'miss', 'ms', 'sir', 'lady', 'lord', 'dr', 'professor', 
              'captain', 'colonel', 'father', 'mother'}
    SPEECH_VERBS = {'said', 'replied', 'asked', 'answered', 'shouted', 'whispered', 
                    'called', 'cried', 'muttered', 'exclaimed', 'told', 'spoke'}
    ACTION_VERBS = {'walked', 'ran', 'looked', 'turned', 'stood', 'sat', 'nodded', 
                    'smiled', 'laughed', 'thought', 'knew', 'saw', 'watched', 'felt'}
    
    high_value_count = 0
    
    for pattern in patterns:
        pattern_lower = pattern.lower()
        words = pattern_lower.split()
        
        # Check for titles, speech verbs, or action verbs
        for word in words:
            if word in TITLES or word in SPEECH_VERBS or word in ACTION_VERBS:
                high_value_count += 1
                break  # Count each pattern max once
    
    # Calculate multiplier: scales from 1.0 to boost_factor based on ratio
    if not patterns:
        return 1.0
    
    high_value_ratio = high_value_count / len(patterns)
    
    # Linear interpolation: 1.0 + (boost_factor - 1.0) * ratio
    multiplier = 1.0 + (boost_factor - 1.0) * high_value_ratio
    
    return multiplier


def extract_context_patterns(tokens: List[str], bio_tags: List[str], window: int = 3) -> Set[str]:
    """
    Extract context patterns around entities (n-grams with placeholder).

    Examples:
    - "said [E] quietly"
    - "[E] walked to"
    - "the brave [E]"
    """
    if len(tokens) != len(bio_tags):
        return set()

    patterns = set()

    # Find entity spans
    entity_spans = []
    start = None

    for i, tag in enumerate(bio_tags):
        if tag.startswith('B-'):
            if start is not None:
                entity_spans.append((start, i))
            start = i
        elif tag == 'O':
            if start is not None:
                entity_spans.append((start, i))
                start = None
    if start is not None:
        entity_spans.append((start, len(bio_tags)))

    # Extract context for each entity
    for ent_start, ent_end in entity_spans:
        left_start = max(0, ent_start - window)
        right_end = min(len(tokens), ent_end + window)

        left_ctx = [t.lower() for t in tokens[left_start:ent_start]]
        right_ctx = [t.lower() for t in tokens[ent_end:right_end]]

        # Full context pattern
        pattern = ' '.join(left_ctx + ['[E]'] + right_ctx)
        patterns.add(pattern)

        # Bigrams (more granular)
        if len(left_ctx) >= 2:
            patterns.add(' '.join(left_ctx[-2:]) + ' [E]')
        if len(right_ctx) >= 2:
            patterns.add('[E] ' + ' '.join(right_ctx[:2]))

        # Unigrams (catch edge cases)
        if left_ctx:
            patterns.add(left_ctx[-1] + ' [E]')
        if right_ctx:
            patterns.add('[E] ' + right_ctx[0])

    return patterns


def extract_entity_names(tokens: List[str], bio_tags: List[str]) -> Set[str]:
    """Extract surface form names."""
    names = set()
    current = []

    for token, tag in zip(tokens, bio_tags):
        if tag.startswith('B-'):
            if current:
                names.add(' '.join(current).lower())
            current = [token]
        elif tag.startswith('I-') and current:
            current.append(token)
        else:
            if current:
                names.add(' '.join(current).lower())
                current = []

    if current:
        names.add(' '.join(current).lower())

    return names


def get_length_bucket(token_count: int, buckets: List[Tuple[int, int]]) -> int:
    """Get bucket index for token count."""
    for i, (low, high) in enumerate(buckets):
        if low <= token_count < high:
            return i
    return len(buckets) - 1


def compute_text_hash(tokens: List[str]) -> str:
    """Fast hash for exact deduplication."""
    key_tokens = tokens[:100] + tokens[-100:] if len(tokens) > 200 else tokens
    normalized = ' '.join(t.lower() for t in key_tokens)

    if HAS_XXHASH:
        return xxhash.xxh64(normalized).hexdigest()[:12]
    else:
        return hashlib.md5(normalized.encode()).hexdigest()[:12]


# =============================================================================
# Pass 1: Build Metadata Index
# =============================================================================

def build_metadata_index(
    input_path: str,
    config: DiversityConfig
) -> Tuple[List[ItemMeta], int]:
    """
    Stream through file once, extract lightweight metadata.

    NOTE: Unlike v1, context patterns are ENABLED for proper scoring.
    """
    print(f"📂 Pass 1: Building metadata index from {input_path}...")

    metadata = []
    seen_hashes = set()
    duplicates = 0

    # Count lines
    import subprocess
    try:
        result = subprocess.run(['wc', '-l', input_path],
                               capture_output=True, text=True, timeout=30)
        line_count = int(result.stdout.split()[0])
    except:
        line_count = None

    print(f"   Found ~{line_count if line_count else '?'} items")
    print("   Indexing with context pattern extraction...")

    with open(input_path, 'r') as f:
        for line_num, line in enumerate(tqdm(f, total=line_count, desc="   Indexing")):
            item = json.loads(line)
            tokens = item.get('tokens', [])
            bio_tags = item.get('bio_tags', [])

            # Exact dedup
            text_hash = compute_text_hash(tokens)
            if text_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(text_hash)

            # Extract features (context patterns ENABLED)
            token_count = len(tokens)
            length_bucket = get_length_bucket(token_count, config.length_buckets)
            entity_names = extract_entity_names(tokens, bio_tags)
            context_patterns = extract_context_patterns(
                tokens, bio_tags, config.context_window
            )
            
            # Extract story_id for duplicate penalty
            story_id = item.get('story_id') or item.get('meta', {}).get('story_id')

            meta = ItemMeta(
                line_number=line_num,
                text_hash=text_hash,
                token_count=token_count,
                length_bucket=length_bucket,
                entity_names=entity_names,
                context_patterns=context_patterns,
                story_id=story_id
            )
            metadata.append(meta)

    print(f"   Indexed {len(metadata)} unique items ({duplicates} exact duplicates removed)")

    # Report bucket distribution
    bucket_counts = Counter(m.length_bucket for m in metadata)
    print("   Length bucket distribution:")
    for bucket_idx in sorted(bucket_counts.keys()):
        low, high = config.length_buckets[bucket_idx]
        print(f"      [{low}-{high}): {bucket_counts[bucket_idx]}")

    # Report context pattern stats
    total_patterns = sum(len(m.context_patterns) for m in metadata)
    avg_patterns = total_patterns / len(metadata) if metadata else 0
    print(f"   Context patterns: {total_patterns} total, {avg_patterns:.1f} avg per item")

    return metadata, line_count


# =============================================================================
# Novelty-Redundancy Balanced Sampling
# =============================================================================

def sample_bucket_auto(
    items: List[ItemMeta],
    config: DiversityConfig,
    max_size: Optional[int] = None
) -> List[ItemMeta]:
    """
    Sample items with auto-stopping based on marginal novelty.

    Uses balanced scoring: (Δ_unique/total)^α * (1-overlap)^β
    Stops when marginal score < epsilon OR reaches max_size.
    """
    if not items:
        return []

    print(f"      Sampling from {len(items)} items...")
    print(f"      Config: α={config.alpha}, β={config.beta}, ε={config.epsilon}, stagnant={config.max_stagnant}")

    # Initialize tracking
    selected = []
    global_context_index = ContextIndex(config)
    name_counts = Counter()
    story_counts = Counter()  # Track scenes per story

    # Shuffle for randomness
    random.shuffle(items)

    # Greedy selection with auto-stopping
    pbar = tqdm(total=max_size if max_size else len(items), desc="      Selecting")

    prev_score = float('inf')
    stagnant_count = 0

    for meta in items:
        # Check stopping conditions
        if max_size and len(selected) >= max_size:
            break

        # Compute novelty-redundancy score
        score, delta_unique, overlap = meta.compute_novelty_score(
            global_context_index, name_counts, config
        )
        
        # Apply story duplication penalty
        if meta.story_id and story_counts[meta.story_id] > 0:
            score *= (1 - config.story_dup_penalty)

        # Auto-stop if novelty exhausted
        if config.epsilon is not None:
            if score < config.epsilon:
                stagnant_count += 1
                if stagnant_count >= config.max_stagnant:
                    print(f"\n      Auto-stopped: {config.max_stagnant} consecutive items below ε={config.epsilon}")
                    break
            else:
                stagnant_count = 0

        # Skip if score is 0 (no novelty)
        if score == 0:
            continue

        # Add item
        selected.append(meta)
        pbar.update(1)

        # Update global state
        for p in meta.context_patterns:
            global_context_index.add(p)
        for n in meta.entity_names:
            name_counts[n] += 1
        if meta.story_id:
            story_counts[meta.story_id] += 1

        # Log progress periodically
        if len(selected) % 1000 == 0:
            unique_contexts = len(global_context_index)
            avg_score = sum(
                m.compute_novelty_score(global_context_index, name_counts, config)[0]
                for m in items[:100]  # Sample
            ) / 100
            pbar.set_postfix({
                'unique_ctx': unique_contexts,
                'avg_score': f'{avg_score:.4f}'
            })

    pbar.close()

    print(f"      Selected {len(selected)} items")
    print(f"      Unique contexts accumulated: {len(global_context_index)}")

    return selected


# =============================================================================
# Main Pipeline
# =============================================================================

def create_diverse_subsets(
    input_path: str,
    output_dir: str,
    config: DiversityConfig
) -> None:
    """Main pipeline with novelty-redundancy balanced sampling."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    random.seed(42)

    # Pass 1: Build metadata index
    all_meta, total_lines = build_metadata_index(input_path, config)

    # Group by length bucket
    buckets = defaultdict(list)
    for meta in all_meta:
        buckets[meta.length_bucket].append(meta)

    del all_meta

    # Calculate allocation per bucket
    total_items = sum(len(b) for b in buckets.values())

    # Fixed size mode vs. Auto-stop mode
    use_fixed_sizes = config.phase1_size is not None
    
    if use_fixed_sizes:
        # Fixed size mode: proportional allocation
        phase1_total = min(config.phase1_size, total_items)
        phase2_total = min(config.phase2_size, total_items - phase1_total) if config.phase2_size else 0
        print(f"\n📊 Sampling strategy: Fixed sizes")
        print(f"   Phase 1: {phase1_total} (fixed)")
        print(f"   Phase 2: {phase2_total} (fixed)")
    else:
        # Auto-stop mode: route by token length
        phase1_total = None
        phase2_total = None
        print(f"\n📊 Sampling strategy: Auto-stop")
        print(f"   Phase 1: items < {config.phase_split_tokens} tokens (until ε={config.epsilon})")
        print(f"   Phase 2: items >= {config.phase_split_tokens} tokens (until ε={config.epsilon})")

    phase1_selected = []
    phase2_selected = []

    # Process each bucket
    for bucket_idx in sorted(buckets.keys()):
        bucket_items = buckets[bucket_idx]
        bucket_size = len(bucket_items)
        bucket_range = config.length_buckets[bucket_idx]

        print(f"\n   Bucket {bucket_range}: {bucket_size} items")

        if use_fixed_sizes:
            # Fixed size mode: proportional split, then secondary sampling
            proportion = bucket_size / total_items
            bucket_phase1_target = max(1, int(phase1_total * proportion))
            bucket_phase2_target = max(1, int(phase2_total * proportion)) if phase2_total else 0
            
            print(f"      Targets - phase1: {bucket_phase1_target}, phase2: {bucket_phase2_target}")

            # Sample phase1
            bucket_phase1 = sample_bucket_auto(bucket_items, config, bucket_phase1_target)
            phase1_selected.extend(bucket_phase1)

            # Sample phase2 from remaining
            if bucket_phase2_target > 0:
                phase1_lines = {m.line_number for m in bucket_phase1}
                remaining = [m for m in bucket_items if m.line_number not in phase1_lines]
                bucket_phase2 = sample_bucket_auto(remaining, config, bucket_phase2_target)
                phase2_selected.extend(bucket_phase2)
        
        else:
            # Auto-stop mode: route by token length
            # Separate items by phase based on token count
            phase1_items = [m for m in bucket_items if m.token_count < config.phase_split_tokens]
            phase2_items = [m for m in bucket_items if m.token_count >= config.phase_split_tokens]
            
            print(f"      Phase 1 candidates: {len(phase1_items)} (< {config.phase_split_tokens} tokens)")
            print(f"      Phase 2 candidates: {len(phase2_items)} (>= {config.phase_split_tokens} tokens)")
            
            # Sample each phase independently until novelty exhausted
            if phase1_items:
                bucket_phase1 = sample_bucket_auto(phase1_items, config, None)  # No limit
                phase1_selected.extend(bucket_phase1)
                print(f"      Phase 1 sampled: {len(bucket_phase1)}")
            
            if phase2_items:
                bucket_phase2 = sample_bucket_auto(phase2_items, config, None)  # No limit
                phase2_selected.extend(bucket_phase2)
                print(f"      Phase 2 sampled: {len(bucket_phase2)}")

        del bucket_items

    # Trim to exact sizes if specified (fixed mode only)
    if use_fixed_sizes:
        if phase1_total:
            random.shuffle(phase1_selected)
            phase1_selected = phase1_selected[:phase1_total]

        if phase2_total and phase2_total > 0:
            random.shuffle(phase2_selected)
            phase2_selected = phase2_selected[:phase2_total]

    print(f"\n   Final: phase1={len(phase1_selected)}, phase2={len(phase2_selected)}")

    # Write outputs
    phase1_lines = {m.line_number for m in phase1_selected}
    phase2_lines = {m.line_number for m in phase2_selected}

    del phase1_selected
    del phase2_selected
    del buckets

    print(f"\n💾 Pass 2: Writing outputs...")

    phase1_path = output_path / 'diverse_phase1.jsonl'
    phase2_path = output_path / 'diverse_phase2.jsonl'
    combined_path = output_path / 'diverse_combined.jsonl'

    phase1_count = 0
    phase2_count = 0

    with open(phase1_path, 'w') as f1, \
         open(phase2_path, 'w') as f2, \
         open(combined_path, 'w') as fc, \
         open(input_path, 'r') as fin:

        for line_num, line in enumerate(tqdm(fin, total=total_lines, desc="   Writing")):
            if line_num in phase1_lines:
                f1.write(line)
                fc.write(line)
                phase1_count += 1
            elif line_num in phase2_lines:
                f2.write(line)
                fc.write(line)
                phase2_count += 1

    print(f"\n✅ Done!")
    print(f"   Phase 1: {phase1_path} ({phase1_count} items)")
    if phase2_count > 0:
        print(f"   Phase 2: {phase2_path} ({phase2_count} items)")
    print(f"   Combined: {combined_path} ({phase1_count + phase2_count} items)")


def main():
    parser = argparse.ArgumentParser(
        description="Create diverse NER subsets with novelty-redundancy balanced scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-stop mode (no size limits, stops when novelty exhausted)
  python create_diverse_subset_v2.py --input training.jsonl --output ./diverse --auto-stop

  # Fixed size mode
  python create_diverse_subset_v2.py --input training.jsonl --output ./diverse --phase1-size 20000 --phase2-size 40000

  # Custom scoring parameters
  python create_diverse_subset_v2.py --input training.jsonl --output ./diverse --alpha 1.5 --beta 0.8 --epsilon 0.005
        """
    )

    parser.add_argument('--input', required=True, help="Input JSONL file")
    parser.add_argument('--output', required=True, help="Output directory")

    # Size control
    parser.add_argument('--phase1-size', type=int, default=None, help="Phase 1 size (None = auto-stop)")
    parser.add_argument('--phase2-size', type=int, default=None, help="Phase 2 size (None = auto-stop)")
    parser.add_argument('--auto-stop', action='store_true', help="Auto-stop mode (ignore size limits)")
    parser.add_argument('--phase-split-tokens', type=int, default=1024, 
                       help="Token threshold for phase split in auto-stop mode (default: 1024)")

    # Scoring parameters
    parser.add_argument('--alpha', type=float, default=1.0, help="Novelty appetite (default: 1.0)")
    parser.add_argument('--beta', type=float, default=0.5, help="Redundancy aversion (default: 0.5)")
    parser.add_argument('--epsilon', type=float, default=0.05, help="Auto-stop threshold (default: 0.05)")
    parser.add_argument('--max-stagnant', type=int, default=50, 
                       help="Max consecutive low-score items before stopping (default: 50)")
    parser.add_argument('--story-dup-penalty', type=float, default=0.3,
                       help="Score penalty for multiple scenes from same story (default: 0.3)")

    # Other settings
    parser.add_argument('--max-docs-per-name', type=int, default=50, help="Max docs per entity name")
    parser.add_argument('--context-window', type=int, default=3, help="Context window around entities")
    parser.add_argument('--title-context-boost', type=float, default=1.3, 
                       help="Quality boost for title/dialogue patterns (1.0=off, default: 1.3)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    # Apply auto-stop mode
    if args.auto_stop:
        phase1_size = None
        phase2_size = None
    else:
        phase1_size = args.phase1_size or 20000
        phase2_size = args.phase2_size or 40000

    config = DiversityConfig(
        alpha=args.alpha,
        beta=args.beta,
        epsilon=args.epsilon,
        max_stagnant=args.max_stagnant,
        story_dup_penalty=args.story_dup_penalty,
        phase1_size=phase1_size,
        phase2_size=phase2_size,
        phase_split_tokens=args.phase_split_tokens,
        max_docs_per_name=args.max_docs_per_name,
        context_window=args.context_window,
        title_context_boost=args.title_context_boost,
    )

    create_diverse_subsets(args.input, args.output, config)


if __name__ == "__main__":
    main()
