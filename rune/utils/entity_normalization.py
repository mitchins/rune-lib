"""
Entity normalization utilities for consistent entity deduplication and matching.

Provides:
- Canonical name variant expansion (titles, first/last name combinations)
- Possessive normalization
- Title filtering
- Entity matching with partial containment
"""

from typing import Set, List


# Common English titles that should be treated as optional modifiers
TITLE_TOKENS = {
    "mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord",
    "captain", "professor", "prof", "rev", "father", "sister",
    "brother", "duke", "duchess", "baron", "count", "king", "queen",
    "prince", "princess", "sergeant", "lieutenant", "colonel", "general",
    "major", "admiral", "commander"
}


def normalize_entity(entity: str) -> str:
    """
    Normalize an entity string for deduplication and matching.
    
    Handles:
    - Trailing/leading whitespace
    - Quote characters (all types)
    - Trailing punctuation
    - Possessive markers ('s)
    - Case normalization
    
    Args:
        entity: Raw entity string
        
    Returns:
        Normalized entity string (lowercased)
        
    Examples:
        >>> normalize_entity("Fox's")
        'fox'
        >>> normalize_entity('"Jackson"')
        'jackson'
        >>> normalize_entity("Andrea,")
        'andrea'
        >>> normalize_entity("  Mr. Bennet.  ")
        'mr. bennet'
    """
    if not entity:
        return ""
    
    # Strip leading/trailing whitespace
    entity = entity.strip()
    
    # Strip all types of quotes (regular and smart quotes)
    quote_chars = ['"', "'", '"', '"', ''', ''', '«', '»']
    for quote in quote_chars:
        entity = entity.strip(quote)
    
    # Strip trailing punctuation (but preserve internal punctuation like "Mr.")
    entity = entity.rstrip('.,;:!?')
    
    # Remove possessive markers
    # Handle both regular and smart apostrophes
    entity = entity.replace("'s", "").replace("'s", "")
    
    # Lowercase for case-insensitive matching
    return entity.lower()


def remove_titles(name: str) -> str:
    """
    Remove title tokens from a name.

    Args:
        name: Full name potentially with title (e.g., "Dr. Watson")

    Returns:
        Name without title (e.g., "Watson")
    """
    tokens = name.split()
    filtered = [tok for tok in tokens if tok.lower().rstrip('.') not in TITLE_TOKENS]
    return ' '.join(filtered)


def expand_name_variants(canonical_name: str) -> Set[str]:
    """
    Expand canonical name into all valid surface forms.

    Generates variants including:
    - Full name
    - First name only
    - Last name only
    - Name with/without title
    - Name with/without punctuation

    Args:
        canonical_name: Canonical form (e.g., "Dr. Elizabeth Bennet")

    Returns:
        Set of normalized variant strings
    """
    variants = set()

    # Normalize the base name
    normalized = normalize_entity(canonical_name)
    variants.add(normalized)

    # Remove title for variant generation
    no_title = remove_titles(canonical_name)
    normalized_no_title = normalize_entity(no_title)
    variants.add(normalized_no_title)

    # Split into parts
    parts = normalized_no_title.split()

    if len(parts) == 0:
        return variants

    # Single-token names
    if len(parts) == 1:
        variants.add(parts[0])
        return variants

    # Multi-token names: add first, last, and combinations
    first = parts[0]
    last = parts[-1]

    variants.add(first)
    variants.add(last)

    # First + Last (skipping middle names)
    if len(parts) > 1:
        variants.add(f"{first} {last}")

    # Last, First format (common in citations)
    variants.add(f"{last} {first}")

    # Middle name combinations (if present)
    if len(parts) > 2:
        middle = parts[1:-1]
        # First Middle Last
        variants.add(' '.join([first] + middle + [last]))
        # First Middle
        variants.add(' '.join([first] + middle))
        # Middle Last
        variants.add(' '.join(middle + [last]))

    return variants


def match_entity(predicted: str, ground_truth_variants: Set[str]) -> bool:
    """
    Check if predicted entity matches any variant of ground truth.

    Uses bidirectional substring matching to handle partial names.

    Args:
        predicted: Predicted entity text
        ground_truth_variants: Set of valid variants for ground truth entity

    Returns:
        True if predicted matches any variant
    """
    pred_norm = normalize_entity(predicted)

    # Exact match
    if pred_norm in ground_truth_variants:
        return True

    # Bidirectional substring match (handles "Elizabeth" → "Elizabeth Bennet")
    for variant in ground_truth_variants:
        if pred_norm in variant or variant in pred_norm:
            return True

    return False


def build_variant_map(character_names: List[str]) -> dict:
    """
    Build a mapping from all variants to canonical names.

    Useful for coreference resolution and entity linking.

    Args:
        character_names: List of canonical character names

    Returns:
        Dict mapping variant → canonical name
    """
    variant_to_canonical = {}

    for canonical in character_names:
        variants = expand_name_variants(canonical)
        for variant in variants:
            # Handle conflicts: prefer longer canonical names
            if variant in variant_to_canonical:
                existing = variant_to_canonical[variant]
                if len(canonical) > len(existing):
                    variant_to_canonical[variant] = canonical
            else:
                variant_to_canonical[variant] = canonical

    return variant_to_canonical
