#!/usr/bin/env python3
"""
Canonical label alignment - THE single source of truth.

This module provides the ONLY alignment functions used by:
- Training (train_modernbert_streaming.py)
- Preprocessing (when generating new data)
- Validation (validate_alignment.py)
- Unit tests (tests/test_label_alignment.py)

Any code that aligns labels to tokenizer output MUST import from here.
"""
from typing import List, Dict, Any

# Map role-aware and non-standard BIO tags to simplified PERSON/LOCATION/AGENT.
# Used when simplify_labels=True in both raw-text and pretokenized paths.
_SIMPLIFY_MAP: Dict[str, str] = {
    "B-PROTAGONIST": "B-PERSON",
    "I-PROTAGONIST": "I-PERSON",
    "B-ANTAGONIST":  "B-PERSON",
    "I-ANTAGONIST":  "I-PERSON",
    "B-SUPPORTING":  "B-PERSON",
    "I-SUPPORTING":  "I-PERSON",
    "B-TITLE":       "B-PERSON",
    "I-TITLE":       "I-PERSON",
    # Already-simplified pass-through
    "B-PERSON":      "B-PERSON",
    "I-PERSON":      "I-PERSON",
    "B-LOCATION":    "B-LOCATION",
    "I-LOCATION":    "I-LOCATION",
    "B-AGENT":       "B-AGENT",
    "I-AGENT":       "I-AGENT",
    "O":             "O",
}


def _apply_simplify(tag: str) -> str:
    """Map a tag through _SIMPLIFY_MAP; fall back to B/I-PERSON for unknowns."""
    simplified = _SIMPLIFY_MAP.get(tag)
    if simplified is not None:
        return simplified
    if tag.startswith("B-"):
        return "B-PERSON"
    if tag.startswith("I-"):
        return "I-PERSON"
    return tag


def align_labels_to_tokenizer(
    text: str,
    tokens: List[str],
    bio_tags: List[str],
    tokenizer,
    max_length: int,
    label_to_id: Dict[str, int],
    simplify_labels: bool = False,
) -> Dict[str, Any]:
    """
    THE canonical alignment function - ONE SOURCE OF TRUTH.

    Aligns pre-labeled tokens to any tokenizer's subword output with proper
    B-/I- continuation logic to prevent spurious entity splits.

    Args:
        text: Raw text string
        tokens: Pre-split tokens (from any tokenizer)
        bio_tags: BIO tags corresponding to tokens (same length as tokens)
        tokenizer: HuggingFace tokenizer to align to
        max_length: Maximum sequence length
        label_to_id: Mapping from BIO tag to integer ID
        simplify_labels: If True, collapse role-aware tags (e.g. B-TITLE,
            B-PROTAGONIST) to B/I-PERSON before alignment.  Unknown B-/I-
            tags also fall back to B/I-PERSON.

    Returns:
        Dict with input_ids, attention_mask, labels (all lists, not tensors)

    Algorithm:
        1. (Optionally) simplify role-aware tags to canonical entity types.
        2. Map characters to BIO tags based on token positions.
        3. Tokenize text with offsets.
        4. For each subword token:
           - Get tag from character at token start position.
           - If subword continues previous entity (no gap + same type), convert B- to I-.
        5. Return aligned labels matching tokenizer output.

    Example:
        >>> text = "Gandalf rode to Rivendell."
        >>> tokens = ["Gandalf", "rode", "to", "Rivendell", "."]
        >>> bio_tags = ["B-PERSON", "O", "O", "B-PERSON", "O"]
        >>> result = align_labels_to_tokenizer(text, tokens, bio_tags, tokenizer, 512, label_to_id)
        >>> # "Gandalf" might tokenize as ["Gand", "alf"]
        >>> # Result: ["B-PERSON", "I-PERSON", "O", "O", "B-PERSON", "O"]
        >>> # First subword gets B-, subsequent subwords within same entity get I-

    Unit tested in tests/test_label_alignment.py with 8 comprehensive test cases.
    """
    # Step 0: Optionally simplify role-aware tags
    if simplify_labels:
        bio_tags = [_apply_simplify(tag) for tag in bio_tags]

    # Step 1: Map characters to BIO tags
    char_to_tag = ['O'] * len(text)
    
    search_pos = 0
    for token, tag in zip(tokens, bio_tags):
        token_start = text.find(token, search_pos)
        if token_start >= 0:
            token_end = token_start + len(token)
            for char_idx in range(token_start, token_end):
                if char_idx < len(char_to_tag):
                    char_to_tag[char_idx] = tag
            search_pos = token_end
    
    # Step 2: Tokenize with offsets
    # Use dynamic padding (pad in batch collator, not here)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding in DataCollator
        return_offsets_mapping=True,
    )
    
    offset_mapping = tokenized["offset_mapping"]
    labels = []
    
    # Step 3: Align labels with continuation logic
    prev_tag = 'O'
    prev_end = -1
    
    for start, end in offset_mapping:
        if start == end == 0:
            # Special token (PAD, CLS, SEP)
            labels.append(-100)
            prev_tag = 'O'
            prev_end = -1
        else:
            tag = char_to_tag[start] if start < len(char_to_tag) else 'O'
            
            # CRITICAL: Continuation logic to prevent spurious entity splits
            # If this token continues the previous entity (no gap AND same entity type),
            # convert B- to I- to maintain single entity
            if prev_end == start and tag.startswith('B-') and prev_tag.endswith(tag[2:]):
                tag = 'I-' + tag[2:]
            
            # Get label ID
            label_id = label_to_id.get(tag)
            if label_id is None:
                # Should not happen - all tags are B-PERSON, I-PERSON, or O
                raise ValueError(
                    f"Tag '{tag}' not found in label_to_id mapping. "
                    f"Available: {list(label_to_id.keys())}"
                )
            labels.append(label_id)
            prev_tag = tag
            prev_end = end
    
    # Remove offset_mapping from output (not needed for training)
    del tokenized["offset_mapping"]
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


def align_labels_pretokenized(
    tokens: List[str],
    bio_tags: List[str],
    tokenizer,
    max_length: int,
    label_to_id: Dict[str, int],
    simplify_labels: bool = False,
) -> Dict[str, Any]:
    """
    Align pretokenized BIO labels to tokenizer subword output.

    THE canonical function for pretokenized inputs (bucket_1024 / prefiltered
    JSONL datasets where ``tokens`` and ``bio_tags`` are already stored).

    Uses HuggingFace ``word_ids()`` for alignment.  Continuation subwords of
    the same word receive label -100 (ignored in loss), following the standard
    HuggingFace NER convention.

    Args:
        tokens: Pre-split tokens (already word-tokenized)
        bio_tags: BIO tags corresponding to tokens (same length as tokens)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        label_to_id: Mapping from BIO tag to integer ID
        simplify_labels: If True, map role-aware tags (e.g. B-PROTAGONIST,
            B-TITLE) to B/I-PERSON before alignment.

    Returns:
        Dict with input_ids, attention_mask, labels (all lists, not tensors).
        Continuation subwords have label -100.
    """
    if simplify_labels:
        bio_tags = [_apply_simplify(tag) for tag in bio_tags]
        # Ensure any remaining unknown tags are caught below via label lookup

    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    word_ids = tokenized.word_ids()
    labels = []
    prev_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != prev_word_idx:
            tag = bio_tags[word_idx]
            label_id = label_to_id.get(tag)
            if label_id is None:
                raise ValueError(
                    f"Tag '{tag}' not found in label_to_id. "
                    f"Available: {list(label_to_id.keys())}"
                )
            labels.append(label_id)
        else:
            labels.append(-100)
        prev_word_idx = word_idx

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


# Expose at module level for clean imports
__all__ = ['align_labels_to_tokenizer', 'align_labels_pretokenized']
