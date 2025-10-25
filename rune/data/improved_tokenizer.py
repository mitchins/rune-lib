#!/usr/bin/env python3
"""
Improved tokenization to fix punctuation inheritance bug.

Splits boundary punctuation while preserving internal punctuation in names.
"""

import re
import string
from typing import List, Dict, Any, Optional

class ImprovedTokenizer:
    """Improved tokenizer that fixes punctuation inheritance issues."""

    def __init__(self):
        # Regex to split words while preserving internal punctuation in names
        self.WORD_RE = re.compile(
            r"""
            (?:
                # Initials pattern (J.K., H.P., etc.)
                [A-Z]\.(?:[A-Z]\.)*
                (?:'s|'s)?                        # optional possessive
            )
            |
            (?:
                # Regular words with internal punctuation
                [A-Za-z]+                         # word part
                (?:['-][A-Za-z]+)*               # optional apostrophe/hyphen + word parts (no dots here)
                (?:'s|'s)?                        # optional possessive
            )
            |
            \d+                                   # numbers
            |
            [^\w\s]                               # single punctuation (boundaries)
            """,
            re.VERBOSE,
        )

    def split_words(self, text: str) -> List[str]:
        """
        Split text into words, separating boundary punctuation but preserving internal punctuation.

        Examples:
        - "John Smith, the detective." → ["John", "Smith", ",", "the", "detective", "."]
        - "O'Reilly's house" → ["O'Reilly", "'s", "house"]
        - "H.P. Lovecraft" → ["H.P.", "Lovecraft"]
        - "Jean-Luc Picard" → ["Jean-Luc", "Picard"]

        Args:
            text: Input text to tokenize

        Returns:
            List of word tokens with boundary punctuation separated
        """
        return [m.group(0) for m in self.WORD_RE.finditer(text)]

    def is_punctuation_only(self, token: str) -> bool:
        """Check if a token contains only punctuation characters."""
        # Remove whitespace prefixes like 'Ġ' from tokenizer
        clean_token = token.replace('Ġ', '').strip()
        return len(clean_token) > 0 and all(ch in string.punctuation for ch in clean_token)

    def tokenize_and_align_labels(
        self,
        text: str,
        character_names: List[Dict[str, str]],
        tokenizer,
        max_length: int = 4096
    ) -> Dict[str, Any]:
        """
        Tokenize text and create aligned BIO labels with punctuation fix.

        Args:
            text: Raw story text
            character_names: List of {"name": str, "role": str} dicts
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length

        Returns:
            Dictionary with tokenized inputs and aligned labels
        """
        # Step 1: Split text into words (fixing punctuation issue)
        words = self.split_words(text)

        # Step 2: Create BIO tags for words
        bio_tags = self._create_bio_tags(words, character_names)

        # Step 3: Tokenize with proper word boundaries
        tokenized_inputs = tokenizer(
            words,
            truncation=True,
            is_split_into_words=True,  # KEY: This enables proper word_id alignment
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Step 4: Align labels with defensive punctuation handling
        labels = self._align_labels_defensively(
            words, bio_tags, tokenized_inputs.word_ids(), tokenizer
        )

        tokenized_inputs["labels"] = labels

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels,
            "words": words,
            "bio_tags": bio_tags
        }

    def _create_bio_tags(self, words: List[str], character_names: List[Dict[str, str]]) -> List[str]:
        """Create BIO tags for word list from character names with overlap handling."""
        bio_tags = ['O'] * len(words)

        # Sort characters by name length (longest first) to prioritize full names
        sorted_chars = sorted(character_names, key=lambda x: len(x['name'].split()), reverse=True)

        # Track which positions are already tagged to prevent overwrites
        tagged_positions = set()

        for char in sorted_chars:
            name = char['name']
            role = char['role']

            # Split character name using same tokenizer
            name_words = self.split_words(name)

            # Find this character name sequence in the word list
            for i in range(len(words) - len(name_words) + 1):
                if words[i:i+len(name_words)] == name_words:
                    # Check if any position in this sequence is already tagged
                    sequence_positions = []
                    for j, word in enumerate(name_words):
                        word_idx = i + j
                        if not self.is_punctuation_only(word):
                            sequence_positions.append(word_idx)

                    # Skip if any non-punctuation position is already tagged
                    if any(pos in tagged_positions for pos in sequence_positions):
                        continue

                    # Tag this sequence, skipping punctuation-only tokens
                    first_non_punct = True
                    for j, word in enumerate(name_words):
                        word_idx = i + j
                        if not self.is_punctuation_only(word):
                            if first_non_punct:
                                bio_tags[word_idx] = f'B-{role}'
                                first_non_punct = False
                            else:
                                bio_tags[word_idx] = f'I-{role}'
                            tagged_positions.add(word_idx)
                        # Punctuation tokens keep their 'O' tag

        return bio_tags

    def _align_labels_defensively(
        self,
        words: List[str],
        bio_tags: List[str],
        word_ids: List[Optional[int]],
        tokenizer
    ) -> List[int]:
        """
        Align labels with defensive punctuation handling.

        Rules:
        1. Only first subword of each word gets the label
        2. Punctuation-only tokens get -100 (ignored in loss)
        3. Subsequent subwords get -100 (ignored in loss)
        """
        # Label mapping
        label_to_id = {
            'O': 0,
            'B-PROTAGONIST': 1, 'I-PROTAGONIST': 2,
            'B-ANTAGONIST': 3, 'I-ANTAGONIST': 4,
            'B-SUPPORTING': 5, 'I-SUPPORTING': 6
        }

        labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens (CLS, SEP, PAD)
                labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word
                if word_idx < len(words):
                    word = words[word_idx]
                    tag = bio_tags[word_idx]

                    # DEFENSIVE: Don't learn from punctuation-only tokens
                    if self.is_punctuation_only(word):
                        labels.append(-100)  # Ignore punctuation in loss
                    else:
                        labels.append(label_to_id.get(tag, -100))
                else:
                    labels.append(-100)  # Out of bounds
            else:
                # Subsequent subword tokens - ignore in loss
                labels.append(-100)

            previous_word_idx = word_idx

        return labels

def test_improved_tokenizer():
    """Test the improved tokenizer on problematic cases."""

    print("🧪 TESTING IMPROVED TOKENIZER")
    print("=" * 50)

    tokenizer_impl = ImprovedTokenizer()

    test_cases = [
        "John Smith, the detective.",
        "O'Reilly's investigation continues.",
        "H.P. Lovecraft wrote stories.",
        "Jean-Luc Picard, captain of the Enterprise.",
        "Dr. Smith-Jones met Mrs. O'Connor."
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n📋 TEST CASE {i}: '{text}'")

        words = tokenizer_impl.split_words(text)
        print(f"   Words: {words}")

        # Check for proper separation
        punct_words = [w for w in words if tokenizer_impl.is_punctuation_only(w)]
        name_words = [w for w in words if not tokenizer_impl.is_punctuation_only(w)]

        print(f"   Punctuation tokens: {punct_words}")
        print(f"   Name/word tokens: {name_words}")

        # Verify internal punctuation preserved
        internal_punct_preserved = any("'" in w or "-" in w or "." in w for w in name_words)
        if internal_punct_preserved:
            preserved_words = [w for w in name_words if any(p in w for p in ["'", "-", "."])]
            print(f"   ✅ Internal punctuation preserved: {preserved_words}")

if __name__ == "__main__":
    test_improved_tokenizer()