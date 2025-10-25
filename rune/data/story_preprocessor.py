"""
Story preprocessing pipeline to convert raw story data into training format.

Takes stories with character metadata and generates tokenized bio-tagged training data.
"""

import json
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from transformers import PreTrainedTokenizer
try:
    import spacy
except ImportError:
    spacy = None

from ..core.schema import BioSequence, TagSchema
from .improved_tokenizer import ImprovedTokenizer


class StoryPreprocessor:
    """Converts raw story data with character metadata into bio-tagged training format."""

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initialize story preprocessor.

        Args:
            tokenizer: Optional tokenizer for alignment (if None, uses whitespace)
            use_spacy: Use spaCy for better sentence and token detection
            spacy_model: spaCy model to use
        """
        self.tokenizer = tokenizer
        self.use_spacy = use_spacy

        # Initialize improved tokenizer for punctuation handling
        self.improved_tokenizer = ImprovedTokenizer()

        if use_spacy and spacy is not None:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"⚠️  spaCy model '{spacy_model}' not found. Using improved tokenization.")
                self.use_spacy = False
                self.nlp = None
        else:
            if use_spacy and spacy is None:
                print("⚠️  spaCy not installed. Using improved tokenization.")
            self.use_spacy = False
            self.nlp = None

    def process_story(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single story into training format.

        Args:
            story_data: Raw story with 'text', 'characters', etc.

        Returns:
            Processed story with tokens and bio_tags
        """
        text = story_data["text"]
        characters = story_data.get("characters", [])

        # Create character name mappings with expanded variants
        char_to_role = {}
        for char in characters:
            name = char["name"]
            role = char["role"]

            # Expand name to all valid variants
            variants = self._expand_name_variants(name)

            # Map all variants to the same role
            for variant in variants:
                char_to_role[variant] = role

        # Tokenize the text
        if self.use_spacy and self.nlp:
            tokens = self._tokenize_with_spacy(text)
        else:
            tokens = self.improved_tokenizer.split_words(text)

        # Generate bio-tags
        bio_tags = self._generate_bio_tags(tokens, char_to_role)

        # Create processed story
        processed = {
            "story_id": story_data.get("story_id", "unknown"),
            "genre": story_data.get("metadata", {}).get("genre", "unknown"),
            "text": text,
            "tokens": tokens,
            "bio_tags": bio_tags,
            "entities": self._extract_entities(tokens, bio_tags),
        }

        return processed

    def _expand_name_variants(self, name: str) -> set:
        """
        Expand a character name into all valid variants for matching.

        This handles cases where the text contains partial name mentions:
        - "Dr. Alessandro Sanna" should match "Dr. Alessandro", "Dr. Sanna", "Alessandro", "Sanna"
        - "Jane Smith" should match "Jane", "Smith"
        - Normalizes punctuation (Dr. vs Dr)

        Args:
            name: Full canonical character name

        Returns:
            Set of all valid name variants
        """
        parts = name.split()
        variants = {name}  # Always include full canonical name

        # Common titles to recognize
        titles = {"Dr.", "Dr", "Mr.", "Mr", "Mrs.", "Mrs", "Ms.", "Ms", "Miss",
                  "Captain", "Detective", "Professor", "Officer", "Agent",
                  "Lieutenant", "Sergeant", "Colonel", "General", "Admiral"}

        if len(parts) >= 3:
            # Check if first part is a title
            if parts[0] in titles or parts[0].rstrip('.') in titles:
                # "Dr. Alessandro Sanna" -> title, first, last
                title = parts[0]
                first = parts[1]
                last = " ".join(parts[2:])  # Handle multi-word last names

                variants.add(f"{title} {first}")      # "Dr. Alessandro"
                variants.add(f"{title} {last}")       # "Dr. Sanna"
                variants.add(first)                    # "Alessandro"
                variants.add(last)                     # "Sanna"
                variants.add(f"{first} {last}")       # "Alessandro Sanna"
            else:
                # "Alessandro Michael Sanna" -> first, middle, last
                first = parts[0]
                last = parts[-1]
                middle = " ".join(parts[1:-1])

                variants.add(first)                    # "Alessandro"
                variants.add(last)                     # "Sanna"
                variants.add(f"{first} {last}")       # "Alessandro Sanna"

        elif len(parts) == 2:
            # "Alessandro Sanna" or "Dr. Alessandro"
            if parts[0] in titles or parts[0].rstrip('.') in titles:
                # "Dr. Alessandro" -> title + name
                title = parts[0]
                name_part = parts[1]
                variants.add(name_part)               # "Alessandro"
                variants.add(f"{title} {name_part}")  # "Dr. Alessandro"
            else:
                # "Alessandro Sanna" -> first + last
                first = parts[0]
                last = parts[1]
                variants.add(first)                   # "Alessandro"
                variants.add(last)                    # "Sanna"

        # Normalize punctuation (Dr. vs Dr, etc.)
        normalized = set()
        for variant in variants:
            normalized.add(variant)
            # Remove trailing periods from titles
            if any(variant.startswith(t) for t in titles):
                normalized.add(variant.rstrip('.'))
                normalized.add(variant.replace('.', ''))
            # Add case variations
            normalized.add(variant.lower())
            normalized.add(variant.upper())

        return normalized

    def _tokenize_with_spacy(self, text: str) -> List[str]:
        """Tokenize text using spaCy."""
        doc = self.nlp(text)
        return [token.text for token in doc]

    def _tokenize_simple(self, text: str) -> List[str]:
        """Simple whitespace tokenization with punctuation separation."""
        # Split on whitespace and separate punctuation
        tokens = []
        words = text.split()

        for word in words:
            # Handle punctuation at the end
            if word and word[-1] in '.!?":;,':
                if len(word) > 1:
                    tokens.append(word[:-1])
                    tokens.append(word[-1])
                else:
                    tokens.append(word)
            else:
                tokens.append(word)

        return tokens

    def _generate_bio_tags(self, tokens: List[str], char_to_role: Dict[str, str]) -> List[str]:
        """
        Generate BIO tags for tokens based on character mappings.

        Args:
            tokens: List of tokens
            char_to_role: Mapping from character names to roles

        Returns:
            List of BIO tags
        """
        bio_tags = ["O"] * len(tokens)

        # Find character mentions
        i = 0
        while i < len(tokens):
            # Try multi-word character names first
            found_match = False

            # Try matching up to 3 consecutive tokens (for names like "Harry Potter")
            for j in range(min(3, len(tokens) - i), 0, -1):
                candidate = " ".join(tokens[i:i+j])

                # Check exact match
                if candidate in char_to_role:
                    role = char_to_role[candidate]
                    bio_tags[i] = f"B-{role}"
                    for k in range(1, j):
                        bio_tags[i + k] = f"I-{role}"
                    i += j
                    found_match = True
                    break

                # Check case-insensitive match
                elif candidate.lower() in char_to_role:
                    role = char_to_role[candidate.lower()]
                    bio_tags[i] = f"B-{role}"
                    for k in range(1, j):
                        bio_tags[i + k] = f"I-{role}"
                    i += j
                    found_match = True
                    break

            if not found_match:
                i += 1

        return bio_tags

    def _extract_entities(self, tokens: List[str], bio_tags: List[str]) -> List[Dict[str, Any]]:
        """Extract entity information from tokens and tags."""
        entities = []
        current_entity = None

        for i, (token, tag) in enumerate(zip(tokens, bio_tags)):
            if tag.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)

                # Start new entity
                role = tag[2:]
                current_entity = {
                    "text": token,
                    "label": "PERSON",  # All characters are persons
                    "role": role,
                    "start": i,
                    "end": i,
                }

            elif tag.startswith("I-") and current_entity:
                # Continue current entity
                current_entity["text"] += " " + token
                current_entity["end"] = i

            else:
                # End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget last entity
        if current_entity:
            entities.append(current_entity)

        return entities

    def process_jsonl_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        limit: Optional[int] = None,
    ) -> None:
        """
        Process a JSONL file of raw stories into training format.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output processed JSONL file
            limit: Optional limit on number of stories to process
        """
        print(f"📚 Processing stories from {input_path}")

        processed_count = 0
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:

            for i, line in enumerate(infile):
                if limit and i >= limit:
                    break

                try:
                    story_data = json.loads(line.strip())
                    processed_story = self.process_story(story_data)

                    outfile.write(json.dumps(processed_story) + "\n")
                    processed_count += 1

                    if processed_count % 100 == 0:
                        print(f"   Processed {processed_count} stories...")

                except Exception as e:
                    print(f"⚠️  Error processing story {i}: {e}")
                    continue

        print(f"✅ Processed {processed_count} stories → {output_path}")

    def get_character_stats(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """Get statistics about characters in the dataset."""
        role_counts = {}
        genre_counts = {}
        total_stories = 0

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                story_data = json.loads(line.strip())
                total_stories += 1

                # Count genres
                genre = story_data.get("metadata", {}).get("genre", "unknown")
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

                # Count character roles
                for char in story_data.get("characters", []):
                    role = char["role"]
                    role_counts[role] = role_counts.get(role, 0) + 1

        return {
            "total_stories": total_stories,
            "role_distribution": role_counts,
            "genre_distribution": genre_counts,
        }