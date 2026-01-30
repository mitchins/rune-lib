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

# Canonical set of title tokens - used for both variant expansion and BIO tagging
TITLE_TOKENS = {
    "mr", "mrs", "ms", "miss", "sir", "lady", "lord", "dr", "professor",
    "captain", "major", "colonel", "commander", "emperor", "duke", "queen", "king",
    "master", "detective", "officer", "agent", "lieutenant", "sergeant", "general", "admiral"
}


class StoryPreprocessor:
    """Converts raw story data with character metadata into bio-tagged training format.
    
    REQUIRES spaCy for accurate surname licensing. Install with:
        pip install spacy
        python -m spacy download en_core_web_sm
    """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initialize story preprocessor.

        Args:
            tokenizer: Optional tokenizer for alignment (if None, uses whitespace)
            spacy_model: spaCy model to use (REQUIRED)
        """
        self.tokenizer = tokenizer

        # Initialize improved tokenizer for punctuation handling
        self.improved_tokenizer = ImprovedTokenizer()

        # spaCy is REQUIRED - fail hard if not available
        if spacy is None:
            raise RuntimeError(
                "spaCy is required for StoryPreprocessor.\n"
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
        
        try:
            self.nlp = spacy.load(spacy_model)
            self.use_spacy = True
        except OSError:
            raise RuntimeError(
                f"spaCy model '{spacy_model}' not found.\n"
                f"Install with: python -m spacy download {spacy_model}"
            )

    def process_story(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single story into training format.

        Args:
            story_data: Raw story with 'text', 'entities'/'characters', etc.

        Returns:
            Processed story with tokens and bio_tags
        """
        text = story_data["text"]

        # STANDARD FORMAT: entities field (type=PERSON for characters)
        # Legacy fallback: characters field (deprecated)
        characters = []

        if "entities" in story_data:
            # Standard format - filter to PERSON entities only
            entities = story_data["entities"]
            for ent in entities:
                if ent.get("type") == "PERSON":
                    characters.append({
                        "name": ent.get("text", ent.get("name", "")),
                        "role": ent.get("role", "supporting")
                    })
        elif "characters" in story_data:
            # Legacy format (deprecated)
            characters = story_data.get("characters", [])

        # Create character name mappings with expanded variants
        char_to_role = {}
        surname_only_variants = set()  # Track standalone surnames for gating
        
        for char in characters:
            name = char["name"]
            role = char["role"]

            # Expand name to all valid variants
            variants = self._expand_name_variants(name)
            
            # Detect surname-only variants (single-token, not first word of canonical name)
            name_parts = name.split()
            canonical_first = name_parts[0] if name_parts else ""
            for variant in variants:
                variant_parts = variant.split()
                # Surname-only if: single token AND (not the first word of a multi-token name OR the name is single-token itself)
                # Conservative: single-token names are treated as potentially surnames and require gating
                if (len(variant_parts) == 1 and 
                    (len(name_parts) == 1 or variant.lower() != canonical_first.lower()) and
                    variant.lower().rstrip('.') not in TITLE_TOKENS):
                    surname_only_variants.add(variant.lower())

            # Map all variants to the same role
            for variant in variants:
                char_to_role[variant] = role

        # Tokenize the text with spaCy (REQUIRED)
        tokens = self._tokenize_with_spacy(text)

        # Generate bio-tags
        bio_tags = self._generate_bio_tags(tokens, char_to_role, surname_only_variants)

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

    def process_stories_batch(self, stories: List[Dict[str, Any]], batch_size: int = 100, n_process: int = 1) -> List[Dict[str, Any]]:
        """
        Process multiple stories with batched spaCy pipe (GPU/multiprocessing optimized).
        
        Uses nlp.pipe() for efficient batch parsing instead of calling nlp() per story.
        Supports CPU parallelism (n_process) and GPU if available.
        
        Args:
            stories: List of story dicts
            batch_size: Batch size for spaCy pipe
            n_process: Number of processes (1=single, >1=multiprocessing)
        
        Returns:
            List of processed stories (same structure as process_story)
        """
        if not stories:
            return []
        
        # Extract texts for batch processing
        texts = [story["text"] for story in stories]
        
        # Batch parse all texts with spaCy pipe
        docs = list(self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process))
        
        # Process each story with its pre-parsed doc
        results = []
        for story, doc in zip(stories, docs):
            # Reuse process_story logic but with pre-parsed doc
            # Extract characters
            characters = []
            if "entities" in story:
                entities = story["entities"]
                for ent in entities:
                    if ent.get("type") == "PERSON":
                        characters.append({
                            "name": ent.get("text", ent.get("name", "")),
                            "role": ent.get("role", "supporting")
                        })
            elif "characters" in story:
                characters = story.get("characters", [])
            
            # Build character mappings
            char_to_role = {}
            surname_only_variants = set()
            for char in characters:
                name = char["name"]
                role = char["role"]
                variants = self._expand_name_variants(name)
                
                # Detect surname-only variants
                name_parts = name.split()
                canonical_first = name_parts[0] if name_parts else ""
                for variant in variants:
                    variant_parts = variant.split()
                    if (len(variant_parts) == 1 and 
                        (len(name_parts) == 1 or variant.lower() != canonical_first.lower()) and
                        variant.lower().rstrip('.') not in TITLE_TOKENS):
                        surname_only_variants.add(variant.lower())
                
                for variant in variants:
                    char_to_role[variant] = role
            
            # Tokenize from pre-parsed doc
            tokens = [token.text for token in doc]
            
            # Generate bio-tags - call existing method with text for lazy init
            bio_tags = self._generate_bio_tags(tokens, char_to_role, surname_only_variants, doc)
            
            # Create processed story
            processed = {
                "story_id": story.get("story_id", "unknown"),
                "genre": story.get("metadata", {}).get("genre", "unknown"),
                "text": story["text"],
                "tokens": tokens,
                "bio_tags": bio_tags,
                "entities": self._extract_entities(tokens, bio_tags),
            }
            results.append(processed)
        
        return results

    def _expand_name_variants(self, name: str) -> set:
        """
        Expand a character name into all valid variants for matching.

        This handles cases where the text contains partial name mentions:
        - "Dr. Alessandro Sanna" should match "Dr. Alessandro", "Dr. Sanna", "Alessandro", "Sanna"
        - "Jane Smith" should match "Jane", "Smith"
        - Normalizes punctuation (Dr. vs Dr)

        ⚠️ KNOWN LIMITATION (V2):
        Currently expands first and last names equally. Last names should require
        contextual "licensing" (title prefix, proximity to canonical form) but this
        gating is not yet implemented. Model learns to discriminate via context despite
        noisy training signal.
        
        First names: High-confidence standalone (contextually unambiguous)
        Last names: Should require anchors (title/proximity) - NOT IMPLEMENTED YET
        
        See README.md § "Name Variant Expansion: First vs. Last Names" for full policy.

        Args:
            name: Full canonical character name

        Returns:
            Set of all valid name variants
        """
        parts = name.split()
        variants = {name}  # Always include full canonical name

        # Use canonical title set (check both original case and lowercase)
        def is_title_word(word):
            return word.lower().rstrip('.') in TITLE_TOKENS

        if len(parts) >= 3:
            # Check if first part is a title
            if is_title_word(parts[0]):
                # "Dr. Alessandro Sanna" -> title, first, last
                title = parts[0]
                first = parts[1]
                last = " ".join(parts[2:])  # Handle multi-word last names

                variants.add(f"{title} {first}")      # "Dr. Alessandro"
                variants.add(f"{title} {last}")       # "Dr. Sanna" ✅ SAFE: title-licensed
                variants.add(first)                    # "Alessandro" ✅ SAFE: first name
                variants.add(last)                     # "Sanna" ⚠️ RISK: last-name-only (V3: needs gating)
                variants.add(f"{first} {last}")       # "Alessandro Sanna"
            else:
                # "Alessandro Michael Sanna" -> first, middle, last
                first = parts[0]
                last = parts[-1]
                middle = " ".join(parts[1:-1])

                variants.add(first)                    # "Alessandro" ✅ SAFE: first name
                variants.add(last)                     # "Sanna" ⚠️ RISK: last-name-only (V3: needs gating)
                variants.add(f"{first} {last}")       # "Alessandro Sanna"

        elif len(parts) == 2:
            # "Alessandro Sanna" or "Dr. Alessandro"
            if is_title_word(parts[0]):
                # "Dr. Alessandro" -> title + name
                title = parts[0]
                name_part = parts[1]
                variants.add(name_part)               # "Alessandro" ✅ SAFE: likely first name with title
                variants.add(f"{title} {name_part}")  # "Dr. Alessandro" ✅ SAFE: title-licensed
            else:
                # "Alessandro Sanna" -> first + last
                first = parts[0]
                last = parts[1]
                # Don't add articles as standalone variants - they're not valid character references
                articles = {"the", "a", "an"}
                if first.lower() not in articles:
                    variants.add(first)               # "Alessandro" ✅ SAFE: first name
                variants.add(last)                    # "Sanna" ⚠️ RISK: last-name-only (V3: needs gating)

        # Normalize punctuation (Dr. vs Dr, etc.)
        normalized = set()
        for variant in variants:
            normalized.add(variant)
            # Remove trailing periods from titles (check against canonical set)
            variant_first = variant.split()[0].lower().rstrip('.') if variant.split() else ''
            if variant_first in TITLE_TOKENS:
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

    def _check_surname_licensing(self, tokens: List[str], lowered: List[str], 
                                 idx: int, text: str, spacy_doc_ref: list, ARTICLES: set) -> bool:
        """
        Check if a surname at position idx has syntactic licensing to be tagged.
        
        Uses spaCy syntax analysis (REQUIRED - fails if spaCy unavailable).
        Lazily initializes spaCy parsing on first call.
        
        Args:
            spacy_doc_ref: list containing [spacy_doc] or [] for lazy init
        
        Returns True if licensed (should tag), False otherwise.
        """
        # Lazy initialization: parse only when first needed
        if not spacy_doc_ref:
            spacy_doc_ref.append(self.nlp(text))
        
        spacy_doc = spacy_doc_ref[0]
        
        # Find the spaCy token that matches our token at idx
        # Use text matching since tokenization might differ
        token_text = tokens[idx]
        spacy_token = None
        
        # Search for matching token in spaCy doc
        # When multiple identical tokens exist, find the closest one
        candidates = []
        for st in spacy_doc:
            if st.text == token_text or st.text.lower() == token_text.lower():
                candidates.append((abs(st.i - idx), st))
        
        if candidates:
            # Pick the closest match
            candidates.sort(key=lambda x: x[0])
            spacy_token = candidates[0][1]
        
        if spacy_token is None:
            # Token not found in spaCy doc - likely tokenization mismatch
            # Check simple pattern: capitalized predecessor or verb neighbor
            prev_token_actual = tokens[idx-1] if idx > 0 else ""
            next_token = lowered[idx+1] if idx+1 < len(lowered) else ""
            prev_lower = lowered[idx-1] if idx > 0 else ""
            prev_is_capitalized = prev_token_actual and prev_token_actual[0].isupper()
            prev_is_article = prev_lower in ARTICLES
            
            # Allow if: capitalized non-article predecessor
            return prev_is_capitalized and not prev_is_article
        
        # Check syntactic roles that license surnames
        # 1. Subject of a verb (nsubj/nsubjpass)
        if spacy_token.dep_ in ("nsubj", "nsubjpass"):
            return True
        
        # 2. Token itself is possessive marker
        # e.g., "Holmes's coat" → Holmes has dep_=poss
        if spacy_token.dep_ == "poss":
            return True
        
        # 3. Vocative or appositive (direct address)
        if spacy_token.dep_ in ("appos", "vocative"):
            return True
        
        # 4. Subject-verb inversion for dialogue tags: "remarked Holmes", "said Watson"
        # In these patterns, the name is dobj (not nsubj) but follows a speech verb
        # We want "remarked Holmes" (Holmes is dobj after speech verb) ✅
        # But NOT "named Norton" (Norton is oprd after naming verb) ❌
        SPEECH_VERBS = {"say", "remark", "ask", "reply", "answer", "whisper", "shout",
                        "exclaim", "mutter", "continue", "call", "cry", "yell"}
        
        if spacy_token.dep_ == "dobj":
            # Check if the head verb is a speech verb
            head_verb = spacy_token.head
            if head_verb.pos_ == "VERB" and head_verb.lemma_.lower() in SPEECH_VERBS:
                return True
        
        # 5. Capitalized non-article predecessor (first name pattern)
        # BUT: Exclude if predecessor is a non-speech verb (e.g., "named Norton", "called John")
        # We want "Elizabeth Bennet" ✅ but NOT "named Norton" ❌
        if idx > 0:
            prev_token = tokens[idx-1]
            prev_lower = lowered[idx-1]
            if prev_token and prev_token[0].isupper() and prev_lower not in ARTICLES:
                # Check if previous token is a verb in spaCy
                # Find the spaCy token for the previous token
                try:
                    prev_spacy_i = max(0, spacy_token.i - 1)
                    prev_spacy = spacy_doc[prev_spacy_i]
                    
                    # Allow if previous is NOT a verb, OR if it IS a speech verb
                    if prev_spacy.pos_ != "VERB":
                        return True
                    elif prev_spacy.lemma_.lower() in SPEECH_VERBS:
                        return True
                    # Otherwise it's a non-speech verb (like "named") - don't license
                except:
                    # If we can't check spaCy, be conservative and allow it
                    return True
        
        # No licensing found
        return False

    def _generate_bio_tags(self, tokens: List[str], char_to_role: Dict[str, str], 
                          surname_only_variants: set = None, spacy_doc_precomputed=None) -> List[str]:
        """
        Generate BIO tags for ALL occurrences of known character names.
        All matched entities are tagged as PERSON (role is metadata, not NER signal).
        
        Strategy:
          - Titles (Mr., Dr., etc.) are kept as O but provide context for next token
          - Always match LONGEST name variant first (greedy matching)
          - Single-token names require capitalization OR title context
          - Surname-only variants require syntactic licensing (verb anchor, possessive, etc.)
          - Uses spaCy POS/dependency parsing (lazy-initialized) ONLY when surname gating is needed
        
        Args:
            spacy_doc_precomputed: Optional pre-parsed spaCy doc (for batched processing)
        """
        if surname_only_variants is None:
            surname_only_variants = set()
            
        ARTICLES = {"the", "a", "an"}  # Lowercase article set
        
        # Lazy initialization: spaCy will be parsed only when actually needed
        text = " ".join(tokens)
        spacy_doc = None
        
        # Use module-level TITLE_TOKENS
        
        # Lazy-initialized spaCy doc (container to hold reference)
        # If pre-computed doc is provided (from batch processing), use it directly
        if spacy_doc_precomputed is not None:
            spacy_doc_ref = [spacy_doc_precomputed]
        else:
            spacy_doc_ref = []

        tags = ["O"] * len(tokens)
        lowered = [t.lower().strip(".,!?;:'\"") for t in tokens]
        
        # Pre-compute name variants grouped by token length (longest first)
        # This ensures we always match the longest possible name
        variants_by_length = {}
        max_length = 1
        for variant in char_to_role.keys():
            # Count tokens in this variant
            var_tokens = variant.split()
            length = len(var_tokens)
            max_length = max(max_length, length)
            if length not in variants_by_length:
                variants_by_length[length] = set()
            variants_by_length[length].add(variant)

        i = 0
        while i < len(tokens):
            tok = lowered[i]

            # (1) Skip title tokens entirely — context only
            if tok in TITLE_TOKENS:
                i += 1
                continue

            # (2) Look behind for title context
            prev_tok = ""
            for k in range(i - 1, max(-1, i - 4), -1):
                if k >= 0 and lowered[k] and lowered[k] not in '.,!?;:\'"':
                    prev_tok = lowered[k]
                    break

            # (3) Check for entity matches, LONGEST FIRST
            matched = False
            window = 0
            
            # Try from longest possible down to 1
            for w in range(min(max_length, len(tokens) - i), 0, -1):
                if w not in variants_by_length:
                    continue
                span = " ".join(lowered[i:i + w])
                if span in variants_by_length[w]:
                    matched = True
                    window = w
                    break

            if not matched:
                i += 1
                continue

            # (4) For single-token matches, apply stricter rules
            if window == 1:
                is_capitalized = tokens[i][0].isupper() if tokens[i] else False
                has_title_context = prev_tok in TITLE_TOKENS
                
                # Skip lowercase without title
                if not is_capitalized and not has_title_context:
                    i += 1
                    continue
                
                # SURNAME GATING: Standalone surnames require licensing context
                # Uses spaCy syntax when available, falls back to pattern matching
                # Blocks: "the Bennet girls", "The Bennet sisters" (article-prefixed)
                # Allows: "Mr. Bennet" (title), "said Holmes" (verb anchor), "Holmes's" (possessive)
                if tok in surname_only_variants:
                    # Exception 1: if preceded by a title, always allow (e.g., "Mr. Bennet")
                    if has_title_context:
                        pass  # Allow - title-licensed surname
                    else:
                        # Check syntactic licensing using spaCy (lazy-initialized)
                        is_licensed = self._check_surname_licensing(
                            tokens, lowered, i, text, spacy_doc_ref, ARTICLES
                        )
                        
                        if not is_licensed:
                            # No licensing context - block
                            i += 1
                            continue
                
                # Additional check: Block names that are objects of naming/calling verbs
                # E.g., "named Norton", "called John" - these are weak introductory mentions
                # Use spaCy to detect oprd/attr dependencies after naming verbs
                if window == 1:  # Only for single-token matches
                    # Lazy-initialize spaCy if needed
                    if not spacy_doc_ref:
                        spacy_doc_ref.append(self.nlp(text))
                    
                    spacy_doc = spacy_doc_ref[0]
                    
                    # Find this token in spaCy doc - use original token text, not lowered
                    # Search near the same position, but be more precise
                    original_tok = tokens[i]
                    spacy_token = None
                    
                    # First try exact position match
                    if i < len(spacy_doc) and spacy_doc[i].text == original_tok:
                        spacy_token = spacy_doc[i]
                    else:
                        # Search nearby (within +/- 2 positions)
                        for offset in range(-2, 3):
                            idx = i + offset
                            if 0 <= idx < len(spacy_doc) and spacy_doc[idx].text == original_tok:
                                spacy_token = spacy_doc[idx]
                                break
                    
                    if spacy_token:
                        # Check if it's an object predicate of a naming verb
                        # NOTE: We check oprd/attr (not dobj) because:
                        # - "named Norton" → Norton is oprd (object predicate) → BLOCK
                        # - "called Norton" → Norton is dobj (direct object in speech) → ALLOW (checked later)
                        NAMING_VERBS = {"name", "call", "dub", "christen"}
                        if spacy_token.dep_ in ("oprd", "attr"):  # Removed "dobj" - that's for speech verbs
                            head_verb = spacy_token.head
                            if head_verb.pos_ == "VERB" and head_verb.lemma_.lower() in NAMING_VERBS:
                                # Block: "named Norton", "a man called John" (naming context)
                                i += 1
                                continue
                
                # Additional check: if token is a common word/number AND no title context, skip
                # This prevents "Three" in "Three more weeks" from being tagged
                ambiguous_tokens = {
                    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                    "eighty", "ninety", "hundred", "thousand"
                }
                
                if tok in ambiguous_tokens and not has_title_context:
                    # Require multi-token match or title context for number names
                    i += 1
                    continue

            # (5) Apply BIO tags - all entities tagged as PERSON
            tags[i] = "B-PERSON"
            for j in range(1, window):
                if i + j < len(tokens):
                    tags[i + j] = "I-PERSON"

            i += window

        return tags

    def _extract_entities(self, tokens: List[str], bio_tags: List[str]) -> List[Dict[str, Any]]:
        """Extract entity information from tokens and tags. All entities are PERSON type."""
        entities = []
        current_entity = None

        for i, (token, tag) in enumerate(zip(tokens, bio_tags)):
            if tag.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)

                # Start new entity
                current_entity = {
                    "text": token,
                    "label": "PERSON",
                    "start": i,
                    "end": i + 1,
                }

            elif tag.startswith("I-") and current_entity:
                # Continue current entity
                current_entity["text"] += " " + token
                current_entity["end"] = i + 1

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