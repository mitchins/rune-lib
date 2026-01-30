#!/usr/bin/env python3
"""
Pytest tests for StoryPreprocessor.

Tests the contextual labeling strategy:
1. Titles are labeled as O (context cues), not B/I-PERSON
2. Contextual gating prevents false positives (e.g., "the cook" vs "Mr. Cook")
3. Name variant expansion works correctly
4. Entity extraction from BIO tags
5. Support for both entities and characters formats
"""

import pytest
from rune.data.story_preprocessor import StoryPreprocessor


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance with spaCy (now mandatory)."""
    return StoryPreprocessor()


class TestContextualLabeling:
    """Test suite for contextual labeling strategy."""

    def test_title_as_context_cue(self, preprocessor):
        """Titles (Mr., Dr., etc.) should be labeled as O, not B/I-PERSON."""
        story = {
            "story_id": "test1",
            "text": "Mr. Bennet arrived at the door.",
            "entities": [
                {"text": "Elizabeth Bennet", "type": "PERSON", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        # spaCy tokenizes "Mr." as one token, followed by "Bennet"
        # Mr. should be O, Bennet should be B-PERSON
        assert processed["tokens"] == ["Mr.", "Bennet", "arrived", "at", "the", "door", "."]
        assert processed["bio_tags"] == ["O", "B-PERSON", "O", "O", "O", "O", "O"]

    def test_title_with_surname(self, preprocessor):
        """Title + surname (Dr. Lamp) should label surname correctly."""
        story = {
            "story_id": "test2",
            "text": "Dr. Lamp entered the room.",
            "entities": [
                {"text": "Helena Lamp", "type": "PERSON", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        # spaCy tokenizes "Dr." as one token
        assert processed["bio_tags"] == ["O", "B-PERSON", "O", "O", "O", "O"]

    def test_common_noun_collision_prevented(self, preprocessor):
        """Common nouns like 'the cook' should NOT be labeled when ambiguous."""
        story = {
            "story_id": "test3",
            "text": "The cook prepared dinner.",
            "entities": [
                {"text": "James Cook", "type": "PERSON", "role": "supporting"}
            ]
        }

        processed = preprocessor.process_story(story)

        # 'cook' should be O because it's lowercase and has no title
        assert processed["bio_tags"] == ["O", "O", "O", "O", "O"]

    def test_title_with_common_noun_surname(self, preprocessor):
        """Title + common noun surname (Mr. Cook) should be labeled correctly."""
        story = {
            "story_id": "test4",
            "text": "Mr. Cook prepared dinner.",
            "entities": [
                {"text": "James Cook", "type": "PERSON", "role": "supporting"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Cook should be B-PERSON because preceded by title (Mr. is one token)
        assert processed["bio_tags"] == ["O", "B-PERSON", "O", "O", "O"]

    def test_full_name_without_title(self, preprocessor):
        """Full names should work normally without titles."""
        story = {
            "story_id": "test5",
            "text": "James Cook prepared dinner.",
            "entities": [
                {"text": "James Cook", "type": "PERSON", "role": "supporting"}
            ]
        }

        processed = preprocessor.process_story(story)

        assert processed["bio_tags"] == ["B-PERSON", "I-PERSON", "O", "O", "O"]

    def test_multiple_titles(self, preprocessor):
        """Multiple titles (Lady/Sir) should be handled correctly."""
        story = {
            "story_id": "test6",
            "text": "Lady Lucas and Sir William arrived.",
            "entities": [
                {"text": "Charlotte Lucas", "type": "PERSON", "role": "supporting"},
                {"text": "William Lucas", "type": "PERSON", "role": "supporting"}
            ]
        }

        processed = preprocessor.process_story(story)

        assert processed["bio_tags"] == ["O", "B-PERSON", "O", "O", "B-PERSON", "O", "O"]

    def test_sentence_start_capitalization_tags_names(self, preprocessor):
        """Capitalized names at sentence start should be tagged (maximize recall)."""
        story = {
            "story_id": "test7",
            "text": "The door opened. Cook arrived with supplies.",
            "entities": [
                {"text": "James Cook", "type": "PERSON", "role": "supporting"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Cook is capitalized and a known character variant - should be tagged
        # The model learns from context whether it's a name or noun
        assert processed["bio_tags"] == ["O", "O", "O", "O", "B-PERSON", "O", "O", "O", "O"]

    def test_title_mid_sentence(self, preprocessor):
        """Title mid-sentence provides clear context."""
        story = {
            "story_id": "test8",
            "text": "The man was Mr. Cook from the village.",
            "entities": [
                {"text": "James Cook", "type": "PERSON", "role": "supporting"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Cook should be labeled because preceded by Mr. (spaCy tokenizes Mr. as one token)
        assert processed["bio_tags"] == ["O", "O", "O", "O", "B-PERSON", "O", "O", "O", "O"]


class TestNameVariantExpansion:
    """Test suite for name variant expansion."""

    def test_two_part_name(self, preprocessor):
        """Two-part names should expand to both parts and full name."""
        variants = preprocessor._expand_name_variants("James Cook")

        # Should include lowercase versions due to normalization
        assert "james" in variants
        assert "cook" in variants
        assert "james cook" in variants

    def test_three_part_name_with_title(self, preprocessor):
        """Names with titles should expand correctly."""
        variants = preprocessor._expand_name_variants("Dr. Alessandro Sanna")

        # Check key variants (case-insensitive)
        variants_lower = {v.lower() for v in variants}
        assert "alessandro" in variants_lower
        assert "sanna" in variants_lower
        assert "dr. alessandro" in variants_lower or "dr alessandro" in variants_lower
        assert "dr. sanna" in variants_lower or "dr sanna" in variants_lower

    def test_single_name(self, preprocessor):
        """Single names should just return themselves."""
        variants = preprocessor._expand_name_variants("Elizabeth")

        assert "elizabeth" in variants

    def test_punctuation_normalization(self, preprocessor):
        """Titles should be normalized with and without punctuation."""
        variants = preprocessor._expand_name_variants("Mr. Smith")

        variants_lower = {v.lower() for v in variants}
        # Should have both "mr. smith" and "mr smith"
        assert any("mr" in v and "smith" in v for v in variants_lower)


class TestEntityExtraction:
    """Test entity extraction from BIO tags."""

    def test_single_entity_extraction(self, preprocessor):
        """Should extract single-token entities correctly when in full name."""
        story = {
            "story_id": "test",
            "text": "Elizabeth Bennet smiled.",
            "entities": [
                {"text": "Elizabeth Bennet", "type": "PERSON", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)
        entities = processed["entities"]

        assert len(entities) == 1
        assert entities[0]["text"] == "Elizabeth Bennet"
        assert entities[0]["label"] == "PERSON"

    def test_multi_token_entity_extraction(self, preprocessor):
        """Should extract multi-token entities correctly."""
        story = {
            "story_id": "test",
            "text": "James Cook smiled.",
            "entities": [
                {"text": "James Cook", "type": "PERSON", "role": "supporting"}
            ]
        }

        processed = preprocessor.process_story(story)
        entities = processed["entities"]

        assert len(entities) == 1
        assert entities[0]["text"] == "James Cook"
        assert entities[0]["label"] == "PERSON"

    def test_multiple_entities(self, preprocessor):
        """Should extract multiple entities in order."""
        story = {
            "story_id": "test",
            "text": "Lady Lucas and Sir William arrived.",
            "entities": [
                {"text": "Charlotte Lucas", "type": "PERSON", "role": "supporting"},
                {"text": "William Lucas", "type": "PERSON", "role": "supporting"}
            ]
        }

        processed = preprocessor.process_story(story)
        entities = processed["entities"]

        assert len(entities) == 2
        assert entities[0]["text"] == "Lucas"
        assert entities[1]["text"] == "William"


class TestDataFormatSupport:
    """Test support for both entities and characters formats."""

    def test_entities_format(self, preprocessor):
        """Should support entities format (type=PERSON)."""
        story = {
            "story_id": "test",
            "text": "James Cook arrived.",
            "entities": [
                {"text": "James Cook", "type": "PERSON", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        assert processed["bio_tags"] == ["B-PERSON", "I-PERSON", "O", "O"]

    def test_characters_format_legacy(self, preprocessor):
        """Should support legacy characters format."""
        story = {
            "story_id": "test",
            "text": "James Cook arrived.",
            "characters": [
                {"name": "James Cook", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        assert processed["bio_tags"] == ["B-PERSON", "I-PERSON", "O", "O"]

    def test_entities_filters_non_person(self, preprocessor):
        """Should filter out non-PERSON entities."""
        story = {
            "story_id": "test",
            "text": "James Cook visited London.",
            "entities": [
                {"text": "James Cook", "type": "PERSON", "role": "protagonist"},
                {"text": "London", "type": "LOCATION", "role": None}
            ]
        }

        processed = preprocessor.process_story(story)

        # Only James Cook should be labeled, not London
        assert processed["bio_tags"] == ["B-PERSON", "I-PERSON", "O", "O", "O"]

    def test_empty_entities(self, preprocessor):
        """Should handle stories with no entities gracefully."""
        story = {
            "story_id": "test",
            "text": "The door opened.",
            "entities": []
        }

        processed = preprocessor.process_story(story)

        assert all(tag == "O" for tag in processed["bio_tags"])


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_text(self, preprocessor):
        """Should handle empty text gracefully."""
        story = {
            "story_id": "test",
            "text": "",
            "entities": []
        }

        processed = preprocessor.process_story(story)

        assert processed["tokens"] == []
        assert processed["bio_tags"] == []

    def test_only_punctuation(self, preprocessor):
        """Should handle text with only punctuation."""
        story = {
            "story_id": "test",
            "text": "... !!!",
            "entities": []
        }

        processed = preprocessor.process_story(story)

        # Should tokenize punctuation
        assert len(processed["tokens"]) > 0
        assert all(tag == "O" for tag in processed["bio_tags"])

    def test_different_roles(self, preprocessor):
        """Should handle different character roles correctly with full names."""
        story = {
            "story_id": "test",
            "text": "Alice Smith met Bob Jones and Carol Lee.",
            "entities": [
                {"text": "Alice Smith", "type": "PERSON", "role": "protagonist"},
                {"text": "Bob Jones", "type": "PERSON", "role": "supporting"},
                {"text": "Carol Lee", "type": "PERSON", "role": "antagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        # All characters should be tagged as PERSON (role-agnostic)
        assert processed["bio_tags"].count("B-PERSON") == 3
        assert processed["bio_tags"].count("I-PERSON") == 3


class TestCommonNounHandling:
    """Test handling of common nouns that can be surnames."""

    def test_common_noun_lowercase(self, preprocessor):
        """Lowercase common nouns should not be labeled."""
        story = {
            "story_id": "test",
            "text": "The lamp flickered.",
            "entities": [
                {"text": "Helena Lamp", "type": "PERSON", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        # 'lamp' should be O
        assert processed["bio_tags"] == ["O", "O", "O", "O"]

    def test_common_noun_with_title(self, preprocessor):
        """Common nouns with titles should be labeled."""
        story = {
            "story_id": "test",
            "text": "Dr. Lamp examined the patient.",
            "entities": [
                {"text": "Helena Lamp", "type": "PERSON", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Lamp should be B-PERSON because preceded by Dr.
        assert processed["tokens"][1] == "Lamp" and processed["bio_tags"][1] == "B-PERSON"

    def test_common_noun_in_full_name(self, preprocessor):
        """Common nouns in full names should be labeled."""
        story = {
            "story_id": "test",
            "text": "Helena Lamp examined the patient.",
            "entities": [
                {"text": "Helena Lamp", "type": "PERSON", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Helena Lamp should both be labeled
        assert processed["bio_tags"][0] == "B-PERSON"
        assert processed["bio_tags"][1] == "I-PERSON"


class TestSpaCySurnameGating:
    """
    Test spaCy-based surname licensing with dependency parsing.
    
    These tests ensure the preprocessor correctly distinguishes:
    - Functional entity mentions (actors with agency) ✅ TAG
    - Weak metadata mentions (names in passing) ❌ BLOCK
    
    Critical for preventing false positives while maintaining high recall.
    """

    def test_surname_as_subject_nsubj(self, preprocessor):
        """
        CASE: Subject of verb (nsubj dependency)
        EXAMPLE: "Holmes rushed at the door"
        JUSTIFICATION: Name is performing an action → functional entity mention → TAG
        spaCy: "Holmes" has dep_=nsubj, head="rushed" (VERB)
        """
        story = {
            "story_id": "test",
            "text": "Holmes rushed at the door.",
            "characters": [
                {"name": "Sherlock Holmes", "role": "PROTAGONIST"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Holmes should be tagged (nsubj of "rushed")
        assert processed["tokens"][0] == "Holmes"
        assert processed["bio_tags"][0] == "B-PERSON", "Subject of verb should be tagged"

    def test_surname_after_speech_verb_dobj(self, preprocessor):
        """
        CASE: Object of speech verb (subject-verb inversion in dialogue)
        EXAMPLE: "remarked Holmes quietly"
        JUSTIFICATION: Standard dialogue attribution pattern → functional entity → TAG
        spaCy: "Holmes" has dep_=dobj, head="remarked" (VERB, lemma in SPEECH_VERBS)
        """
        story = {
            "story_id": "test",
            "text": "The case was simple, remarked Holmes quietly.",
            "characters": [
                {"name": "Sherlock Holmes", "role": "PROTAGONIST"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Find Holmes in tokens
        holmes_idx = processed["tokens"].index("Holmes")
        assert processed["bio_tags"][holmes_idx] == "B-PERSON", "Surname after speech verb should be tagged"

    def test_naming_verb_blocks_single_token_name(self, preprocessor):
        """
        CASE: Object of naming verb (oprd/attr dependency)
        EXAMPLE: "To an English lawyer named Norton"
        JUSTIFICATION: Weak introductory metadata, not functional mention → BLOCK
        spaCy: "Norton" has dep_=oprd, head="named" (VERB, lemma="name")
        
        CRITICAL: This prevents tagging names that appear only as labels/identifiers
        in byline/introduction context. The functional mention comes later when
        the character actually acts: "Norton entered the room" → that one gets tagged.
        """
        story = {
            "story_id": "test",
            "text": "To an English lawyer named Norton.",
            "characters": [
                {"name": "Norton", "role": "MINOR"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Norton should NOT be tagged (oprd of naming verb "named")
        norton_idx = processed["tokens"].index("Norton")
        assert processed["bio_tags"][norton_idx] == "O", "Object of naming verb should be blocked"

    def test_naming_verb_blocks_but_later_action_tags(self, preprocessor):
        """
        CASE: Same name first introduced via naming verb, then used functionally
        EXAMPLE: "lawyer named Norton. Norton entered"
        JUSTIFICATION: First mention is metadata → BLOCK. Second is action → TAG.
        
        This validates that our naming-verb filter is position-specific (uses
        exact token alignment) and doesn't globally block all instances of a name.
        """
        story = {
            "story_id": "test",
            "text": "To a lawyer named Norton. Norton entered the room.",
            "characters": [
                {"name": "Norton", "role": "MINOR"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Find both Norton instances
        norton_indices = [i for i, tok in enumerate(processed["tokens"]) if tok == "Norton"]
        assert len(norton_indices) == 2, "Should find two Norton tokens"

        # First Norton (after "named") should be blocked
        assert processed["bio_tags"][norton_indices[0]] == "O", "First Norton (named) should be blocked"

        # Second Norton (subject of "entered") should be tagged
        assert processed["bio_tags"][norton_indices[1]] == "B-PERSON", "Second Norton (subject) should be tagged"

    def test_article_prefix_blocks_surname(self, preprocessor):
        """
        CASE: Article-prefixed surname
        EXAMPLE: "the Bennet girls", "The Bennet sisters"
        JUSTIFICATION: "the X" indicates role/descriptor, not a name → BLOCK
        
        Policy heuristic: "Can you add 'the' and it still makes sense?"
        If yes → role/title → O-tag
        """
        story = {
            "story_id": "test",
            "text": "The Bennet sisters arrived.",
            "characters": [
                {"name": "Elizabeth Bennet", "role": "PROTAGONIST"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Bennet should NOT be tagged (article-prefixed)
        bennet_idx = processed["tokens"].index("Bennet")
        assert processed["bio_tags"][bennet_idx] == "O", "Article-prefixed surname should be blocked"

    def test_title_prefix_allows_surname(self, preprocessor):
        """
        CASE: Title-prefixed surname
        EXAMPLE: "Mr. Bennet arrived"
        JUSTIFICATION: Title explicitly marks this as a name → TAG
        
        This is one of the strongest licensing contexts for standalone surnames.
        """
        story = {
            "story_id": "test",
            "text": "Mr. Bennet arrived at the door.",
            "characters": [
                {"name": "Fitzwilliam Bennet", "role": "SUPPORTING"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Bennet should be tagged (title-licensed)
        bennet_idx = processed["tokens"].index("Bennet")
        assert processed["bio_tags"][bennet_idx] == "B-PERSON", "Title-prefixed surname should be tagged"

    def test_surname_with_possessive(self, preprocessor):
        """
        CASE: Surname with possessive marker
        EXAMPLE: "Holmes's coat", "Bennet's eyes"
        JUSTIFICATION: Possessive indicates ownership/agency → functional entity → TAG
        spaCy: Token has child with dep_=poss
        """
        story = {
            "story_id": "test",
            "text": "Holmes's coat was threadbare.",
            "characters": [
                {"name": "Sherlock Holmes", "role": "PROTAGONIST"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Holmes should be tagged (has possessive marker)
        holmes_idx = processed["tokens"].index("Holmes")
        assert processed["bio_tags"][holmes_idx] == "B-PERSON", "Surname with possessive should be tagged"

    def test_first_name_prefix_allows_surname(self, preprocessor):
        """
        CASE: First name + surname combination
        EXAMPLE: "Elizabeth Bennet smiled"
        JUSTIFICATION: Multi-word names always tagged → both tokens get labels
        
        This is the most reliable pattern - full names should always be tagged.
        """
        story = {
            "story_id": "test",
            "text": "Elizabeth Bennet smiled at him.",
            "characters": [
                {"name": "Elizabeth Bennet", "role": "PROTAGONIST"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Both tokens should be tagged
        assert processed["bio_tags"][0] == "B-PERSON"
        assert processed["bio_tags"][1] == "I-PERSON"

    def test_multiple_speech_verbs(self, preprocessor):
        """
        CASE: Various speech verbs in dialogue attribution
        EXAMPLES: "said Holmes", "whispered Watson", "exclaimed Lestrade"
        JUSTIFICATION: All standard dialogue verbs should license surnames
        
        SPEECH_VERBS = {say, remark, ask, reply, answer, whisper, shout, exclaim, mutter, continue}
        """
        test_cases = [
            ("She was wrong, said Holmes.", "Holmes"),
            ("Not quite, whispered Watson.", "Watson"),
            ("Impossible, exclaimed Lestrade.", "Lestrade"),
        ]

        for text, expected_name in test_cases:
            story = {
                "story_id": "test",
                "text": text,
                "characters": [
                    {"name": f"John {expected_name}", "role": "SUPPORTING"}
                ]
            }

            processed = preprocessor.process_story(story)

            # Find the surname in tokens
            name_idx = processed["tokens"].index(expected_name)
            assert processed["bio_tags"][name_idx] == "B-PERSON", \
                f"Surname after speech verb should be tagged in: {text}"

    def test_called_vs_named_distinction(self, preprocessor):
        """
        CASE: "called" as naming verb vs. speech verb
        EXAMPLES: 
          - "a man called Norton" (naming) → BLOCK
          - "called Sherlock Holmes" (speech, multi-word name) → TAG
        
        JUSTIFICATION: Same verb, different syntactic roles
        - Naming verb: Norton is object predicate (oprd/attr)
        - Speech verb: Holmes is dobj (direct object), but full name always tags
        
        NOTE: Single-word names after "called" are ambiguous - we test the clearer cases
        """
        # Naming verb context - should block single-word name
        story1 = {
            "story_id": "test1",
            "text": "A man called Norton arrived.",
            "characters": [{"name": "Norton", "role": "MINOR"}]
        }
        processed1 = preprocessor.process_story(story1)
        norton_idx1 = processed1["tokens"].index("Norton")
        assert processed1["bio_tags"][norton_idx1] == "O", \
            "Norton in 'called Norton' (naming verb) should be blocked"

        # Speech verb with multi-word name - should tag
        story2 = {
            "story_id": "test2",
            "text": "Come here, called Sherlock Holmes.",
            "characters": [{"name": "Sherlock Holmes", "role": "PROTAGONIST"}]
        }
        processed2 = preprocessor.process_story(story2)
        sherlock_idx = processed2["tokens"].index("Sherlock")
        assert processed2["bio_tags"][sherlock_idx] == "B-PERSON"
        assert processed2["bio_tags"][sherlock_idx + 1] == "I-PERSON", \
            "Multi-word name after 'called' should be fully tagged"

    def test_article_vs_first_name_capitalized_predecessor(self, preprocessor):
        """
        CASE: Distinguishing capitalized articles from first names
        EXAMPLES:
          - "The Bennet sisters" (article) → BLOCK
          - "Elizabeth Bennet" (first name) → TAG
        
        JUSTIFICATION: Check #5 in surname licensing requires non-article predecessor
        ARTICLES = {"the", "a", "an"} (case-insensitive check)
        """
        # Article case
        story1 = {
            "story_id": "test1",
            "text": "The Bennet family was large.",
            "characters": [{"name": "Elizabeth Bennet", "role": "PROTAGONIST"}]
        }
        processed1 = preprocessor.process_story(story1)
        bennet_idx1 = processed1["tokens"].index("Bennet")
        assert processed1["bio_tags"][bennet_idx1] == "O", \
            "'The Bennet' should be blocked (article prefix)"

        # First name case
        story2 = {
            "story_id": "test2",
            "text": "Elizabeth Bennet was clever.",
            "characters": [{"name": "Elizabeth Bennet", "role": "PROTAGONIST"}]
        }
        processed2 = preprocessor.process_story(story2)
        assert processed2["bio_tags"][0] == "B-PERSON"
        assert processed2["bio_tags"][1] == "I-PERSON", \
            "'Elizabeth Bennet' should be fully tagged (first name + surname)"


class TestCommonNounHandling:
    """Test handling of common nouns that can be surnames."""

    def test_common_noun_lowercase(self, preprocessor):
        """Lowercase common nouns should not be labeled."""
        story = {
            "story_id": "test",
            "text": "The lamp flickered.",
            "entities": [
                {"text": "Helena Lamp", "type": "PERSON", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        # 'lamp' should be O
        assert processed["bio_tags"] == ["O", "O", "O", "O"]

    def test_common_noun_with_title(self, preprocessor):
        """Common nouns with titles should be labeled."""
        story = {
            "story_id": "test",
            "text": "Dr. Lamp examined the patient.",
            "entities": [
                {"text": "Helena Lamp", "type": "PERSON", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Lamp should be B-PERSON because preceded by Dr.
        assert processed["tokens"][1] == "Lamp" and processed["bio_tags"][1] == "B-PERSON"

    def test_common_noun_in_full_name(self, preprocessor):
        """Common nouns in full names should be labeled."""
        story = {
            "story_id": "test",
            "text": "Helena Lamp examined the patient.",
            "entities": [
                {"text": "Helena Lamp", "type": "PERSON", "role": "protagonist"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Helena Lamp should both be labeled
        assert processed["bio_tags"][0] == "B-PERSON"
        assert processed["bio_tags"][1] == "I-PERSON"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
