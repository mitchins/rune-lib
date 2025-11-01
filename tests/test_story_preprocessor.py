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
    """Create a preprocessor instance without spaCy for consistent testing."""
    return StoryPreprocessor(use_spacy=False)


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

        # Mr. and . should be O, Bennet should be B-protagonist
        assert processed["tokens"] == ["Mr", ".", "Bennet", "arrived", "at", "the", "door", "."]
        assert processed["bio_tags"] == ["O", "O", "B-protagonist", "O", "O", "O", "O", "O"]

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

        assert processed["bio_tags"] == ["O", "O", "B-protagonist", "O", "O", "O", "O"]

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

        # Cook should be B-supporting because preceded by title
        assert processed["bio_tags"] == ["O", "O", "B-supporting", "O", "O", "O"]

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

        assert processed["bio_tags"] == ["B-supporting", "I-supporting", "O", "O", "O"]

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

        assert processed["bio_tags"] == ["O", "B-supporting", "O", "O", "B-supporting", "O", "O"]

    def test_sentence_start_capitalization_ambiguous(self, preprocessor):
        """Capitalized surnames at sentence start should be skipped if ambiguous."""
        story = {
            "story_id": "test7",
            "text": "The door opened. Cook arrived with supplies.",
            "entities": [
                {"text": "James Cook", "type": "PERSON", "role": "supporting"}
            ]
        }

        processed = preprocessor.process_story(story)

        # Cook at sentence start without title should be skipped
        assert processed["bio_tags"] == ["O", "O", "O", "O", "O", "O", "O", "O", "O"]

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

        # Cook should be labeled because preceded by Mr.
        assert processed["bio_tags"] == ["O", "O", "O", "O", "O", "B-supporting", "O", "O", "O", "O"]


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
        assert entities[0]["role"] == "protagonist"
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
        assert entities[0]["role"] == "supporting"

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

        assert processed["bio_tags"] == ["B-protagonist", "I-protagonist", "O", "O"]

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

        assert processed["bio_tags"] == ["B-protagonist", "I-protagonist", "O", "O"]

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
        assert processed["bio_tags"] == ["B-protagonist", "I-protagonist", "O", "O", "O"]

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

        # Check each character has correct role
        assert "B-protagonist" in processed["bio_tags"]
        assert "B-supporting" in processed["bio_tags"]
        assert "B-antagonist" in processed["bio_tags"]


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

        # Lamp should be B-protagonist because preceded by Dr.
        assert processed["bio_tags"][2] == "B-protagonist"

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
        assert processed["bio_tags"][0] == "B-protagonist"
        assert processed["bio_tags"][1] == "I-protagonist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
