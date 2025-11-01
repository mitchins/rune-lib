"""Tests for schema module."""

import pytest
from rune.core.schema import BioTag, BioSequence, TagSchema, BioPrefix


class TestBioTag:
    """Test BioTag class."""

    def test_create_outside_tag(self):
        tag = BioTag.outside()
        assert tag.prefix == BioPrefix.O
        assert tag.entity_type is None
        assert str(tag) == "O"

    def test_create_beginning_tag(self):
        tag = BioTag.beginning("PERSON")
        assert tag.prefix == BioPrefix.B
        assert tag.entity_type == "PERSON"
        assert str(tag) == "B-PERSON"

    def test_create_inside_tag(self):
        tag = BioTag.inside("LOCATION")
        assert tag.prefix == BioPrefix.I
        assert tag.entity_type == "LOCATION"
        assert str(tag) == "I-LOCATION"

    def test_from_string(self):
        # Test O tag
        tag = BioTag.from_string("O")
        assert tag.prefix == BioPrefix.O
        assert tag.entity_type is None

        # Test B tag
        tag = BioTag.from_string("B-PERSON")
        assert tag.prefix == BioPrefix.B
        assert tag.entity_type == "PERSON"

        # Test I tag
        tag = BioTag.from_string("I-LOCATION")
        assert tag.prefix == BioPrefix.I
        assert tag.entity_type == "LOCATION"

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            BioTag.from_string("INVALID")

    def test_validation(self):
        # O tag with entity type should fail
        with pytest.raises(ValueError):
            BioTag(prefix=BioPrefix.O, entity_type="PERSON")

        # B tag without entity type should fail
        with pytest.raises(ValueError):
            BioTag(prefix=BioPrefix.B, entity_type=None)

        # I tag without entity type should fail
        with pytest.raises(ValueError):
            BioTag(prefix=BioPrefix.I, entity_type=None)


class TestBioSequence:
    """Test BioSequence class."""

    def test_create_sequence(self):
        tokens = ["John", "Doe"]
        tags = [BioTag.beginning("PERSON"), BioTag.inside("PERSON")]
        sequence = BioSequence(tokens=tokens, tags=tags)

        assert sequence.tokens == tokens
        assert sequence.tags == tags

    def test_mismatched_length(self):
        tokens = ["John", "Doe"]
        tags = [BioTag.beginning("PERSON")]  # Only one tag for two tokens

        with pytest.raises(ValueError):
            BioSequence(tokens=tokens, tags=tags)

    def test_to_iob_format(self):
        tokens = ["John", "Doe", "lives", "here"]
        tags = [
            BioTag.beginning("PERSON"),
            BioTag.inside("PERSON"),
            BioTag.outside(),
            BioTag.outside(),
        ]
        sequence = BioSequence(tokens=tokens, tags=tags)

        iob_format = sequence.to_iob_format()
        expected = ["B-PERSON", "I-PERSON", "O", "O"]
        assert iob_format == expected

    def test_from_iob_format(self):
        tokens = ["John", "Doe", "lives", "here"]
        iob_tags = ["B-PERSON", "I-PERSON", "O", "O"]

        sequence = BioSequence.from_iob_format(tokens, iob_tags)

        assert sequence.tokens == tokens
        assert len(sequence.tags) == len(tokens)
        assert str(sequence.tags[0]) == "B-PERSON"
        assert str(sequence.tags[1]) == "I-PERSON"
        assert str(sequence.tags[2]) == "O"
        assert str(sequence.tags[3]) == "O"


class TestTagSchema:
    """Test TagSchema class."""

    def test_create_schema(self):
        entity_types = ["PERSON", "LOCATION"]
        schema = TagSchema.create_standard_schema(entity_types, "Test schema")

        assert schema.entity_types == entity_types
        assert schema.description == "Test schema"

    def test_duplicate_entity_types(self):
        with pytest.raises(ValueError):
            TagSchema(entity_types=["PERSON", "PERSON"])

    def test_get_all_tags(self):
        schema = TagSchema.create_standard_schema(["PERSON", "LOCATION"])
        all_tags = schema.get_all_tags()

        expected = ["O", "B-PERSON", "I-PERSON", "B-LOCATION", "I-LOCATION"]
        assert all_tags == expected

    def test_get_mappings(self):
        schema = TagSchema.create_standard_schema(["PERSON"])

        tag_to_id = schema.get_tag_to_id_mapping()
        id_to_tag = schema.get_id_to_tag_mapping()

        expected_tags = ["O", "B-PERSON", "I-PERSON"]
        assert list(tag_to_id.keys()) == expected_tags
        assert list(id_to_tag.values()) == expected_tags

        # Check bidirectional mapping
        for tag, tag_id in tag_to_id.items():
            assert id_to_tag[tag_id] == tag

    def test_protagonist_schema(self):
        schema = TagSchema.create_protagonist_schema()

        assert schema.entity_types == ["PROTAGONIST"]
        assert "protagonist" in schema.description.lower()

        all_tags = schema.get_all_tags()
        expected = ["O", "B-PROTAGONIST", "I-PROTAGONIST"]
        assert all_tags == expected