"""Tests for entity normalization and lemmatization."""

import pytest
from rune.core.normalization import (
    EntityNormalizer,
    NormalizedEntity,
    SalutationType,
    NameVariantGenerator,
)


class TestEntityNormalizer:
    """Test EntityNormalizer class."""

    def test_basic_normalization(self):
        """Test basic entity normalization."""
        normalizer = EntityNormalizer()

        entity = normalizer.normalize_entity("John Smith", "PERSON")

        assert entity.surface_form == "John Smith"
        assert entity.canonical_form == "John Smith"
        assert entity.lemma == "John Smith"
        assert entity.entity_type == "PERSON"
        assert entity.salutation is None
        assert not entity.possessive

    def test_possessive_handling(self):
        """Test possessive marker removal."""
        normalizer = EntityNormalizer(remove_possessives=True)

        # Test 's removal
        entity = normalizer.normalize_entity("Lenny's", "PERSON")
        assert entity.surface_form == "Lenny's"
        assert entity.canonical_form == "Lenny"
        assert entity.lemma == "Lenny"
        assert entity.possessive

        # Test ' removal (plural possessive)
        entity = normalizer.normalize_entity("friends'", "PERSON")
        assert entity.surface_form == "friends'"
        assert entity.canonical_form == "friends"
        assert entity.lemma == "friends"
        assert entity.possessive

    def test_salutation_handling(self):
        """Test salutation extraction."""
        normalizer = EntityNormalizer()

        # Test Mr.
        entity = normalizer.normalize_entity("Mr. Smith", "PERSON")
        assert entity.surface_form == "Mr. Smith"
        assert entity.canonical_form == "Smith"
        assert entity.lemma == "Smith"
        assert entity.salutation == "Mr"

        # Test Dr without period
        entity = normalizer.normalize_entity("Dr Johnson", "PERSON")
        assert entity.surface_form == "Dr Johnson"
        assert entity.canonical_form == "Johnson"
        assert entity.salutation == "Dr"

        # Test case insensitive
        entity = normalizer.normalize_entity("mrs. Williams", "PERSON")
        assert entity.canonical_form == "Williams"
        assert entity.salutation == "mrs"

    def test_combined_salutation_and_possessive(self):
        """Test combined salutation and possessive handling."""
        normalizer = EntityNormalizer()

        entity = normalizer.normalize_entity("Dr. Smith's", "PERSON")
        assert entity.surface_form == "Dr. Smith's"
        assert entity.canonical_form == "Smith"
        assert entity.lemma == "Smith"
        assert entity.salutation == "Dr"
        assert entity.possessive

    def test_non_person_entities(self):
        """Test that salutations aren't extracted from non-person entities."""
        normalizer = EntityNormalizer()

        # Organization with "Dr" in name
        entity = normalizer.normalize_entity("Dr. Pepper Corp", "ORG")
        assert entity.canonical_form == "Dr. Pepper Corp"  # No salutation removal
        assert entity.salutation is None

    def test_custom_salutations(self):
        """Test custom salutation lists."""
        custom_salutations = {"Prof", "Wizard", "Lord"}
        normalizer = EntityNormalizer(salutations=custom_salutations)

        entity = normalizer.normalize_entity("Wizard Gandalf", "PERSON")
        assert entity.canonical_form == "Gandalf"
        assert entity.salutation == "Wizard"

        # Default salutations shouldn't work
        entity = normalizer.normalize_entity("Mr. Smith", "PERSON")
        assert entity.canonical_form == "Mr. Smith"  # Not removed
        assert entity.salutation is None

    def test_custom_rules(self):
        """Test custom normalization rules."""
        custom_rules = {"Lenny's office": "Leonard's office"}
        normalizer = EntityNormalizer(custom_rules=custom_rules)

        entity = normalizer.normalize_entity("Lenny's office", "LOCATION")
        assert entity.canonical_form == "Leonard's office"
        assert entity.lemma == "Leonard's office"

    def test_lowercase_matching(self):
        """Test lowercase canonical forms."""
        normalizer = EntityNormalizer(lowercase_matching=True)

        entity = normalizer.normalize_entity("John SMITH", "PERSON")
        assert entity.surface_form == "John SMITH"
        assert entity.canonical_form == "john smith"
        assert entity.lemma == "john smith"

    def test_normalize_entities_list(self):
        """Test normalizing a list of entities."""
        normalizer = EntityNormalizer()

        entities = [
            {"text": "Dr. Smith", "label": "PERSON", "start": 0, "end": 1},
            {"text": "Lenny's", "label": "PERSON", "start": 3, "end": 3},
        ]

        normalized = normalizer.normalize_entities(entities)

        assert len(normalized) == 2
        assert normalized[0].canonical_form == "Smith"
        assert normalized[0].span == (0, 1)
        assert normalized[1].canonical_form == "Lenny"
        assert normalized[1].possessive

    def test_group_by_canonical(self):
        """Test grouping entities by canonical form."""
        normalizer = EntityNormalizer()

        entities = [
            normalizer.normalize_entity("Dr. Smith", "PERSON"),
            normalizer.normalize_entity("Smith", "PERSON"),
            normalizer.normalize_entity("Mr. Jones", "PERSON"),
        ]

        groups = normalizer.group_by_canonical(entities)

        # Should have 2 groups: Smith and Jones
        assert len(groups) == 2
        assert ("Smith", "PERSON") in groups
        assert ("Jones", "PERSON") in groups
        assert len(groups[("Smith", "PERSON")]) == 2  # Dr. Smith and Smith

    def test_find_coreferent_candidates(self):
        """Test finding coreferent candidates."""
        normalizer = EntityNormalizer()

        target = normalizer.normalize_entity("Smith", "PERSON")
        candidates = [
            normalizer.normalize_entity("Dr. Smith", "PERSON"),
            normalizer.normalize_entity("John Smith", "PERSON"),
            normalizer.normalize_entity("Jones", "PERSON"),
            normalizer.normalize_entity("Google", "ORG"),
        ]

        # Exact matching
        matches = normalizer.find_coreferent_candidates(target, candidates)
        assert len(matches) == 1  # Only Dr. Smith (same canonical form)

        # Fuzzy matching
        matches = normalizer.find_coreferent_candidates(target, candidates, fuzzy_match=True)
        assert len(matches) == 2  # Dr. Smith and John Smith


class TestNameVariantGenerator:
    """Test NameVariantGenerator class."""

    def test_get_canonical_name(self):
        """Test getting canonical form of names."""
        generator = NameVariantGenerator()

        assert generator.get_canonical_name("Bob") == "Robert"
        assert generator.get_canonical_name("Lenny") == "Leonard"
        assert generator.get_canonical_name("John") == "John"  # No variant

    def test_get_variants(self):
        """Test getting all variants of a name."""
        generator = NameVariantGenerator()

        variants = generator.get_variants("Robert")
        assert "Robert" in variants
        assert "Bob" in variants
        assert "Bobby" in variants

        variants = generator.get_variants("Lenny")
        assert "Leonard" in variants
        assert "Len" in variants
        assert "Lenny" in variants

    def test_are_name_variants(self):
        """Test checking if names are variants."""
        generator = NameVariantGenerator()

        assert generator.are_name_variants("Bob", "Robert")
        assert generator.are_name_variants("Lenny", "Leonard")
        assert generator.are_name_variants("Robert", "Bobby")
        assert not generator.are_name_variants("John", "Bob")

    def test_case_insensitive(self):
        """Test case insensitive variant matching."""
        generator = NameVariantGenerator()

        assert generator.are_name_variants("bob", "ROBERT")
        assert generator.get_canonical_name("lenny") == "Leonard"


class TestSalutationType:
    """Test SalutationType enum."""

    def test_default_salutations(self):
        """Test default salutation set."""
        defaults = SalutationType.get_default_set()

        assert "Mr" in defaults
        assert "Dr" in defaults
        assert "Prof" in defaults
        assert len(defaults) > 5  # Should have several salutations

    def test_enum_values(self):
        """Test enum values."""
        assert SalutationType.MR.value == "Mr"
        assert SalutationType.DR.value == "Dr"
        assert SalutationType.PROF.value == "Prof"


class TestNormalizedEntity:
    """Test NormalizedEntity dataclass."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        entity = NormalizedEntity(
            surface_form="Dr. Smith's",
            canonical_form="Smith",
            lemma="Smith",
            entity_type="PERSON",
            salutation="Dr",
            possessive=True,
            span=(0, 2)
        )

        result = entity.to_dict()

        assert result["surface_form"] == "Dr. Smith's"
        assert result["canonical_form"] == "Smith"
        assert result["lemma"] == "Smith"
        assert result["entity_type"] == "PERSON"
        assert result["salutation"] == "Dr"
        assert result["possessive"] is True
        assert result["span"] == (0, 2)