"""Tests for coreference resolution utilities."""

import pytest
from rune.core.coreference import (
    EntityMention,
    CoreferenceCluster,
    SimpleCoreferenceResolver,
    CoreferenceHook,
    create_coreference_report,
)
from rune.core.schema import BioSequence
from rune.core.normalization import EntityNormalizer


class TestEntityMention:
    """Test EntityMention dataclass."""

    def test_create_mention(self):
        """Test creating an entity mention."""
        mention = EntityMention(
            mention_id="s0_e0",
            entity_id="cluster_0",
            surface_form="Dr. Smith",
            canonical_form="Smith",
            entity_type="PERSON",
            sentence_idx=0,
            token_span=(0, 1),
            confidence=0.95,
        )

        assert mention.mention_id == "s0_e0"
        assert mention.entity_id == "cluster_0"
        assert mention.surface_form == "Dr. Smith"
        assert mention.canonical_form == "Smith"
        assert mention.entity_type == "PERSON"
        assert mention.confidence == 0.95


class TestCoreferenceCluster:
    """Test CoreferenceCluster class."""

    def test_create_cluster(self):
        """Test creating a coreference cluster."""
        cluster = CoreferenceCluster(
            cluster_id="cluster_0",
            entity_type="PERSON",
            canonical_name="Smith",
        )

        assert cluster.cluster_id == "cluster_0"
        assert cluster.entity_type == "PERSON"
        assert cluster.canonical_name == "Smith"
        assert len(cluster.mentions) == 0

    def test_add_mention(self):
        """Test adding mentions to cluster."""
        cluster = CoreferenceCluster(
            cluster_id="cluster_0",
            entity_type="PERSON",
            canonical_name="Smith",
        )

        mention = EntityMention(
            mention_id="s0_e0",
            entity_id=None,
            surface_form="Dr. Smith",
            canonical_form="Smith",
            entity_type="PERSON",
            sentence_idx=0,
            token_span=(0, 1),
        )

        cluster.add_mention(mention)

        assert len(cluster.mentions) == 1
        assert mention.entity_id == "cluster_0"
        assert cluster.mentions[0] == mention

    def test_get_most_complete_form(self):
        """Test getting most complete entity form."""
        cluster = CoreferenceCluster(
            cluster_id="cluster_0",
            entity_type="PERSON",
            canonical_name="Smith",
        )

        # Add mentions of varying lengths
        mention1 = EntityMention(
            mention_id="s0_e0",
            entity_id="cluster_0",
            surface_form="Smith",
            canonical_form="Smith",
            entity_type="PERSON",
            sentence_idx=0,
            token_span=(0, 0),
        )

        mention2 = EntityMention(
            mention_id="s1_e0",
            entity_id="cluster_0",
            surface_form="Dr. John Smith",
            canonical_form="John Smith",
            entity_type="PERSON",
            sentence_idx=1,
            token_span=(0, 2),
        )

        cluster.add_mention(mention1)
        cluster.add_mention(mention2)

        # Should return the longer form
        assert cluster.get_most_complete_form() == "Dr. John Smith"


class TestSimpleCoreferenceResolver:
    """Test SimpleCoreferenceResolver class."""

    def create_sample_mentions(self):
        """Create sample mentions for testing."""
        mentions = [
            EntityMention(
                mention_id="s0_e0",
                entity_id=None,
                surface_form="Dr. Smith",
                canonical_form="Smith",
                entity_type="PERSON",
                sentence_idx=0,
                token_span=(0, 1),
            ),
            EntityMention(
                mention_id="s1_e0",
                entity_id=None,
                surface_form="Smith",
                canonical_form="Smith",
                entity_type="PERSON",
                sentence_idx=1,
                token_span=(5, 5),
            ),
            EntityMention(
                mention_id="s2_e0",
                entity_id=None,
                surface_form="John Smith",
                canonical_form="John Smith",
                entity_type="PERSON",
                sentence_idx=2,
                token_span=(0, 1),
            ),
            EntityMention(
                mention_id="s3_e0",
                entity_id=None,
                surface_form="Google",
                canonical_form="Google",
                entity_type="ORG",
                sentence_idx=3,
                token_span=(0, 0),
            ),
        ]
        return mentions

    def test_exact_matching(self):
        """Test exact canonical form matching."""
        resolver = SimpleCoreferenceResolver(match_partial_names=False)
        mentions = self.create_sample_mentions()

        clusters = resolver.resolve(mentions)

        # Should have 3 clusters: Smith (2 mentions), John Smith (1), Google (1)
        assert len(clusters) == 3

        # Find Smith cluster
        smith_cluster = next(c for c in clusters if c.canonical_name == "Smith")
        assert len(smith_cluster.mentions) == 2

    def test_partial_name_matching(self):
        """Test partial name matching for PERSON entities."""
        resolver = SimpleCoreferenceResolver(match_partial_names=True)
        mentions = self.create_sample_mentions()

        clusters = resolver.resolve(mentions)

        # With partial matching, Smith and John Smith should be in same cluster
        # Should have 2 clusters: Smith group (3 mentions), Google (1)
        assert len(clusters) == 2

        # Find the person cluster
        person_cluster = next(c for c in clusters if c.entity_type == "PERSON")
        assert len(person_cluster.mentions) == 3

    def test_different_entity_types(self):
        """Test that different entity types don't cluster together."""
        resolver = SimpleCoreferenceResolver()

        mentions = [
            EntityMention(
                mention_id="s0_e0",
                entity_id=None,
                surface_form="Apple",
                canonical_form="Apple",
                entity_type="ORG",
                sentence_idx=0,
                token_span=(0, 0),
            ),
            EntityMention(
                mention_id="s1_e0",
                entity_id=None,
                surface_form="Apple",
                canonical_form="Apple",
                entity_type="PRODUCT",  # Different entity type
                sentence_idx=1,
                token_span=(0, 0),
            ),
        ]

        clusters = resolver.resolve(mentions)

        # Should have 2 separate clusters despite same surface form
        assert len(clusters) == 2


class TestCoreferenceHook:
    """Test CoreferenceHook class."""

    def test_process_sequence(self):
        """Test processing a sequence to extract mentions."""
        hook = CoreferenceHook()

        sequence = BioSequence.from_iob_format(
            ["Dr", "Smith", "met", "with", "Jones"],
            ["B-PERSON", "I-PERSON", "O", "O", "B-PERSON"]
        )

        entities = [
            {"text": "Dr Smith", "label": "PERSON", "start": 0, "end": 1},
            {"text": "Jones", "label": "PERSON", "start": 4, "end": 4},
        ]

        mentions = hook.process_sequence(sequence, entities, sentence_idx=0)

        assert len(mentions) == 2
        assert mentions[0].mention_id == "s0_e0"
        assert mentions[0].surface_form == "Dr Smith"
        assert mentions[0].sentence_idx == 0
        assert mentions[1].surface_form == "Jones"

    def test_resolve_document(self):
        """Test resolving coreferences across a document."""
        hook = CoreferenceHook()

        sequences = [
            BioSequence.from_iob_format(
                ["Dr", "Smith", "arrived"],
                ["B-PERSON", "I-PERSON", "O"]
            ),
            BioSequence.from_iob_format(
                ["Smith", "was", "late"],
                ["B-PERSON", "O", "O"]
            ),
        ]

        all_entities = [
            [{"text": "Dr Smith", "label": "PERSON", "start": 0, "end": 1}],
            [{"text": "Smith", "label": "PERSON", "start": 0, "end": 0}],
        ]

        clusters = hook.resolve_document(sequences, all_entities)

        # Should have 1 cluster with 2 mentions
        assert len(clusters) == 1
        assert len(clusters[0].mentions) == 2


class TestCoreferenceReport:
    """Test coreference report generation."""

    def test_create_report(self):
        """Test creating a human-readable report."""
        cluster = CoreferenceCluster(
            cluster_id="cluster_0",
            entity_type="PERSON",
            canonical_name="Smith",
        )

        mention1 = EntityMention(
            mention_id="s0_e0",
            entity_id="cluster_0",
            surface_form="Dr. Smith",
            canonical_form="Smith",
            entity_type="PERSON",
            sentence_idx=0,
            token_span=(0, 1),
        )

        mention2 = EntityMention(
            mention_id="s1_e0",
            entity_id="cluster_0",
            surface_form="Smith",
            canonical_form="Smith",
            entity_type="PERSON",
            sentence_idx=1,
            token_span=(0, 0),
        )

        cluster.add_mention(mention1)
        cluster.add_mention(mention2)

        report = create_coreference_report([cluster])

        assert "Dr. Smith (PERSON)" in report
        assert "Canonical: Smith" in report
        assert "Mentions (2)" in report
        assert "sentence 0" in report
        assert "sentence 1" in report