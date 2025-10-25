"""Coreference resolution utilities and hooks."""

from typing import List, Dict, Optional, Set, Tuple, Any, Protocol
from dataclasses import dataclass, field
from collections import defaultdict

from .schema import BioSequence
from .normalization import NormalizedEntity, EntityNormalizer


@dataclass
class EntityMention:
    """Represents a mention of an entity with coreference information."""
    mention_id: str
    entity_id: Optional[str]  # Cluster ID for coreference
    surface_form: str
    canonical_form: str
    entity_type: str
    sentence_idx: int
    token_span: Tuple[int, int]
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreferenceCluster:
    """Represents a cluster of coreferent entity mentions."""
    cluster_id: str
    entity_type: str
    canonical_name: str
    mentions: List[EntityMention] = field(default_factory=list)

    def add_mention(self, mention: EntityMention) -> None:
        """Add a mention to this cluster."""
        mention.entity_id = self.cluster_id
        self.mentions.append(mention)

    def get_most_complete_form(self) -> str:
        """Get the most complete/informative form of the entity."""
        # Prefer longer forms as they're often more complete
        if not self.mentions:
            return self.canonical_name

        longest = max(self.mentions, key=lambda m: len(m.surface_form))
        return longest.surface_form


class CoreferenceResolver(Protocol):
    """Protocol for pluggable coreference resolution systems."""

    def resolve(
        self,
        mentions: List[EntityMention],
        context: Optional[Any] = None,
    ) -> List[CoreferenceCluster]:
        """Resolve coreferences among mentions."""
        ...


class SimpleCoreferenceResolver:
    """Simple rule-based coreference resolution."""

    def __init__(
        self,
        normalizer: Optional[EntityNormalizer] = None,
        match_partial_names: bool = True,
        match_pronouns: bool = False,
    ):
        """
        Initialize simple coreference resolver.

        Args:
            normalizer: Entity normalizer to use
            match_partial_names: Match partial names (e.g., "Smith" to "John Smith")
            match_pronouns: Attempt pronoun resolution (experimental)
        """
        self.normalizer = normalizer or EntityNormalizer()
        self.match_partial_names = match_partial_names
        self.match_pronouns = match_pronouns

    def resolve(
        self,
        mentions: List[EntityMention],
        context: Optional[Any] = None,
    ) -> List[CoreferenceCluster]:
        """
        Resolve coreferences using simple rules.

        Args:
            mentions: List of entity mentions
            context: Optional context (unused in simple resolver)

        Returns:
            List of coreference clusters
        """
        clusters = []
        mention_to_cluster = {}

        for mention in mentions:
            # Find matching cluster
            matched_cluster = self._find_matching_cluster(
                mention, clusters, mention_to_cluster
            )

            if matched_cluster:
                matched_cluster.add_mention(mention)
                mention_to_cluster[mention.mention_id] = matched_cluster
            else:
                # Create new cluster
                cluster = CoreferenceCluster(
                    cluster_id=f"cluster_{len(clusters)}",
                    entity_type=mention.entity_type,
                    canonical_name=mention.canonical_form,
                )
                cluster.add_mention(mention)
                clusters.append(cluster)
                mention_to_cluster[mention.mention_id] = cluster

        return clusters

    def _find_matching_cluster(
        self,
        mention: EntityMention,
        clusters: List[CoreferenceCluster],
        mention_to_cluster: Dict[str, CoreferenceCluster],
    ) -> Optional[CoreferenceCluster]:
        """Find a matching cluster for a mention."""
        for cluster in clusters:
            # Must be same entity type
            if cluster.entity_type != mention.entity_type:
                continue

            # Check exact canonical match
            if cluster.canonical_name == mention.canonical_form:
                return cluster

            # Check partial name matching for PERSON entities
            if self.match_partial_names and mention.entity_type == "PERSON":
                if self._is_partial_name_match(
                    mention.canonical_form,
                    cluster.canonical_name
                ):
                    return cluster

        return None

    def _is_partial_name_match(self, name1: str, name2: str) -> bool:
        """Check if names partially match."""
        parts1 = name1.lower().split()
        parts2 = name2.lower().split()

        # Check if all parts of shorter name are in longer name
        if len(parts1) < len(parts2):
            return all(part in parts2 for part in parts1)
        elif len(parts2) < len(parts1):
            return all(part in parts1 for part in parts2)

        return False


class CoreferenceHook:
    """Hook system for integrating with external coreference resolvers."""

    def __init__(self):
        """Initialize coreference hook."""
        self.normalizer = EntityNormalizer()
        self.resolver: Optional[CoreferenceResolver] = None

    def set_resolver(self, resolver: CoreferenceResolver) -> None:
        """Set the coreference resolver to use."""
        self.resolver = resolver

    def process_sequence(
        self,
        sequence: BioSequence,
        entities: List[Dict],
        sentence_idx: int = 0,
    ) -> List[EntityMention]:
        """
        Process a sequence to extract entity mentions.

        Args:
            sequence: Bio-tagged sequence
            entities: Extracted entities from the sequence
            sentence_idx: Index of this sentence in document

        Returns:
            List of entity mentions ready for coreference
        """
        mentions = []

        # Normalize entities first
        normalized = self.normalizer.normalize_entities(entities)

        for i, norm_entity in enumerate(normalized):
            mention = EntityMention(
                mention_id=f"s{sentence_idx}_e{i}",
                entity_id=None,  # Will be set by resolver
                surface_form=norm_entity.surface_form,
                canonical_form=norm_entity.canonical_form,
                entity_type=norm_entity.entity_type,
                sentence_idx=sentence_idx,
                token_span=norm_entity.span or (0, 0),
                attributes={
                    "salutation": norm_entity.salutation,
                    "possessive": norm_entity.possessive,
                    "lemma": norm_entity.lemma,
                }
            )
            mentions.append(mention)

        return mentions

    def resolve_document(
        self,
        sequences: List[BioSequence],
        all_entities: List[List[Dict]],
    ) -> List[CoreferenceCluster]:
        """
        Resolve coreferences across a document.

        Args:
            sequences: List of bio-tagged sequences (sentences)
            all_entities: List of entity lists for each sequence

        Returns:
            List of coreference clusters
        """
        # Extract all mentions
        all_mentions = []
        for i, (seq, entities) in enumerate(zip(sequences, all_entities)):
            mentions = self.process_sequence(seq, entities, i)
            all_mentions.extend(mentions)

        # Use resolver if available, otherwise use simple resolver
        if self.resolver:
            clusters = self.resolver.resolve(all_mentions)
        else:
            simple_resolver = SimpleCoreferenceResolver(self.normalizer)
            clusters = simple_resolver.resolve(all_mentions)

        return clusters


def create_coreference_report(clusters: List[CoreferenceCluster]) -> str:
    """
    Create a human-readable coreference report.

    Args:
        clusters: List of coreference clusters

    Returns:
        Formatted report string
    """
    report_lines = ["=== Coreference Report ===\n"]

    for cluster in clusters:
        report_lines.append(f"Entity: {cluster.get_most_complete_form()} ({cluster.entity_type})")
        report_lines.append(f"  Canonical: {cluster.canonical_name}")
        report_lines.append(f"  Mentions ({len(cluster.mentions)}):")

        for mention in cluster.mentions:
            report_lines.append(
                f"    - \"{mention.surface_form}\" (sentence {mention.sentence_idx})"
            )
        report_lines.append("")

    return "\n".join(report_lines)