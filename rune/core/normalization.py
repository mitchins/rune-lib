"""Entity normalization and lemmatization utilities."""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import re
from enum import Enum


class SalutationType(str, Enum):
    """Common salutation types."""
    MR = "Mr"
    MRS = "Mrs"
    MS = "Ms"
    DR = "Dr"
    PROF = "Prof"
    SIR = "Sir"
    DAME = "Dame"
    LORD = "Lord"
    LADY = "Lady"

    @classmethod
    def get_default_set(cls) -> Set[str]:
        """Get default salutations."""
        return {s.value for s in cls}


@dataclass
class NormalizedEntity:
    """Represents a normalized entity with surface and canonical forms."""
    surface_form: str  # Original text as it appears
    canonical_form: str  # Normalized/lemmatized form
    lemma: str  # Base lemma (e.g., "Lenny" from "Lenny's")
    entity_type: str
    salutation: Optional[str] = None
    possessive: bool = False
    span: Optional[Tuple[int, int]] = None  # Token indices

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "surface_form": self.surface_form,
            "canonical_form": self.canonical_form,
            "lemma": self.lemma,
            "entity_type": self.entity_type,
            "salutation": self.salutation,
            "possessive": self.possessive,
            "span": self.span,
        }


class EntityNormalizer:
    """Normalizes and lemmatizes entities for better coreference resolution."""

    def __init__(
        self,
        salutations: Optional[Set[str]] = None,
        remove_possessives: bool = True,
        lowercase_matching: bool = False,
        custom_rules: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize normalizer.

        Args:
            salutations: Set of salutation prefixes to recognize
            remove_possessives: Strip possessive markers ('s, ')
            lowercase_matching: Use lowercase for canonical matching
            custom_rules: Custom normalization rules (surface -> canonical)
        """
        self.salutations = salutations or SalutationType.get_default_set()
        self.remove_possessives = remove_possessives
        self.lowercase_matching = lowercase_matching
        self.custom_rules = custom_rules or {}

        # Compile regex patterns
        self.possessive_pattern = re.compile(r"'s?$")
        self.salutation_pattern = self._compile_salutation_pattern()

    def _compile_salutation_pattern(self) -> re.Pattern:
        """Compile regex for salutation detection."""
        salutations_regex = "|".join(re.escape(s) for s in self.salutations)
        # Match salutation with optional period, followed by space
        return re.compile(rf"^({salutations_regex})\.?\s+", re.IGNORECASE)

    def normalize_entity(self, text: str, entity_type: str) -> NormalizedEntity:
        """
        Normalize a single entity.

        Args:
            text: Entity text to normalize
            entity_type: Type of entity (PERSON, ORG, etc.)

        Returns:
            NormalizedEntity with surface and canonical forms
        """
        surface_form = text
        canonical_form = text
        lemma = text
        salutation = None
        possessive = False

        # Check custom rules first
        if text in self.custom_rules:
            canonical_form = self.custom_rules[text]
            lemma = canonical_form
        else:
            # Extract salutation if present
            salutation_match = self.salutation_pattern.match(text)
            if salutation_match and entity_type == "PERSON":
                salutation = salutation_match.group(1)
                # Remove salutation from canonical form
                canonical_form = text[salutation_match.end():]
                lemma = canonical_form

            # Handle possessives
            if self.remove_possessives and self.possessive_pattern.search(lemma):
                possessive = True
                lemma = self.possessive_pattern.sub("", lemma)
                if canonical_form == surface_form:  # No salutation was removed
                    canonical_form = lemma

            # Apply lowercase matching if configured
            if self.lowercase_matching:
                canonical_form = canonical_form.lower()
                lemma = lemma.lower()

        return NormalizedEntity(
            surface_form=surface_form,
            canonical_form=canonical_form,
            lemma=lemma,
            entity_type=entity_type,
            salutation=salutation,
            possessive=possessive,
        )

    def normalize_entities(
        self,
        entities: List[Dict],
    ) -> List[NormalizedEntity]:
        """
        Normalize a list of entities.

        Args:
            entities: List of entity dictionaries from BioTagger

        Returns:
            List of normalized entities
        """
        normalized = []
        for entity in entities:
            norm_entity = self.normalize_entity(
                entity["text"],
                entity["label"]
            )
            # Preserve span information if available
            if "start" in entity and "end" in entity:
                norm_entity.span = (entity["start"], entity["end"])
            normalized.append(norm_entity)

        return normalized

    def group_by_canonical(
        self,
        entities: List[NormalizedEntity],
    ) -> Dict[str, List[NormalizedEntity]]:
        """
        Group entities by their canonical form.

        Args:
            entities: List of normalized entities

        Returns:
            Dict mapping canonical forms to lists of entities
        """
        groups = {}
        for entity in entities:
            key = (entity.canonical_form, entity.entity_type)
            if key not in groups:
                groups[key] = []
            groups[key].append(entity)

        return groups

    def find_coreferent_candidates(
        self,
        entity: NormalizedEntity,
        entities: List[NormalizedEntity],
        fuzzy_match: bool = False,
    ) -> List[NormalizedEntity]:
        """
        Find potential coreferent entities.

        Args:
            entity: Entity to find coreferences for
            entities: List of entities to search
            fuzzy_match: Allow partial matching (e.g., "Smith" matches "John Smith")

        Returns:
            List of potentially coreferent entities
        """
        candidates = []

        for candidate in entities:
            # Skip if different entity types
            if candidate.entity_type != entity.entity_type:
                continue

            # Exact canonical match
            if candidate.canonical_form == entity.canonical_form:
                candidates.append(candidate)
            # Fuzzy matching for names
            elif fuzzy_match and entity.entity_type == "PERSON":
                # Check if one is substring of the other
                if (entity.lemma in candidate.lemma or
                    candidate.lemma in entity.lemma):
                    candidates.append(candidate)
                # Check last name matching (simple heuristic)
                entity_parts = entity.lemma.split()
                candidate_parts = candidate.lemma.split()
                if (len(entity_parts) > 1 and len(candidate_parts) > 1 and
                    entity_parts[-1] == candidate_parts[-1]):
                    candidates.append(candidate)

        return candidates


class NameVariantGenerator:
    """Generate name variants for better matching."""

    def __init__(self):
        """Initialize name variant generator."""
        # Common name variations
        self.common_variants = {
            "Robert": ["Rob", "Bob", "Bobby"],
            "William": ["Will", "Bill", "Billy"],
            "James": ["Jim", "Jimmy"],
            "Michael": ["Mike", "Mickey"],
            "Richard": ["Rick", "Dick", "Ricky"],
            "Leonard": ["Len", "Lenny"],
            "Thomas": ["Tom", "Tommy"],
            "Charles": ["Charlie", "Chuck"],
            "Joseph": ["Joe", "Joey"],
            "Benjamin": ["Ben", "Benny"],
        }

        # Build reverse mapping
        self.variant_to_canonical = {}
        for canonical, variants in self.common_variants.items():
            for variant in variants:
                self.variant_to_canonical[variant.lower()] = canonical

    def get_canonical_name(self, name: str) -> str:
        """Get canonical form of a name."""
        name_lower = name.lower()
        return self.variant_to_canonical.get(name_lower, name)

    def get_variants(self, name: str) -> List[str]:
        """Get all variants of a name."""
        # Check if this is a variant
        canonical = self.get_canonical_name(name)

        # Get all variants for the canonical form
        if canonical in self.common_variants:
            variants = [canonical] + self.common_variants[canonical]
            return variants

        # Return just the name if no variants known
        return [name]

    def are_name_variants(self, name1: str, name2: str) -> bool:
        """Check if two names are variants of each other."""
        canonical1 = self.get_canonical_name(name1)
        canonical2 = self.get_canonical_name(name2)
        return canonical1.lower() == canonical2.lower()