"""Schema definitions for bio-tagging."""

from typing import List, Dict, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class BioPrefix(str, Enum):
    """Bio-tagging prefixes following IOB2 format."""
    B = "B"  # Beginning of entity
    I = "I"  # Inside entity
    O = "O"  # Outside entity


class BioTag(BaseModel):
    """Represents a single bio-tag with prefix and entity type."""
    prefix: BioPrefix
    entity_type: Optional[str] = None

    @validator('entity_type')
    def validate_entity_type(cls, v, values):
        if values.get('prefix') == BioPrefix.O and v is not None:
            raise ValueError("O tags cannot have an entity type")
        if values.get('prefix') in [BioPrefix.B, BioPrefix.I] and v is None:
            raise ValueError("B and I tags must have an entity type")
        return v

    def __str__(self) -> str:
        if self.prefix == BioPrefix.O:
            return "O"
        return f"{self.prefix.value}-{self.entity_type}"

    @classmethod
    def from_string(cls, tag_str: str) -> "BioTag":
        """Create BioTag from string representation like 'B-PERSON' or 'O'."""
        if tag_str == "O":
            return cls(prefix=BioPrefix.O)

        if "-" not in tag_str:
            raise ValueError(f"Invalid tag format: {tag_str}")

        prefix_str, entity_type = tag_str.split("-", 1)
        return cls(prefix=BioPrefix(prefix_str), entity_type=entity_type)

    @classmethod
    def outside(cls) -> "BioTag":
        """Create an O (outside) tag."""
        return cls(prefix=BioPrefix.O)

    @classmethod
    def beginning(cls, entity_type: str) -> "BioTag":
        """Create a B (beginning) tag for given entity type."""
        return cls(prefix=BioPrefix.B, entity_type=entity_type)

    @classmethod
    def inside(cls, entity_type: str) -> "BioTag":
        """Create an I (inside) tag for given entity type."""
        return cls(prefix=BioPrefix.I, entity_type=entity_type)


class BioSequence(BaseModel):
    """Represents a sequence of tokens with their bio-tags."""
    tokens: List[str] = Field(..., min_items=1)
    tags: List[BioTag] = Field(..., min_items=1)

    @validator('tags')
    def validate_tags_length(cls, v, values):
        tokens = values.get('tokens', [])
        if len(v) != len(tokens):
            raise ValueError("Number of tags must match number of tokens")
        return v

    def to_iob_format(self) -> List[str]:
        """Convert to standard IOB format strings."""
        return [str(tag) for tag in self.tags]

    @classmethod
    def from_iob_format(cls, tokens: List[str], iob_tags: List[str]) -> "BioSequence":
        """Create from tokens and IOB format tag strings."""
        tags = [BioTag.from_string(tag) for tag in iob_tags]
        return cls(tokens=tokens, tags=tags)


class TagSchema(BaseModel):
    """Defines the schema for entity types and their properties."""
    entity_types: List[str] = Field(..., min_items=1)
    description: Optional[str] = None

    @validator('entity_types')
    def validate_entity_types(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("Entity types must be unique")
        return v

    def get_all_tags(self) -> List[str]:
        """Get all possible tag strings for this schema."""
        tags = ["O"]
        for entity_type in self.entity_types:
            tags.extend([f"B-{entity_type}", f"I-{entity_type}"])
        return tags

    def get_tag_to_id_mapping(self) -> Dict[str, int]:
        """Get mapping from tag strings to IDs."""
        return {tag: i for i, tag in enumerate(self.get_all_tags())}

    def get_id_to_tag_mapping(self) -> Dict[int, str]:
        """Get mapping from IDs to tag strings."""
        return {i: tag for i, tag in enumerate(self.get_all_tags())}

    @classmethod
    def create_standard_schema(cls, entity_types: List[str], description: Optional[str] = None) -> "TagSchema":
        """Create a standard schema with given entity types."""
        return cls(entity_types=entity_types, description=description)

    @classmethod
    def create_protagonist_schema(cls) -> "TagSchema":
        """Create a schema for protagonist tagging."""
        return cls(
            entity_types=["PROTAGONIST"],
            description="Schema for identifying protagonists in text"
        )