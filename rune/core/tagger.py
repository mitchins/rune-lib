"""Main BioTagger class for named entity recognition."""

from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from .schema import BioTag, BioSequence, TagSchema
from .normalization import EntityNormalizer, NormalizedEntity
from .coreference import CoreferenceHook, CoreferenceCluster


class BioTagger:
    """Main class for bio-tagging with BERT-based models."""

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        tag_schema: TagSchema,
        device: Optional[str] = None,
        max_length: int = 512,
        enable_normalization: bool = True,
        enable_coreference: bool = False,
    ):
        """
        Initialize the BioTagger.

        Args:
            model_name_or_path: HuggingFace model name or path to local model
            tag_schema: Schema defining the entity types
            device: Device to run inference on (auto-detected if None)
            max_length: Maximum sequence length for tokenization
            enable_normalization: Enable entity normalization/lemmatization
            enable_coreference: Enable coreference resolution hooks
        """
        self.tag_schema = tag_schema
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        # Create mappings
        self.id_to_label = tag_schema.get_id_to_tag_mapping()
        self.label_to_id = tag_schema.get_tag_to_id_mapping()

        # Optional components
        self.normalizer = EntityNormalizer() if enable_normalization else None
        self.coreference_hook = CoreferenceHook() if enable_coreference else None

    def tag_tokens(self, tokens: List[str]) -> BioSequence:
        """
        Tag a pre-tokenized sequence of tokens.

        Args:
            tokens: List of string tokens

        Returns:
            BioSequence with predicted tags
        """
        # Tokenize with the model's tokenizer
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Map predictions back to original tokens
        word_ids = encoding.word_ids()
        predicted_tags = []

        for i, token in enumerate(tokens):
            # Find the first subtoken for this word
            token_predictions = []
            for j, word_id in enumerate(word_ids):
                if word_id == i:
                    token_predictions.append(predictions[0][j].item())

            if token_predictions:
                # Use the prediction from the first subtoken
                tag_id = token_predictions[0]
                tag_str = self.id_to_label[tag_id]
                predicted_tags.append(BioTag.from_string(tag_str))
            else:
                # Fallback to O tag
                predicted_tags.append(BioTag.outside())

        return BioSequence(tokens=tokens, tags=predicted_tags)

    def tag_text(self, text: str) -> BioSequence:
        """
        Tag a raw text string.

        Args:
            text: Input text to tag

        Returns:
            BioSequence with predicted tags
        """
        # Simple whitespace tokenization
        tokens = text.split()
        return self.tag_tokens(tokens)

    def extract_entities(self, bio_sequence: BioSequence) -> List[Dict[str, Any]]:
        """
        Extract entities from a tagged sequence.

        Args:
            bio_sequence: Tagged sequence

        Returns:
            List of entity dictionaries with 'text', 'label', 'start', 'end'
        """
        entities = []
        current_entity = None

        for i, (token, tag) in enumerate(zip(bio_sequence.tokens, bio_sequence.tags)):
            if tag.prefix.value == "B":
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "label": tag.entity_type,
                    "start": i,
                    "end": i,
                    "tokens": [token],
                }
            elif tag.prefix.value == "I" and current_entity and tag.entity_type == current_entity["label"]:
                # Continuation of current entity
                current_entity["text"] += " " + token
                current_entity["end"] = i
                current_entity["tokens"].append(token)
            else:
                # End of current entity (O tag or different entity type)
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)

        return entities

    def tag_and_extract(self, text: str, normalize: bool = True) -> Dict[str, Any]:
        """
        Tag text and extract entities in one step.

        Args:
            text: Input text
            normalize: Apply normalization if available

        Returns:
            Dictionary with 'sequence', 'entities', and optionally 'normalized_entities'
        """
        sequence = self.tag_text(text)
        entities = self.extract_entities(sequence)

        result = {"sequence": sequence, "entities": entities}

        # Add normalized entities if normalization is enabled
        if normalize and self.normalizer:
            normalized = self.normalizer.normalize_entities(entities)
            result["normalized_entities"] = normalized

            # Group by canonical form
            canonical_groups = self.normalizer.group_by_canonical(normalized)
            result["canonical_groups"] = {
                f"{canon}_{etype}": [e.to_dict() for e in group]
                for (canon, etype), group in canonical_groups.items()
            }

        return result

    def tag_document(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Tag a multi-sentence document with coreference support.

        Args:
            sentences: List of sentences

        Returns:
            Dictionary with sequences, entities, and coreference clusters
        """
        sequences = []
        all_entities = []

        for sentence in sentences:
            seq = self.tag_text(sentence)
            entities = self.extract_entities(seq)
            sequences.append(seq)
            all_entities.append(entities)

        result = {
            "sequences": sequences,
            "all_entities": all_entities,
        }

        # Add coreference clusters if enabled
        if self.coreference_hook:
            clusters = self.coreference_hook.resolve_document(sequences, all_entities)
            result["coreference_clusters"] = clusters

        return result

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        entity_types: List[str],
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> "BioTagger":
        """
        Create a BioTagger from a pretrained model with custom entity types.

        Args:
            model_name_or_path: HuggingFace model name or path
            entity_types: List of entity types to recognize
            device: Device to run on
            max_length: Maximum sequence length

        Returns:
            Configured BioTagger instance
        """
        schema = TagSchema.create_standard_schema(entity_types)
        return cls(model_name_or_path, schema, device, max_length)