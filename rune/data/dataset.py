"""Dataset utilities for NER training and evaluation."""

from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import json
from datasets import Dataset
from transformers import PreTrainedTokenizer
import torch

from ..core.schema import BioSequence, TagSchema, BioTag


class NerDataset:
    """Dataset class for NER training and evaluation."""

    def __init__(
        self,
        sequences: List[BioSequence],
        tag_schema: TagSchema,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize NER dataset.

        Args:
            sequences: List of bio-tagged sequences
            tag_schema: Schema defining entity types
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.tag_schema = tag_schema
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = tag_schema.get_tag_to_id_mapping()
        self.id_to_label = tag_schema.get_id_to_tag_mapping()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized example."""
        sequence = self.sequences[idx]
        return self._tokenize_and_align_labels(sequence)

    def _tokenize_and_align_labels(self, sequence: BioSequence) -> Dict[str, torch.Tensor]:
        """
        Tokenize tokens and align labels with subword tokens.

        Args:
            sequence: Bio-tagged sequence

        Returns:
            Dictionary with tokenized inputs and aligned labels
        """
        tokenized_inputs = self.tokenizer(
            sequence.tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        labels = []
        word_ids = tokenized_inputs.word_ids()

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens (CLS, SEP, PAD)
                labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword token of a word
                tag_str = str(sequence.tags[word_idx])
                labels.append(self.label_to_id[tag_str])
            else:
                # Subsequent subword tokens - use -100 to ignore in loss
                labels.append(-100)
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

        # Remove batch dimension added by return_tensors="pt"
        for key in tokenized_inputs:
            tokenized_inputs[key] = tokenized_inputs[key].squeeze(0)

        return tokenized_inputs

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset."""
        tokenized_data = [self[i] for i in range(len(self))]

        # Convert to the format expected by HuggingFace
        hf_data = {
            "input_ids": [item["input_ids"] for item in tokenized_data],
            "attention_mask": [item["attention_mask"] for item in tokenized_data],
            "labels": [item["labels"] for item in tokenized_data],
        }

        if "token_type_ids" in tokenized_data[0]:
            hf_data["token_type_ids"] = [item["token_type_ids"] for item in tokenized_data]

        return Dataset.from_dict(hf_data)

    @classmethod
    def from_conll_format(
        cls,
        file_path: Union[str, Path],
        tag_schema: TagSchema,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        separator: str = "\t",
    ) -> "NerDataset":
        """
        Load dataset from CoNLL format file.

        Args:
            file_path: Path to CoNLL format file
            tag_schema: Schema defining entity types
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            separator: Column separator (default: tab)

        Returns:
            NerDataset instance
        """
        sequences = []
        current_tokens = []
        current_tags = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    # Empty line indicates end of sequence
                    if current_tokens:
                        sequence = BioSequence.from_iob_format(current_tokens, current_tags)
                        sequences.append(sequence)
                        current_tokens = []
                        current_tags = []
                else:
                    parts = line.split(separator)
                    if len(parts) >= 2:
                        token = parts[0]
                        tag = parts[-1]  # Last column is typically the tag
                        current_tokens.append(token)
                        current_tags.append(tag)

        # Don't forget the last sequence
        if current_tokens:
            sequence = BioSequence.from_iob_format(current_tokens, current_tags)
            sequences.append(sequence)

        return cls(sequences, tag_schema, tokenizer, max_length)

    @classmethod
    def from_json_format(
        cls,
        file_path: Union[str, Path],
        tag_schema: TagSchema,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> "NerDataset":
        """
        Load dataset from JSON format.

        Expected format:
        [
            {"tokens": ["Hello", "world"], "tags": ["O", "O"]},
            {"tokens": ["John", "Doe"], "tags": ["B-PERSON", "I-PERSON"]}
        ]

        Args:
            file_path: Path to JSON file
            tag_schema: Schema defining entity types
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length

        Returns:
            NerDataset instance
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sequences = []
        for item in data:
            sequence = BioSequence.from_iob_format(item["tokens"], item["tags"])
            sequences.append(sequence)

        return cls(sequences, tag_schema, tokenizer, max_length)

    def train_test_split(self, test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple["NerDataset", "NerDataset"]:
        """
        Split dataset into train and test sets.

        Args:
            test_size: Proportion of test set
            random_state: Random seed

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        import random

        if random_state is not None:
            random.seed(random_state)

        indices = list(range(len(self.sequences)))
        random.shuffle(indices)

        split_idx = int(len(indices) * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_sequences = [self.sequences[i] for i in train_indices]
        test_sequences = [self.sequences[i] for i in test_indices]

        train_dataset = NerDataset(train_sequences, self.tag_schema, self.tokenizer, self.max_length)
        test_dataset = NerDataset(test_sequences, self.tag_schema, self.tokenizer, self.max_length)

        return train_dataset, test_dataset

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        label_counts = {}
        for sequence in self.sequences:
            for tag in sequence.tags:
                tag_str = str(tag)
                label_counts[tag_str] = label_counts.get(tag_str, 0) + 1
        return label_counts