"""Dataset loader for story NER with Longformer support."""

from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import json
from datasets import Dataset
from transformers import PreTrainedTokenizer
import torch

from ..core.schema import BioSequence, TagSchema
from .improved_tokenizer import ImprovedTokenizer


class StoryNerDataset:
    """Dataset class for story-based NER with long sequences."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tag_schema: Optional[TagSchema] = None,
        max_length: int = 4096,
        include_genre: bool = True,
        auto_detect_tags: bool = True,
    ):
        """
        Initialize story NER dataset.

        Args:
            tokenizer: Tokenizer (should be Longformer tokenizer for best results)
            tag_schema: Optional custom schema (will auto-detect from data if None)
            max_length: Maximum sequence length (4096 for Longformer)
            include_genre: Include genre information
            auto_detect_tags: Automatically detect tag schema from data
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_genre = include_genre
        self.auto_detect_tags = auto_detect_tags

        self.tag_schema = tag_schema
        self.label_to_id = None
        self.id_to_label = None

        self.stories: List[Dict[str, Any]] = []

        # Initialize improved tokenizer for defensive label alignment
        self.improved_tokenizer = ImprovedTokenizer()

    def _detect_tags_from_data(self) -> TagSchema:
        """Auto-detect tag schema from the loaded data."""
        all_tags = set()
        entity_types = set()

        for story in self.stories:
            for tag in story.get("bio_tags", []):
                all_tags.add(tag)
                if tag != "O" and "-" in tag:
                    entity_type = tag.split("-", 1)[1]
                    entity_types.add(entity_type)

        entity_types = sorted(list(entity_types))
        if not entity_types:
            # Fallback to default
            entity_types = ["PROTAGONIST", "ANTAGONIST", "SUPPORTING"]

        return TagSchema.create_standard_schema(
            entity_types=entity_types,
            description=f"Auto-detected schema from data: {entity_types}"
        )

    def _create_label_mappings(self) -> None:
        """Create label mappings from detected or provided schema."""
        if self.tag_schema is None:
            self.tag_schema = self._detect_tags_from_data()

        # Create mappings directly from the unique tags in data
        all_tags = set(["O"])  # Always include O tag

        for story in self.stories:
            for tag in story.get("bio_tags", []):
                all_tags.add(tag)

        # CRITICAL: O tag must be label 0 in NER systems
        sorted_tags = ["O"] + sorted([tag for tag in all_tags if tag != "O"])
        self.label_to_id = {tag: i for i, tag in enumerate(sorted_tags)}
        self.id_to_label = {i: tag for i, tag in enumerate(sorted_tags)}

        print(f"📋 Detected tags: {sorted_tags}")

    @classmethod
    def from_jsonl(
        cls,
        file_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        limit: Optional[int] = None,
        auto_preprocess: bool = True,
    ) -> "StoryNerDataset":
        """
        Load dataset from JSONL file with auto-preprocessing support.

        Args:
            file_path: Path to JSONL file
            tokenizer: Longformer tokenizer
            max_length: Maximum sequence length
            limit: Optional limit on number of stories to load
            auto_preprocess: Automatically preprocess raw story data

        Returns:
            StoryNerDataset instance
        """
        dataset = cls(tokenizer=tokenizer, max_length=max_length)

        # Check if data needs preprocessing
        needs_preprocessing = False

        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                first_story = json.loads(first_line)
                # Check if it's raw data (has 'characters' but no 'bio_tags')
                if "characters" in first_story and "bio_tags" not in first_story:
                    needs_preprocessing = True
                    print("📋 Detected raw story data - auto-preprocessing enabled")

        if needs_preprocessing and auto_preprocess:
            from .story_preprocessor import StoryPreprocessor
            preprocessor = StoryPreprocessor(use_spacy=False)  # Use simple tokenization

            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break

                    raw_story = json.loads(line.strip())
                    # Preprocess on the fly
                    processed_story = preprocessor.process_story(raw_story)
                    dataset.stories.append(processed_story)

                    if (i + 1) % 1000 == 0:
                        print(f"   Preprocessed {i + 1} stories...")
        else:
            # Load preprocessed data
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break

                    data = json.loads(line.strip())
                    dataset.stories.append(data)

        print(f"📚 Loaded {len(dataset.stories)} stories")

        # Auto-detect and create label mappings after loading data
        dataset._create_label_mappings()

        return dataset

    def __len__(self) -> int:
        return len(self.stories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized story."""
        story = self.stories[idx]
        return self._tokenize_and_align_labels(story)

    def _tokenize_and_align_labels(self, story: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Tokenize story and align labels with defensive punctuation handling.

        Args:
            story: Story dictionary with tokens and bio_tags

        Returns:
            Dictionary with tokenized inputs and aligned labels
        """
        tokens = story["tokens"]
        bio_tags = story["bio_tags"]

        # Tokenize with the model's tokenizer (e.g., Longformer)
        tokenized_inputs = self.tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Use defensive label alignment from improved tokenizer
        labels = self.improved_tokenizer._align_labels_defensively(
            tokens, bio_tags, tokenized_inputs.word_ids(), self.tokenizer
        )

        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

        # Add genre information if requested
        if self.include_genre and "genre" in story:
            tokenized_inputs["genre"] = story["genre"]

        # Add story ID for tracking
        if "story_id" in story:
            tokenized_inputs["story_id"] = story["story_id"]

        # Remove batch dimension
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in tokenized_inputs:
                tokenized_inputs[key] = tokenized_inputs[key].squeeze(0)

        return tokenized_inputs

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset."""
        processed_stories = []

        for i in range(len(self)):
            item = self[i]
            # Convert tensors to lists for HF Dataset
            processed_item = {
                "input_ids": item["input_ids"].tolist(),
                "attention_mask": item["attention_mask"].tolist(),
                "labels": item["labels"].tolist(),
            }

            # Add metadata if available
            for key in ["genre", "story_id"]:
                if key in item:
                    processed_item[key] = item[key]

            processed_stories.append(processed_item)

        return Dataset.from_list(processed_stories)

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        label_counts = {}

        for story in self.stories:
            for tag in story["bio_tags"]:
                label_counts[tag] = label_counts.get(tag, 0) + 1

        return label_counts

    def split_by_genre(self) -> Dict[str, "StoryNerDataset"]:
        """Split dataset by genre for genre-specific evaluation."""
        genre_datasets = {}

        for story in self.stories:
            genre = story.get("genre", "unknown")
            if genre not in genre_datasets:
                genre_datasets[genre] = StoryNerDataset(
                    tokenizer=self.tokenizer,
                    tag_schema=self.tag_schema,
                    max_length=self.max_length,
                    include_genre=self.include_genre,
                    include_role=self.include_role,
                )
            genre_datasets[genre].stories.append(story)

        return genre_datasets