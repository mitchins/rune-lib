#!/usr/bin/env python3
"""
Train RUNE NER with ModernBERT streaming dataset.
Uses ModernBERT's 8192 token context to handle longer stories.
"""

import os
import json
import torch
import argparse
from typing import Iterator, Dict, Any, List, Tuple
from torch.utils.data import IterableDataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import f1_score, accuracy_score
from rune.data.story_preprocessor import StoryPreprocessor


class ModernBERTStreamingDataset(IterableDataset):
    """Streaming dataset for ModernBERT with optional length filtering."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int = 8192,
        skip: int = 0,
        limit: int = None,
        prefiltered: bool = False,
        simplify_labels: bool = False,
    ):
        """
        Initialize streaming dataset.

        Args:
            jsonl_path: Path to JSONL file
            tokenizer: ModernBERT tokenizer
            max_length: Max sequence length (8192 for ModernBERT)
            skip: Number of stories to skip
            limit: Max stories to process
            prefiltered: If True, assumes data is pre-processed with tokens/bio_tags
            simplify_labels: If True, collapse role labels to B-PERSON/I-PERSON
        """
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skip = skip
        self.limit = limit
        self.prefiltered = prefiltered
        self.simplify_labels = simplify_labels

        # Initialize preprocessor if not prefiltered
        if not prefiltered:
            self.preprocessor = StoryPreprocessor(use_spacy=False)

        # Label mappings
        if simplify_labels:
            # Simplified B/I/O for entity detection only
            self.label_to_id = {
                'O': 0,
                'B-PERSON': 1,
                'I-PERSON': 2,
                'B-LOCATION': 3,
                'I-LOCATION': 4,
            }
            # Map role labels to simplified labels
            self.role_to_simple = {
                'B-PROTAGONIST': 'B-PERSON',
                'I-PROTAGONIST': 'I-PERSON',
                'B-ANTAGONIST': 'B-PERSON',
                'I-ANTAGONIST': 'I-PERSON',
                'B-SUPPORTING': 'B-PERSON',
                'I-SUPPORTING': 'I-PERSON',
                'B-LOCATION': 'B-LOCATION',
                'I-LOCATION': 'I-LOCATION',
                'O': 'O',
            }
        else:
            # Full role-aware labels
            self.label_to_id = {
                'O': 0,
                'B-ANTAGONIST': 1,
                'B-PROTAGONIST': 2,
                'B-SUPPORTING': 3,
                'I-ANTAGONIST': 4,
                'I-PROTAGONIST': 5,
                'I-SUPPORTING': 6,
            }
            self.role_to_simple = None

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over stories."""
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            # Skip for train/val split
            for _ in range(self.skip):
                next(f)

            count = 0
            filtered_count = 0

            for line in f:
                if self.limit and count >= self.limit:
                    break

                try:
                    raw_story = json.loads(line.strip())

                    # Get or create processed story
                    if self.prefiltered:
                        # Already has tokens and bio_tags
                        processed = raw_story
                    else:
                        # Need to preprocess
                        processed = self.preprocessor.process_story(raw_story)

                    # Check token length
                    tokens = processed["tokens"]
                    test_tokenized = self.tokenizer(
                        tokens,
                        is_split_into_words=True,
                        truncation=False,
                        add_special_tokens=True
                    )
                    token_length = len(test_tokenized["input_ids"])

                    # Filter if too long
                    if token_length > self.max_length:
                        filtered_count += 1
                        continue

                    # Tokenize and align labels
                    tokenized = self._tokenize_and_align_labels(processed)

                    yield tokenized
                    count += 1

                except Exception as e:
                    print(f"⚠️  Error processing story: {e}")
                    continue

            if filtered_count > 0:
                print(f"📊 Filtered {filtered_count} stories exceeding {self.max_length} tokens")
            print(f"✅ Yielded {count} stories within token limit")

    def _tokenize_and_align_labels(self, story: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize story and align labels."""
        tokens = story["tokens"]
        bio_tags = story["bio_tags"]

        # Tokenize
        tokenized_inputs = self.tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Align labels
        word_ids = tokenized_inputs.word_ids()
        labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                tag = bio_tags[word_idx]
                # Apply simplification if enabled
                if self.simplify_labels and self.role_to_simple:
                    tag = self.role_to_simple.get(tag, tag)
                labels.append(self.label_to_id.get(tag, 0))
            else:
                labels.append(-100)
            previous_word_idx = word_idx

        return {
            "input_ids": tokenized_inputs["input_ids"].squeeze(0),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_compute_metrics(label_list):
    """Create metrics computation function."""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        flat_true = [item for sublist in true_labels for item in sublist]
        flat_pred = [item for sublist in true_predictions for item in sublist]

        if len(flat_true) == 0:
            return {"accuracy": 0.0}

        fine_grained_f1 = f1_score(true_labels, true_predictions)
        token_accuracy = accuracy_score(true_labels, true_predictions)

        true_binary = [
            ["ENTITY" if label != "O" else "O" for label in sequence]
            for sequence in true_labels
        ]
        pred_binary = [
            ["ENTITY" if label != "O" else "O" for label in sequence]
            for sequence in true_predictions
        ]
        entity_detection_f1 = f1_score(true_binary, pred_binary)

        entity_correct = sum(1 for t, p in zip(flat_true, flat_pred) if t != 'O' and t == p)
        entity_total = sum(1 for t in flat_true if t != 'O')
        entity_accuracy = entity_correct / entity_total if entity_total > 0 else 0

        return {
            "f1": fine_grained_f1,
            "accuracy": token_accuracy,
            "entity_detection_f1": entity_detection_f1,
            "entity_accuracy": entity_accuracy,
        }

    return compute_metrics


class OODValidationDataset(Dataset):
    """Static OOD validation dataset (non-streaming)."""
    
    def __init__(
        self,
        ood_examples: List[Dict[str, Any]],
        tokenizer,
        max_length: int,
        simplify_labels: bool = False
    ):
        """
        Initialize OOD dataset.
        
        Args:
            ood_examples: List of preprocessed OOD examples
            tokenizer: Tokenizer
            max_length: Max sequence length (truncate if longer)
            simplify_labels: If True, collapse role labels
        """
        self.examples = ood_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.simplify_labels = simplify_labels
        
        # Label mappings
        if simplify_labels:
            self.label_to_id = {
                'O': 0,
                'B-PERSON': 1,
                'I-PERSON': 2,
                'B-LOCATION': 3,
                'I-LOCATION': 4,
            }
            self.role_to_simple = {
                'B-PROTAGONIST': 'B-PERSON',
                'I-PROTAGONIST': 'I-PERSON',
                'B-ANTAGONIST': 'B-PERSON',
                'I-ANTAGONIST': 'I-PERSON',
                'B-SUPPORTING': 'B-PERSON',
                'I-SUPPORTING': 'I-PERSON',
                'B-LOCATION': 'B-LOCATION',
                'I-LOCATION': 'I-LOCATION',
                'O': 'O',
            }
        else:
            self.label_to_id = {
                'O': 0,
                'B-ANTAGONIST': 1,
                'B-PROTAGONIST': 2,
                'B-SUPPORTING': 3,
                'I-ANTAGONIST': 4,
                'I-PROTAGONIST': 5,
                'I-SUPPORTING': 6,
            }
            self.role_to_simple = None
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a single example."""
        example = self.examples[idx]
        tokens = example['tokens']
        bio_tags = example['bio_tags']
        
        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            bio_tags = bio_tags[:self.max_length]
        
        # Simplify labels if needed
        if self.simplify_labels and self.role_to_simple:
            bio_tags = [self.role_to_simple.get(tag, tag) for tag in bio_tags]
        
        # Convert to IDs
        label_ids = [self.label_to_id.get(tag, 0) for tag in bio_tags]
        
        # Tokenize (already tokenized, so just convert to input_ids)
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False
        )
        
        # Align labels with subword tokens
        word_ids = tokenized.word_ids()
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(label_ids[word_id])
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': aligned_labels
        }


def load_and_preprocess_ood(
    ood_path: str,
    max_length: int,
    warn_truncation: bool = True
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Load OOD validation data and preprocess it.
    
    Args:
        ood_path: Path to OOD JSONL file
        max_length: Max token length for this stage
        warn_truncation: Whether to warn about truncated examples
    
    Returns:
        Tuple of (examples, usable_count, truncated_count)
    """
    preprocessor = StoryPreprocessor(use_spacy=True)
    examples = []
    truncated = 0
    
    with open(ood_path) as f:
        for line in f:
            raw = json.loads(line)
            
            # Preprocess to get tokens and bio_tags
            processed = preprocessor.process_story(
                text=raw['text'],
                characters=[
                    {
                        'name': char['name'],
                        'role': char.get('role', 'SUPPORTING')
                    }
                    for char in raw.get('characters', [])
                ],
                story_id=raw.get('story_id', 'ood'),
                genre=raw.get('genre', 'classic')
            )
            
            if processed and 'tokens' in processed and 'bio_tags' in processed:
                examples.append(processed)
                if len(processed['tokens']) > max_length:
                    truncated += 1
    
    if warn_truncation and truncated > 0:
        print(f"⚠️  {truncated}/{len(examples)} OOD examples exceed {max_length} tokens (will be truncated)")
    
    return examples, len(examples), truncated


def main():
    """Main training function with curriculum support."""

    parser = argparse.ArgumentParser(
        description="Train NER model with optional curriculum learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-stage training
  python3 train_modernbert_streaming.py --input data.jsonl --output ./model --model answerdotai/ModernBERT-base
  
  # Curriculum training (2 stages)
  python3 train_modernbert_streaming.py \\
    --model allenai/longformer-base-4096 \\
    --curriculum \\
    --curriculum-stage1-input diverse_phase1.jsonl \\
    --curriculum-stage2-input diverse_phase2.jsonl \\
    --curriculum-stage1-epochs 2 \\
    --curriculum-stage2-epochs 3 \\
    --output ./model \\
    --batch-size 4 \\
    --use-bf16
"""
    )
    
    # Model and data
    parser.add_argument('--model', default="answerdotai/ModernBERT-base", help="HuggingFace model name")
    parser.add_argument('--input', default=None, help="Input JSONL file (single-stage)")
    parser.add_argument('--output', default="./story_ner_model", help="Output directory")
    parser.add_argument('--max-length', type=int, default=None, help="Max token length (auto-detect from data if not set)")
    parser.add_argument('--stage1-max-length', type=int, default=1024, help="Stage 1 max length (short examples)")
    parser.add_argument('--stage2-max-length', type=int, default=4096, help="Stage 2 max length (long examples)")
    
    # Curriculum learning
    parser.add_argument('--curriculum', action='store_true', help="Enable curriculum learning (2 stages)")
    parser.add_argument('--curriculum-stage1-input', default=None, help="Stage 1 input (easy examples)")
    parser.add_argument('--curriculum-stage2-input', default=None, help="Stage 2 input (hard examples)")
    parser.add_argument('--curriculum-stage1-epochs', type=int, default=2, help="Stage 1 epochs")
    parser.add_argument('--curriculum-stage2-epochs', type=int, default=3, help="Stage 2 epochs")
    
    # Training config
    parser.add_argument('--batch-size', type=int, default=None, help="Per-device batch size (both stages if not specified separately)")
    parser.add_argument('--stage1-batch-size', type=int, default=None, help="Stage 1 batch size (defaults to --batch-size or 8)")
    parser.add_argument('--stage2-batch-size', type=int, default=None, help="Stage 2 batch size (defaults to --batch-size or 4)")
    parser.add_argument('--gradient-accumulation', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--learning-rate', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--use-bf16', action='store_true', help="Use bfloat16 mixed precision")
    
    # Resuming
    parser.add_argument('--resume-from-checkpoint', default=None, help="Resume from checkpoint path")
    
    # Labels
    parser.add_argument('--simplify-labels', action='store_true', help="Collapse role labels to B/I-PERSON")
    
    # Validation
    parser.add_argument('--eval-ood', default=None, help="Path to OOD validation set (e.g., ood_validation_ground_truth_balanced.jsonl)")
    parser.add_argument('--no-test-split', action='store_true', help="Don't split data into train/val (use 100%% for training)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.curriculum:
        if not args.curriculum_stage1_input or not args.curriculum_stage2_input:
            parser.error("--curriculum requires --curriculum-stage1-input and --curriculum-stage2-input")
    elif not args.input:
        parser.error("--input required for single-stage training (or use --curriculum)")
    
    # Validate OOD flags
    if args.no_test_split and not args.eval_ood:
        print("⚠️  WARNING: --no-test-split without --eval-ood means NO evaluation will occur")
    
    print("🚀 TRAINING RUNE NER")
    print("=" * 70)
    print(f"🤖 Model: {args.model}")
    print(f"💾 Max length: {args.max_length} tokens")
    if args.curriculum:
        print("🎓 Mode: Curriculum learning (2 stages)")
        print(f"   Stage 1: {args.curriculum_stage1_input} ({args.curriculum_stage1_epochs} epochs)")
        print(f"   Stage 2: {args.curriculum_stage2_input} ({args.curriculum_stage2_epochs} epochs)")
    else:
        print("📖 Mode: Single-stage training")
        print(f"   Input: {args.input}")
    if args.simplify_labels:
        print("🔄 Labels: Simplified B/I-PERSON (role-agnostic)")
    else:
        print("🎭 Labels: Full role-aware (B-PROTAGONIST, etc.)")
    
    # Validation strategy
    if args.eval_ood and args.no_test_split:
        print("📊 Validation: OOD only (100% training data)")
    elif args.eval_ood:
        print("📊 Validation: 90/10 split + OOD holdout")
    elif args.no_test_split:
        print("📊 Validation: None (100% training data)")
    else:
        print("📊 Validation: 90/10 split")
    
    print(f"💾 Output: {args.output}")
    if args.resume_from_checkpoint:
        print(f"♻️  Resuming from: {args.resume_from_checkpoint}")
    print("=" * 70)
    print()

    # Configuration
    model_name = args.model
    # Set max_length based on mode
    if args.curriculum:
        stage1_max_length = args.stage1_max_length
        stage2_max_length = args.stage2_max_length
        max_length = stage1_max_length  # For initial display
    else:
        max_length = args.max_length or 8192
    
    training_file = args.input if not args.curriculum else args.curriculum_stage1_input

    # Count total stories
    with open(training_file, "r") as f:
        total_stories = sum(1 for _ in f)

    print(f"📚 Total clean stories: {total_stories}")
    if args.curriculum:
        print(f"   Stage 1 max length: {stage1_max_length} tokens")
        print(f"   Stage 2 max length: {stage2_max_length} tokens")
    else:
        print(f"   Max token length: {max_length}")
    print(f"   All stories fit within token limit (pre-filtered)")
    print()

    # Calculate split (if needed)
    if args.no_test_split:
        val_split_idx = total_stories
        train_count = total_stories
        val_count = 0
        print(f"📊 Train/Val Split:")
        print(f"   Training: {train_count} stories (100%)")
        print(f"   Validation: Using OOD holdout only" if args.eval_ood else "   Validation: None")
    else:
        val_split_idx = int(0.9 * total_stories)
        train_count = val_split_idx
        val_count = total_stories - val_split_idx
        print(f"📊 Train/Val Split:")
        print(f"   Training: {train_count} stories")
        print(f"   Validation: {val_count} stories")
    print()

    # Load ModernBERT tokenizer and model
    print(f"🔄 Loading ModernBERT...")
    # Load tokenizer with add_prefix_space for Longformer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_prefix_space=True  # Required for Longformer with pretokenized inputs
    )

    # Configure labels based on simplify_labels flag
    if args.simplify_labels:
        label_to_id = {
            'O': 0,
            'B-PERSON': 1,
            'I-PERSON': 2,
            'B-LOCATION': 3,
            'I-LOCATION': 4,
        }
    else:
        label_to_id = {
            'O': 0,
            'B-ANTAGONIST': 1,
            'B-PROTAGONIST': 2,
            'B-SUPPORTING': 3,
            'I-ANTAGONIST': 4,
            'I-PROTAGONIST': 5,
            'I-SUPPORTING': 6,
        }
    id_to_label = {v: k for k, v in label_to_id.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    print(f"✅ ModernBERT loaded with {len(label_to_id)} labels")
    print(f"   Labels: {', '.join(label_to_id.keys())}")
    print(f"   Max position embeddings: {model.config.max_position_embeddings}")
    print()

    # Create streaming datasets
    print("🔄 Creating streaming datasets...")
    train_dataset = ModernBERTStreamingDataset(
        training_file,
        tokenizer,
        max_length=max_length,
        skip=0,
        limit=train_count,
        prefiltered=True,  # Data is already pre-processed
        simplify_labels=args.simplify_labels
    )

    # Create validation dataset (if not using OOD-only)
    if args.no_test_split and not args.eval_ood:
        val_dataset = None
        print("✅ Training dataset created (no validation)")
    elif args.no_test_split and args.eval_ood:
        val_dataset = None  # Will be replaced with OOD below
        print("✅ Training dataset created (OOD validation only)")
    else:
        val_dataset = ModernBERTStreamingDataset(
            training_file,
            tokenizer,
            max_length=max_length,
            skip=val_split_idx,
            limit=val_count,
            prefiltered=True,  # Data is already pre-processed
            simplify_labels=args.simplify_labels
        )
        print("✅ Streaming datasets created (using pre-filtered data)")
    print()

    # Output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    bf16_enabled = args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    fp16_enabled = not bf16_enabled and torch.cuda.is_available()
    
    # Determine eval strategy based on validation
    has_eval = args.eval_ood or not args.no_test_split
    eval_strategy = "steps" if has_eval else "no"
    load_best = has_eval
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Overridden per stage in curriculum mode
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy=eval_strategy,
        eval_steps=500 if has_eval else None,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=load_best,
        metric_for_best_model="f1" if has_eval else None,
        greater_is_better=True if has_eval else None,
        gradient_accumulation_steps=args.gradient_accumulation,
        report_to=None,
        save_total_limit=3,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        dataloader_num_workers=0,
    )
    
    if bf16_enabled:
        print(f"✅ Using bfloat16 mixed precision")
    elif fp16_enabled:
        print(f"✅ Using float16 mixed precision")
    print()

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    # Load OOD validation if specified
    ood_dataset_stage1 = None
    ood_dataset_stage2 = None
    if args.eval_ood:
        print(f"🔄 Loading OOD validation from: {args.eval_ood}")
        
        # For curriculum: load OOD for both stages
        if args.curriculum:
            # Stage 1 OOD (filter to stage1_max_length)
            ood_examples_s1, s1_count, s1_trunc = load_and_preprocess_ood(
                args.eval_ood,
                stage1_max_length,
                warn_truncation=True
            )
            ood_dataset_stage1 = OODValidationDataset(
                ood_examples_s1,
                tokenizer,
                stage1_max_length,
                simplify_labels=args.simplify_labels
            )
            print(f"✅ Stage 1 OOD: {s1_count} examples (max_length={stage1_max_length})")
            if s1_trunc > 0:
                print(f"   ⚠️  {s1_trunc} examples truncated")
            
            # Stage 2 OOD (filter to stage2_max_length)
            ood_examples_s2, s2_count, s2_trunc = load_and_preprocess_ood(
                args.eval_ood,
                stage2_max_length,
                warn_truncation=False  # Don't warn twice
            )
            ood_dataset_stage2 = OODValidationDataset(
                ood_examples_s2,
                tokenizer,
                stage2_max_length,
                simplify_labels=args.simplify_labels
            )
            print(f"✅ Stage 2 OOD: {s2_count} examples (max_length={stage2_max_length})")
            if s2_trunc > 0:
                print(f"   ⚠️  {s2_trunc} examples truncated")
        else:
            # Single-stage: load OOD once
            ood_examples, ood_count, ood_trunc = load_and_preprocess_ood(
                args.eval_ood,
                max_length,
                warn_truncation=True
            )
            ood_dataset_stage1 = OODValidationDataset(
                ood_examples,
                tokenizer,
                max_length,
                simplify_labels=args.simplify_labels
            )
            print(f"✅ OOD validation: {ood_count} examples (max_length={max_length})")
            if ood_trunc > 0:
                print(f"   ⚠️  {ood_trunc} examples truncated")
        print()

    # CURRICULUM TRAINING or SINGLE-STAGE
    if args.curriculum:
        # Determine batch sizes per stage
        if args.batch_size is not None:
            # --batch-size overrides both
            stage1_batch_size = args.batch_size
            stage2_batch_size = args.batch_size
        else:
            # Use stage-specific or defaults
            stage1_batch_size = args.stage1_batch_size or 8  # Bigger for short sequences
            stage2_batch_size = args.stage2_batch_size or 4  # Smaller for long sequences
        
        # Allow override
        if args.stage1_batch_size:
            stage1_batch_size = args.stage1_batch_size
        if args.stage2_batch_size:
            stage2_batch_size = args.stage2_batch_size
        
        print(f"🎓 CURRICULUM LEARNING: Stage 1 (Easy)")
        print("=" * 70)
        print(f"📊 Stage 1 batch size: {stage1_batch_size} (short sequences, 1024 tokens)")
        print(f"📊 Stage 2 batch size: {stage2_batch_size} (long sequences, 4096 tokens)")
        print()
        
        # Count examples in each stage (wc -l)
        with open(args.curriculum_stage1_input) as f:
            stage1_count = sum(1 for _ in f)
        with open(args.curriculum_stage2_input) as f:
            stage2_count = sum(1 for _ in f)
        
        print(f"📊 Stage 1: {stage1_count:,} examples")
        print(f"📊 Stage 2: {stage2_count:,} examples")
        
        # Calculate max_steps for each stage
        stage1_effective_batch = stage1_batch_size * args.gradient_accumulation
        stage2_effective_batch = stage2_batch_size * args.gradient_accumulation
        stage1_steps = (stage1_count // stage1_effective_batch) * args.curriculum_stage1_epochs
        stage2_steps = (stage2_count // stage2_effective_batch) * args.curriculum_stage2_epochs
        
        print(f"📈 Stage 1: batch={stage1_batch_size}, grad_accum={args.gradient_accumulation}, effective={stage1_effective_batch}, steps={stage1_steps:,}")
        print(f"📈 Stage 2: batch={stage2_batch_size}, grad_accum={args.gradient_accumulation}, effective={stage2_effective_batch}, steps={stage2_steps:,}")
        print()
        
        # Stage 1: Easy examples
        stage1_output = f"{output_dir}/stage1"
        os.makedirs(stage1_output, exist_ok=True)
        
        stage1_args = TrainingArguments(
            output_dir=stage1_output,
            max_steps=stage1_steps,
            per_device_train_batch_size=stage1_batch_size,
            per_device_eval_batch_size=stage1_batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=args.learning_rate,
            logging_dir=f"{stage1_output}/logs",
            logging_steps=100,
            eval_strategy=eval_strategy,
            eval_steps=500 if has_eval else None,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=load_best,
            metric_for_best_model="f1" if has_eval else None,
            greater_is_better=True if has_eval else None,
            gradient_accumulation_steps=args.gradient_accumulation,
            report_to=None,
            save_total_limit=3,
            bf16=bf16_enabled,
            fp16=fp16_enabled,
            dataloader_num_workers=0,
        )
        
        # Stage 1 datasets
        stage1_train = ModernBERTStreamingDataset(
            args.curriculum_stage1_input,
            tokenizer,
            max_length=stage1_max_length,
            prefiltered=True,
            simplify_labels=args.simplify_labels
        )
        
        # Stage 1 validation: Use OOD if specified, otherwise 5% holdout
        if args.eval_ood:
            stage1_val = ood_dataset_stage1
        elif not args.no_test_split:
            stage1_val = ModernBERTStreamingDataset(
                args.curriculum_stage1_input,
                tokenizer,
                max_length=stage1_max_length,
                skip=int(0.95 * stage1_count),
                prefiltered=True,
                simplify_labels=args.simplify_labels
            )
        else:
            stage1_val = None
        
        stage1_trainer = Trainer(
            model=model,
            args=stage1_args,
            train_dataset=stage1_train,
            eval_dataset=stage1_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=create_compute_metrics(list(id_to_label.values())),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if stage1_val else [],
        )
        
        print(f"🏋️ Stage 1 training ({args.curriculum_stage1_epochs} epochs)...")
        if args.curriculum_stage1_epochs > 0:
            resume_checkpoint = args.resume_from_checkpoint if args.resume_from_checkpoint and "stage1" in args.resume_from_checkpoint else None
            stage1_trainer.train(resume_from_checkpoint=resume_checkpoint)
            stage1_trainer.save_model()
            tokenizer.save_pretrained(stage1_output)
            print(f"✅ Stage 1 complete. Saved to: {stage1_output}\n")
        else:
            print(f"⏭️  Skipping Stage 1 (0 epochs)\n")
        
        # Stage 2: Hard examples (fine-tune from stage 1)
        print(f"🎓 CURRICULUM LEARNING: Stage 2 (Hard)")
        print("=" * 70)
        
        stage2_output = f"{output_dir}/stage2"
        os.makedirs(stage2_output, exist_ok=True)
        
        # Load stage 1 model or resume checkpoint
        if args.resume_from_checkpoint and "stage2" in args.resume_from_checkpoint:
            print(f"♻️  Resuming Stage 2 from: {args.resume_from_checkpoint}")
            model_stage2 = AutoModelForTokenClassification.from_pretrained(args.resume_from_checkpoint)
        elif args.curriculum_stage1_epochs > 0:
            print(f"📥 Loading Stage 1 model from: {stage1_output}")
            model_stage2 = AutoModelForTokenClassification.from_pretrained(stage1_output)
        else:
            print(f"📥 Using base model (Stage 1 skipped)")
            model_stage2 = model
        
        stage2_args = TrainingArguments(
            output_dir=stage2_output,
            max_steps=stage2_steps,
            per_device_train_batch_size=stage2_batch_size,
            per_device_eval_batch_size=stage2_batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=args.learning_rate,
            logging_dir=f"{stage2_output}/logs",
            logging_steps=100,
            eval_strategy=eval_strategy,
            eval_steps=500 if has_eval else None,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=load_best,
            metric_for_best_model="f1" if has_eval else None,
            greater_is_better=True if has_eval else None,
            gradient_accumulation_steps=args.gradient_accumulation,
            report_to=None,
            save_total_limit=3,
            bf16=bf16_enabled,
            fp16=fp16_enabled,
            dataloader_num_workers=0,
        )
        
        # Stage 2 datasets
        stage2_train = ModernBERTStreamingDataset(
            args.curriculum_stage2_input,
            tokenizer,
            max_length=stage2_max_length,
            prefiltered=True,
            simplify_labels=args.simplify_labels
        )
        
        # Stage 2 validation: Use OOD if specified, otherwise 5% holdout
        if args.eval_ood:
            stage2_val = ood_dataset_stage2
        elif not args.no_test_split:
            stage2_val = ModernBERTStreamingDataset(
                args.curriculum_stage2_input,
                tokenizer,
                max_length=stage2_max_length,
                skip=int(0.95 * stage2_count),
                prefiltered=True,
                simplify_labels=args.simplify_labels
            )
        else:
            stage2_val = None
        
        stage2_trainer = Trainer(
            model=model_stage2,
            args=stage2_args,
            train_dataset=stage2_train,
            eval_dataset=stage2_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=create_compute_metrics(list(id_to_label.values())),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if stage2_val else [],
        )
        
        print(f"🏋️ Stage 2 training ({args.curriculum_stage2_epochs} epochs)...")
        resume_checkpoint = args.resume_from_checkpoint if args.resume_from_checkpoint and "stage2" in args.resume_from_checkpoint else None
        stage2_trainer.train(resume_from_checkpoint=resume_checkpoint)
        stage2_trainer.save_model()
        tokenizer.save_pretrained(stage2_output)
        
        # Final eval
        print(f"\n📊 Final evaluation on Stage 2...")
        eval_results = stage2_trainer.evaluate()
        
        print(f"\n📈 FINAL RESULTS:")
        for key, value in eval_results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '').upper()
                print(f"   {metric_name}: {value:.4f}")
        
        print(f"\n🎉 CURRICULUM TRAINING COMPLETE!")
        print(f"📁 Stage 1: {stage1_output}")
        print(f"📁 Stage 2 (final): {stage2_output}")
        
        final_model_path = stage2_output
        
    else:
        # SINGLE-STAGE TRAINING
        print(f"🏋️ Single-stage training...")
        print("=" * 70)
        
        # Use OOD dataset if specified and no split, otherwise use val_dataset
        final_eval_dataset = ood_dataset_stage1 if args.eval_ood else val_dataset
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=final_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=create_compute_metrics(list(id_to_label.values())),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if final_eval_dataset else [],
        )
        
        resume_checkpoint = args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Final eval
        print(f"\n📊 Final evaluation...")
        eval_results = trainer.evaluate()
        
        print(f"\n📈 FINAL RESULTS:")
        for key, value in eval_results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '').upper()
                print(f"   {metric_name}: {value:.4f}")
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"📁 Model saved to: {output_dir}")
        
        final_model_path = output_dir

    # Metadata
    import json as js
    metadata = {
        "model": model_name,
        "max_token_length": max_length,
        "total_stories": total_stories,
        "streaming": True,
        "label_mapping": label_to_id,
        "curriculum": args.curriculum,
    }
    with open(f"{final_model_path}/training_metadata.json", "w") as f:
        js.dump(metadata, f, indent=2)

    print()
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
