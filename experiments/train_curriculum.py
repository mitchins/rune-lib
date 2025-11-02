#!/usr/bin/env python3
"""
Curriculum learning trainer for ModernBERT NER.

Three-phase curriculum:
  Phase 1: Short scenes (avg 884 tokens) - learn entity boundaries
  Phase 2: Longer scenes + shorter stories (1162-4590 tokens) - bridge to full context
  Phase 3: Full stories (avg 4572 tokens) - long-range consistency
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from typing import Dict, List, Iterator, Any
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import re


class CurriculumStreamingDataset(IterableDataset):
    """Streaming dataset for curriculum learning phases."""

    def __init__(
        self,
        scenes_file: str,
        stories_file: str,
        phase: int,
        tokenizer,
        max_length: int,
        simplify_labels: bool,
        label_to_id: Dict[str, int]
    ):
        """
        Initialize streaming dataset for a specific phase.

        Args:
            scenes_file: Path to scene-level JSONL
            stories_file: Path to story-level JSONL
            phase: Phase number (1, 2, or 3)
            tokenizer: ModernBERT tokenizer
            max_length: Max sequence length
            simplify_labels: If True, collapse role labels to B-PERSON/I-PERSON
            label_to_id: Label to ID mapping
        """
        self.scenes_file = scenes_file
        self.stories_file = stories_file
        self.phase = phase
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.simplify_labels = simplify_labels
        self.label_to_id = label_to_id

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples for this phase."""

        if self.phase == 1:
            # Phase 1: All scenes
            yield from self._iter_file(self.scenes_file, min_tokens=0, max_tokens=999999)

        elif self.phase == 2:
            # Phase 2: Longer scenes (>= 1162 tokens) + shorter stories (<= 4590 tokens)
            # First: longer scenes
            yield from self._iter_file(self.scenes_file, min_tokens=1162, max_tokens=999999)
            # Then: shorter stories
            yield from self._iter_file(self.stories_file, min_tokens=0, max_tokens=4590)

        elif self.phase == 3:
            # Phase 3: All full stories
            yield from self._iter_file(self.stories_file, min_tokens=0, max_tokens=999999)

    def _iter_file(self, filepath: str, min_tokens: int, max_tokens: int) -> Iterator[Dict[str, Any]]:
        """Stream samples from a file with token length filtering."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())

                    # Filter by token length
                    token_count = len(sample['tokens'])
                    if token_count < min_tokens or token_count > max_tokens:
                        continue

                    # Tokenize and align
                    tokenized = self._tokenize_and_align(sample)

                    # HARD LIMIT: Skip if tokenized length exceeds phase-specific max for memory safety
                    actual_length = len(tokenized['input_ids'])
                    phase_limits = {
                        1: 1500,  # Phase 1: short scenes, batch=12, max 18k tokens/batch
                        2: 3000,  # Phase 2: mixed scenes/stories, batch=6, max 18k tokens/batch
                        3: 8000,  # Phase 3: full stories, batch=4, max 32k tokens/batch
                    }
                    if actual_length > phase_limits.get(self.phase, 8000):
                        continue

                    yield tokenized

                except Exception as e:
                    print(f"⚠️  Error processing sample: {e}")
                    continue

    def _tokenize_and_align(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize and align labels."""
        tokens = sample['tokens']
        bio_tags = sample['bio_tags']

        # Simplify labels if needed
        if self.simplify_labels:
            simplified_tags = []
            for tag in bio_tags:
                if tag.startswith('B-') and tag not in ['B-LOCATION']:
                    simplified_tags.append('B-PERSON')
                elif tag.startswith('I-') and tag not in ['I-LOCATION']:
                    simplified_tags.append('I-PERSON')
                else:
                    simplified_tags.append(tag)
            bio_tags = simplified_tags

        # Tokenize (NO padding here - let DataCollator handle it dynamically)
        tokenized_inputs = self.tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            padding=False,  # Dynamic padding by DataCollator
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
                labels.append(self.label_to_id.get(tag, 0))
            else:
                labels.append(-100)
            previous_word_idx = word_idx

        return {
            "input_ids": tokenized_inputs["input_ids"].squeeze(0),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def count_phase_samples(scenes_file: str, stories_file: str, phase: int) -> tuple:
    """
    Count samples for a specific training phase WITHOUT loading into memory.

    Returns:
        (phase_name, sample_count)
    """
    print(f"\n{'='*80}")
    print(f"COUNTING DATA FOR PHASE {phase}")
    print(f"{'='*80}\n")

    count = 0

    if phase == 1:
        # Phase 1: All scenes
        print(f"Counting all scenes from {scenes_file}...")
        with open(scenes_file, 'r') as f:
            for line in f:
                count += 1

        phase_name = "phase1_scenes"
        print(f"Found {count} scenes (avg ~884 tokens)")

    elif phase == 2:
        # Phase 2: Longer scenes (>= 1162 tokens) + shorter stories (<= 4590 tokens)
        print(f"Counting longer scenes and shorter stories...")

        # Count longer scenes
        scene_count = 0
        with open(scenes_file, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                if len(sample['tokens']) >= 1162:
                    scene_count += 1

        print(f"  Found {scene_count} longer scenes (>= 1162 tokens)")

        # Count shorter stories
        story_count = 0
        with open(stories_file, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                if len(sample['tokens']) <= 4590:
                    story_count += 1

        print(f"  Found {story_count} shorter stories (<= 4590 tokens)")

        count = scene_count + story_count
        phase_name = "phase2_mixed"
        print(f"Total Phase 2 samples: {count}")

    elif phase == 3:
        # Phase 3: All full stories
        print(f"Counting all full stories from {stories_file}...")
        with open(stories_file, 'r') as f:
            for line in f:
                count += 1

        phase_name = "phase3_stories"
        print(f"Found {count} full stories (avg ~4572 tokens)")

    else:
        raise ValueError(f"Invalid phase: {phase}")

    return phase_name, count


def compute_metrics(eval_pred):
    """Compute F1, precision, recall for evaluation."""
    from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    # Remove ignored index (special tokens)
    true_labels = [[label_list[l] for l in label if l != -100]
                   for label, in zip(labels)]
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]

    # Compute entity-level detection F1 (collapse all entity types to generic ENTITY)
    entity_true_labels = []
    entity_true_predictions = []

    for true_seq, pred_seq in zip(true_labels, true_predictions):
        entity_true = []
        entity_pred = []

        for t, p in zip(true_seq, pred_seq):
            # Map all B-/I- tags to generic entity tags
            if t.startswith('B-'):
                entity_true.append('B-ENTITY')
            elif t.startswith('I-'):
                entity_true.append('I-ENTITY')
            else:
                entity_true.append(t)

            if p.startswith('B-'):
                entity_pred.append('B-ENTITY')
            elif p.startswith('I-'):
                entity_pred.append('I-ENTITY')
            else:
                entity_pred.append(p)

        entity_true_labels.append(entity_true)
        entity_true_predictions.append(entity_pred)

    return {
        'f1': f1_score(true_labels, true_predictions),
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'accuracy': accuracy_score(true_labels, true_predictions),
        'entity_detection_f1': f1_score(entity_true_labels, entity_true_predictions),
        'entity_accuracy': accuracy_score(entity_true_labels, entity_true_predictions),
    }


def main():
    parser = argparse.ArgumentParser(description="Curriculum NER training with ModernBERT")
    parser.add_argument('--scenes-file', default='ner_training_scenes.jsonl', help="Scene-level data file")
    parser.add_argument('--stories-file', default='ner_training_ready.jsonl', help="Story-level data file")
    parser.add_argument('--output', required=True, help="Output directory for final model")
    parser.add_argument('--validation-file', help="Optional validation file for eval during training")
    parser.add_argument('--max-length', type=int, default=8000, help="Max sequence length")
    parser.add_argument('--simplify-labels', action='store_true', help="Collapse all roles to B/I-PERSON")
    parser.add_argument('--start-phase', type=int, default=1, choices=[1, 2, 3],
                        help="Which phase to start from (for resuming)")
    parser.add_argument('--resume-from', help="Checkpoint to resume from (for continuing a phase)")

    args = parser.parse_args()

    # Setup label schema
    global label_list, label_to_id
    if args.simplify_labels:
        label_list = ['O', 'B-PERSON', 'I-PERSON']
    else:
        # Will be populated from data
        label_list = ['O', 'B-PERSON', 'I-PERSON', 'B-LOCATION', 'I-LOCATION']

    label_to_id = {label: i for i, label in enumerate(label_list)}

    # Load tokenizer and model base
    model_name = "answerdotai/ModernBERT-base"
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create output directory structure
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Training phases
    phases = [1, 2, 3]
    if args.start_phase > 1:
        phases = [p for p in phases if p >= args.start_phase]
        print(f"\nResuming from Phase {args.start_phase}")

    current_model_path = args.resume_from if args.resume_from else model_name

    for phase in phases:
        print(f"\n{'#'*80}")
        print(f"# PHASE {phase}")
        print(f"{'#'*80}\n")

        # Count phase samples (without loading into memory)
        phase_name, sample_count = count_phase_samples(
            args.scenes_file, args.stories_file, phase
        )

        # Create streaming dataset
        print(f"Creating streaming dataset...")
        train_dataset = CurriculumStreamingDataset(
            scenes_file=args.scenes_file,
            stories_file=args.stories_file,
            phase=phase,
            tokenizer=tokenizer,
            max_length=args.max_length,
            simplify_labels=args.simplify_labels,
            label_to_id=label_to_id
        )
        print(f"✅ Streaming dataset created (will stream {sample_count:,} samples)")

        # Load model (from checkpoint if continuing)
        print(f"Loading model from {current_model_path}...")
        model = AutoModelForTokenClassification.from_pretrained(
            current_model_path,
            num_labels=len(label_list),
            id2label={i: label for i, label in enumerate(label_list)},
            label2id=label_to_id
        )

        # Calculate steps for this phase (1 epoch)
        # Phase-specific batch sizes to maximize VRAM usage (32GB VRAM)
        if phase == 1:
            # Phase 1: Short scenes (~884 tokens avg) - dynamic padding
            # Reduced to 12 for memory safety (max scene: 2358 BERT tokens)
            batch_size = 12
            gradient_accumulation_steps = 1
            eval_batch_size = 1  # Minimal eval batch to avoid OOM during checkpoint
        elif phase == 2:
            # Phase 2: Medium length (~2876 tokens avg)
            batch_size = 6
            gradient_accumulation_steps = 3
            eval_batch_size = 1  # Minimal eval batch
        else:
            # Phase 3: Full stories (~4572 tokens avg, max 8000 tokens)
            # Reduced to 2 for memory safety with 8000 token sequences
            batch_size = 2
            gradient_accumulation_steps = 8
            eval_batch_size = 1  # Very small eval batch for long sequences

        effective_batch_size = batch_size * gradient_accumulation_steps
        steps_per_epoch = sample_count // effective_batch_size
        max_steps = steps_per_epoch
        save_steps = max(steps_per_epoch // 10, 100)  # Save 10 times per phase
        eval_steps = save_steps

        print(f"\nPhase {phase} Training Config:")
        print(f"  Samples: {sample_count:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch:,}")
        print(f"  Max steps: {max_steps:,}")
        print(f"  Save every: {save_steps} steps")
        print(f"  Eval every: {eval_steps} steps")

        # Phase-specific output directory
        phase_output_dir = output_dir / phase_name
        phase_output_dir.mkdir(exist_ok=True, parents=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(phase_output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,  # Smaller eval batch to avoid OOM
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type='cosine',
            save_steps=save_steps,
            save_total_limit=5,
            eval_strategy='steps' if args.validation_file else 'no',
            eval_steps=eval_steps if args.validation_file else None,
            eval_accumulation_steps=1,  # Clear memory after each eval step
            logging_steps=50,
            load_best_model_at_end=True if args.validation_file else False,
            metric_for_best_model='f1' if args.validation_file else None,
            greater_is_better=True,
            push_to_hub=False,
            report_to='none',
            fp16=True,
            dataloader_num_workers=0,  # Required for IterableDataset
            remove_unused_columns=False,  # Streaming dataset manages its own columns
        )

        # Data collator (dynamic padding to longest in batch)
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,  # Dynamic padding to longest sequence in batch
            max_length=args.max_length,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics if args.validation_file else None,
        )

        # Train this phase
        print(f"\n🚀 Starting Phase {phase} training...")
        trainer.train()

        # Save phase checkpoint
        phase_checkpoint = output_dir / f"phase{phase}_complete"
        print(f"\n💾 Saving Phase {phase} checkpoint to {phase_checkpoint}...")
        trainer.save_model(str(phase_checkpoint))

        # Update current model path for next phase
        current_model_path = str(phase_checkpoint)

        print(f"\n✅ Phase {phase} complete!")

    # Save final model
    final_output = output_dir / "final"
    print(f"\n💾 Saving final curriculum-trained model to {final_output}...")
    trainer.save_model(str(final_output))

    print(f"\n{'='*80}")
    print(f"🎉 CURRICULUM TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Final model saved to: {final_output}")
    print(f"\nPhase checkpoints:")
    for p in phases:
        print(f"  Phase {p}: {output_dir / f'phase{p}_complete'}")


if __name__ == '__main__':
    main()
