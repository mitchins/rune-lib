#!/usr/bin/env python3
"""
Train RUNE NER with ModernBERT streaming dataset.
Uses ModernBERT's 8192 token context to handle longer stories.
"""

import os
import json
import torch
from typing import Iterator, Dict, Any
from torch.utils.data import IterableDataset
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
        """
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skip = skip
        self.limit = limit
        self.prefiltered = prefiltered

        # Initialize preprocessor if not prefiltered
        if not prefiltered:
            self.preprocessor = StoryPreprocessor(use_spacy=False)

        # Label mappings
        self.label_to_id = {
            'O': 0,
            'B-ANTAGONIST': 1,
            'B-PROTAGONIST': 2,
            'B-SUPPORTING': 3,
            'I-ANTAGONIST': 4,
            'I-PROTAGONIST': 5,
            'I-SUPPORTING': 6,
        }
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


def main():
    """Main training function with ModernBERT."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ModernBERT NER model")
    parser.add_argument('--input', default="ner_training_filtered_8k.jsonl", help="Input JSONL file")
    parser.add_argument('--output', default="./story_ner_model_modernbert", help="Output directory")
    parser.add_argument('--max-length', type=int, default=8192, help="Max token length")
    args = parser.parse_args()

    print("🚀 TRAINING RUNE NER WITH MODERNBERT")
    print("=" * 70)
    print("🤖 Model: ModernBERT-base (8192 token context)")
    print("💾 Memory-efficient streaming")
    print("📊 Dataset: Synthetic stories with ground truth")
    print(f"📁 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    print("=" * 70)
    print()

    # Configuration
    model_name = "answerdotai/ModernBERT-base"
    max_length = args.max_length
    training_file = args.input

    # Count total stories
    with open(training_file, "r") as f:
        total_stories = sum(1 for _ in f)

    print(f"📚 Total clean stories: {total_stories}")
    print(f"   Max token length: {max_length}")
    print(f"   All stories fit within token limit (pre-filtered)")
    print(f"   Compare: Longformer had only 5,028 stories (4096 limit)")
    print()

    # Calculate split
    val_split_idx = int(0.9 * total_stories)
    train_count = val_split_idx
    val_count = total_stories - val_split_idx

    print(f"📊 Train/Val Split:")
    print(f"   Training: {train_count} stories")
    print(f"   Validation: {val_count} stories")
    print()

    # Load ModernBERT tokenizer and model
    print(f"🔄 Loading ModernBERT...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        prefiltered=True  # Data is already pre-processed
    )

    val_dataset = ModernBERTStreamingDataset(
        training_file,
        tokenizer,
        max_length=max_length,
        skip=val_split_idx,
        limit=val_count,
        prefiltered=True  # Data is already pre-processed
    )
    print("✅ Streaming datasets created (using pre-filtered data)")
    print()

    # Output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments - Conservative for variable-length sequences
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=3,  # Conservative for uneven batch lengths
        per_device_eval_batch_size=3,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Use F1 for span integrity (not entity_accuracy)
        greater_is_better=True,
        gradient_accumulation_steps=5,  # Effective batch size: 15 (3*5)
        report_to=None,
        save_total_limit=3,
        fp16=True,
        dataloader_num_workers=0,
        max_steps=10000,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=create_compute_metrics(list(id_to_label.values())),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"🏋️ Starting ModernBERT training...")
    print(f"   Max steps: {training_args.max_steps}")
    print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   Evaluation every: {training_args.eval_steps} steps")
    print("-" * 70)
    print()

    try:
        # Train
        trainer.train()

        print(f"\n✅ Training completed!")
        print()

        # Save
        print(f"💾 Saving model to {output_dir}...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Metadata
        import json as js
        metadata = {
            "model": model_name,
            "max_token_length": max_length,
            "total_stories": total_stories,
            "streaming": True,
            "label_mapping": label_to_id,
        }
        with open(f"{output_dir}/training_metadata.json", "w") as f:
            js.dump(metadata, f, indent=2)

        # Final evaluation
        print(f"📊 Running final evaluation...")
        eval_results = trainer.evaluate()
        print()

        print(f"📈 FINAL RESULTS:")
        for key, value in eval_results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '').upper()
                print(f"   {metric_name}: {value:.4f}")
        print()

        print(f"🎉 SUCCESS! ModernBERT training completed")
        print(f"📁 Model saved to: {output_dir}")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
