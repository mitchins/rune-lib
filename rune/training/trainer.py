"""Training utilities for bio-tagging models."""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import accuracy_score, classification_report, f1_score
import numpy as np

from ..models.bert import BertNerModel
from ..data.dataset import NerDataset
from ..core.schema import TagSchema


class BioTaggerTrainer:
    """Trainer for bio-tagging models using HuggingFace Transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        tag_schema: TagSchema,
        output_dir: Union[str, Path] = "./output",
        max_length: int = 512,
    ):
        """
        Initialize the trainer.

        Args:
            model_name_or_path: Pre-trained model name or path
            tag_schema: Schema defining entity types
            output_dir: Directory to save outputs
            max_length: Maximum sequence length
        """
        self.model_name_or_path = model_name_or_path
        self.tag_schema = tag_schema
        self.output_dir = Path(output_dir)
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Initialize model
        self.model = BertNerModel.from_pretrained_bert(
            model_name_or_path, tag_schema
        )

        # Data collator
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=max_length,
        )

        # Mappings
        self.id_to_label = tag_schema.get_id_to_tag_mapping()
        self.label_to_id = tag_schema.get_tag_to_id_mapping()

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id_to_label[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id_to_label[l] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Compute metrics using seqeval
        f1 = f1_score(true_labels, true_predictions)
        accuracy = accuracy_score(true_labels, true_predictions)

        return {
            "f1": f1,
            "accuracy": accuracy,
        }

    def train(
        self,
        train_dataset: NerDataset,
        eval_dataset: Optional[NerDataset] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 500,
        early_stopping_patience: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Training batch size per device
            per_device_eval_batch_size: Evaluation batch size per device
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency
            eval_steps: Evaluation frequency
            save_steps: Save frequency
            early_stopping_patience: Early stopping patience
            **kwargs: Additional training arguments

        Returns:
            Training history
        """
        # Convert to HuggingFace datasets
        train_hf_dataset = train_dataset.to_hf_dataset()
        eval_hf_dataset = eval_dataset.to_hf_dataset() if eval_dataset else None

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps if eval_hf_dataset else None,
            save_steps=save_steps,
            evaluation_strategy="steps" if eval_hf_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_hf_dataset else False,
            metric_for_best_model="f1" if eval_hf_dataset else None,
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard by default
            **kwargs,
        )

        # Callbacks
        callbacks = []
        if eval_hf_dataset and early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_hf_dataset,
            eval_dataset=eval_hf_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics if eval_hf_dataset else None,
            callbacks=callbacks,
        )

        # Train
        train_result = trainer.train()

        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        return train_result

    def evaluate(self, eval_dataset: NerDataset) -> Dict[str, Any]:
        """
        Evaluate the model.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Evaluation metrics
        """
        eval_hf_dataset = eval_dataset.to_hf_dataset()

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=16,
            report_to=None,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_hf_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        return trainer.evaluate()

    def save_model(self, save_path: Union[str, Path]) -> None:
        """Save the trained model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save tag schema
        import json
        schema_path = save_path / "tag_schema.json"
        with open(schema_path, "w") as f:
            json.dump({
                "entity_types": self.tag_schema.entity_types,
                "description": self.tag_schema.description,
            }, f, indent=2)

    @classmethod
    def load_model(cls, model_path: Union[str, Path]) -> "BioTaggerTrainer":
        """Load a trained model."""
        model_path = Path(model_path)

        # Load tag schema
        import json
        schema_path = model_path / "tag_schema.json"
        with open(schema_path, "r") as f:
            schema_data = json.load(f)

        tag_schema = TagSchema.create_standard_schema(
            schema_data["entity_types"],
            schema_data.get("description")
        )

        trainer = cls(str(model_path), tag_schema)
        trainer.model = BertNerModel.from_pretrained_bert(str(model_path), tag_schema)

        return trainer