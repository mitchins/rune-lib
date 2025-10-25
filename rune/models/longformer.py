"""Longformer-based model for long sequence NER."""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from transformers import (
    LongformerModel,
    LongformerTokenizer,
    LongformerConfig,
    PreTrainedModel,
)

from ..core.schema import TagSchema


class LongformerNerModel(PreTrainedModel):
    """Longformer model for Named Entity Recognition with long sequences."""

    config_class = LongformerConfig
    base_model_prefix = "longformer"

    def __init__(self, config, tag_schema: TagSchema):
        """
        Initialize Longformer NER model.

        Args:
            config: Longformer configuration
            tag_schema: Schema defining entity types
        """
        super().__init__(config)
        self.tag_schema = tag_schema
        self.num_labels = len(tag_schema.get_all_tags())

        # Longformer backbone
        self.longformer = LongformerModel(config)

        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids: Token ids
            attention_mask: Attention mask
            global_attention_mask: Global attention for Longformer (optional)
            labels: Ground truth labels for training
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with loss and logits
        """
        # If no global attention mask provided, use CLS token
        if global_attention_mask is None and input_ids is not None:
            global_attention_mask = torch.zeros_like(input_ids)
            # Set global attention on CLS token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Use class weights and label smoothing to handle class imbalance
            label_smoothing = getattr(self, 'label_smoothing', 0.0)

            if hasattr(self, 'class_weights') and self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=-100,
                    weight=self.class_weights,
                    label_smoothing=label_smoothing
                )
            else:
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=-100,
                    label_smoothing=label_smoothing
                )

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

    @classmethod
    def from_pretrained_longformer(
        cls,
        model_name_or_path: str,
        tag_schema: TagSchema,
        **kwargs,
    ) -> "LongformerNerModel":
        """
        Create a LongformerNerModel from pretrained Longformer.

        Args:
            model_name_or_path: Name or path of pretrained Longformer
            tag_schema: Schema defining entity types
            **kwargs: Additional arguments for model loading

        Returns:
            LongformerNerModel instance
        """
        config = LongformerConfig.from_pretrained(model_name_or_path, **kwargs)
        model = cls(config, tag_schema)

        # Load pretrained Longformer weights
        longformer_model = LongformerModel.from_pretrained(model_name_or_path, **kwargs)
        model.longformer = longformer_model

        return model

    def get_label_mappings(self) -> Dict[str, Dict]:
        """Get label to ID and ID to label mappings."""
        return {
            "label_to_id": self.tag_schema.get_tag_to_id_mapping(),
            "id_to_label": self.tag_schema.get_id_to_tag_mapping(),
        }

    def resize_position_embeddings(self, new_max_position_embeddings: int):
        """
        Resize position embeddings for longer sequences.

        Args:
            new_max_position_embeddings: New maximum sequence length
        """
        old_embeddings = self.longformer.embeddings.position_embeddings
        new_embeddings = self.longformer._get_resized_embeddings(
            old_embeddings, new_max_position_embeddings
        )
        self.longformer.embeddings.position_embeddings = new_embeddings
        self.longformer.embeddings.position_ids = torch.arange(
            new_max_position_embeddings
        ).expand((1, -1))
        self.config.max_position_embeddings = new_max_position_embeddings