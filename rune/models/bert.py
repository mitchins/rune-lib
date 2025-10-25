"""BERT-based model for named entity recognition."""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..core.schema import TagSchema


class BertNerModel(PreTrainedModel):
    """BERT model for Named Entity Recognition with configurable tag schemas."""

    def __init__(self, config, tag_schema: TagSchema):
        super().__init__(config)
        self.tag_schema = tag_schema
        self.num_labels = len(tag_schema.get_all_tags())

        # BERT backbone
        self.bert = AutoModel.from_config(config)

        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids: Token ids
            attention_mask: Attention mask
            token_type_ids: Token type ids
            labels: Ground truth labels for training
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with loss and logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

    @classmethod
    def from_pretrained_bert(
        cls,
        model_name_or_path: str,
        tag_schema: TagSchema,
        **kwargs,
    ) -> "BertNerModel":
        """
        Create a BertNerModel from a pretrained BERT model.

        Args:
            model_name_or_path: Name or path of pretrained BERT model
            tag_schema: Schema defining the entity types
            **kwargs: Additional arguments for model loading

        Returns:
            BertNerModel instance
        """
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        model = cls(config, tag_schema)

        # Load pretrained BERT weights
        bert_model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        model.bert = bert_model

        return model

    def get_tokenizer(self, model_name_or_path: str) -> PreTrainedTokenizer:
        """Get the tokenizer for this model."""
        return AutoTokenizer.from_pretrained(model_name_or_path)

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """Resize token embeddings if needed."""
        return self.bert.resize_token_embeddings(new_num_tokens)

    def get_label_mappings(self) -> Dict[str, Dict]:
        """Get label to ID and ID to label mappings."""
        return {
            "label_to_id": self.tag_schema.get_tag_to_id_mapping(),
            "id_to_label": self.tag_schema.get_id_to_tag_mapping(),
        }