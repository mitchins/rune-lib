"""Evaluation metrics for NER tasks."""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from ..core.schema import BioSequence


def evaluate_ner(
    true_sequences: List[BioSequence],
    pred_sequences: List[BioSequence],
) -> Dict[str, Any]:
    """
    Evaluate NER predictions against ground truth.

    Args:
        true_sequences: Ground truth sequences
        pred_sequences: Predicted sequences

    Returns:
        Dictionary with evaluation metrics
    """
    if len(true_sequences) != len(pred_sequences):
        raise ValueError("Number of true and predicted sequences must match")

    # Convert to seqeval format
    true_labels = []
    pred_labels = []

    for true_seq, pred_seq in zip(true_sequences, pred_sequences):
        if len(true_seq.tokens) != len(pred_seq.tokens):
            raise ValueError("Token sequences must have the same length")

        true_tags = [str(tag) for tag in true_seq.tags]
        pred_tags = [str(tag) for tag in pred_seq.tags]

        true_labels.append(true_tags)
        pred_labels.append(pred_tags)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
    }

    # Detailed classification report
    report = classification_report(true_labels, pred_labels, output_dict=True)
    metrics["classification_report"] = report

    return metrics


def compute_entity_metrics(
    true_sequences: List[BioSequence],
    pred_sequences: List[BioSequence],
) -> Dict[str, Any]:
    """
    Compute entity-level metrics.

    Args:
        true_sequences: Ground truth sequences
        pred_sequences: Predicted sequences

    Returns:
        Dictionary with entity-level metrics
    """
    true_entities = []
    pred_entities = []

    for true_seq, pred_seq in zip(true_sequences, pred_sequences):
        # Extract entities from sequences
        true_ents = _extract_entities_with_positions(true_seq)
        pred_ents = _extract_entities_with_positions(pred_seq)

        true_entities.extend(true_ents)
        pred_entities.extend(pred_ents)

    # Convert to sets for comparison
    true_set = set(true_entities)
    pred_set = set(pred_entities)

    # Compute metrics
    tp = len(true_set & pred_set)  # True positives
    fp = len(pred_set - true_set)  # False positives
    fn = len(true_set - pred_set)  # False negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-entity type metrics
    entity_type_metrics = _compute_per_entity_type_metrics(true_entities, pred_entities)

    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "per_entity_type": entity_type_metrics,
    }


def _extract_entities_with_positions(sequence: BioSequence) -> List[Tuple[str, int, int]]:
    """
    Extract entities with their positions from a sequence.

    Args:
        sequence: Bio-tagged sequence

    Returns:
        List of (entity_type, start_pos, end_pos) tuples
    """
    entities = []
    current_entity = None

    for i, tag in enumerate(sequence.tags):
        if tag.prefix.value == "B":
            # Start of new entity
            if current_entity:
                entities.append(current_entity)
            current_entity = (tag.entity_type, i, i)
        elif tag.prefix.value == "I" and current_entity and tag.entity_type == current_entity[0]:
            # Continuation of current entity
            current_entity = (current_entity[0], current_entity[1], i)
        else:
            # End of current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    # Don't forget the last entity
    if current_entity:
        entities.append(current_entity)

    return entities


def _compute_per_entity_type_metrics(
    true_entities: List[Tuple[str, int, int]],
    pred_entities: List[Tuple[str, int, int]],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per entity type."""
    # Group by entity type
    true_by_type = defaultdict(set)
    pred_by_type = defaultdict(set)

    for entity in true_entities:
        true_by_type[entity[0]].add(entity)

    for entity in pred_entities:
        pred_by_type[entity[0]].add(entity)

    # Get all entity types
    all_types = set(true_by_type.keys()) | set(pred_by_type.keys())

    metrics = {}
    for entity_type in all_types:
        true_set = true_by_type[entity_type]
        pred_set = pred_by_type[entity_type]

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(true_set),
        }

    return metrics