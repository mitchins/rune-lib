"""Tests for utility functions."""

import pytest
from rune.core.schema import BioSequence, BioTag
from rune.utils.metrics import evaluate_ner, compute_entity_metrics


class TestMetrics:
    """Test evaluation metrics."""

    def create_sample_sequences(self):
        """Create sample sequences for testing."""
        # Ground truth sequence
        true_seq = BioSequence.from_iob_format(
            ["John", "Doe", "works", "at", "Google", "in", "California"],
            ["B-PERSON", "I-PERSON", "O", "O", "B-ORG", "O", "B-LOC"]
        )

        # Perfect prediction
        pred_seq_perfect = BioSequence.from_iob_format(
            ["John", "Doe", "works", "at", "Google", "in", "California"],
            ["B-PERSON", "I-PERSON", "O", "O", "B-ORG", "O", "B-LOC"]
        )

        # Prediction with some errors
        pred_seq_errors = BioSequence.from_iob_format(
            ["John", "Doe", "works", "at", "Google", "in", "California"],
            ["B-PERSON", "O", "O", "O", "B-ORG", "B-LOC", "I-LOC"]
        )

        return true_seq, pred_seq_perfect, pred_seq_errors

    def test_evaluate_ner_perfect(self):
        """Test evaluation with perfect predictions."""
        true_seq, pred_seq_perfect, _ = self.create_sample_sequences()

        metrics = evaluate_ner([true_seq], [pred_seq_perfect])

        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_evaluate_ner_with_errors(self):
        """Test evaluation with some prediction errors."""
        true_seq, _, pred_seq_errors = self.create_sample_sequences()

        metrics = evaluate_ner([true_seq], [pred_seq_errors])

        # Should have some errors
        assert metrics["accuracy"] < 1.0
        assert metrics["f1"] < 1.0

        # Should still have classification report
        assert "classification_report" in metrics

    def test_evaluate_ner_mismatched_length(self):
        """Test evaluation with mismatched sequence lengths."""
        true_seq, pred_seq_perfect, _ = self.create_sample_sequences()

        # Create prediction with different length
        pred_seq_short = BioSequence.from_iob_format(
            ["John", "Doe"],
            ["B-PERSON", "I-PERSON"]
        )

        with pytest.raises(ValueError):
            evaluate_ner([true_seq], [pred_seq_short])

    def test_compute_entity_metrics_perfect(self):
        """Test entity-level metrics with perfect predictions."""
        true_seq, pred_seq_perfect, _ = self.create_sample_sequences()

        metrics = compute_entity_metrics([true_seq], [pred_seq_perfect])

        assert metrics["entity_precision"] == 1.0
        assert metrics["entity_recall"] == 1.0
        assert metrics["entity_f1"] == 1.0
        assert metrics["true_positives"] == 3  # PERSON, ORG, LOC
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0

    def test_compute_entity_metrics_with_errors(self):
        """Test entity-level metrics with some errors."""
        true_seq, _, pred_seq_errors = self.create_sample_sequences()

        metrics = compute_entity_metrics([true_seq], [pred_seq_errors])

        # Should have some errors at entity level
        assert metrics["entity_precision"] < 1.0
        assert metrics["entity_recall"] < 1.0
        assert metrics["entity_f1"] < 1.0

        # Should have per-entity-type metrics
        assert "per_entity_type" in metrics

    def test_empty_sequences(self):
        """Test metrics with empty sequences."""
        empty_seq = BioSequence.from_iob_format(["hello", "world"], ["O", "O"])

        metrics = evaluate_ner([empty_seq], [empty_seq])
        assert metrics["accuracy"] == 1.0

        entity_metrics = compute_entity_metrics([empty_seq], [empty_seq])
        assert entity_metrics["entity_precision"] == 0.0  # No entities to measure
        assert entity_metrics["true_positives"] == 0

    def test_per_entity_type_metrics(self):
        """Test per-entity-type breakdown."""
        true_seq, _, pred_seq_errors = self.create_sample_sequences()

        metrics = compute_entity_metrics([true_seq], [pred_seq_errors])
        per_type = metrics["per_entity_type"]

        # Should have metrics for each entity type that appears
        assert "PERSON" in per_type or "ORG" in per_type or "LOC" in per_type

        # Each entity type should have precision, recall, f1, support
        for entity_type, type_metrics in per_type.items():
            assert "precision" in type_metrics
            assert "recall" in type_metrics
            assert "f1" in type_metrics
            assert "support" in type_metrics