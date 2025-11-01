#!/usr/bin/env python3
"""
Unit tests for the improved tokenizer with synthetic diverse passage.
Core test for punctuation inheritance bug fix.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import LongformerTokenizerFast
from rune.data.improved_tokenizer import ImprovedTokenizer
from rune.data.story_dataset import StoryNerDataset

class TestImprovedTokenizer:
    """Core unit tests for improved tokenizer functionality."""

    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        cls.improved_tokenizer = ImprovedTokenizer()

        # Create synthetic diverse passage with complex names
        cls.test_passage = """Detective Maria O'Sullivan arrived at the crime scene just as the rain began to fall. She had worked with forensic expert Dr. J.K. Morrison on dozens of cases, but this one felt different. The victim, wealthy businessman Robert Chen, lay motionless in his study.

"What do you think, J.K.?" Maria asked, her Irish accent thick with concern. Dr. Morrison adjusted his glasses and knelt beside the body. "This wasn't random, O'Sullivan. Look at the positioning."

Chen's business partner, the notorious Vincent D'Angelo, had an alibi that seemed too convenient. Maria made a mental note to question D'Angelo again. Something about his story didn't add up, and she trusted her instincts."""

        # Define character roles for testing
        cls.test_characters = [
            {"name": "Maria O'Sullivan", "role": "PROTAGONIST"},
            {"name": "O'Sullivan", "role": "PROTAGONIST"},  # Last name reference
            {"name": "Maria", "role": "PROTAGONIST"},       # First name reference
            {"name": "Dr. J.K. Morrison", "role": "SUPPORTING"},
            {"name": "J.K. Morrison", "role": "SUPPORTING"},
            {"name": "J.K.", "role": "SUPPORTING"},         # Initials only
            {"name": "Dr. Morrison", "role": "SUPPORTING"},
            {"name": "Morrison", "role": "SUPPORTING"},
            {"name": "Robert Chen", "role": "SUPPORTING"},
            {"name": "Chen", "role": "SUPPORTING"},
            {"name": "Robert", "role": "SUPPORTING"},
            {"name": "Vincent D'Angelo", "role": "ANTAGONIST"},
            {"name": "D'Angelo", "role": "ANTAGONIST"},
            {"name": "Vincent", "role": "ANTAGONIST"}
        ]

    def test_word_splitting_preserves_names(self):
        """Test that complex names are preserved while punctuation is separated."""
        words = self.improved_tokenizer.split_words(self.test_passage)

        # Check that complex names are preserved as single tokens
        assert "O'Sullivan" in words, "O'Sullivan should be preserved as single token"
        assert "J.K." in words, "J.K. should be preserved as single token"
        assert "D'Angelo" in words, "D'Angelo should be preserved as single token"

        # Check that boundary punctuation is separated
        assert "," in words, "Commas should be separated"
        assert "." in words, "Periods should be separated"
        assert "?" in words, "Question marks should be separated"

        # Verify specific patterns don't get over-split
        assert "J" not in words or words.count("J") == 0, "J.K. should not be split into J + . + K + ."

        print(f"✅ Word splitting test passed. Found {len(words)} tokens.")

    def test_punctuation_not_in_entity_names(self):
        """Test that standalone punctuation doesn't inherit entity labels."""
        words = self.improved_tokenizer.split_words(self.test_passage)
        bio_tags = self.improved_tokenizer._create_bio_tags(words, self.test_characters)

        # Find punctuation tokens and their tags
        punctuation_tags = []
        for word, tag in zip(words, bio_tags):
            if self.improved_tokenizer.is_punctuation_only(word):
                punctuation_tags.append((word, tag))

        # All punctuation should have 'O' tags
        for punct, tag in punctuation_tags:
            assert tag == 'O', f"Punctuation '{punct}' should have O tag, got {tag}"

        print(f"✅ Punctuation tagging test passed. {len(punctuation_tags)} punctuation tokens all tagged as O.")

    def test_entity_detection_accuracy(self):
        """Test that all expected entities are correctly detected."""
        words = self.improved_tokenizer.split_words(self.test_passage)
        bio_tags = self.improved_tokenizer._create_bio_tags(words, self.test_characters)

        # Count detected entities by role
        entity_counts = {"PROTAGONIST": 0, "SUPPORTING": 0, "ANTAGONIST": 0}

        for tag in bio_tags:
            if tag.startswith("B-"):
                role = tag.split("-")[1]
                if role in entity_counts:
                    entity_counts[role] += 1

        # Should find at least one of each role
        assert entity_counts["PROTAGONIST"] >= 1, f"Should detect PROTAGONIST entities, found {entity_counts['PROTAGONIST']}"
        assert entity_counts["SUPPORTING"] >= 1, f"Should detect SUPPORTING entities, found {entity_counts['SUPPORTING']}"
        assert entity_counts["ANTAGONIST"] >= 1, f"Should detect ANTAGONIST entities, found {entity_counts['ANTAGONIST']}"

        print(f"✅ Entity detection test passed: {entity_counts}")

    def test_integration_with_dataset(self):
        """Test integration with StoryNerDataset for end-to-end validation."""
        # Create mock tokenizer for testing
        model_path = os.path.join(os.path.dirname(__file__), "..", "story_ner_model_augmented")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Required model not found at {model_path}. Cannot skip core integration test.")

        tokenizer = LongformerTokenizerFast.from_pretrained(model_path, add_prefix_space=True)

        # Create test story in dataset format
        words = self.improved_tokenizer.split_words(self.test_passage)
        bio_tags = self.improved_tokenizer._create_bio_tags(words, self.test_characters)

        test_story = {
            "story_id": "unit_test_synthetic",
            "genre": "mystery",
            "text": self.test_passage,
            "tokens": words,
            "bio_tags": bio_tags,
            "entities": []
        }

        # Create dataset and process
        dataset = StoryNerDataset(tokenizer=tokenizer, max_length=256)
        dataset.stories = [test_story]

        # Setup label mapping
        dataset.label_to_id = {
            'O': 0,
            'B-PROTAGONIST': 1, 'I-PROTAGONIST': 2,
            'B-ANTAGONIST': 3, 'I-ANTAGONIST': 4,
            'B-SUPPORTING': 5, 'I-SUPPORTING': 6
        }
        dataset.id_to_label = {v: k for k, v in dataset.label_to_id.items()}

        # Process story
        processed = dataset[0]

        # Verify no punctuation gets entity labels
        input_ids = processed["input_ids"]
        labels = processed["labels"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        punctuation_issues = []
        for token, label_id in zip(tokens, labels):
            if token in [',', '.', '!', '?', ';', ':', '"', "'"] and label_id.item() > 0:
                punctuation_issues.append((token, label_id.item()))

        assert len(punctuation_issues) == 0, f"Found punctuation with entity labels: {punctuation_issues}"

        print(f"✅ Integration test passed. No punctuation inheritance detected.")

    def test_complex_name_patterns(self):
        """Test specific complex name patterns that caused issues."""
        test_cases = [
            ("Dr. J.K. Morrison examined the evidence.", ["J.K."]),
            ("Maria O'Sullivan's investigation continues.", ["O'Sullivan's"]),
            ("Vincent D'Angelo, the suspect, fled.", ["D'Angelo"]),
            ("She met with Prof. Smith-Jones yesterday.", ["Smith-Jones"]),
        ]

        for text, expected_names in test_cases:
            words = self.improved_tokenizer.split_words(text)

            for expected_name in expected_names:
                assert expected_name in words, f"Expected '{expected_name}' to be preserved in: {words}"

            # Verify punctuation is separated
            assert "," in words or "." in words, f"Expected punctuation to be separated in: {words}"

        print(f"✅ Complex name pattern test passed for {len(test_cases)} cases.")

def test_synthetic_passage_coverage():
    """Standalone test function for pytest discovery."""
    tester = TestImprovedTokenizer()
    tester.setup_class()

    print("\n🧪 RUNNING SYNTHETIC PASSAGE UNIT TESTS")
    print("=" * 50)

    tester.test_word_splitting_preserves_names()
    tester.test_punctuation_not_in_entity_names()
    tester.test_entity_detection_accuracy()
    tester.test_complex_name_patterns()

    tester.test_integration_with_dataset()  # No skipping allowed

    print("\n🎉 ALL SYNTHETIC PASSAGE TESTS PASSED!")
    print("✅ Improved tokenizer ready for production use.")

if __name__ == "__main__":
    test_synthetic_passage_coverage()