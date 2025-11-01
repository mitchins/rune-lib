#!/usr/bin/env python3
"""
Unit tests for name augmentation system.

Tests the critical lemmatization-safe replacement cases:
- Possessive forms ('s)
- First name only occurrences
- Title consistency (Miss/Mr./Dr. etc.)
- Partial name matches
- Edge cases and boundaries
"""

import unittest
import json
from unittest.mock import Mock, patch

from rune.augment.name_inventory import NameInventory
from rune.augment.augment_utils import AugmentUtils


class TestNameAugmentation(unittest.TestCase):
    """Test suite for name augmentation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.inventory = NameInventory()
        self.augmenter = AugmentUtils(self.inventory)

        # Mock name generation for predictable tests
        self.test_names = {
            ('male', 'Anglo/Western'): {
                'first_name': 'John',
                'last_name': 'Smith',
                'title': 'Mr.',
                'full_name': 'Mr. John Smith'
            },
            ('female', 'Hispanic/Latino'): {
                'first_name': 'Maria',
                'last_name': 'Garcia',
                'title': 'Ms.',
                'full_name': 'Ms. Maria Garcia'
            }
        }

    def create_test_story(self, text, characters):
        """Create a test story with given text and characters."""
        return {
            'story_id': 'test_001',
            'text': text,
            'characters': characters,
            'metadata': {'genre': 'test'}
        }

    def test_possessive_forms_replacement(self):
        """Test that possessive forms like 'John's' are properly replaced."""

        # Test case: "John's car" should become "Maria's car"
        original_text = "John's car was parked outside. He loved John's vehicle."
        characters = [{'name': 'John Smith', 'role': 'PROTAGONIST'}]

        story = self.create_test_story(original_text, characters)

        # Mock the name generation
        with patch.object(self.inventory, 'generate_name', return_value=self.test_names[('female', 'Hispanic/Latino')]):
            augmented_story = self.augmenter.augment_story_data(story)

        augmented_text = augmented_story['text']

        # Should replace both "John's" occurrences
        self.assertNotIn("John's", augmented_text, "Possessive 'John's' should be replaced")
        self.assertIn("Maria's", augmented_text, "Should contain new possessive 'Maria's'")

        # Verify count
        self.assertEqual(augmented_text.count("Maria's"), 2, "Should replace all possessive occurrences")

    def test_first_name_only_occurrences(self):
        """Test replacement of first name only references."""

        original_text = "Hello John! said Mary. John nodded back at Mary enthusiastically."
        characters = [
            {'name': 'John Smith', 'role': 'PROTAGONIST'},
            {'name': 'Mary Johnson', 'role': 'SUPPORTING'}
        ]

        story = self.create_test_story(original_text, characters)

        # Mock name generation for both characters
        mock_calls = [
            self.test_names[('male', 'Anglo/Western')],     # For John
            self.test_names[('female', 'Hispanic/Latino')]   # For Mary
        ]

        with patch.object(self.inventory, 'generate_name', side_effect=mock_calls):
            augmented_story = self.augmenter.augment_story_data(story)

        augmented_text = augmented_story['text']

        # Original names should be gone
        self.assertNotIn(" John ", augmented_text, "First name 'John' should be replaced")
        self.assertNotIn(" Mary ", augmented_text, "First name 'Mary' should be replaced")

        # New names should be present
        self.assertIn("John", augmented_text, "New first name should be present")
        self.assertIn("Maria", augmented_text, "New first name should be present")

    def test_title_consistency(self):
        """Test that titles are consistently replaced (Mr./Ms./Dr./Miss etc.)."""

        original_text = "Dr. Sarah Williams entered. Miss Williams was brilliant. Dr. Williams had studied for years."
        characters = [{'name': 'Dr. Sarah Williams', 'role': 'PROTAGONIST'}]

        story = self.create_test_story(original_text, characters)

        with patch.object(self.inventory, 'generate_name', return_value=self.test_names[('female', 'Hispanic/Latino')]):
            augmented_story = self.augmenter.augment_story_data(story)

        augmented_text = augmented_story['text']

        # Original should be gone
        self.assertNotIn("Dr. Sarah Williams", augmented_text, "Full original name should be replaced")
        self.assertNotIn("Miss Williams", augmented_text, "Title variations should be replaced")
        self.assertNotIn("Dr. Williams", augmented_text, "Title + surname should be replaced")

        # New name should appear consistently
        self.assertIn("Ms. Maria Garcia", augmented_text, "Full new name should appear")

    def test_partial_name_boundary_issues(self):
        """Test that partial matches don't cause issues (e.g., 'Johnson' vs 'John')."""

        original_text = "John met Johnson at the restaurant. Johnson's friend John was late."
        characters = [{'name': 'John Smith', 'role': 'PROTAGONIST'}]

        story = self.create_test_story(original_text, characters)

        with patch.object(self.inventory, 'generate_name', return_value=self.test_names[('female', 'Hispanic/Latino')]):
            augmented_story = self.augmenter.augment_story_data(story)

        augmented_text = augmented_story['text']

        # Should replace "John" but NOT "Johnson"
        self.assertNotIn(" John ", augmented_text, "'John' should be replaced")
        self.assertIn("Johnson", augmented_text, "'Johnson' should NOT be replaced (different name)")

        # New name should be present
        self.assertIn("Maria", augmented_text, "New name should be present")

    def test_quoted_names(self):
        """Test names in quotes are properly handled."""

        original_text = 'She said "Hello, John!" and John replied "Hi there!"'
        characters = [{'name': 'John Smith', 'role': 'PROTAGONIST'}]

        story = self.create_test_story(original_text, characters)

        with patch.object(self.inventory, 'generate_name', return_value=self.test_names[('female', 'Hispanic/Latino')]):
            augmented_story = self.augmenter.augment_story_data(story)

        augmented_text = augmented_story['text']

        # Should replace quoted names too
        self.assertNotIn('"Hello, John!"', augmented_text, "Quoted names should be replaced")
        self.assertIn('"Hello, Maria!"', augmented_text, "Should contain new quoted name")

    def test_multiple_characters_no_cross_contamination(self):
        """Test that multiple character replacements don't interfere with each other."""

        original_text = "John met Sarah. Sarah liked John. John and Sarah became friends."
        characters = [
            {'name': 'John Smith', 'role': 'PROTAGONIST'},
            {'name': 'Sarah Williams', 'role': 'SUPPORTING'}
        ]

        story = self.create_test_story(original_text, characters)

        # Different mock names for each character
        mock_calls = [
            {'first_name': 'Carlos', 'last_name': 'Rodriguez', 'title': 'Mr.', 'full_name': 'Mr. Carlos Rodriguez'},
            {'first_name': 'Aisha', 'last_name': 'Patel', 'title': 'Ms.', 'full_name': 'Ms. Aisha Patel'}
        ]

        with patch.object(self.inventory, 'generate_name', side_effect=mock_calls):
            augmented_story = self.augmenter.augment_story_data(story)

        augmented_text = augmented_story['text']

        # Original names should be gone
        self.assertNotIn("John", augmented_text, "Original 'John' should be replaced")
        self.assertNotIn("Sarah", augmented_text, "Original 'Sarah' should be replaced")

        # New names should be present
        self.assertIn("Carlos", augmented_text, "New name 'Carlos' should be present")
        self.assertIn("Aisha", augmented_text, "New name 'Aisha' should be present")

        # Verify character counts
        self.assertEqual(augmented_text.count("Carlos"), 3, "Carlos should appear 3 times (replacing John)")
        self.assertEqual(augmented_text.count("Aisha"), 3, "Aisha should appear 3 times (replacing Sarah)")

    def test_case_sensitivity(self):
        """Test that name replacement handles different cases properly."""

        original_text = "JOHN shouted. john whispered. John spoke normally."
        characters = [{'name': 'John Smith', 'role': 'PROTAGONIST'}]

        story = self.create_test_story(original_text, characters)

        with patch.object(self.inventory, 'generate_name', return_value=self.test_names[('female', 'Hispanic/Latino')]):
            augmented_story = self.augmenter.augment_story_data(story)

        augmented_text = augmented_story['text']

        # Should handle all cases appropriately
        self.assertIn("MARIA", augmented_text, "Should handle uppercase")
        self.assertIn("maria", augmented_text, "Should handle lowercase")
        self.assertIn("Maria", augmented_text, "Should handle normal case")

    def test_character_metadata_update(self):
        """Test that character metadata is properly updated in augmented story."""

        original_text = "John walked to the store."
        characters = [{'name': 'John Smith', 'role': 'PROTAGONIST', 'culture': 'Anglo/Western'}]

        story = self.create_test_story(original_text, characters)

        with patch.object(self.inventory, 'generate_name', return_value=self.test_names[('female', 'Hispanic/Latino')]):
            augmented_story = self.augmenter.augment_story_data(story)

        # Check character metadata was updated
        updated_char = augmented_story['characters'][0]
        self.assertEqual(updated_char['name'], 'Ms. Maria Garcia', "Character name should be updated")
        self.assertEqual(updated_char['original_name'], 'John Smith', "Original name should be preserved")
        self.assertEqual(updated_char['role'], 'PROTAGONIST', "Role should be preserved")

    def test_no_replacement_when_probability_zero(self):
        """Test that names are not replaced when replacement probability is 0."""

        original_text = "John walked to the store."
        characters = [{'name': 'John Smith', 'role': 'PROTAGONIST'}]
        story = self.create_test_story(original_text, characters)

        # Test with 0% replacement probability
        augmented_story = self.augmenter.augment_story_data(story, replacement_probability=0.0)

        # Should be unchanged
        self.assertEqual(augmented_story['text'], original_text, "Text should be unchanged with 0% probability")
        self.assertEqual(augmented_story['characters'][0]['name'], 'John Smith', "Character name should be unchanged")

    def test_empty_character_list(self):
        """Test handling of stories with no characters."""

        original_text = "The weather was nice today."
        story = self.create_test_story(original_text, [])

        augmented_story = self.augmenter.augment_story_data(story)

        # Should return unchanged
        self.assertEqual(augmented_story['text'], original_text, "Text should be unchanged with no characters")
        self.assertEqual(augmented_story['characters'], [], "Characters should remain empty")

    def test_missing_name_error_handling(self):
        """Test error handling when name generation fails."""

        original_text = "John walked to the store."
        characters = [{'name': 'John Smith', 'role': 'PROTAGONIST'}]
        story = self.create_test_story(original_text, characters)

        # Mock name generation failure
        with patch.object(self.inventory, 'generate_name', side_effect=Exception("Name generation failed")):
            augmented_story = self.augmenter.augment_story_data(story)

        # Should handle gracefully - either unchanged or with error logging
        self.assertIsNotNone(augmented_story, "Should return a story even on name generation failure")

    def test_augmentation_preserves_story_structure(self):
        """Test that augmentation preserves all story fields."""

        original_story = {
            'story_id': 'test_123',
            'text': 'John walked.',
            'characters': [{'name': 'John Smith', 'role': 'PROTAGONIST'}],
            'metadata': {'genre': 'drama', 'token_count': 10},
            'custom_field': 'should_be_preserved'
        }

        with patch.object(self.inventory, 'generate_name', return_value=self.test_names[('female', 'Hispanic/Latino')]):
            augmented_story = self.augmenter.augment_story_data(original_story)

        # Check all fields are preserved
        self.assertEqual(augmented_story['story_id'], 'test_123', "Story ID should be preserved")
        self.assertEqual(augmented_story['metadata'], original_story['metadata'], "Metadata should be preserved")
        self.assertEqual(augmented_story['custom_field'], 'should_be_preserved', "Custom fields should be preserved")
        self.assertIn('augmentation_metadata', augmented_story, "Should add augmentation metadata")


class TestNameReplacementEdgeCases(unittest.TestCase):
    """Test edge cases in name replacement logic."""

    def test_miss_nguyen_case(self):
        """Test the specific 'Miss Nguyen' case mentioned by user."""

        inventory = NameInventory()
        augmenter = AugmentUtils(inventory)

        original_text = "Miss Nguyen entered the classroom. The students greeted Miss Nguyen warmly."
        characters = [{'name': 'Miss Nguyen', 'role': 'SUPPORTING'}]

        story = {
            'story_id': 'test_miss_nguyen',
            'text': original_text,
            'characters': characters
        }

        # Mock replacement name
        mock_name = {
            'first_name': 'Elena',
            'last_name': 'Rossi',
            'title': 'Ms.',
            'full_name': 'Ms. Elena Rossi'
        }

        with patch.object(inventory, 'generate_name', return_value=mock_name):
            augmented_story = augmenter.augment_story_data(story)

        augmented_text = augmented_story['text']

        # Original should be completely gone
        self.assertNotIn("Miss Nguyen", augmented_text, "'Miss Nguyen' should be completely replaced")
        self.assertNotIn("Nguyen", augmented_text, "Surname 'Nguyen' should be replaced")

        # New name should be present
        self.assertIn("Ms. Elena Rossi", augmented_text, "Full new name should be present")
        self.assertEqual(augmented_text.count("Ms. Elena Rossi"), 2, "Should replace both occurrences")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)