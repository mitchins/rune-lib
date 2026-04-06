"""
Unit tests for entity extraction round-trip consistency.

Tests that entity indices are correctly computed as EXCLUSIVE end indices,
and that round-trip conversion (bio_tags -> entities -> bio_tags) is consistent.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rune.data.story_preprocessor import StoryPreprocessor


class TestEntityExtraction(unittest.TestCase):
    """Test entity extraction with correct exclusive end indices."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = StoryPreprocessor(use_spacy=False)
        
        # Real example from dataset (story with Dmitri Volkov)
        self.test_tokens = [
            'The', 'scent', 'of', 'simmering', 'mirepoix', ',', 'a', 'comforting',
            'symphony', 'of', 'slow-cooked', 'onions', ',', 'carrots', ',', 'and',
            'celery', ',', 'usually', 'anchored', 'Dmitri', 'Volkov', '.', 'It',
            'was', 'the', 'olfactory', 'equivalent', 'of', 'a', 'perfectly'
        ]
        
        self.test_bio_tags = [
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'B-protagonist', 'I-protagonist', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O'
        ]
    
    def test_extract_entities_exclusive_indices(self):
        """Test that entity indices use exclusive end (Python convention)."""
        entities = self.preprocessor._extract_entities(self.test_tokens, self.test_bio_tags)
        
        # Should find "Dmitri Volkov"
        self.assertEqual(len(entities), 1, "Should extract exactly one entity")
        
        entity = entities[0]
        self.assertEqual(entity['text'], 'Dmitri Volkov')
        self.assertEqual(entity['role'], 'protagonist')
        self.assertEqual(entity['start'], 20)
        self.assertEqual(entity['end'], 22, "End index should be EXCLUSIVE (22, not 21)")
        
        # Verify the indices work with Python slicing
        extracted_tokens = self.test_tokens[entity['start']:entity['end']]
        expected_tokens = ['Dmitri', 'Volkov']
        self.assertEqual(extracted_tokens, expected_tokens,
                        f"tokens[{entity['start']}:{entity['end']}] should give full name")
    
    def test_single_token_entity(self):
        """Test single-token entity has correct exclusive end."""
        tokens = ['Hello', 'Sarah', 'said', 'goodbye']
        bio_tags = ['O', 'B-protagonist', 'O', 'O']
        
        entities = self.preprocessor._extract_entities(tokens, bio_tags)
        
        self.assertEqual(len(entities), 1)
        entity = entities[0]
        self.assertEqual(entity['text'], 'Sarah')
        self.assertEqual(entity['start'], 1)
        self.assertEqual(entity['end'], 2, "Single token: end should be start + 1")
        
        # Verify slicing
        self.assertEqual(tokens[entity['start']:entity['end']], ['Sarah'])
    
    def test_multi_word_entity(self):
        """Test multi-word entity has correct span."""
        tokens = ['Dr', '.', 'Jane', 'Smith', 'arrived', 'yesterday']
        bio_tags = ['B-protagonist', 'I-protagonist', 'I-protagonist', 'I-protagonist', 'O', 'O']
        
        entities = self.preprocessor._extract_entities(tokens, bio_tags)
        
        self.assertEqual(len(entities), 1)
        entity = entities[0]
        self.assertEqual(entity['text'], 'Dr . Jane Smith')
        self.assertEqual(entity['start'], 0)
        self.assertEqual(entity['end'], 4, "4-token entity: end = start + 4")
        
        # Verify slicing
        self.assertEqual(tokens[entity['start']:entity['end']], 
                        ['Dr', '.', 'Jane', 'Smith'])
    
    def test_multiple_entities(self):
        """Test extraction of multiple entities with correct indices."""
        tokens = ['Sarah', 'met', 'John', 'and', 'Mary', 'yesterday']
        bio_tags = ['B-protagonist', 'O', 'B-antagonist', 'O', 'B-ally', 'O']
        
        entities = self.preprocessor._extract_entities(tokens, bio_tags)
        
        self.assertEqual(len(entities), 3)
        
        # Check each entity
        self.assertEqual(entities[0]['text'], 'Sarah')
        self.assertEqual(tokens[entities[0]['start']:entities[0]['end']], ['Sarah'])
        
        self.assertEqual(entities[1]['text'], 'John')
        self.assertEqual(tokens[entities[1]['start']:entities[1]['end']], ['John'])
        
        self.assertEqual(entities[2]['text'], 'Mary')
        self.assertEqual(tokens[entities[2]['start']:entities[2]['end']], ['Mary'])
    
    def test_consecutive_entities(self):
        """Test consecutive multi-word entities."""
        tokens = ['Mr', 'Smith', 'and', 'Mrs', 'Jones', 'arrived']
        bio_tags = ['B-protagonist', 'I-protagonist', 'O', 
                   'B-antagonist', 'I-antagonist', 'O']
        
        entities = self.preprocessor._extract_entities(tokens, bio_tags)
        
        self.assertEqual(len(entities), 2)
        
        # Mr Smith
        self.assertEqual(entities[0]['text'], 'Mr Smith')
        self.assertEqual(entities[0]['start'], 0)
        self.assertEqual(entities[0]['end'], 2)
        self.assertEqual(tokens[0:2], ['Mr', 'Smith'])
        
        # Mrs Jones
        self.assertEqual(entities[1]['text'], 'Mrs Jones')
        self.assertEqual(entities[1]['start'], 3)
        self.assertEqual(entities[1]['end'], 5)
        self.assertEqual(tokens[3:5], ['Mrs', 'Jones'])
    
    def test_entity_at_end(self):
        """Test entity at the end of token list."""
        tokens = ['Hello', 'world', 'from', 'Dr', 'Watson']
        bio_tags = ['O', 'O', 'O', 'B-protagonist', 'I-protagonist']
        
        entities = self.preprocessor._extract_entities(tokens, bio_tags)
        
        self.assertEqual(len(entities), 1)
        entity = entities[0]
        self.assertEqual(entity['text'], 'Dr Watson')
        self.assertEqual(entity['start'], 3)
        self.assertEqual(entity['end'], 5)
        self.assertEqual(tokens[3:5], ['Dr', 'Watson'])
    
    def test_round_trip_consistency(self):
        """Test that bio_tags -> entities -> bio_tags is consistent."""
        from create_hybrid_dataset import (
            extract_entities_from_bio_tags,
            reconstruct_bio_tags_from_entities
        )
        
        # Original bio_tags
        original_tags = self.test_bio_tags
        
        # Extract entities
        entities = extract_entities_from_bio_tags(self.test_tokens, original_tags)
        
        # Reconstruct bio_tags
        reconstructed_tags = reconstruct_bio_tags_from_entities(self.test_tokens, entities)
        
        # Normalize case (original may be lowercase, reconstructed uppercase)
        original_normalized = [t.upper() for t in original_tags]
        
        self.assertEqual(reconstructed_tags, original_normalized,
                        "Round-trip should preserve BIO tags")
    
    def test_empty_input(self):
        """Test handling of empty input."""
        entities = self.preprocessor._extract_entities([], [])
        self.assertEqual(entities, [])
    
    def test_no_entities(self):
        """Test tokens with no entities."""
        tokens = ['The', 'quick', 'brown', 'fox']
        bio_tags = ['O', 'O', 'O', 'O']
        
        entities = self.preprocessor._extract_entities(tokens, bio_tags)
        self.assertEqual(entities, [])
    
    def test_incomplete_entity_at_end(self):
        """Test handling of incomplete entity (I- without B-)."""
        tokens = ['Hello', 'world', 'Smith']
        bio_tags = ['O', 'O', 'I-protagonist']  # Missing B- tag
        
        # Should handle gracefully (skip orphan I- tag)
        entities = self.preprocessor._extract_entities(tokens, bio_tags)
        # Orphan I- tags should be ignored
        self.assertEqual(len(entities), 0)


class TestEntityIndexValidation(unittest.TestCase):
    """Test that entity indices can be validated."""
    
    def test_validate_exclusive_end_index(self):
        """Test validation of exclusive end indices."""
        tokens = ['Dmitri', 'Volkov', 'walked']
        
        # Correct: exclusive end
        entity_correct = {
            'text': 'Dmitri Volkov',
            'start': 0,
            'end': 2
        }
        actual = ' '.join(tokens[entity_correct['start']:entity_correct['end']])
        self.assertEqual(actual, entity_correct['text'])
        
        # Incorrect: inclusive end
        entity_wrong = {
            'text': 'Dmitri Volkov',
            'start': 0,
            'end': 1  # WRONG: inclusive
        }
        actual = ' '.join(tokens[entity_wrong['start']:entity_wrong['end']])
        self.assertNotEqual(actual, entity_wrong['text'],
                           "Inclusive end index should NOT work")


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestEntityExtraction))
    test_suite.addTest(unittest.makeSuite(TestEntityIndexValidation))
    return test_suite


if __name__ == '__main__':
    unittest.main()
