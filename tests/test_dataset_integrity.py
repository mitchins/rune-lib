"""
Integration tests for dataset integrity.

Tests that the generated datasets have correct entity indices and BIO tag alignment.
"""

import unittest
import json
from pathlib import Path


class TestDatasetIntegrity(unittest.TestCase):
    """Test integrity of generated datasets."""
    
    def test_entity_indices_exclusive(self):
        """Test that entity indices in dataset use exclusive end."""
        dataset_paths = [
            'diverse_hybrid_fixed/diverse_phase1_20k.jsonl',
            'diverse_hybrid_fixed/diverse_combined_60k.jsonl'
        ]
        
        for dataset_path in dataset_paths:
            if not Path(dataset_path).exists():
                self.skipTest(f"Dataset not found: {dataset_path}")
            
            with self.subTest(dataset=dataset_path):
                errors = []
                checked = 0
                
                with open(dataset_path) as f:
                    for i, line in enumerate(f):
                        if i >= 100:  # Sample first 100 for speed
                            break
                        
                        item = json.loads(line)
                        tokens = item['tokens']
                        entities = item.get('entities', [])
                        
                        for ent in entities:
                            checked += 1
                            start = ent.get('start')
                            end = ent.get('end')
                            expected_text = ent.get('text')
                            
                            # Validate indices exist
                            if start is None or end is None:
                                errors.append(f"Line {i}: Missing start/end indices")
                                continue
                            
                            # Validate indices are sane
                            if start < 0 or end > len(tokens) or start >= end:
                                errors.append(f"Line {i}: Invalid indices [{start}:{end}] for {len(tokens)} tokens")
                                continue
                            
                            # Validate text matches (exclusive end)
                            actual_text = ' '.join(tokens[start:end])
                            if actual_text != expected_text:
                                errors.append(
                                    f"Line {i}: Entity text mismatch\n"
                                    f"  Expected: '{expected_text}'\n"
                                    f"  Got:      '{actual_text}'\n"
                                    f"  Indices:  [{start}:{end}]"
                                )
                
                self.assertEqual(len(errors), 0, 
                               f"\n{dataset_path} has {len(errors)} errors:\n" + "\n".join(errors[:5]))
                self.assertGreater(checked, 0, "Should check at least some entities")
    
    def test_bio_tags_alignment(self):
        """Test that BIO tags align with entity indices."""
        dataset_path = 'diverse_hybrid_fixed/diverse_phase1_20k.jsonl'
        
        if not Path(dataset_path).exists():
            self.skipTest(f"Dataset not found: {dataset_path}")
        
        errors = []
        
        with open(dataset_path) as f:
            for i, line in enumerate(f):
                if i >= 100:  # Sample
                    break
                
                item = json.loads(line)
                tokens = item['tokens']
                bio_tags = item.get('bio_tags', [])
                entities = item.get('entities', [])
                
                # Check tokens and tags match length
                if len(tokens) != len(bio_tags):
                    errors.append(f"Line {i}: Length mismatch - {len(tokens)} tokens, {len(bio_tags)} tags")
                    continue
                
                # For each entity, verify BIO tags are correct
                for ent in entities:
                    start = ent.get('start')
                    end = ent.get('end')
                    role = ent.get('role', '').upper()
                    
                    if start is None or end is None:
                        continue
                    
                    if start >= len(bio_tags) or end > len(bio_tags):
                        continue
                    
                    # First token should be B-
                    if not bio_tags[start].startswith('B-'):
                        errors.append(
                            f"Line {i}: Entity at [{start}:{end}] should start with B- tag, got '{bio_tags[start]}'"
                        )
                    
                    # Subsequent tokens should be I-
                    for j in range(start + 1, end):
                        if not bio_tags[j].startswith('I-'):
                            errors.append(
                                f"Line {i}: Entity at [{start}:{end}] should have I- tag at position {j}, got '{bio_tags[j]}'"
                            )
                            break
        
        self.assertEqual(len(errors), 0, 
                        f"Found {len(errors)} BIO tag alignment errors:\n" + "\n".join(errors[:5]))


if __name__ == '__main__':
    unittest.main()
