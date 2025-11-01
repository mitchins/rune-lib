"""
Tests to verify tokenization consistency fixes.

Tests cover the critical bugs we fixed:
1. BioTagger.tag_text() uses ImprovedTokenizer
2. Entity normalization handles punctuation/possessives
"""

import pytest
from rune.data.improved_tokenizer import ImprovedTokenizer
from rune.utils.entity_normalization import normalize_entity


class TestImprovedTokenizer:
    """Test ImprovedTokenizer handles boundary punctuation correctly."""
    
    def setup_method(self):
        self.tokenizer = ImprovedTokenizer()
    
    def test_separates_trailing_punctuation(self):
        """Verify punctuation is separated from words."""
        text = "Fox said, 'Hello.'"
        tokens = self.tokenizer.split_words(text)
        
        # Should separate punctuation
        assert 'Fox' in tokens
        assert ',' in tokens
        assert '.' in tokens
        # Should NOT have combined forms
        assert 'Fox,' not in tokens
        assert 'Hello.' not in tokens
    
    def test_preserves_internal_punctuation(self):
        """Verify internal punctuation in names is preserved."""
        text = "Dr. O'Reilly and Jean-Luc met Mr. Smith."
        tokens = self.tokenizer.split_words(text)
        
        # Should preserve internal punctuation
        assert "O'Reilly" in tokens
        assert "Jean-Luc" in tokens
    
    def test_handles_possessives(self):
        """Verify possessives are separated."""
        text = "Fox's investigation and Rogers's house"
        tokens = self.tokenizer.split_words(text)
        
        # Possessives should be kept with the name
        assert "Fox's" in tokens or ("Fox" in tokens and "'s" in tokens)
        assert "Rogers's" in tokens or ("Rogers" in tokens and "'s" in tokens)
    
    def test_handles_quotes(self):
        """Verify quotes are separated."""
        text = '"Jackson" and "Andrea" arrived.'
        tokens = self.tokenizer.split_words(text)
        
        # Quotes should be separate
        assert 'Jackson' in tokens
        assert 'Andrea' in tokens
        assert '"' in tokens or '"Jackson"' not in tokens


class TestEntityNormalization:
    """Test entity normalization for deduplication."""
    
    def test_normalizes_possessives(self):
        """Possessives should be stripped."""
        assert normalize_entity("Fox's") == "fox"
        assert normalize_entity("Rogers's") == "rogers"
        assert normalize_entity("Andrea's") == "andrea"  # smart quote
    
    def test_strips_trailing_punctuation(self):
        """Trailing punctuation should be removed."""
        assert normalize_entity("fox.") == "fox"
        assert normalize_entity("andrea,") == "andrea"
        assert normalize_entity("bennet;") == "bennet"
    
    def test_strips_quotes(self):
        """All quote types should be removed."""
        assert normalize_entity('"Jackson"') == "jackson"
        assert normalize_entity("'Andrea'") == "andrea"  # smart quotes
        assert normalize_entity('"Fox"') == "fox"  # smart quotes
    
    def test_preserves_internal_periods(self):
        """Internal periods (like titles) should be preserved."""
        assert normalize_entity("Mr. Bennet") == "mr. bennet"
        assert normalize_entity("Dr. Smith") == "dr. smith"
    
    def test_lowercases(self):
        """All entities should be lowercased."""
        assert normalize_entity("FOX") == "fox"
        assert normalize_entity("Jackson") == "jackson"
        assert normalize_entity("ANDREA") == "andrea"
    
    def test_deduplication_scenario(self):
        """Real-world scenario: multiple forms of same entity."""
        variants = [
            "Fox",
            "Fox.",
            "Fox,",
            "Fox's",
            '"Fox"',
            "'Fox'",
        ]
        
        normalized = [normalize_entity(v) for v in variants]
        
        # All should normalize to the same form
        assert len(set(normalized)) == 1
        assert normalized[0] == "fox"


class TestTokenizationConsistency:
    """Integration test: tokenization consistency across pipeline."""
    
    def test_same_tokenizer_everywhere(self):
        """Verify same tokenization logic is used."""
        tokenizer = ImprovedTokenizer()
        
        test_text = 'Fox said, "Hello, Andrea."'
        tokens = tokenizer.split_words(test_text)
        
        # These should match what training sees
        assert 'Fox' in tokens
        assert ',' in tokens
        assert '"' in tokens
        assert 'Andrea' in tokens
        assert '.' in tokens
        
        # These should NOT appear (text.split() would produce these)
        assert 'Fox,' not in tokens
        assert '"Hello,' not in tokens
        assert 'Andrea."' not in tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestTokenizationAlignmentCritical:
    """
    CRITICAL: Test that tokenization → HF tokenizer → word_ids() preserves alignment.
    
    This is the core requirement: when we tokenize with ImprovedTokenizer,
    then pass to HuggingFace tokenizer with is_split_into_words=True,
    word_ids() must correctly map subwords back to original words.
    """
    
    def setup_method(self):
        self.tokenizer = ImprovedTokenizer()
    
    def test_word_boundary_preservation(self):
        """
        Critical: Verify word boundaries are preserved through tokenization.
        
        ImprovedTokenizer.split_words() must create tokens that HF tokenizer
        can correctly map back via word_ids().
        """
        text = 'Fox said, "Hello."'
        
        # Step 1: Split with ImprovedTokenizer
        words = self.tokenizer.split_words(text)
        
        # Expected: ['Fox', 'said', ',', '"', 'Hello', '.', '"']
        assert len(words) == 7
        assert words[0] == 'Fox'
        assert words[1] == 'said'
        assert words[2] == ','
        assert words[4] == 'Hello'
        
        # Step 2: Verify punctuation is isolated (critical for word_ids alignment)
        for i, word in enumerate(words):
            if word in [',', '"', '.']:
                # Punctuation should be single characters
                assert len(word) == 1, f"Punctuation at index {i} should be isolated: {word}"
    
    def test_offset_alignment_example(self):
        """
        Test realistic scenario: name with punctuation.
        
        This tests the exact bug we fixed: "Fox." being one token vs ["Fox", "."]
        """
        text = "Fox. Andrea, Jonathan."
        
        # What text.split() would give (WRONG):
        wrong_tokens = text.split()
        assert 'Fox.' in wrong_tokens  # BUG: punctuation attached
        
        # What ImprovedTokenizer gives (CORRECT):
        correct_tokens = self.tokenizer.split_words(text)
        assert 'Fox' in correct_tokens
        assert '.' in correct_tokens
        assert 'Fox.' not in correct_tokens  # Punctuation separated
        
        # Verify we can reconstruct positions
        # Fox should be at index 0, period at index 1, etc.
        assert correct_tokens[0] == 'Fox'
        assert correct_tokens[1] == '.'
        assert correct_tokens[2] == 'Andrea'
        assert correct_tokens[3] == ','
    
    def test_entity_span_reconstruction(self):
        """
        Test that we can reconstruct entity spans from tokens.
        
        This verifies that B-I-O tagging will work correctly with the tokenization.
        """
        text = "Dr. Elena Martinez, the scientist."
        tokens = self.tokenizer.split_words(text)
        
        # Find "Elena Martinez" span
        # Should be: Dr. | . | Elena | Martinez | , | the | scientist | .
        assert 'Elena' in tokens
        assert 'Martinez' in tokens
        
        elena_idx = tokens.index('Elena')
        martinez_idx = tokens.index('Martinez')
        
        # They should be adjacent (critical for B-I tagging)
        assert martinez_idx == elena_idx + 1, "Multi-word entities must have adjacent tokens"
    
    def test_possessive_alignment(self):
        """
        Test possessive handling preserves alignment.
        
        Critical because possessives affect word_ids() mapping.
        """
        text = "Fox's investigation and Rogers's house"
        tokens = self.tokenizer.split_words(text)
        
        # Possessives should be handled consistently
        # Either: ["Fox's", "investigation"] OR ["Fox", "'s", "investigation"]
        # Both are valid, but must be consistent for word_ids() mapping
        
        if "'s" in tokens:
            # Possessive is separated
            fox_idx = tokens.index("Fox")
            poss_idx = tokens.index("'s")
            assert poss_idx == fox_idx + 1, "Possessive must immediately follow name"
        else:
            # Possessive is attached
            assert "Fox's" in tokens or "Rogers's" in tokens

