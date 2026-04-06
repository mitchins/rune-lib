#!/usr/bin/env python3
"""
Stress tests for NER tagging using classic literature excerpts.

These tests use real literary text to validate that:
1. ALL mentions of character names are tagged (not just first mention)
2. Multi-word names (first + last) are properly tagged with B-I sequences
3. Title prefixes (Mr., Mrs., Miss) are excluded but surnames after them are tagged
4. The preprocessor handles complex Victorian-era names and dialogue correctly

Uses excerpts from:
- Sherlock Holmes (Arthur Conan Doyle) - complex Victorian names, titles
- Pride and Prejudice (Jane Austen) - formal names, family references
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rune.data.story_preprocessor import StoryPreprocessor


# =============================================================================
# SHERLOCK HOLMES - A Study in Scarlet, Chapter 1 (excerpt)
# =============================================================================

SHERLOCK_HOLMES_TEXT = """
In the year 1878 I took my degree of Doctor of Medicine of the University of 
London, and proceeded to Netley to go through the course prescribed for surgeons 
in the army. Having completed my studies there, I was duly attached to the Fifth 
Northumberland Fusiliers as Assistant Surgeon. The regiment was stationed in India 
at the time, and before I could join it, the second Afghan war had broken out. 
On landing at Bombay, I learned that my corps had advanced through the passes, 
and was already deep in the enemy's country. I followed, however, with many other 
officers who were in the same situation as myself, and succeeded in reaching 
Candahar in safety, where I found my regiment, and at once entered upon my new duties.

The campaign brought honours and promotion to many, but for me it had nothing but 
misfortune and disaster. I was removed from my brigade and attached to the Berkshires, 
with whom I served at the fatal battle of Maiwand. There I was struck on the shoulder 
by a Jezail bullet, which shattered the bone and grazed the subclavian artery. I 
should have fallen into the hands of the murderous Ghazis had it not been for the 
devotion and courage shown by Murray, my orderly, who threw me across a pack-horse, 
and succeeded in bringing me safely to the British lines.

Young Stamford had been a dresser under me at Barts, and before I knew where I was, 
he had shown me upstairs into the sitting-room. As I entered, a tall, gaunt young 
man rose from a chemical apparatus.

"Dr. Watson, I presume?" said he, rising and extending a hand.

"Indeed," I answered, "and you must be Mr. Sherlock Holmes."

Holmes laughed. "I have heard of you from Stamford. I hear you are looking for 
lodgings. I have my eye on a suite in Baker Street which would suit us down to 
the ground."

"Excellent," said I. "I should be happy to share with you, Mr. Holmes."

Holmes nodded. "Then I shall expect you tomorrow at noon. Stamford here can give 
you the address."

Stamford grinned. "I knew you two would get along. Watson, you won't find a more 
singular companion than Sherlock Holmes."
"""

SHERLOCK_CHARACTERS = [
    {"name": "John Watson", "role": "protagonist"},
    {"name": "Sherlock Holmes", "role": "protagonist"},
    {"name": "Murray", "role": "supporting"},
    {"name": "Stamford", "role": "supporting"},
]


# =============================================================================
# PRIDE AND PREJUDICE - Chapter 1 (excerpt)
# =============================================================================

PRIDE_PREJUDICE_TEXT = """
It is a truth universally acknowledged, that a single man in possession of a 
good fortune, must be in want of a wife.

However little known the feelings or views of such a man may be on his first 
entering a neighbourhood, this truth is so well fixed in the minds of the 
surrounding families, that he is considered as the rightful property of some 
one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that 
Netherfield Park is let at last?"

Mr. Bennet replied that he had not.

"But it is," returned she; "for Mrs. Long has just been here, and she told 
me all about it."

Mr. Bennet made no answer.

"Do not you want to know who has taken it?" cried his wife impatiently.

"You want to tell me, and I have no objection to hearing it."

This was invitation enough.

"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a 
young man of large fortune from the north of England; that he came down on 
Monday in a chaise and four to see the place, and was so much delighted with 
it that he agreed with Mr. Morris immediately; that he is to take possession 
before Michaelmas, and some of his servants are to be in the house by the end 
of next week."

"What is his name?"

"Bingley."

"Is he married or single?"

"Oh! single, my dear, to be sure! A single man of large fortune; four or five 
thousand a year. What a fine thing for our girls!"

"How so? how can it affect them?"

"My dear Mr. Bennet," replied his wife, "how can you be so tiresome! You must 
know that I am thinking of his marrying one of them."

"Is that his design in settling here?"

"Design! nonsense, how can you talk so! But it is very likely that he may fall 
in love with one of them, and therefore you must visit him as soon as he comes."

"I see no occasion for that. You and the girls may go, or you may send them by 
themselves, which perhaps will be still better, for as you are as handsome as 
any of them, Mr. Bingley might like you the best of the party."

"My dear, you flatter me. I certainly have had my share of beauty, but I do not 
pretend to be anything extraordinary now. When a woman has five grown up daughters, 
she ought to give over thinking of her own beauty."

"In such cases, a woman has not often much beauty to think of."

"But, my dear, you must indeed go and see Mr. Bingley when he comes into the 
neighbourhood."

"It is more than I engage for, I assure you."

"But consider your daughters. Only think what an establishment it would be for 
one of them. Sir William and Lady Lucas are determined to go, merely on that 
account, for in general you know they visit no new comers. Indeed you must go, 
for it will be impossible for us to visit him, if you do not."

"You are over scrupulous surely. I dare say Mr. Bingley will be very glad to 
see you; and I will send a few lines by you to assure him of my hearty consent 
to his marrying which ever he chooses of the girls; though I must throw in a 
good word for my little Lizzy."

"I desire you will do no such thing. Lizzy is not a bit better than the others; 
and I am sure she is not half so handsome as Jane, nor half so good humoured as 
Lydia. But you are always giving her the preference."

"They have none of them much to recommend them," replied he; "they are all silly 
and ignorant like other girls; but Lizzy has something more of quickness than 
her sisters."

"Mr. Bennet, how can you abuse your own children in such a way? You take delight 
in vexing me. You have no compassion on my poor nerves."

"You mistake me, my dear. I have a high respect for your nerves. They are my old 
friends. I have heard you mention them with consideration these twenty years at least."

"Ah! you do not know what I suffer."

"But I hope you will get over it, and live to see many young men of four thousand 
a year come into the neighbourhood."
"""

PRIDE_PREJUDICE_CHARACTERS = [
    {"name": "Mr. Bennet", "role": "protagonist"},
    {"name": "Mrs. Bennet", "role": "protagonist"},
    {"name": "Charles Bingley", "role": "catalyst"},
    {"name": "Mrs. Long", "role": "supporting"},
    {"name": "Mr. Morris", "role": "supporting"},
    {"name": "Lizzy Bennet", "role": "protagonist"},  # Elizabeth's nickname
    {"name": "Jane Bennet", "role": "supporting"},
    {"name": "Lydia Bennet", "role": "supporting"},
    {"name": "William Lucas", "role": "supporting"},  # "Sir" is a title, not part of name
    {"name": "Lady Lucas", "role": "supporting"},
]


class TestSherlockHolmes:
    """Test NER tagging on Sherlock Holmes text."""
    
    @pytest.fixture
    def preprocessor(self):
        return StoryPreprocessor(use_spacy=False)
    
    @pytest.fixture
    def processed(self, preprocessor):
        story = {
            "story_id": "sherlock_test",
            "text": SHERLOCK_HOLMES_TEXT,
            "entities": [
                {"text": c["name"], "type": "PERSON", "role": c["role"]}
                for c in SHERLOCK_CHARACTERS
            ]
        }
        return preprocessor.process_story(story)
    
    def _count_mentions(self, tokens, tags, name_part):
        """Count how many times a name part is tagged as B- or I-."""
        count = 0
        for tok, tag in zip(tokens, tags):
            if tok == name_part and tag != 'O':
                count += 1
        return count
    
    def _find_all_mentions(self, tokens, tags, name_part):
        """Find all positions where a name part appears."""
        mentions = []
        for i, (tok, tag) in enumerate(zip(tokens, tags)):
            if tok == name_part:
                mentions.append((i, tag))
        return mentions
    
    def test_holmes_tagged_multiple_times(self, processed):
        """Holmes appears multiple times and should be tagged each time."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        mentions = self._find_all_mentions(tokens, tags, 'Holmes')
        
        # Holmes appears at least 4 times in the text
        assert len(mentions) >= 4, f"Expected at least 4 'Holmes' mentions, found {len(mentions)}"
        
        # ALL mentions should be tagged (not O)
        tagged = [m for m in mentions if m[1] != 'O']
        assert len(tagged) == len(mentions), (
            f"Expected all {len(mentions)} 'Holmes' mentions to be tagged, "
            f"but only {len(tagged)} were. Untagged: {[m for m in mentions if m[1] == 'O']}"
        )
    
    def test_watson_tagged_multiple_times(self, processed):
        """Watson appears multiple times and should be tagged each time."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        mentions = self._find_all_mentions(tokens, tags, 'Watson')
        
        # Watson appears at least 2 times
        assert len(mentions) >= 2, f"Expected at least 2 'Watson' mentions, found {len(mentions)}"
        
        # ALL mentions should be tagged
        tagged = [m for m in mentions if m[1] != 'O']
        assert len(tagged) == len(mentions), (
            f"Expected all {len(mentions)} 'Watson' mentions to be tagged, "
            f"but only {len(tagged)} were."
        )
    
    def test_stamford_tagged_multiple_times(self, processed):
        """Stamford appears 4 times and should be tagged each time."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        mentions = self._find_all_mentions(tokens, tags, 'Stamford')
        
        # Stamford appears 4 times in the text
        assert len(mentions) >= 3, f"Expected at least 3 'Stamford' mentions, found {len(mentions)}"
        
        # ALL mentions should be tagged
        tagged = [m for m in mentions if m[1] != 'O']
        assert len(tagged) == len(mentions), (
            f"Expected all {len(mentions)} 'Stamford' mentions to be tagged, "
            f"but only {len(tagged)} were."
        )
    
    def test_murray_single_mention_tagged(self, processed):
        """Murray appears once and should be tagged."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        mentions = self._find_all_mentions(tokens, tags, 'Murray')
        
        assert len(mentions) >= 1, "Expected at least 1 'Murray' mention"
        assert mentions[0][1] != 'O', f"Murray should be tagged, got {mentions[0][1]}"
    
    def test_sherlock_holmes_full_name_bio_sequence(self, processed):
        """'Sherlock Holmes' should have B-I tag sequence."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        # Find "Sherlock Holmes" sequence
        found_bi_sequence = False
        for i in range(len(tokens) - 1):
            if tokens[i] == 'Sherlock' and tokens[i+1] == 'Holmes':
                if tags[i].startswith('B-') and tags[i+1].startswith('I-'):
                    found_bi_sequence = True
                    break
        
        assert found_bi_sequence, (
            "Expected 'Sherlock Holmes' to have B-I tag sequence"
        )
    
    def test_title_mr_not_tagged(self, processed):
        """'Mr.' title should NOT be tagged as part of entity."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        for i, tok in enumerate(tokens):
            if tok in ('Mr', 'Mr.'):
                assert tags[i] == 'O', f"Title 'Mr.' at pos {i} should be O, got {tags[i]}"
    
    def test_dr_title_not_tagged(self, processed):
        """'Dr.' title should NOT be tagged as part of entity."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        for i, tok in enumerate(tokens):
            if tok in ('Dr', 'Dr.'):
                assert tags[i] == 'O', f"Title 'Dr.' at pos {i} should be O, got {tags[i]}"


class TestPrideAndPrejudice:
    """Test NER tagging on Pride and Prejudice text."""
    
    @pytest.fixture
    def preprocessor(self):
        return StoryPreprocessor(use_spacy=False)
    
    @pytest.fixture
    def processed(self, preprocessor):
        story = {
            "story_id": "pride_prejudice_test",
            "text": PRIDE_PREJUDICE_TEXT,
            "entities": [
                {"text": c["name"], "type": "PERSON", "role": c["role"]}
                for c in PRIDE_PREJUDICE_CHARACTERS
            ]
        }
        return preprocessor.process_story(story)
    
    def _find_all_mentions(self, tokens, tags, name_part):
        """Find all positions where a name part appears."""
        mentions = []
        for i, (tok, tag) in enumerate(zip(tokens, tags)):
            if tok == name_part:
                mentions.append((i, tag))
        return mentions
    
    def test_bennet_tagged_many_times(self, processed):
        """'Bennet' appears many times (Mr. Bennet, Mrs. Bennet) - all should be tagged."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        mentions = self._find_all_mentions(tokens, tags, 'Bennet')
        
        # Bennet appears at least 5 times in the text excerpt
        assert len(mentions) >= 5, f"Expected at least 5 'Bennet' mentions, found {len(mentions)}"
        
        # ALL mentions should be tagged
        tagged = [m for m in mentions if m[1] != 'O']
        assert len(tagged) == len(mentions), (
            f"Expected all {len(mentions)} 'Bennet' mentions to be tagged, "
            f"but only {len(tagged)} were."
        )
    
    def test_bingley_tagged_multiple_times(self, processed):
        """'Bingley' appears multiple times and should always be tagged."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        mentions = self._find_all_mentions(tokens, tags, 'Bingley')
        
        # Bingley appears at least 3 times
        assert len(mentions) >= 3, f"Expected at least 3 'Bingley' mentions, found {len(mentions)}"
        
        # ALL mentions should be tagged
        tagged = [m for m in mentions if m[1] != 'O']
        assert len(tagged) == len(mentions), (
            f"Expected all {len(mentions)} 'Bingley' mentions to be tagged, "
            f"but only {len(tagged)} were."
        )
    
    def test_lizzy_nickname_tagged(self, processed):
        """'Lizzy' (nickname for Elizabeth) should be tagged."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        mentions = self._find_all_mentions(tokens, tags, 'Lizzy')
        
        # Lizzy appears at least twice
        assert len(mentions) >= 2, f"Expected at least 2 'Lizzy' mentions, found {len(mentions)}"
        
        # ALL mentions should be tagged  
        tagged = [m for m in mentions if m[1] != 'O']
        assert len(tagged) == len(mentions), (
            f"Expected all {len(mentions)} 'Lizzy' mentions to be tagged, "
            f"but only {len(tagged)} were."
        )
    
    def test_jane_lydia_tagged(self, processed):
        """'Jane' and 'Lydia' should be tagged."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        for name in ['Jane', 'Lydia']:
            mentions = self._find_all_mentions(tokens, tags, name)
            assert len(mentions) >= 1, f"Expected at least 1 '{name}' mention"
            assert mentions[0][1] != 'O', f"'{name}' should be tagged, got {mentions[0][1]}"
    
    def test_mrs_long_tagged(self, processed):
        """'Mrs. Long' - Long should be tagged, Mrs. should not."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        mentions = self._find_all_mentions(tokens, tags, 'Long')
        
        assert len(mentions) >= 2, f"Expected at least 2 'Long' mentions"
        
        # All 'Long' mentions should be tagged
        tagged = [m for m in mentions if m[1] != 'O']
        assert len(tagged) == len(mentions), (
            f"Expected all 'Long' mentions to be tagged"
        )
    
    def test_sir_william_lady_lucas(self, processed):
        """'Sir William' and 'Lady Lucas' - names tagged, titles not."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        # Check Sir is not tagged
        for i, tok in enumerate(tokens):
            if tok == 'Sir':
                assert tags[i] == 'O', f"'Sir' at pos {i} should be O"
        
        # Check Lady is not tagged
        for i, tok in enumerate(tokens):
            if tok == 'Lady':
                assert tags[i] == 'O', f"'Lady' at pos {i} should be O"
        
        # Check William and Lucas ARE tagged
        william_mentions = self._find_all_mentions(tokens, tags, 'William')
        lucas_mentions = self._find_all_mentions(tokens, tags, 'Lucas')
        
        # William should be tagged (it's part of "William Lucas")
        assert len(william_mentions) >= 1, "'William' should appear at least once"
        assert any(m[1] != 'O' for m in william_mentions), "'William' should be tagged"
        
        # Lucas should be tagged
        assert len(lucas_mentions) >= 1, "'Lucas' should appear at least once"
        assert any(m[1] != 'O' for m in lucas_mentions), "'Lucas' should be tagged"
    
    def test_mr_mrs_titles_not_tagged(self, processed):
        """All 'Mr.' and 'Mrs.' titles should NOT be tagged."""
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        for i, tok in enumerate(tokens):
            if tok in ('Mr', 'Mr.', 'Mrs', 'Mrs.'):
                assert tags[i] == 'O', f"Title '{tok}' at pos {i} should be O, got {tags[i]}"


class TestAllMentionsTagging:
    """Cross-cutting tests to verify ALL mentions are tagged, not just first."""
    
    @pytest.fixture
    def preprocessor(self):
        return StoryPreprocessor(use_spacy=False)
    
    def test_repeated_name_all_tagged(self, preprocessor):
        """A name repeated 10 times should be tagged 10 times."""
        text = "John walked in. John sat down. John smiled. John spoke. John listened. " \
               "John agreed. John stood. John left. John returned. John laughed."
        
        story = {
            "story_id": "repeat_test",
            "text": text,
            "entities": [{"text": "John Smith", "type": "PERSON", "role": "protagonist"}]
        }
        
        processed = preprocessor.process_story(story)
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        john_mentions = [(i, t) for i, (tok, t) in enumerate(zip(tokens, tags)) if tok == 'John']
        
        assert len(john_mentions) == 10, f"Expected 10 'John' mentions, found {len(john_mentions)}"
        
        tagged = [m for m in john_mentions if m[1] != 'O']
        assert len(tagged) == 10, (
            f"Expected all 10 'John' mentions to be tagged, but only {len(tagged)} were"
        )
    
    def test_mixed_full_and_partial_names(self, preprocessor):
        """Mix of 'John Smith', 'John', and 'Smith' - all should be tagged."""
        text = "John Smith entered. Smith looked around. John nodded. " \
               "Mr. Smith spoke first. John Smith agreed."
        
        story = {
            "story_id": "mixed_test",
            "text": text,
            "entities": [{"text": "John Smith", "type": "PERSON", "role": "protagonist"}]
        }
        
        processed = preprocessor.process_story(story)
        tokens = processed['tokens']
        tags = processed['bio_tags']
        
        # Count tagged mentions
        john_tagged = sum(1 for tok, tag in zip(tokens, tags) if tok == 'John' and tag != 'O')
        smith_tagged = sum(1 for tok, tag in zip(tokens, tags) if tok == 'Smith' and tag != 'O')
        
        # John appears 3 times (John Smith, John, John Smith)
        assert john_tagged == 3, f"Expected 3 'John' tagged, got {john_tagged}"
        
        # Smith appears 4 times (John Smith, Smith, Mr. Smith, John Smith)
        assert smith_tagged == 4, f"Expected 4 'Smith' tagged, got {smith_tagged}"
    
    def test_supervision_signal_ratio(self, preprocessor):
        """Verify we get high supervision signal (most name tokens tagged)."""
        story = {
            "story_id": "signal_test",
            "text": SHERLOCK_HOLMES_TEXT,
            "entities": [
                {"text": c["name"], "type": "PERSON", "role": c["role"]}
                for c in SHERLOCK_CHARACTERS
            ]
        }
        
        processed = preprocessor.process_story(story)
        tags = processed['bio_tags']
        
        # Count entity tags
        entity_tags = sum(1 for t in tags if t != 'O')
        total_tags = len(tags)
        
        # We should have a reasonable number of entity tags
        # (not just 4-5 for first mentions, but many more)
        assert entity_tags >= 10, (
            f"Expected at least 10 entity tags for supervision, got {entity_tags}. "
            f"This suggests only first mentions are being tagged."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
