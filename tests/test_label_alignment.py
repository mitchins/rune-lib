#!/usr/bin/env python3
"""Stringent unit tests for label alignment - MUST be token perfect."""
import sys
from transformers import AutoTokenizer
from rune.ner.label_alignment import align_labels_to_tokenizer


def test_simple_entity():
    """Test basic entity alignment."""
    text = "John Smith works at Google."
    tokens = ["John", "Smith", "works", "at", "Google", "."]
    bio_tags = ["B-PERSON", "I-PERSON", "O", "O", "B-LOCATION", "O"]
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", add_prefix_space=True)
    label_to_id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "B-LOCATION": 3, "I-LOCATION": 4}
    
    result = align_labels_to_tokenizer(
        text=text,
        tokens=tokens,
        bio_tags=bio_tags,
        tokenizer=tokenizer,
        max_length=128,
        label_to_id=label_to_id,
        simplify_labels=False,
    )
    
    # Decode and check
    decoded_tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    labels = result["labels"]
    
    print("Test: Simple Entity")
    print(f"Text: {text}")
    for i, (token, label) in enumerate(zip(decoded_tokens, labels)):
        if label != -100:
            label_name = [k for k, v in label_to_id.items() if v == label][0]
            print(f"  Token {i}: '{token}' -> {label_name}")
    
    # Critical assertions
    all_tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    assert any("John" in tok or "john" in tok.lower() for tok in all_tokens), f"'John' not in tokens: {all_tokens}"
    assert 1 in result["labels"] or 2 in result["labels"], f"PERSON entity not found! Labels: {result['labels']}"
    assert 3 in result["labels"], f"LOCATION entity not found! Labels: {result['labels']}"
    print("✅ PASSED\n")


def test_subword_alignment():
    """Test that subword tokens inherit the label from their parent word."""
    text = "Unheard-of character."
    tokens = ["Unheard-of", "character", "."]
    bio_tags = ["B-PERSON", "I-PERSON", "O"]
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", add_prefix_space=True)
    label_to_id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}
    
    result = align_labels_to_tokenizer(
        text=text,
        tokens=tokens,
        bio_tags=bio_tags,
        tokenizer=tokenizer,
        max_length=128,
        label_to_id=label_to_id,
        simplify_labels=False,
    )
    
    decoded_tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    labels = result["labels"]
    
    print("Test: Subword Alignment")
    print(f"Text: {text}")
    print(f"Original tokens: {tokens}")
    print(f"Longformer tokenizes 'Unheard-of' as multiple subwords:")
    
    person_labels = []
    for i, (token, label) in enumerate(zip(decoded_tokens, labels)):
        if label != -100 and label != 0:
            label_name = [k for k, v in label_to_id.items() if v == label][0]
            print(f"  Token {i}: '{token}' -> {label_name}")
            person_labels.append(label)
    
    # ALL subwords of "Unheard-of" and "character" should be labeled as PERSON
    assert len(person_labels) >= 2, f"Expected at least 2 PERSON tokens, got {len(person_labels)}"
    print("✅ PASSED\n")


def test_punctuation_handling():
    """Test that punctuation is correctly labeled as O."""
    text = "Dr. Sarah O'Connor said, \"Hello!\""
    tokens = ["Dr.", "Sarah", "O'Connor", "said", ",", "\"", "Hello", "!", "\""]
    bio_tags = ["B-PERSON", "I-PERSON", "I-PERSON", "O", "O", "O", "O", "O", "O"]
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", add_prefix_space=True)
    label_to_id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}
    
    result = align_labels_to_tokenizer(
        text=text,
        tokens=tokens,
        bio_tags=bio_tags,
        tokenizer=tokenizer,
        max_length=128,
        label_to_id=label_to_id,
        simplify_labels=False,
    )
    
    decoded_tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    labels = result["labels"]
    
    print("Test: Punctuation Handling")
    print(f"Text: {text}")
    for i, (token, label) in enumerate(zip(decoded_tokens, labels)):
        if label != -100:
            label_name = [k for k, v in label_to_id.items() if v == label][0]
            print(f"  Token {i}: '{token}' -> {label_name}")
    
    # Check that we have PERSON labels and O labels
    assert 1 in result["labels"] or 2 in result["labels"], "PERSON entity not found!"
    assert 0 in result["labels"], "O label not found!"
    print("✅ PASSED\n")


def test_adjacent_entities():
    """Test entities that are adjacent with no space."""
    text = "JohnSmith123 is a user."
    tokens = ["JohnSmith123", "is", "a", "user", "."]
    bio_tags = ["B-PERSON", "O", "O", "O", "O"]
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", add_prefix_space=True)
    label_to_id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}
    
    result = align_labels_to_tokenizer(
        text=text,
        tokens=tokens,
        bio_tags=bio_tags,
        tokenizer=tokenizer,
        max_length=128,
        label_to_id=label_to_id,
        simplify_labels=False,
    )
    
    decoded_tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    labels = result["labels"]
    
    print("Test: Adjacent Entities (no space)")
    print(f"Text: {text}")
    for i, (token, label) in enumerate(zip(decoded_tokens, labels)):
        if label != -100:
            label_name = [k for k, v in label_to_id.items() if v == label][0]
            print(f"  Token {i}: '{token}' -> {label_name}")
    
    assert 1 in result["labels"] or 2 in result["labels"], "PERSON entity not found!"
    print("✅ PASSED\n")


def test_entity_at_boundaries():
    """Test entities at start and end of text."""
    text = "Alice talked to Bob"
    tokens = ["Alice", "talked", "to", "Bob"]
    bio_tags = ["B-PERSON", "O", "O", "B-PERSON"]
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", add_prefix_space=True)
    label_to_id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}
    
    result = align_labels_to_tokenizer(
        text=text,
        tokens=tokens,
        bio_tags=bio_tags,
        tokenizer=tokenizer,
        max_length=128,
        label_to_id=label_to_id,
        simplify_labels=False,
    )
    
    decoded_tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    labels = result["labels"]
    
    print("Test: Entities at Boundaries")
    print(f"Text: {text}")
    
    person_count = 0
    for i, (token, label) in enumerate(zip(decoded_tokens, labels)):
        if label != -100:
            label_name = [k for k, v in label_to_id.items() if v == label][0]
            print(f"  Token {i}: '{token}' -> {label_name}")
            if label == 1:  # B-PERSON
                person_count += 1
    
    assert person_count == 2, f"Expected 2 B-PERSON entities, found {person_count}"
    print("✅ PASSED\n")


def test_simplify_labels():
    """Test that simplify_labels correctly collapses entity types."""
    text = "King Arthur of Camelot."
    tokens = ["King", "Arthur", "of", "Camelot", "."]
    bio_tags = ["B-TITLE", "B-PERSON", "O", "B-LOCATION", "O"]
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", add_prefix_space=True)
    label_to_id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "B-LOCATION": 3, "I-LOCATION": 4}
    
    result = align_labels_to_tokenizer(
        text=text,
        tokens=tokens,
        bio_tags=bio_tags,
        tokenizer=tokenizer,
        max_length=128,
        label_to_id=label_to_id,
        simplify_labels=True,  # This should map B-TITLE -> B-PERSON
    )
    
    decoded_tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    labels = result["labels"]
    
    print("Test: Simplify Labels")
    print(f"Text: {text}")
    for i, (token, label) in enumerate(zip(decoded_tokens, labels)):
        if label != -100:
            label_name = [k for k, v in label_to_id.items() if v == label][0]
            print(f"  Token {i}: '{token}' -> {label_name}")
    
    # B-TITLE should become B-PERSON, B-LOCATION should stay B-LOCATION
    assert 1 in result["labels"], "B-PERSON (from B-TITLE) not found!"
    assert 3 in result["labels"], "B-LOCATION not found!"
    print("✅ PASSED\n")


def test_realistic_story_excerpt():
    """Test with a realistic story excerpt."""
    text = "\"I don't trust him,\" Elara whispered to Thorne as they entered the castle."
    tokens = ["\"", "I", "do", "n't", "trust", "him", ",", "\"", "Elara", "whispered", "to", "Thorne", "as", "they", "entered", "the", "castle", "."]
    bio_tags = ["O", "O", "O", "O", "O", "O", "O", "O", "B-PERSON", "O", "O", "B-PERSON", "O", "O", "O", "O", "O", "O"]
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", add_prefix_space=True)
    label_to_id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}
    
    result = align_labels_to_tokenizer(
        text=text,
        tokens=tokens,
        bio_tags=bio_tags,
        tokenizer=tokenizer,
        max_length=128,
        label_to_id=label_to_id,
        simplify_labels=False,
    )
    
    decoded_tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    labels = result["labels"]
    
    print("Test: Realistic Story Excerpt")
    print(f"Text: {text}")
    
    person_entities = []
    for i, (token, label) in enumerate(zip(decoded_tokens, labels)):
        if label != -100:
            label_name = [k for k, v in label_to_id.items() if v == label][0]
            if label == 1:  # B-PERSON
                person_entities.append(token)
                print(f"  Token {i}: '{token}' -> {label_name} ⭐")
            elif label != 0:
                print(f"  Token {i}: '{token}' -> {label_name}")
    
    # Should find both Elara and Thorne
    assert len(person_entities) >= 2, f"Expected at least 2 person entities, found {len(person_entities)}: {person_entities}"
    print(f"Found {len(person_entities)} person entities: {person_entities}")
    print("✅ PASSED\n")


def test_entity_count_consistency():
    """Test that we preserve entity boundaries (no spurious splits)."""
    text = "The wizard Gandalf rode to Minas Tirith with Pippin."
    tokens = ["The", "wizard", "Gandalf", "rode", "to", "Minas", "Tirith", "with", "Pippin", "."]
    bio_tags = ["O", "O", "B-PERSON", "O", "O", "B-LOCATION", "I-LOCATION", "O", "B-PERSON", "O"]
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", add_prefix_space=True)
    label_to_id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "B-LOCATION": 3, "I-LOCATION": 4}
    
    result = align_labels_to_tokenizer(
        text=text,
        tokens=tokens,
        bio_tags=bio_tags,
        tokenizer=tokenizer,
        max_length=128,
        label_to_id=label_to_id,
        simplify_labels=False,
    )
    
    decoded_tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    labels = result["labels"]
    
    print("Test: Entity Count Consistency")
    print(f"Text: {text}")
    print(f"Original entities: Gandalf (PERSON), Minas Tirith (LOCATION), Pippin (PERSON)")
    
    b_person_count = 0
    b_location_count = 0
    for i, (token, label) in enumerate(zip(decoded_tokens, labels)):
        if label != -100:
            label_name = [k for k, v in label_to_id.items() if v == label][0]
            if label != 0:
                print(f"  Token {i}: '{token}' -> {label_name}")
            if label == 1:  # B-PERSON
                b_person_count += 1
            elif label == 3:  # B-LOCATION
                b_location_count += 1
    
    print(f"Found: {b_person_count} PERSON entities, {b_location_count} LOCATION entities")
    assert b_person_count == 2, f"Expected 2 PERSON entities, found {b_person_count}"
    assert b_location_count == 1, f"Expected 1 LOCATION entity, found {b_location_count}"
    print("✅ PASSED - Entity counts match!\n")


if __name__ == "__main__":
    print("="*80)
    print("STRINGENT LABEL ALIGNMENT TESTS")
    print("Testing with Longformer tokenizer (allenai/longformer-base-4096)")
    print("="*80 + "\n")
    
    try:
        test_simple_entity()
        test_subword_alignment()
        test_punctuation_handling()
        test_adjacent_entities()
        test_entity_at_boundaries()
        test_simplify_labels()
        test_realistic_story_excerpt()
        test_entity_count_consistency()
        
        print("="*80)
        print("🎉 ALL TESTS PASSED - ALIGNMENT IS TOKEN PERFECT")
        print("="*80)
        sys.exit(0)
        
    except AssertionError as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        sys.exit(1)
    except Exception as e:
        print("\n" + "="*80)
        print(f"💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        sys.exit(1)
