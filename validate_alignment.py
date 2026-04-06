#!/usr/bin/env python3
"""Validate canonical label alignment."""
import json
import sys
import argparse
from transformers import AutoTokenizer
from rune.ner.label_alignment import align_labels_to_tokenizer


def validate_with_canonical(data_file, model_name, num_samples=100, simplify=False):
    """Validate alignment using canonical function."""
    print(f"🔍 Validating Canonical Label Alignment")
    print(f"📁 Data: {data_file}")
    print(f"🤖 Model: {model_name}")
    print(f"📊 Samples: {num_samples}")
    print(f"🔄 Simplify labels: {simplify}")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    
    # Build label_to_id from actual data
    all_tags = set()
    with open(data_file, 'r') as f:
        for line in list(f)[-num_samples:]:
            story = json.loads(line)
            all_tags.update(story['bio_tags'])
    
    label_to_id = {"O": 0}
    for tag in sorted(all_tags):
        if tag != "O" and tag not in label_to_id:
            label_to_id[tag] = len(label_to_id)
    
    print(f"\n📋 Found {len(label_to_id)} unique labels")
    print(f"   Sample: {list(label_to_id.keys())[:10]}...")
    
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    samples = lines[-num_samples:]
    
    total_orig_entities = 0
    total_aligned_entities = 0
    total_entity_tokens = 0
    
    print("\n📊 Processing samples...")
    for idx, line in enumerate(samples[:10]):
        story = json.loads(line)
        raw_text = story.get("text", "")
        if not raw_text:
            raw_text = " ".join(story["tokens"])
        
        orig_entities = sum(1 for tag in story["bio_tags"] if tag.startswith('B-'))
        
        result = align_labels_to_tokenizer(
            text=raw_text,
            tokens=story["tokens"],
            bio_tags=story["bio_tags"],
            tokenizer=tokenizer,
            max_length=4096,
            label_to_id=label_to_id,
            simplify_labels=simplify,
        )
        
        aligned_entities = sum(1 for lbl in result["labels"] if lbl > 0 and lbl != -100 and (lbl == 1 or any(label_to_id[k] == lbl for k in label_to_id if k.startswith('B-'))))
        labeled_tokens = sum(1 for lbl in result["labels"] if lbl > 0 and lbl != -100)
        
        total_orig_entities += orig_entities
        total_aligned_entities += aligned_entities
        total_entity_tokens += labeled_tokens
        
        if idx < 5:
            print(f"\nStory {idx + 1}:")
            print(f"  Original entities: {orig_entities}")
            print(f"  Aligned entities: {aligned_entities}")
            print(f"  Total entity tokens: {labeled_tokens}")
            if aligned_entities != orig_entities:
                print(f"  ⚠️  Count mismatch!")
    
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY ({len(samples)} stories):")
    print(f"   Original B- entities: {total_orig_entities}")
    print(f"   Aligned B- entities: {total_aligned_entities}")
    print(f"   Total entity tokens: {total_entity_tokens}")
    print(f"   Match rate: {100 * total_aligned_entities / total_orig_entities if total_orig_entities > 0 else 0:.1f}%")
    print("=" * 80)
    
    if total_aligned_entities == total_orig_entities:
        print("\n✅ PERFECT ALIGNMENT - Entity counts match exactly!")
    else:
        print(f"\n⚠️  Alignment has {abs(total_aligned_entities - total_orig_entities)} entity count difference")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ner_training_latest_processed.jsonl")
    parser.add_argument("--model", default="allenai/longformer-base-4096")
    parser.add_argument("--test-run", type=int, default=100)
    parser.add_argument("--simplify-labels", action="store_true")
    args = parser.parse_args()
    
    sys.path.insert(0, "/mnt/storage/rune-lib")
    
    validate_with_canonical(
        args.input,
        args.model,
        num_samples=args.test_run,
        simplify=args.simplify_labels,
    )
