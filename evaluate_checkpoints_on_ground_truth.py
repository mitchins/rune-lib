#!/usr/bin/env python3
"""
Evaluate all model checkpoints against ground truth validation data.
Stores results in structured JSON for analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from rune.data.improved_tokenizer import ImprovedTokenizer
from collections import defaultdict
import gc


def evaluate_model_on_story(model_path: str, story: Dict) -> Dict:
    """
    Evaluate a single model checkpoint on one story.

    Returns:
        {
            'predicted_entities': [(name, role), ...],
            'ground_truth_entities': [(name, role), ...],
            'true_positives': int,
            'false_positives': int,
            'false_negatives': int,
            'precision': float,
            'recall': float,
            'f1': float
        }
    """
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    id_to_label = {v: k for k, v in model.config.label2id.items()}

    # Tokenize text
    improved_tokenizer = ImprovedTokenizer()
    text = story['text']
    words = improved_tokenizer.split_words(text)

    inputs = tokenizer(words, is_split_into_words=True, truncation=True, max_length=8192, return_tensors='pt')

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0]

    # Extract predicted entities
    word_ids = inputs.word_ids()
    predicted_entities = set()

    current_entity = []
    current_role = None

    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            pred_id = predictions[idx].item()
            label = id_to_label[pred_id]

            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entity_name = ' '.join(current_entity)
                    predicted_entities.add((entity_name.lower(), current_role))

                # Start new entity
                current_role = label.split('-')[1]
                current_entity = [words[word_id]]

            elif label.startswith('I-'):
                role = label.split('-')[1]
                if role == current_role and current_entity:
                    current_entity.append(words[word_id])
                else:
                    # Mismatched I- tag, start new entity
                    if current_entity:
                        entity_name = ' '.join(current_entity)
                        predicted_entities.add((entity_name.lower(), current_role))
                    current_role = role
                    current_entity = [words[word_id]]

            else:  # O tag
                if current_entity:
                    entity_name = ' '.join(current_entity)
                    predicted_entities.add((entity_name.lower(), current_role))
                    current_entity = []
                    current_role = None

    # Save last entity
    if current_entity:
        entity_name = ' '.join(current_entity)
        predicted_entities.add((entity_name.lower(), current_role))

    # Extract ground truth entities
    ground_truth = set()
    for char in story['characters']:
        name = char['name'].lower()
        role = char['role']
        ground_truth.add((name, role))

    # Calculate metrics
    true_positives = len(predicted_entities & ground_truth)
    false_positives = len(predicted_entities - ground_truth)
    false_negatives = len(ground_truth - predicted_entities)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Clean up
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'predicted_entities': list(predicted_entities),
        'ground_truth_entities': list(ground_truth),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predicted_count': len(predicted_entities),
        'ground_truth_count': len(ground_truth)
    }


def evaluate_checkpoint(checkpoint_path: str, validation_data: List[Dict]) -> Dict:
    """Evaluate a single checkpoint on all validation stories."""

    print(f"\n{'='*80}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"{'='*80}")

    results_by_story = {}

    for story in validation_data:
        story_id = story['story_id']
        print(f"  Processing: {story_id}...")

        try:
            results_by_story[story_id] = evaluate_model_on_story(checkpoint_path, story)
        except Exception as e:
            print(f"    ERROR: {e}")
            results_by_story[story_id] = {
                'error': str(e),
                'precision': 0,
                'recall': 0,
                'f1': 0
            }

    # Calculate aggregate metrics
    total_tp = sum(r.get('true_positives', 0) for r in results_by_story.values())
    total_fp = sum(r.get('false_positives', 0) for r in results_by_story.values())
    total_fn = sum(r.get('false_negatives', 0) for r in results_by_story.values())

    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0

    # Micro-averaged F1 (average across stories)
    story_f1s = [r.get('f1', 0) for r in results_by_story.values() if 'f1' in r]
    micro_f1 = sum(story_f1s) / len(story_f1s) if story_f1s else 0

    print(f"\n  Aggregate Metrics:")
    print(f"    Precision: {agg_precision:.4f}")
    print(f"    Recall:    {agg_recall:.4f}")
    print(f"    F1 Score:  {agg_f1:.4f}")
    print(f"    Micro F1:  {micro_f1:.4f}")

    return {
        'checkpoint': checkpoint_path,
        'aggregate_metrics': {
            'precision': agg_precision,
            'recall': agg_recall,
            'f1': agg_f1,
            'micro_f1': micro_f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        },
        'by_story': results_by_story
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on ground truth validation data")
    parser.add_argument('--model-dir', required=True, help="Model directory (e.g., ./story_ner_model_modernbert_fixed)")
    parser.add_argument('--validation-file', default='validation_ground_truth.jsonl', help="Ground truth validation file")
    parser.add_argument('--output', default='checkpoint_evaluation_results.json', help="Output JSON file")
    parser.add_argument('--checkpoint-interval', type=int, default=500, help="Evaluate every N checkpoints")

    args = parser.parse_args()

    # Load validation data
    print(f"Loading validation data from {args.validation_file}...")
    validation_data = []
    with open(args.validation_file, 'r') as f:
        for line in f:
            validation_data.append(json.loads(line.strip()))

    print(f"Loaded {len(validation_data)} validation stories")
    print(f"Total ground truth characters: {sum(len(s['characters']) for s in validation_data)}")

    # Find all checkpoints
    model_dir = Path(args.model_dir)
    checkpoints = sorted([
        str(cp) for cp in model_dir.glob('checkpoint-*')
        if cp.is_dir()
    ], key=lambda x: int(x.split('-')[-1]))

    # Filter by interval
    if args.checkpoint_interval > 1:
        filtered_checkpoints = []
        for cp in checkpoints:
            step = int(cp.split('-')[-1])
            if step % args.checkpoint_interval == 0:
                filtered_checkpoints.append(cp)
        checkpoints = filtered_checkpoints

    print(f"\nFound {len(checkpoints)} checkpoints to evaluate")

    if not checkpoints:
        print("No checkpoints found. Make sure training has started.")
        return

    # Evaluate all checkpoints
    all_results = []

    for checkpoint_path in checkpoints:
        result = evaluate_checkpoint(checkpoint_path, validation_data)
        all_results.append(result)

    # Save results
    output_data = {
        'validation_file': args.validation_file,
        'model_dir': args.model_dir,
        'num_validation_stories': len(validation_data),
        'total_ground_truth_entities': sum(len(s['characters']) for s in validation_data),
        'checkpoints_evaluated': len(all_results),
        'results': all_results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"RESULTS SAVED TO: {args.output}")
    print(f"{'='*80}")

    # Print summary table
    print(f"\n{'Checkpoint':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 80)
    for result in all_results:
        cp_name = Path(result['checkpoint']).name
        metrics = result['aggregate_metrics']
        print(f"{cp_name:<20} {metrics['f1']:>8.4f} {metrics['precision']:>10.4f} {metrics['recall']:>8.4f} "
              f"{metrics['true_positives']:>6} {metrics['false_positives']:>6} {metrics['false_negatives']:>6}")

    # Find best checkpoint
    best_checkpoint = max(all_results, key=lambda x: x['aggregate_metrics']['f1'])
    print(f"\n🏆 BEST CHECKPOINT: {Path(best_checkpoint['checkpoint']).name}")
    print(f"   F1 Score: {best_checkpoint['aggregate_metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
