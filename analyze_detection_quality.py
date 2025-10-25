#!/usr/bin/env python3
"""
Analyze model's actual detection capability and role accuracy.
Goes beyond exact name matching to understand practical usability.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from rune.data.improved_tokenizer import ImprovedTokenizer
import gc


def get_detailed_predictions(model_path: str, text: str) -> List[Dict]:
    """
    Get detailed token-level predictions with confidences.

    Returns list of:
        {
            'token': str,
            'label': str (BIO tag),
            'confidence': float,
            'role': str or None,
            'tag_type': str ('B', 'I', 'O')
        }
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    id_to_label = {v: k for k, v in model.config.label2id.items()}

    improved_tokenizer = ImprovedTokenizer()
    words = improved_tokenizer.split_words(text)

    inputs = tokenizer(words, is_split_into_words=True, truncation=True, max_length=8192, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

    word_ids = inputs.word_ids()
    token_predictions = []

    for idx, word_id in enumerate(word_ids):
        if word_id is not None and (not token_predictions or word_id != word_ids[idx-1]):
            pred_id = predictions[idx].item()
            label = id_to_label[pred_id]
            confidence = probs[idx, pred_id].item()

            tag_type = 'O'
            role = None
            if label != 'O':
                tag_type = label[0]  # 'B' or 'I'
                role = label.split('-')[1]

            token_predictions.append({
                'token': words[word_id],
                'label': label,
                'confidence': confidence,
                'role': role,
                'tag_type': tag_type
            })

    # Clean up
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return token_predictions


def analyze_name_components(predictions: List[Dict], ground_truth_names: Set[str]) -> Dict:
    """
    Check if individual name components (first/last names) are being detected.

    Returns:
        {
            'detected_tokens': set of tokens tagged as entities,
            'name_component_matches': {ground_truth_name: [matching_tokens]},
            'partial_coverage': float (% of ground truth names with at least one component detected)
        }
    """
    detected_tokens = set()
    for pred in predictions:
        if pred['tag_type'] in ('B', 'I'):
            detected_tokens.add(pred['token'].lower())

    name_component_matches = {}
    for gt_name in ground_truth_names:
        gt_name_lower = gt_name.lower()
        components = gt_name_lower.split()

        # Check which components were detected
        matches = []
        for component in components:
            if component in detected_tokens:
                matches.append(component)

        if matches:
            name_component_matches[gt_name] = matches

    partial_coverage = len(name_component_matches) / len(ground_truth_names) if ground_truth_names else 0

    return {
        'detected_tokens': detected_tokens,
        'name_component_matches': name_component_matches,
        'partial_coverage': partial_coverage,
        'total_detected_tokens': len(detected_tokens),
        'total_ground_truth_names': len(ground_truth_names)
    }


def analyze_confidence_near_misses(predictions: List[Dict], ground_truth_names: Set[str]) -> List[Dict]:
    """
    Find tokens that are part of ground truth names but were tagged as O.
    Shows how close the model was to detecting them.
    """
    near_misses = []

    for pred in predictions:
        token_lower = pred['token'].lower()

        # Check if this token appears in any ground truth name
        for gt_name in ground_truth_names:
            if token_lower in gt_name.lower().split():
                if pred['label'] == 'O':
                    near_misses.append({
                        'token': pred['token'],
                        'ground_truth_name': gt_name,
                        'confidence_in_O': pred['confidence'],
                        'was_tagged_as': pred['label']
                    })
                else:
                    # Was tagged as entity but maybe wrong role
                    near_misses.append({
                        'token': pred['token'],
                        'ground_truth_name': gt_name,
                        'predicted_label': pred['label'],
                        'confidence': pred['confidence'],
                        'was_tagged_as': pred['label']
                    })

    return near_misses


def analyze_role_accuracy(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """
    Analyze role classification accuracy for detected entities.
    """
    # Extract entities from predictions
    predicted_entities = []
    current_entity = []
    current_role = None

    for pred in predictions:
        if pred['tag_type'] == 'B':
            if current_entity:
                predicted_entities.append({
                    'name': ' '.join(current_entity).lower(),
                    'role': current_role
                })
            current_entity = [pred['token']]
            current_role = pred['role']
        elif pred['tag_type'] == 'I' and current_entity:
            current_entity.append(pred['token'])
        else:  # O tag
            if current_entity:
                predicted_entities.append({
                    'name': ' '.join(current_entity).lower(),
                    'role': current_role
                })
                current_entity = []
                current_role = None

    if current_entity:
        predicted_entities.append({
            'name': ' '.join(current_entity).lower(),
            'role': current_role
        })

    # Create ground truth lookup
    gt_lookup = {char['name'].lower(): char['role'] for char in ground_truth}

    # Check role accuracy for partial matches
    role_correct = 0
    role_wrong = 0
    not_in_gt = 0

    for pred_entity in predicted_entities:
        pred_name = pred_entity['name']
        pred_role = pred_entity['role']

        # Check if this predicted name is a partial match for any ground truth
        found_match = False
        for gt_name, gt_role in gt_lookup.items():
            # Check if predicted name is substring of GT or vice versa
            if pred_name in gt_name or gt_name in pred_name or any(
                comp in pred_name.split() for comp in gt_name.split()
            ):
                found_match = True
                if pred_role == gt_role:
                    role_correct += 1
                else:
                    role_wrong += 1
                break

        if not found_match:
            not_in_gt += 1

    return {
        'predicted_entity_count': len(predicted_entities),
        'role_correct': role_correct,
        'role_wrong': role_wrong,
        'not_in_ground_truth': not_in_gt,
        'role_accuracy': role_correct / (role_correct + role_wrong) if (role_correct + role_wrong) > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze detection quality and role accuracy")
    parser.add_argument('--model', required=True, help="Model path")
    parser.add_argument('--validation-file', default='validation_ground_truth.jsonl', help="Ground truth file")
    parser.add_argument('--output', required=True, help="Output JSON file")

    args = parser.parse_args()

    # Load validation data
    print(f"Loading validation data from {args.validation_file}...")
    validation_data = []
    with open(args.validation_file, 'r') as f:
        for line in f:
            validation_data.append(json.loads(line.strip()))

    print(f"\n{'='*80}")
    print(f"Analyzing: {args.model}")
    print(f"{'='*80}\n")

    all_results = {}

    for story in validation_data:
        story_id = story['story_id']
        print(f"Processing: {story_id}...")

        # Get detailed predictions
        predictions = get_detailed_predictions(args.model, story['text'])

        # Get ground truth names
        gt_names = {char['name'] for char in story['characters']}

        # Analyze name component detection
        component_analysis = analyze_name_components(predictions, gt_names)

        # Analyze near misses
        near_misses = analyze_confidence_near_misses(predictions, gt_names)

        # Analyze role accuracy
        role_analysis = analyze_role_accuracy(predictions, story['characters'])

        all_results[story_id] = {
            'ground_truth_count': len(story['characters']),
            'component_analysis': {
                'partial_coverage': component_analysis['partial_coverage'],
                'total_detected_tokens': component_analysis['total_detected_tokens'],
                'name_component_matches': component_analysis['name_component_matches']
            },
            'role_analysis': role_analysis,
            'near_misses': near_misses[:20]  # First 20 near misses
        }

        print(f"  Partial Name Coverage: {component_analysis['partial_coverage']:.1%}")
        print(f"  Role Accuracy (for detected): {role_analysis['role_accuracy']:.1%}")
        print(f"  Detected Tokens: {component_analysis['total_detected_tokens']}")
        print(f"  Near Misses: {len(near_misses)}")
        print()

    # Save results
    output_data = {
        'model': args.model,
        'validation_file': args.validation_file,
        'results': all_results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"{'='*80}")
    print(f"RESULTS SAVED TO: {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
