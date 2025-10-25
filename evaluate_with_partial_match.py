#!/usr/bin/env python3
"""
Re-evaluate models with partial name matching.

Gives credit when:
1. Model detects ANY component of a ground truth name (e.g., "shrja" matches "Shrja Torel")
2. Role is correct for that component

This better reflects actual usability: if the model finds "Elizabeth" as PROTAGONIST,
that's valuable even if ground truth is "Elizabeth Bennet".
"""

import json
import argparse
from typing import Dict, List, Set, Tuple


def extract_predicted_entities(predictions_file: str) -> Dict:
    """Load predictions from existing evaluation file."""
    with open(predictions_file, 'r') as f:
        return json.load(f)


def partial_match_evaluation(predicted_entities: List[Tuple], ground_truth_entities: List[Tuple]) -> Dict:
    """
    Evaluate with partial name matching.

    Rules:
    - If ANY word in predicted name matches ANY word in ground truth name
    - AND roles match
    - THEN count as a match

    Returns:
        {
            'exact_matches': int,
            'partial_matches': int,
            'role_correct_partial': int,
            'role_wrong': int,
            'false_positives': int,
            'false_negatives': int,
            'partial_f1': float,
            'partial_precision': float,
            'partial_recall': float
        }
    """
    pred_set = set(predicted_entities)
    gt_set = set(ground_truth_entities)

    # Exact matches (current evaluation)
    exact_matches = pred_set & gt_set

    # Track which GT entities were matched (partially or exactly)
    matched_gt = set()
    matched_pred = set()

    # Exact matches
    for match in exact_matches:
        matched_gt.add(match)
        matched_pred.add(match)

    # Partial matches
    partial_matches = []
    role_correct_partial = 0
    role_wrong_partial = 0

    for pred_name, pred_role in pred_set:
        if (pred_name, pred_role) in matched_pred:
            continue  # Already exact matched

        pred_components = set(pred_name.lower().split())

        for gt_name, gt_role in gt_set:
            if (gt_name, gt_role) in matched_gt:
                continue  # GT already matched

            gt_components = set(gt_name.lower().split())

            # Check if any component matches
            if pred_components & gt_components:
                # At least one word matches
                partial_matches.append({
                    'predicted': (pred_name, pred_role),
                    'ground_truth': (gt_name, gt_role),
                    'matching_components': pred_components & gt_components
                })

                matched_pred.add((pred_name, pred_role))
                matched_gt.add((gt_name, gt_role))

                if pred_role == gt_role:
                    role_correct_partial += 1
                else:
                    role_wrong_partial += 1

                break  # Each prediction can only match one GT

    # Calculate metrics
    total_correct = len(exact_matches) + role_correct_partial
    total_matches = len(exact_matches) + len(partial_matches)

    false_positives = len(pred_set) - len(matched_pred)
    false_negatives = len(gt_set) - len(matched_gt)

    partial_precision = total_correct / len(pred_set) if pred_set else 0
    partial_recall = total_correct / len(gt_set) if gt_set else 0
    partial_f1 = 2 * partial_precision * partial_recall / (partial_precision + partial_recall) if (partial_precision + partial_recall) > 0 else 0

    return {
        'exact_matches': len(exact_matches),
        'partial_matches': len(partial_matches),
        'role_correct_partial': role_correct_partial,
        'role_wrong_partial': role_wrong_partial,
        'total_correct': total_correct,
        'total_matches': total_matches,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'partial_precision': partial_precision,
        'partial_recall': partial_recall,
        'partial_f1': partial_f1,
        'partial_match_details': partial_matches
    }


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate with partial name matching")
    parser.add_argument('--eval-file', required=True, help="Existing evaluation JSON file")
    parser.add_argument('--output', required=True, help="Output file for partial match results")

    args = parser.parse_args()

    # Load existing evaluation
    with open(args.eval_file, 'r') as f:
        eval_data = json.load(f)

    print(f"Re-evaluating {eval_data['model']} with partial matching...\n")

    results_by_story = {}
    total_metrics = {
        'exact_matches': 0,
        'partial_matches': 0,
        'role_correct_partial': 0,
        'total_correct': 0,
        'total_predictions': 0,
        'total_ground_truth': 0
    }

    for story_id, story_data in eval_data['by_story'].items():
        if 'predicted_entities' not in story_data:
            continue

        predicted = [tuple(e) for e in story_data['predicted_entities']]
        ground_truth = [tuple(e) for e in story_data['ground_truth_entities']]

        partial_metrics = partial_match_evaluation(predicted, ground_truth)
        results_by_story[story_id] = partial_metrics

        # Aggregate
        total_metrics['exact_matches'] += partial_metrics['exact_matches']
        total_metrics['partial_matches'] += partial_metrics['partial_matches']
        total_metrics['role_correct_partial'] += partial_metrics['role_correct_partial']
        total_metrics['total_correct'] += partial_metrics['total_correct']
        total_metrics['total_predictions'] += len(predicted)
        total_metrics['total_ground_truth'] += len(ground_truth)

        print(f"{story_id}:")
        print(f"  Exact matches: {partial_metrics['exact_matches']}")
        print(f"  Partial matches (role correct): {partial_metrics['role_correct_partial']}")
        print(f"  Partial matches (role wrong): {partial_metrics['role_wrong_partial']}")
        print(f"  Total correct: {partial_metrics['total_correct']} / {len(ground_truth)} GT")
        print(f"  Partial F1: {partial_metrics['partial_f1']:.3f}")
        print()

    # Calculate aggregate metrics
    agg_precision = total_metrics['total_correct'] / total_metrics['total_predictions'] if total_metrics['total_predictions'] > 0 else 0
    agg_recall = total_metrics['total_correct'] / total_metrics['total_ground_truth'] if total_metrics['total_ground_truth'] > 0 else 0
    agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0

    print("="*80)
    print("AGGREGATE RESULTS (Partial Matching)")
    print("="*80)
    print(f"Exact matches: {total_metrics['exact_matches']}")
    print(f"Partial matches (role correct): {total_metrics['role_correct_partial']}")
    print(f"Total correct detections: {total_metrics['total_correct']} / {total_metrics['total_ground_truth']} GT")
    print(f"Precision: {agg_precision:.3f}")
    print(f"Recall: {agg_recall:.3f}")
    print(f"F1 Score: {agg_f1:.3f}")
    print()

    # Compare to strict evaluation
    strict_metrics = eval_data['aggregate_metrics']
    print("COMPARISON TO STRICT EVALUATION:")
    print(f"  Strict F1:   {strict_metrics['f1']:.3f}")
    print(f"  Partial F1:  {agg_f1:.3f}")
    print(f"  Improvement: +{(agg_f1 - strict_metrics['f1'])*100:.1f} percentage points")

    # Save results
    output_data = {
        'model': eval_data['model'],
        'evaluation_type': 'partial_name_matching',
        'aggregate_metrics': {
            'exact_matches': total_metrics['exact_matches'],
            'partial_matches': total_metrics['partial_matches'],
            'role_correct_partial': total_metrics['role_correct_partial'],
            'total_correct': total_metrics['total_correct'],
            'precision': agg_precision,
            'recall': agg_recall,
            'f1': agg_f1,
            'strict_f1_for_comparison': strict_metrics['f1']
        },
        'by_story': results_by_story
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
