#!/usr/bin/env python3
"""
Threshold calibration sweep for NER models.

Finds optimal confidence thresholds per genre by sweeping from 0.25 to 0.55
and measuring both strict and partial F1 scores.

Saves results to: model_dir/thresholds_eval.json
"""

import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from rune.data.improved_tokenizer import ImprovedTokenizer
from rune.utils.entity_normalization import normalize_entity, expand_name_variants, match_entity
from pathlib import Path
import numpy as np


def evaluate_with_threshold(model, tokenizer, story, threshold, mode='partial'):
    """Evaluate a single story with a specific confidence threshold."""
    
    id_to_label = {v: k for k, v in model.config.label2id.items()}
    
    # Tokenize text
    improved_tokenizer = ImprovedTokenizer()
    text = story['text']
    words = improved_tokenizer.split_words(text)
    
    inputs = tokenizer(words, is_split_into_words=True, truncation=True, max_length=8192, return_tensors='pt')
    
    # Get predictions with probabilities
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        
        # Apply threshold to entity probabilities
        max_probs, max_indices = torch.max(probs, dim=-1)
        predictions = max_indices.clone()
        
        # Set to O (background) if confidence below threshold
        o_label_id = model.config.label2id['O']
        below_threshold = max_probs < threshold
        predictions[below_threshold] = o_label_id
    
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
                if current_entity:
                    entity_name = ' '.join(current_entity)
                    normalized = normalize_entity(entity_name)
                    predicted_entities.add((normalized, current_role))
                
                current_role = label.split('-')[1]
                current_entity = [words[word_id]]
                
            elif label.startswith('I-'):
                role = label.split('-')[1]
                if role == current_role and current_entity:
                    current_entity.append(words[word_id])
                else:
                    if current_entity:
                        entity_name = ' '.join(current_entity)
                        normalized = normalize_entity(entity_name)
                        predicted_entities.add((normalized, current_role))
                    current_role = role
                    current_entity = [words[word_id]]
                    
            else:  # O tag
                if current_entity:
                    entity_name = ' '.join(current_entity)
                    normalized = normalize_entity(entity_name)
                    predicted_entities.add((normalized, current_role))
                    current_entity = []
                    current_role = None
    
    if current_entity:
        entity_name = ' '.join(current_entity)
        normalized = normalize_entity(entity_name)
        predicted_entities.add((normalized, current_role))
    
    # Extract ground truth
    ground_truth = set()
    ground_truth_variants = {}
    
    for char in story['characters']:
        canonical_name = char['name']
        normalized_canonical = normalize_entity(canonical_name)
        ground_truth.add((normalized_canonical, 'PERSON'))
        
        if mode == 'partial':
            variants = expand_name_variants(canonical_name)
            ground_truth_variants[normalized_canonical] = variants
    
    # Calculate metrics
    if mode == 'strict':
        true_positives = len(predicted_entities & ground_truth)
        false_positives = len(predicted_entities - ground_truth)
        false_negatives = len(ground_truth - predicted_entities)
    else:  # partial
        matched_predictions = set()
        matched_ground_truth = set()
        
        for pred_name, pred_role in predicted_entities:
            for gt_name, gt_role in ground_truth:
                if pred_role == gt_role:
                    variants = ground_truth_variants[gt_name]
                    if match_entity(pred_name, variants):
                        matched_predictions.add((pred_name, pred_role))
                        matched_ground_truth.add((gt_name, gt_role))
                        break
        
        true_positives = len(matched_predictions)
        false_positives = len(predicted_entities) - true_positives
        false_negatives = len(ground_truth) - len(matched_ground_truth)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def sweep_thresholds(model_path, validation_file, output_file=None):
    """Sweep thresholds from 0.25 to 0.55 and find optimal per genre."""
    
    # Load model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    
    # Load validation data
    print(f"Loading validation data from {validation_file}...")
    validation_data = []
    with open(validation_file, 'r') as f:
        for line in f:
            validation_data.append(json.loads(line.strip()))
    
    # Define genre mapping (can be expanded)
    genre_map = {
        'pride_prejudice_sample': 'classic',
        'colour_of_magic_extract': 'fantasy',
        'the_ships_of_merior_extract': 'fantasy',
        'the_ghost_in_the_doll_extract': 'modern',
        'a_memory_called_empire_extract': 'sci-fi'
    }
    
    # Threshold range
    thresholds = np.arange(0.25, 0.56, 0.05)
    
    print(f"\nSweeping thresholds: {thresholds}")
    print("="*80)
    
    # Results storage
    results_by_threshold = {}
    results_by_genre = {}
    
    for threshold in thresholds:
        threshold = round(threshold, 2)  # Avoid float precision issues
        print(f"\nThreshold: {threshold:.2f}")
        
        threshold_results = {
            'threshold': threshold,
            'by_story': {},
            'by_genre': {},
            'aggregate': {'strict': {}, 'partial': {}}
        }
        
        for story in validation_data:
            story_id = story['story_id']
            genre = genre_map.get(story_id, 'unknown')
            
            # Evaluate with both strict and partial modes
            strict_metrics = evaluate_with_threshold(model, tokenizer, story, threshold, mode='strict')
            partial_metrics = evaluate_with_threshold(model, tokenizer, story, threshold, mode='partial')
            
            threshold_results['by_story'][story_id] = {
                'genre': genre,
                'strict': strict_metrics,
                'partial': partial_metrics
            }
            
            # Accumulate by genre
            if genre not in threshold_results['by_genre']:
                threshold_results['by_genre'][genre] = {
                    'strict': {'tp': 0, 'fp': 0, 'fn': 0},
                    'partial': {'tp': 0, 'fp': 0, 'fn': 0}
                }
            
            threshold_results['by_genre'][genre]['strict']['tp'] += strict_metrics['true_positives']
            threshold_results['by_genre'][genre]['strict']['fp'] += strict_metrics['false_positives']
            threshold_results['by_genre'][genre]['strict']['fn'] += strict_metrics['false_negatives']
            
            threshold_results['by_genre'][genre]['partial']['tp'] += partial_metrics['true_positives']
            threshold_results['by_genre'][genre]['partial']['fp'] += partial_metrics['false_positives']
            threshold_results['by_genre'][genre]['partial']['fn'] += partial_metrics['false_negatives']
        
        # Calculate aggregate metrics
        for mode in ['strict', 'partial']:
            total_tp = sum(s[mode]['true_positives'] for s in threshold_results['by_story'].values())
            total_fp = sum(s[mode]['false_positives'] for s in threshold_results['by_story'].values())
            total_fn = sum(s[mode]['false_negatives'] for s in threshold_results['by_story'].values())
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_results['aggregate'][mode] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn
            }
        
        # Calculate genre-level F1
        for genre, counts in threshold_results['by_genre'].items():
            for mode in ['strict', 'partial']:
                tp = counts[mode]['tp']
                fp = counts[mode]['fp']
                fn = counts[mode]['fn']
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                
                counts[mode].update({'precision': p, 'recall': r, 'f1': f})
        
        results_by_threshold[threshold] = threshold_results
        
        # Print summary
        print(f"  Aggregate Partial F1: {threshold_results['aggregate']['partial']['f1']:.3f} "
              f"(P: {threshold_results['aggregate']['partial']['precision']:.3f}, "
              f"R: {threshold_results['aggregate']['partial']['recall']:.3f})")
    
    # Find optimal thresholds
    print("\n" + "="*80)
    print("OPTIMAL THRESHOLDS")
    print("="*80)
    
    optimal_thresholds = {
        'aggregate': {},
        'by_genre': {}
    }
    
    # Overall optimal
    for mode in ['strict', 'partial']:
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold, results in results_by_threshold.items():
            f1 = results['aggregate'][mode]['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds['aggregate'][mode] = {
            'threshold': best_threshold,
            'f1': best_f1
        }
        
        print(f"\nAggregate ({mode}):")
        print(f"  Optimal threshold: {best_threshold:.2f}")
        print(f"  F1 score: {best_f1:.3f}")
    
    # Per-genre optimal
    genres = set()
    for results in results_by_threshold.values():
        genres.update(results['by_genre'].keys())
    
    for genre in genres:
        optimal_thresholds['by_genre'][genre] = {}
        
        print(f"\n{genre.capitalize()}:")
        
        for mode in ['strict', 'partial']:
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold, results in results_by_threshold.items():
                if genre in results['by_genre']:
                    f1 = results['by_genre'][genre][mode]['f1']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
            
            optimal_thresholds['by_genre'][genre][mode] = {
                'threshold': best_threshold,
                'f1': best_f1
            }
            
            print(f"  {mode}: threshold={best_threshold:.2f}, F1={best_f1:.3f}")
    
    # Save results
    output_data = {
        'model': model_path,
        'validation_file': validation_file,
        'optimal_thresholds': optimal_thresholds,
        'full_sweep': results_by_threshold
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
    
    # Save simplified thresholds config
    thresholds_config_path = Path(model_path) / 'thresholds_eval.json'
    thresholds_config = {
        'default': optimal_thresholds['aggregate']['partial']['threshold'],
        'by_genre': {
            genre: data['partial']['threshold']
            for genre, data in optimal_thresholds['by_genre'].items()
        }
    }
    
    with open(thresholds_config_path, 'w') as f:
        json.dump(thresholds_config, f, indent=2)
    
    print(f"Threshold config saved to: {thresholds_config_path}")
    
    return optimal_thresholds


def main():
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds to find optimal per genre")
    parser.add_argument('--model', required=True, help="Model path")
    parser.add_argument('--validation-file', default='validation_ground_truth.jsonl', help="Validation file")
    parser.add_argument('--output', help="Output JSON file for full results (optional)")
    
    args = parser.parse_args()
    
    sweep_thresholds(args.model, args.validation_file, args.output)


if __name__ == "__main__":
    main()
