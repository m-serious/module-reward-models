#!/usr/bin/env python3
import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reward_model import MultiModuleRewardModel
from data_loader import PairwisePreferenceDataset

class Evaluator:
    def __init__(
        self,
        model: MultiModuleRewardModel,
        dataset: PairwisePreferenceDataset,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_preferences(self, save_results: bool = True) -> Dict:
        print("Evaluating preference predictions...")
        
        module_predictions = {
            'reflection': {'labels': [], 'scores': [], 'correct': []},
            'planner': {'labels': [], 'scores': [], 'correct': []},
            'executor': {'labels': [], 'scores': [], 'correct': []},
            'memory': {'labels': [], 'scores': [], 'correct': []}
        }
        
        with torch.no_grad():
            for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
                sample = self.dataset[idx]
                module = sample['target_module']
                
                # Get positive and negative samples
                pos_context = sample['positive']['trajectory_context']
                pos_module_output = {module: sample['positive']['module_output']}
                
                neg_context = sample['negative']['trajectory_context']
                neg_module_output = {module: sample['negative']['module_output']}
                
                # Compute scores
                pos_scores = self.model(pos_context, pos_module_output, [module])
                neg_scores = self.model(neg_context, neg_module_output, [module])
                
                # Compute preference probability
                logit = pos_scores[module] - neg_scores[module]
                prob = torch.sigmoid(logit).item()
                
                # Record predictions
                module_predictions[module]['labels'].append(1)  # Positive should be preferred
                module_predictions[module]['scores'].append(prob)
                module_predictions[module]['correct'].append(prob > 0.5)
        
        # Compute metrics
        results = {}
        overall_correct = []
        overall_labels = []
        overall_scores = []
        
        for module, preds in module_predictions.items():
            if len(preds['labels']) > 0:
                accuracy = accuracy_score([1] * len(preds['correct']), preds['correct'])
                auc = roc_auc_score([1] * len(preds['scores']), preds['scores'])
                
                results[module] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'num_samples': len(preds['labels'])
                }
                
                overall_correct.extend(preds['correct'])
                overall_labels.extend(preds['labels'])
                overall_scores.extend(preds['scores'])
        
        # Overall metrics
        if overall_labels:
            results['overall'] = {
                'accuracy': accuracy_score([1] * len(overall_correct), overall_correct),
                'auc': roc_auc_score([1] * len(overall_scores), overall_scores),
                'num_samples': len(overall_labels)
            }
        
        # Save results
        if save_results:
            results_path = Path("logs/evaluation_results.json")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_path}")
        
        return results
    
    def analyze_module_scores(self) -> Dict:
        print("Analyzing module score distributions...")
        
        module_scores = {
            'reflection': {'positive': [], 'negative': []},
            'planner': {'positive': [], 'negative': []},
            'executor': {'positive': [], 'negative': []},
            'memory': {'positive': [], 'negative': []}
        }
        
        with torch.no_grad():
            for idx in tqdm(range(len(self.dataset)), desc="Computing scores"):
                sample = self.dataset[idx]
                module = sample['target_module']
                
                # Positive sample
                pos_context = sample['positive']['trajectory_context']
                pos_module_output = {module: sample['positive']['module_output']}
                pos_scores = self.model(pos_context, pos_module_output, [module])
                module_scores[module]['positive'].append(pos_scores[module].item())
                
                # Negative sample
                neg_context = sample['negative']['trajectory_context']
                neg_module_output = {module: sample['negative']['module_output']}
                neg_scores = self.model(neg_context, neg_module_output, [module])
                module_scores[module]['negative'].append(neg_scores[module].item())
        
        # Compute statistics
        stats = {}
        for module, scores in module_scores.items():
            if scores['positive'] and scores['negative']:
                pos_scores = np.array(scores['positive'])
                neg_scores = np.array(scores['negative'])
                
                stats[module] = {
                    'positive': {
                        'mean': float(np.mean(pos_scores)),
                        'std': float(np.std(pos_scores)),
                        'min': float(np.min(pos_scores)),
                        'max': float(np.max(pos_scores))
                    },
                    'negative': {
                        'mean': float(np.mean(neg_scores)),
                        'std': float(np.std(neg_scores)),
                        'min': float(np.min(neg_scores)),
                        'max': float(np.max(neg_scores))
                    },
                    'separation': float(np.mean(pos_scores) - np.mean(neg_scores))
                }
        
        return stats
    
    def plot_score_distributions(self, save_path: str = "logs/score_distributions.png"):
        print("Plotting score distributions...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        modules = ['reflection', 'planner', 'executor', 'memory']
        
        for idx, module in enumerate(modules):
            ax = axes[idx]
            
            # Collect scores
            pos_scores = []
            neg_scores = []
            
            with torch.no_grad():
                for sample_idx in range(len(self.dataset)):
                    sample = self.dataset[sample_idx]
                    if sample['target_module'] == module:
                        # Positive
                        pos_context = sample['positive']['trajectory_context']
                        pos_module_output = {module: sample['positive']['module_output']}
                        pos_score = self.model(pos_context, pos_module_output, [module])
                        pos_scores.append(pos_score[module].item())
                        
                        # Negative
                        neg_context = sample['negative']['trajectory_context']
                        neg_module_output = {module: sample['negative']['module_output']}
                        neg_score = self.model(neg_context, neg_module_output, [module])
                        neg_scores.append(neg_score[module].item())
            
            if pos_scores and neg_scores:
                # Plot distributions
                ax.hist(pos_scores, bins=20, alpha=0.5, label='Positive', color='green')
                ax.hist(neg_scores, bins=20, alpha=0.5, label='Negative', color='red')
                ax.set_title(f'{module.capitalize()} Score Distribution')
                ax.set_xlabel('Score')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    
    def compute_pairwise_accuracy(self) -> Dict:
        print("Computing pairwise accuracy...")
        
        correct_predictions = {
            'reflection': 0,
            'planner': 0,
            'executor': 0,
            'memory': 0
        }
        total_pairs = {
            'reflection': 0,
            'planner': 0,
            'executor': 0,
            'memory': 0
        }
        
        with torch.no_grad():
            # Process pairs
            for i in range(0, len(self.dataset), 2):
                if i + 1 >= len(self.dataset):
                    break
                
                # Assuming pairs are consecutive
                sample1 = self.dataset[i]
                sample2 = self.dataset[i + 1]
                
                if sample1['pair_id'] != sample2['pair_id']:
                    continue
                
                module = sample1['target_module']
                
                # Determine which is positive and negative
                if sample1.get('is_positive', sample1['positive']['outcome'] == 1):
                    pos_sample = sample1
                    neg_sample = sample2
                else:
                    pos_sample = sample2
                    neg_sample = sample1
                
                # Compute scores
                pos_context = pos_sample['positive']['trajectory_context']
                pos_module_output = {module: pos_sample['positive']['module_output']}
                pos_scores = self.model(pos_context, pos_module_output, [module])
                
                neg_context = neg_sample['negative']['trajectory_context']
                neg_module_output = {module: neg_sample['negative']['module_output']}
                neg_scores = self.model(neg_context, neg_module_output, [module])
                
                # Check if positive is preferred
                if pos_scores[module] > neg_scores[module]:
                    correct_predictions[module] += 1
                total_pairs[module] += 1
        
        # Compute accuracies
        accuracies = {}
        for module in correct_predictions:
            if total_pairs[module] > 0:
                accuracies[module] = correct_predictions[module] / total_pairs[module]
            else:
                accuracies[module] = 0.0
        
        # Overall accuracy
        total_correct = sum(correct_predictions.values())
        total = sum(total_pairs.values())
        if total > 0:
            accuracies['overall'] = total_correct / total
        
        return accuracies

def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Module Reward Model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str,
                        default='dataset/training_pairs.json')
    parser.add_argument('--plot', action='store_true',
                        help='Generate score distribution plots')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = MultiModuleRewardModel(
        encoder_name="Qwen/Qwen3-Embedding-0.6B",
        emb_dim=1024,
        hidden_dim=2048,
        freeze_encoder=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded from {args.checkpoint}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = PairwisePreferenceDataset(args.data_path, shuffle_pairs=False)
    
    # Create evaluator
    evaluator = Evaluator(model, dataset, device=args.device)
    
    # Run evaluations
    print("\n" + "="*50)
    print("Running Evaluations")
    print("="*50)
    
    # Preference accuracy
    pref_results = evaluator.evaluate_preferences()
    print("\nPreference Prediction Results:")
    for module, metrics in pref_results.items():
        if isinstance(metrics, dict):
            print(f"  {module}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    AUC: {metrics['auc']:.4f}")
            print(f"    Samples: {metrics['num_samples']}")
    
    # Score distribution analysis
    score_stats = evaluator.analyze_module_scores()
    print("\nScore Distribution Statistics:")
    for module, stats in score_stats.items():
        print(f"  {module}:")
        print(f"    Positive: mean={stats['positive']['mean']:.4f}, std={stats['positive']['std']:.4f}")
        print(f"    Negative: mean={stats['negative']['mean']:.4f}, std={stats['negative']['std']:.4f}")
        print(f"    Separation: {stats['separation']:.4f}")
    
    # Pairwise accuracy
    pairwise_acc = evaluator.compute_pairwise_accuracy()
    print("\nPairwise Accuracy:")
    for module, acc in pairwise_acc.items():
        print(f"  {module}: {acc:.4f}")
    
    # Generate plots if requested
    if args.plot:
        evaluator.plot_score_distributions()
        print("\nPlots generated successfully!")
    
    # Save all results
    all_results = {
        'checkpoint': args.checkpoint,
        'preference_results': pref_results,
        'score_statistics': score_stats,
        'pairwise_accuracy': pairwise_acc
    }
    
    results_path = Path("logs/full_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComplete results saved to {results_path}")

if __name__ == "__main__":
    main()