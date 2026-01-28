# ablation_experiments/evaluate_ablations.py
"""
Evaluation script for ablation experiments using BERAT framework.
Evaluates all trained ablation models and generates comparison results.
"""

import sys
import os
import torch
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.bacterial_evolution_env import BacterialEvolutionEnv
from digital_twin.twin_interface import DigitalTwin
from eval.main_evaluator import BacterialEvolutionEvaluator
from ablation_experiments.configs.ablation_configs import get_all_ablations


def load_ablation_model(model_path: str, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Load a trained ablation model.
    
    Args:
        model_path: Path to saved model
        config: Model configuration
        
    Returns:
        Loaded model
    """
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Create model based on configuration
    if config['use_graph']:
        from models.policy_network import BioPolicyNet
        
        # Determine input dimension from checkpoint or config
        if 'input_dim' in config:
            input_dim = config['input_dim']
        else:
            # Extract from model state dict
            first_layer_weight = None
            for key, value in checkpoint['model_state_dict'].items():
                if 'linear' in key.lower() and 'weight' in key:
                    first_layer_weight = value
                    break
            
            if first_layer_weight is not None:
                input_dim = first_layer_weight.shape[1]
            else:
                input_dim = 8  # Default fallback
        
        model = BioPolicyNet(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_actions=config['num_actions']
        )
    else:
        # Create MLP model for no_graph ablation
        class MLPPolicyNet(torch.nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(), 
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, num_actions),
                    torch.nn.Softmax(dim=-1)
                )
            
            def forward(self, x, edge_index=None):
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)
                return self.network(x)
        
        # Determine input dimension
        input_dim = 8  # Default
        for key, value in checkpoint['model_state_dict'].items():
            if 'network.0.weight' in key:
                input_dim = value.shape[1]
                break
        
        model = MLPPolicyNet(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_actions=config['num_actions']
        )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def generate_ablation_trajectories(model_path: str, 
                                  config: Dict[str, Any],
                                  n_trajectories: int = 30) -> List[Dict]:
    """
    Generate evaluation trajectories for an ablation model.
    
    Args:
        model_path: Path to trained model
        config: Model configuration  
        n_trajectories: Number of trajectories to generate
        
    Returns:
        List of trajectory dictionaries
    """
    
    # Create environment
    env = BacterialEvolutionEnv(
        data_path=config['data_path'],
        amr_path=config['amr_path'],
        sample_size=config['sample_size'],
        max_steps=config['max_steps'],
        antibiotic_pressure=config['antibiotic_pressure']
    )
    
    # Load model
    model = load_ablation_model(model_path, config)
    
    # Create Digital Twin
    twin = DigitalTwin(model, env)
    
    # Generate trajectories
    trajectories = []
    n_available_samples = len(env.features)
    
    for i in range(n_trajectories):
        isolate_idx = i % min(50, n_available_samples)  # Use first 50 isolates
        
        try:
            trajectory = twin.simulate_evolution(
                isolate_idx=isolate_idx,
                generations=8,  # Shorter for ablation evaluation
                antibiotic_pressure=config['antibiotic_pressure']
            )
            
            trajectory['trajectory_id'] = i
            trajectory['isolate_idx'] = isolate_idx
            trajectories.append(trajectory)
            
        except Exception as e:
            print(f"   Failed to generate trajectory {i}: {str(e)}")
            continue
    
    return trajectories


def evaluate_single_ablation(ablation_name: str,
                            ablation_dir: str,
                            config: Dict[str, Any],
                            num_seeds: int = 3) -> Dict[str, Any]:
    """
    Evaluate a single ablation across all seeds.
    
    Args:
        ablation_name: Name of ablation
        ablation_dir: Directory containing ablation results
        config: Ablation configuration
        num_seeds: Number of seeds to evaluate
        
    Returns:
        Evaluation results dictionary
    """
    
    print(f"\n Evaluating {ablation_name}")
    
    # Initialize evaluator
    evaluator = BacterialEvolutionEvaluator(confidence_level=0.95, random_seed=42)
    
    seed_results = []
    
    for seed in range(num_seeds):
        model_path = os.path.join(ablation_dir, f"{ablation_name}_seed{seed}.pt")
        
        if not os.path.exists(model_path):
            print(f" Model not found for seed {seed}: {model_path}")
            continue
        
        try:
            print(f" Generating trajectories for seed {seed}...")
            trajectories = generate_ablation_trajectories(
                model_path=model_path,
                config=config,
                n_trajectories=30
            )
            
            if len(trajectories) < 10:
                print(f" Too few trajectories for seed {seed}: {len(trajectories)}")
                continue
            
            print(f" Evaluating {len(trajectories)} trajectories...")
            results = evaluator.evaluate_model(trajectories)
            
            # Extract key metrics
            seed_result = {
                'seed': seed,
                'n_trajectories': len(trajectories),
                'etci_score': results.get('etci', {}).get('mean_etci', 0.0),
                'gpac_score': results.get('gpac', {}).get('gpac_score', 0.0),
                'aei_score': results.get('aei', {}).get('mean_aei', 0.0),
                'temporal_score': results.get('temporal', {}).get('overall_temporal_score', 0.0),
                'overall_score': results.get('overall_assessment', {}).get('overall_score', 0.0),
                'detailed_results': results
            }
            
            seed_results.append(seed_result)
            
            print(f" Seed {seed}: Overall={seed_result['overall_score']:.3f}")
            
        except Exception as e:
            print(f" Failed to evaluate seed {seed}: {str(e)}")
            continue
    
    if not seed_results:
        print(f" No successful evaluations for {ablation_name}")
        return None
    
    # Aggregate results across seeds
    aggregated_results = {
        'ablation_name': ablation_name,
        'description': config.get('description', ''),
        'n_seeds': len(seed_results),
        'seed_results': seed_results
    }
    
    # Compute mean and std across seeds
    metrics = ['etci_score', 'gpac_score', 'aei_score', 'temporal_score', 'overall_score']
    
    for metric in metrics:
        values = [result[metric] for result in seed_results]
        aggregated_results[f'{metric}_mean'] = np.mean(values) if values else 0.0
        aggregated_results[f'{metric}_std'] = np.std(values) if values else 0.0
    
    print(f" {ablation_name} Summary:")
    print(f" Overall: {aggregated_results['overall_score_mean']:.3f} ± {aggregated_results['overall_score_std']:.3f}")
    print(f" ETCI: {aggregated_results['etci_score_mean']:.3f} ± {aggregated_results['etci_score_std']:.3f}")
    print(f" GPAC: {aggregated_results['gpac_score_mean']:.3f} ± {aggregated_results['gpac_score_std']:.3f}")
    print(f" AEI: {aggregated_results['aei_score_mean']:.3f} ± {aggregated_results['aei_score_std']:.3f}")
    
    return aggregated_results


def evaluate_all_ablations(results_dir: str, 
                          ablation_names: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate all trained ablation experiments.
    
    Args:
        results_dir: Directory containing ablation training results
        ablation_names: Specific ablations to evaluate (None = all available)
        
    Returns:
        Complete evaluation results
    """
    
    print(" Starting Ablation Evaluation")
    print("=" * 60)
    
    # Get available ablations
    all_ablations = get_all_ablations()
    
    # Find available ablation directories
    if ablation_names is None:
        available_ablations = []
        for name in all_ablations.keys():
            ablation_dir = os.path.join(results_dir, name)
            if os.path.exists(ablation_dir):
                available_ablations.append(name)
        ablation_names = available_ablations
    
    print(f"Found {len(ablation_names)} ablations to evaluate: {ablation_names}")
    
    # Evaluate each ablation
    evaluation_results = {}
    
    for ablation_name in ablation_names:
        ablation_dir = os.path.join(results_dir, ablation_name)
        
        if not os.path.exists(ablation_dir):
            print(f" Ablation directory not found: {ablation_dir}")
            continue
        
        config = all_ablations[ablation_name].get_config()
        config['description'] = all_ablations[ablation_name].description
        
        result = evaluate_single_ablation(
            ablation_name=ablation_name,
            ablation_dir=ablation_dir, 
            config=config
        )
        
        if result:
            evaluation_results[ablation_name] = result
    
    print(f"\n Evaluation complete! {len(evaluation_results)} ablations evaluated.")
    
    return evaluation_results


def save_evaluation_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save evaluation results in multiple formats."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed JSON results
    results_path = os.path.join(output_dir, 'ablation_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary CSV
    summary_data = []
    
    for ablation_name, result in results.items():
        if result is None:
            continue
            
        summary_data.append({
            'Ablation': ablation_name,
            'Description': result['description'],
            'Seeds': result['n_seeds'],
            'Overall_Mean': f"{result['overall_score_mean']:.3f}",
            'Overall_Std': f"{result['overall_score_std']:.3f}",
            'ETCI_Mean': f"{result['etci_score_mean']:.3f}",
            'ETCI_Std': f"{result['etci_score_std']:.3f}",
            'GPAC_Mean': f"{result['gpac_score_mean']:.3f}",
            'GPAC_Std': f"{result['gpac_score_std']:.3f}",
            'AEI_Mean': f"{result['aei_score_mean']:.3f}",
            'AEI_Std': f"{result['aei_score_std']:.3f}",
            'Temporal_Mean': f"{result['temporal_score_mean']:.3f}",
            'Temporal_Std': f"{result['temporal_score_std']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'ablation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f" Results saved:")
    print(f" {results_path}")
    print(f" {summary_path}")
    
    # Print summary table
    print(f"\n Ablation Results Summary:")
    print(summary_df.to_string(index=False))


def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(description="Evaluate ablation experiments")
    parser.add_argument('--results_dir', default='results_icml/ablation_results',
                       help='Directory containing ablation training results')
    parser.add_argument('--output_dir', default='results_icml/ablation_evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--ablations', nargs='+', default=None,
                       help='Specific ablations to evaluate')
    
    args = parser.parse_args()
    
    # Evaluate ablations
    results = evaluate_all_ablations(
        results_dir=args.results_dir,
        ablation_names=args.ablations
    )
    
    # Save results
    save_evaluation_results(results, args.output_dir)


if __name__ == "__main__":
    # Add numpy import for aggregation
    import numpy as np
    main()
