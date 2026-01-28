# ablation_experiments/train_ablations.py
"""
FIXED Training script for ablation experiments.
Trains multiple model variants and saves results for comparison.
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime
import json
import argparse
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# FIX: Ensure utils package is importable
utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
if not os.path.exists(os.path.join(utils_dir, '__init__.py')):
    with open(os.path.join(utils_dir, '__init__.py'), 'w') as f:
        f.write('# Utils package\n')
    print(" Created missing utils/__init__.py")

from env.bacterial_evolution_env import BacterialEvolutionEnv
from models.policy_network import BioPolicyNet
from ablation_experiments.configs.ablation_configs import get_all_ablations, get_priority_ablations


def create_ablated_model(config: Dict[str, Any], input_dim: int) -> torch.nn.Module:
    """Create a model based on ablation configuration."""
    
    if not config['use_graph']:
        return create_mlp_model(config, input_dim)
    else:
        return BioPolicyNet(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'], 
            num_actions=config['num_actions']
        )


def create_mlp_model(config: Dict[str, Any], input_dim: int) -> torch.nn.Module:
    """Create MLP baseline model without graph structure."""
    
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
    
    return MLPPolicyNet(input_dim=input_dim, hidden_dim=config['hidden_dim'], num_actions=config['num_actions'])


def create_ablated_environment(config: Dict[str, Any]) -> BacterialEvolutionEnv:
    """Create environment with ablation-specific modifications."""
    return BacterialEvolutionEnv(
        data_path=config['data_path'],
        amr_path=config['amr_path'],
        sample_size=config['sample_size'],
        max_steps=config['max_steps'],
        antibiotic_pressure=config['antibiotic_pressure']
    )


def train_single_ablation(ablation_name: str, 
                         config: Dict[str, Any],
                         seed: int,
                         output_dir: str) -> Dict[str, Any]:
    """Train a single ablation variant."""
    
    print(f"\n Training {ablation_name} (seed {seed})")
    print(f"   Configuration: {config['hidden_dim']} hidden, {len(config['edge_types'])} edge types")
    
    try:
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create environment
        env = create_ablated_environment(config)
        
        # Create model
        input_dim = env.features.shape[1]
        model = create_ablated_model(config, input_dim)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Training data
        episode_rewards = []
        episode_survivals = []
        episode_gene_counts = []
        
        # Get environment data
        x = torch.tensor(env.features.values, dtype=torch.float32)
        edge_index = env.edge_index if config['use_graph'] else None
        
        print(f"   Starting training for {config['num_episodes']} episodes...")
        
        for episode in range(config['num_episodes']):
            try:
                state, _ = env.reset()
                total_reward = 0.0
                log_probs = []
                rewards = []
                episode_survival = []
                episode_genes = []
                
                done = False
                step_count = 0
                max_steps_per_episode = 20
                
                while not done and step_count < max_steps_per_episode:
                    step_count += 1
                    
                    # FIXED: Get action probabilities (WITH gradients for training)
                    if config['use_graph'] and edge_index is not None:
                        action_probs = model(x, edge_index)
                    else:
                        action_probs = model(x)
                    
                    node_idx = env.current_sample_idx
                    if node_idx >= len(action_probs):
                        node_idx = 0
                    probs = action_probs[node_idx]
                    
                    # Sample action (disable gradients only for sampling)
                    with torch.no_grad():
                        action_dist = torch.distributions.Categorical(probs + 1e-8)
                        action = action_dist.sample()
                    
                    # FIXED: Get log probability (WITH gradients for training)
                    log_prob = torch.distributions.Categorical(probs + 1e-8).log_prob(action)
                    
                    # Take step
                    next_state, reward, done, truncated, info = env.step(action.item())
                    
                    # Store experience
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    total_reward += reward
                    
                    # Store additional info
                    if hasattr(info, '__contains__'):
                        if 'survival_prob' in info:
                            episode_survival.append(info['survival_prob'])
                        if 'n_genes' in info:
                            episode_genes.append(info['n_genes'])
                    
                    state = next_state
                    
                    if truncated:
                        done = True
                
                # REINFORCE update
                if log_probs and len(rewards) > 0:
                    returns = []
                    G = 0
                    gamma = 0.99
                    for r in reversed(rewards):
                        G = r + gamma * G
                        returns.insert(0, G)
                    
                    if len(returns) > 1:
                        returns = torch.tensor(returns, dtype=torch.float32)
                        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                    else:
                        returns = torch.tensor(returns, dtype=torch.float32)
                    
                    # Compute policy loss
                    policy_loss = []
                    for log_prob, G in zip(log_probs, returns):
                        policy_loss.append(-log_prob * G)
                    
                    if policy_loss:
                        policy_loss = torch.stack(policy_loss).sum()
                        
                        # Update model
                        optimizer.zero_grad()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()
                
                # Record episode statistics
                episode_rewards.append(total_reward)
                episode_survivals.append(np.mean(episode_survival) if episode_survival else 0.0)
                episode_gene_counts.append(np.mean(episode_genes) if episode_genes else 0.0)
                
                # Print progress
                if (episode + 1) % 100 == 0:
                    recent_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
                    recent_survival = np.mean(episode_survivals[-50:]) if len(episode_survivals) >= 50 else np.mean(episode_survivals)
                    print(f"   Episode {episode + 1}: Reward={recent_reward:.3f}, Survival={recent_survival:.3f}")
                    
            except Exception as e:
                print(f"   Warning: Episode {episode} failed: {str(e)}")
                episode_rewards.append(0.0)
                episode_survivals.append(0.0)
                episode_gene_counts.append(0.0)
                continue
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{ablation_name}_seed{seed}.pt")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'episode': config['num_episodes'],
            'final_reward': episode_rewards[-1] if episode_rewards else 0.0
        }, model_path)
        
        # Save training metrics
        metrics = {
            'episode_rewards': episode_rewards,
            'episode_survivals': episode_survivals, 
            'episode_gene_counts': episode_gene_counts,
            'final_reward': episode_rewards[-1] if episode_rewards else 0.0,
            'final_survival': episode_survivals[-1] if episode_survivals else 0.0,
            'final_genes': episode_gene_counts[-1] if episode_gene_counts else 0.0,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0
        }
        
        metrics_path = os.path.join(output_dir, f"{ablation_name}_seed{seed}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   Training complete! Final reward: {metrics['final_reward']:.3f}")
        
        return {
            'ablation_name': ablation_name,
            'seed': seed,
            'config': config,
            'metrics': metrics,
            'model_path': model_path
        }
        
    except Exception as e:
        print(f"   Training failed: {str(e)}")
        return None


def train_all_ablations(ablation_names: List[str], 
                       output_dir: str,
                       num_seeds: int = 3) -> Dict[str, List[Dict]]:
    """Train all specified ablation experiments."""
    
    print("🔬 Starting Ablation Training")
    print("=" * 60)
    print(f"Ablations to train: {ablation_names}")
    print(f"Seeds per ablation: {num_seeds}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Get ablation configurations
    all_ablations = get_all_ablations()
    
    # Validate ablation names
    invalid_names = [name for name in ablation_names if name not in all_ablations]
    if invalid_names:
        raise ValueError(f"Invalid ablation names: {invalid_names}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train each ablation
    results = {}
    
    total_experiments = len(ablation_names) * num_seeds
    current_experiment = 0
    
    for ablation_name in ablation_names:
        print(f"\n Processing ablation: {ablation_name}")
        print(f"   Description: {all_ablations[ablation_name].description}")
        
        ablation_results = []
        config = all_ablations[ablation_name].get_config()
        
        # Create ablation-specific directory
        ablation_dir = os.path.join(output_dir, ablation_name)
        os.makedirs(ablation_dir, exist_ok=True)
        
        for seed in range(num_seeds):
            current_experiment += 1
            print(f"   [{current_experiment}/{total_experiments}] Training seed {seed}...")
            
            result = train_single_ablation(
                ablation_name=ablation_name,
                config=config,
                seed=seed,
                output_dir=ablation_dir
            )
            
            if result:
                ablation_results.append(result)
        
        results[ablation_name] = ablation_results
        print(f"   Completed {ablation_name}: {len(ablation_results)}/{num_seeds} seeds successful")
    
    # Save overall results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_ablations': len(ablation_names),
        'num_seeds': num_seeds,
        'results_summary': {
            name: {
                'successful_seeds': len(results.get(name, [])),
                'description': all_ablations[name].description
            }
            for name in ablation_names
        }
    }
    
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n Ablation Training Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"Successful ablations: {sum(len(results.get(name, [])) for name in ablation_names)}/{total_experiments} total")
    
    return results


def main():
    """Main training function with command line interface."""
    
    parser = argparse.ArgumentParser(description="Train ablation experiments")
    parser.add_argument('--ablations', nargs='+', default=None,
                       help='Specific ablations to train (default: priority ablations)')
    parser.add_argument('--output_dir', default='results_icml/ablation_results',
                       help='Output directory for results')
    parser.add_argument('--num_seeds', type=int, default=3,
                       help='Number of random seeds per ablation')
    parser.add_argument('--priority_only', action='store_true',
                       help='Train only priority ablations')
    
    args = parser.parse_args()
    
    # Determine which ablations to train
    if args.ablations:
        ablation_names = args.ablations
    elif args.priority_only:
        ablation_names = list(get_priority_ablations().keys())
    else:
        ablation_names = list(get_priority_ablations().keys())
    
    # Train ablations
    train_all_ablations(
        ablation_names=ablation_names,
        output_dir=args.output_dir,
        num_seeds=args.num_seeds
    )


if __name__ == "__main__":
    main()
