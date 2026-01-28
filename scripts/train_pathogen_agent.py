# scripts/train_pathogen_agent_icml.py
# Training with Full Logging

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


import torch
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
from tqdm import trange
from env.bacterial_evolution_env import BacterialEvolutionEnv
from models.policy_network import BioPolicyNet

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    "sample_size": 1000,           # Number of bacterial isolates
    "num_episodes": 500,           # Training episodes
    "max_steps": 20,               # Steps per episode
    "antibiotic_pressure": 0.5,    # Selection pressure
    "learning_rate": 1e-4,         # Adam LR
    "hidden_dim": 128,             # GNN hidden dimension
    "random_seed": None,           # Set per run
    "checkpoint_interval": 50,     # Save every N episodes
}

# Directories
OUTPUT_DIR = "results_icml"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_checkpoint(model, optimizer, episode, metrics, config, seed):
    """Save model checkpoint with metadata."""
    checkpoint = {
        "episode": episode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_seed{seed}_ep{episode}.pt")
    torch.save(checkpoint, ckpt_path)
    
    # Also save as latest
    latest_path = os.path.join(CHECKPOINT_DIR, f"latest_seed{seed}.pt")
    torch.save(checkpoint, latest_path)
    
    return ckpt_path


def save_metrics(metrics, config, seed):
    """Save training metrics to JSON and numpy files."""
    # Save as JSON (human-readable)
    json_path = os.path.join(METRICS_DIR, f"metrics_seed{seed}.json")
    with open(json_path, 'w') as f:
        json.dump({
            "config": config,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    # Save as numpy (for plotting)
    np.save(os.path.join(METRICS_DIR, f"rewards_seed{seed}.npy"), metrics["rewards"])
    np.save(os.path.join(METRICS_DIR, f"survival_seed{seed}.npy"), metrics["survival_probs"])
    np.save(os.path.join(METRICS_DIR, f"gene_counts_seed{seed}.npy"), metrics["gene_counts"])
    
    return json_path


def train_single_seed(seed):
    """Train model with a single random seed."""
    print("\n" + "="*80)
    print(f"TRAINING WITH SEED {seed}")
    print("="*80)
    
    # Update config with seed
    config = CONFIG.copy()
    config["random_seed"] = seed
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # ============================================
    # 1. Initialize Environment
    # ============================================
    print("\n[1/4] Initializing environment...")
    env = BacterialEvolutionEnv(
        data_path="data/processed/merged_dataset.csv",
        amr_path="data/processed/amr_clean.csv",
        sample_size=config["sample_size"],
        max_steps=config["max_steps"],
        antibiotic_pressure=config["antibiotic_pressure"]
    )
    
    # ============================================
    # 2. Prepare Graph and Model
    # ============================================
    print("\n[2/4] Preparing model...")
    x = torch.tensor(env.features.values, dtype=torch.float32)
    edge_index = env.get_edge_index()
    
    input_dim = x.size(1)
    num_actions = env.action_space.n
    
    model = BioPolicyNet(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_actions=num_actions
    )
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print(f"  ✓ Model: {input_dim} features → {config['hidden_dim']} hidden → {num_actions} actions")
    print(f"  ✓ Graph: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
    
    # ============================================
    # 3. Training Loop
    # ============================================
    print(f"\n[3/4] Training for {config['num_episodes']} episodes...")
    
    # Metrics storage
    metrics = {
        "rewards": [],
        "survival_probs": [],
        "gene_counts": [],
        "actions": [],
        "episode_lengths": []
    }
    
    for ep in trange(config["num_episodes"], desc=f"Seed {seed}"):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        ep_survival = []
        ep_genes = []
        ep_actions = []
        steps = 0
        
        while not done:
            # Get action
            probs_all = model(x, edge_index)
            
            # Safety
            if torch.isnan(probs_all).any() or torch.isinf(probs_all).any():
                probs_all = torch.ones_like(probs_all) / probs_all.size(-1)
            
            node_idx = np.random.randint(0, x.size(0))
            probs = probs_all[node_idx]
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Execute action
            try:
                next_state, reward, done, truncated, info = env.step(action.item())
            except ValueError:
                next_state, reward, done, info = env.step(action.item())
            
            # Track metrics
            if 'survival_prob' in info:
                ep_survival.append(info['survival_prob'])
            if 'n_genes' in info:
                ep_genes.append(info['n_genes'])
            ep_actions.append(action.item())
            
            # Policy gradient update
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            loss = -log_prob * reward_tensor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += reward
            steps += 1
        
        # Store episode metrics
        metrics["rewards"].append(total_reward)
        metrics["survival_probs"].append(np.mean(ep_survival) if ep_survival else 0.0)
        metrics["gene_counts"].append(np.mean(ep_genes) if ep_genes else 0.0)
        metrics["actions"].extend(ep_actions)
        metrics["episode_lengths"].append(steps)
        
        # Periodic saving
        if (ep + 1) % config["checkpoint_interval"] == 0:
            save_checkpoint(model, optimizer, ep + 1, metrics, config, seed)
            
            # Print progress
            recent_reward = np.mean(metrics["rewards"][-config["checkpoint_interval"]:])
            recent_survival = np.mean(metrics["survival_probs"][-config["checkpoint_interval"]:])
            recent_genes = np.mean(metrics["gene_counts"][-config["checkpoint_interval"]:])
            
            print(f"\n  Episode {ep+1:04d}/{config['num_episodes']}")
            print(f"    Reward: {recent_reward:.3f}")
            print(f"    Survival: {recent_survival:.3f}")
            print(f"    Genes: {recent_genes:.1f}")
    
    # ============================================
    # 4. Save Final Results
    # ============================================
    print(f"\n[4/4] Saving results...")
    
    # Save final checkpoint
    final_ckpt = save_checkpoint(model, optimizer, config["num_episodes"], metrics, config, seed)
    print(f"  ✓ Final checkpoint: {final_ckpt}")
    
    # Save metrics
    metrics_path = save_metrics(metrics, config, seed)
    print(f"  ✓ Metrics saved: {metrics_path}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"SEED {seed} COMPLETE")
    print("="*80)
    print(f"  Final Reward (last 50 eps): {np.mean(metrics['rewards'][-50:]):.3f}")
    print(f"  Final Survival: {np.mean(metrics['survival_probs'][-50:]):.3f}")
    print(f"  Final Gene Count: {np.mean(metrics['gene_counts'][-50:]):.1f}")
    print(f"  Best Episode Reward: {np.max(metrics['rewards']):.3f}")
    print("="*80 + "\n")
    
    return metrics


def main():
    """Run training for multiple seeds."""
    print("\n" + "="*80)
    print("PathoGen ICML Training")
    print("="*80)
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Run multiple seeds
    num_seeds = 3
    all_metrics = {}
    
    for seed in range(num_seeds):
        metrics = train_single_seed(seed)
        all_metrics[f"seed_{seed}"] = metrics
    
    # Save combined results
    combined_path = os.path.join(OUTPUT_DIR, "all_seeds_summary.json")
    with open(combined_path, 'w') as f:
        json.dump({
            "config": CONFIG,
            "num_seeds": num_seeds,
            "seeds": list(range(num_seeds)),
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)
    print(f"  Results saved to: {OUTPUT_DIR}/")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/")
    print(f"  Metrics: {METRICS_DIR}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()