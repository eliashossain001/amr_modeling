# baseline_experiments/evaluate_baselines.py
"""
Evaluate all baseline policies and compare to trained model.
Updated to include PPO comparison.
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import pandas as pd
from tqdm import trange
import json

from env.bacterial_evolution_env import BacterialEvolutionEnv
from models.policy_network import BioPolicyNet
from baseline.policies.random_policy import RandomPolicy
from baseline.policies.fixed_policies import (
    AlwaysMutatePolicy, AlwaysTransferPolicy, AlwaysStablePolicy
)
from baseline.policies.greedy_policy import GreedyPolicy

# PPO imports (only if available)
try:
    from stable_baselines3 import PPO
    from baseline.policies.ppo_policy import PPOPolicy
    PPO_AVAILABLE = True
except ImportError:
    print("stable-baselines3 not installed. PPO comparison will be skipped.")
    print("Install with: pip install stable-baselines3")
    PPO_AVAILABLE = False


def evaluate_policy(policy, env, num_episodes=50, policy_name="Policy"):
    """
    Evaluate a policy over multiple episodes.
    
    Returns:
        dict with metrics
    """
    all_rewards = []
    all_survival = []
    all_genes = []
    all_actions = []
    
    for ep in trange(num_episodes, desc=f"Evaluating {policy_name}"):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        ep_survival = []
        ep_genes = []
        ep_actions = []
        
        while not done:
            # Get action from policy
            action = policy.predict(state)
            
            # Execute
            try:
                next_state, reward, done, truncated, info = env.step(action)
            except ValueError:
                # Handle older gym API
                next_state, reward, done, info = env.step(action)
            
            # Track metrics
            total_reward += reward
            if 'survival_prob' in info:
                ep_survival.append(info['survival_prob'])
            if 'n_genes' in info:
                ep_genes.append(info['n_genes'])
            ep_actions.append(action)
            
            state = next_state
        
        all_rewards.append(total_reward)
        all_survival.append(np.mean(ep_survival) if ep_survival else 0.0)
        all_genes.append(np.mean(ep_genes) if ep_genes else 0.0)
        all_actions.extend(ep_actions)
    
    # Compute action distribution
    action_dist = np.bincount(all_actions, minlength=env.action_space.n)
    action_dist = action_dist / action_dist.sum()
    
    return {
        "policy": policy_name,
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_survival": np.mean(all_survival),
        "std_survival": np.std(all_survival),
        "mean_genes": np.mean(all_genes),
        "std_genes": np.std(all_genes),
        "action_dist": action_dist.tolist(),
        "all_rewards": all_rewards,
        "all_survival": all_survival
    }


def evaluate_trained_model(model, env, x, edge_index, num_episodes=50):
    """Evaluate the trained BioPolicyNet (REINFORCE)."""
    all_rewards = []
    all_survival = []
    all_genes = []
    all_actions = []
    
    model.eval()
    
    for ep in trange(num_episodes, desc="Evaluating REINFORCE Model"):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        ep_survival = []
        ep_genes = []
        ep_actions = []
        
        while not done:
            # Get action from trained model
            with torch.no_grad():
                probs_all = model(x, edge_index)
                node_idx = env.current_sample_idx
                probs = probs_all[node_idx].cpu().numpy()
                action = np.argmax(probs)
            
            # Execute
            try:
                next_state, reward, done, truncated, info = env.step(action)
            except ValueError:
                next_state, reward, done, info = env.step(action)
            
            # Track metrics
            total_reward += reward
            if 'survival_prob' in info:
                ep_survival.append(info['survival_prob'])
            if 'n_genes' in info:
                ep_genes.append(info['n_genes'])
            ep_actions.append(action)
            
            state = next_state
        
        all_rewards.append(total_reward)
        all_survival.append(np.mean(ep_survival) if ep_survival else 0.0)
        all_genes.append(np.mean(ep_genes) if ep_genes else 0.0)
        all_actions.extend(ep_actions)
    
    action_dist = np.bincount(all_actions, minlength=env.action_space.n)
    action_dist = action_dist / action_dist.sum()
    
    return {
        "policy": "PathoGen (REINFORCE)",
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_survival": np.mean(all_survival),
        "std_survival": np.std(all_survival),
        "mean_genes": np.mean(all_genes),
        "std_genes": np.std(all_genes),
        "action_dist": action_dist.tolist(),
        "all_rewards": all_rewards,
        "all_survival": all_survival
    }


def evaluate_ppo_model(ppo_path, env, num_episodes=50):
    """Evaluate trained PPO model."""
    
    if not PPO_AVAILABLE:
        print("Cannot evaluate PPO: stable-baselines3 not available")
        return None
    
    try:
        # Load PPO model
        model = PPO.load(ppo_path)
        policy = PPOPolicy(model)
        
        all_rewards = []
        all_survival = []
        all_genes = []
        all_actions = []
        
        for ep in trange(num_episodes, desc="Evaluating PPO"):
            state, _ = env.reset()
            total_reward = 0.0
            done = False
            ep_survival = []
            ep_genes = []
            ep_actions = []
            
            while not done:
                # Get action from PPO
                action = policy.predict(state)
                
                # Execute
                try:
                    next_state, reward, done, truncated, info = env.step(action)
                except ValueError:
                    next_state, reward, done, info = env.step(action)
                
                # Track metrics
                total_reward += reward
                if 'survival_prob' in info:
                    ep_survival.append(info['survival_prob'])
                if 'n_genes' in info:
                    ep_genes.append(info['n_genes'])
                ep_actions.append(action)
                
                state = next_state
            
            all_rewards.append(total_reward)
            all_survival.append(np.mean(ep_survival) if ep_survival else 0.0)
            all_genes.append(np.mean(ep_genes) if ep_genes else 0.0)
            all_actions.extend(ep_actions)
        
        action_dist = np.bincount(all_actions, minlength=env.action_space.n)
        action_dist = action_dist / action_dist.sum()
        
        return {
            "policy": "PPO (stable-baselines3)",
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_survival": np.mean(all_survival),
            "std_survival": np.std(all_survival),
            "mean_genes": np.mean(all_genes),
            "std_genes": np.std(all_genes),
            "action_dist": action_dist.tolist(),
            "all_rewards": all_rewards,
            "all_survival": all_survival
        }
    
    except Exception as e:
        print(f"Error loading PPO model: {e}")
        return None


def main():
    print("\n" + "="*70)
    print("Baseline Experiments - Policy Comparison")
    print("="*70)
    
    # ============================================
    # 1. Setup Environment
    # ============================================
    print("\n[1/5] Initializing environment...")
    env = BacterialEvolutionEnv(
        data_path="data/processed/merged_dataset.csv",
        amr_path="data/processed/amr_clean.csv",
        sample_size=1000,
        max_steps=20,
        antibiotic_pressure=0.5
    )
    
    # ============================================
    # 2. Load Trained REINFORCE Model
    # ============================================
    print("\n[2/5] Loading trained REINFORCE model...")
    checkpoint_path = "results_icml/checkpoints/latest_seed0.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"REINFORCE model not found at: {checkpoint_path}")
        print(" Train it first with: python scripts/train_pathogen_agent_icml.py")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    input_dim = env.features.shape[1]
    num_actions = env.action_space.n
    model = BioPolicyNet(input_dim=input_dim, hidden_dim=128, num_actions=num_actions)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    x = torch.tensor(env.features.values, dtype=torch.float32)
    edge_index = env.get_edge_index()
    
    print(f" Loaded REINFORCE model from episode {checkpoint.get('episode', 'unknown')}")
    
    # ============================================
    # 3. Check for PPO Model
    # ============================================
    print("\n[3/5] Checking for PPO model...")
    ppo_path = "results_icml/baseline_results/ppo/ppo_final.zip"
    ppo_available = os.path.exists(ppo_path) and PPO_AVAILABLE
    
    if ppo_available:
        print(f" PPO model found: {ppo_path}")
    else:
        if not PPO_AVAILABLE:
            print(" PPO evaluation skipped: stable-baselines3 not installed")
        else:
            print(f" PPO model not found: {ppo_path}")
            print("  Train it first with: python baseline_experiments/train_ppo.py")
    
    # ============================================
    # 4. Initialize Baseline Policies
    # ============================================
    print("\n[4/5] Initializing baseline policies...")
    
    policies = [
        ("Random", RandomPolicy(num_actions)),
        ("Always Mutate", AlwaysMutatePolicy()),
        ("Always Transfer", AlwaysTransferPolicy()),
        ("Always Stable", AlwaysStablePolicy()),
        ("Greedy Heuristic", GreedyPolicy(env))
    ]
    
    print(f" Initialized {len(policies)} baseline policies")
    
    # ============================================
    # 5. Evaluate All Policies
    # ============================================
    print("\n[5/5] Running evaluations...")
    
    results = []
    num_episodes = 50
    
    # Evaluate trained REINFORCE model
    print("\n" + "-"*70)
    result = evaluate_trained_model(model, env, x, edge_index, num_episodes=num_episodes)
    results.append(result)
    print(f"REINFORCE: {result['mean_reward']:.3f}±{result['std_reward']:.3f}")
    
    # Evaluate PPO model (if available)
    if ppo_available:
        print("\n" + "-"*70)
        ppo_result = evaluate_ppo_model(ppo_path, env, num_episodes=num_episodes)
        if ppo_result:
            results.append(ppo_result)
            print(f"PPO: {ppo_result['mean_reward']:.3f}±{ppo_result['std_reward']:.3f}")
    
    # Evaluate baselines
    for policy_name, policy in policies:
        print("\n" + "-"*70)
        result = evaluate_policy(policy, env, num_episodes=num_episodes, policy_name=policy_name)
        results.append(result)
        print(f"{policy_name}: {result['mean_reward']:.3f}±{result['std_reward']:.3f}")
    
    # ============================================
    # 6. Save Results
    # ============================================
    print("\n" + "="*70)
    print("Saving results...")
    print("="*70)
    
    output_dir = "results_icml/baseline_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    with open(f"{output_dir}/baseline_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary table
    summary_data = []
    for r in results:
        summary_data.append({
            "Policy": r["policy"],
            "Reward": f"{r['mean_reward']:.3f} ± {r['std_reward']:.3f}",
            "Survival": f"{r['mean_survival']:.3f} ± {r['std_survival']:.3f}",
            "Genes": f"{r['mean_genes']:.1f} ± {r['std_genes']:.1f}",
            "Act0_Mutate": f"{r['action_dist'][0]:.2%}",
            "Act1_Transfer": f"{r['action_dist'][1]:.2%}",
            "Act2_Stable": f"{r['action_dist'][2]:.2%}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/baseline_summary.csv", index=False)
    
    print("\n Results saved:")
    print(f"  {output_dir}/baseline_comparison.json")
    print(f"  {output_dir}/baseline_summary.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    best_result = max(results, key=lambda x: x['mean_reward'])
    worst_result = min(results, key=lambda x: x['mean_reward'])
    
    print(f"Best performing: {best_result['policy']} ({best_result['mean_reward']:.3f})")
    print(f"Worst performing: {worst_result['policy']} ({worst_result['mean_reward']:.3f})")
    
    # Find our models
    reinforce_result = next((r for r in results if "REINFORCE" in r['policy']), None)
    ppo_result = next((r for r in results if "PPO" in r['policy']), None)
    
    if reinforce_result and ppo_result:
        diff = ppo_result['mean_reward'] - reinforce_result['mean_reward']
        print(f"PPO vs REINFORCE: {diff:+.3f} difference")
        
        if abs(diff) < 0.05:
            print(" Similar performance")
        elif diff > 0:
            print(" PPO performs better")
        else:
            print(" REINFORCE performs better")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()