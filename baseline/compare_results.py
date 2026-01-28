# baseline_experiments/compare_results.py
"""
Generate comparison plots for baseline experiments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_comparison():
    """Generate comparison plots."""
    
    # Load results
    results_path = "results_icml/baseline_results/baseline_comparison.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    output_dir = "results_icml/baseline_results"
    
    # Extract data
    policies = [r["policy"] for r in results]
    rewards = [r["mean_reward"] for r in results]
    reward_stds = [r["std_reward"] for r in results]
    survivals = [r["mean_survival"] for r in results]
    survival_stds = [r["std_survival"] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Reward Comparison
    ax = axes[0]
    colors = ['#2E86AB' if 'PathoGen' in p else '#A23B72' for p in policies]
    bars = ax.bar(range(len(policies)), rewards, yerr=reward_stds, 
                   capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha='right')
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Policy Comparison: Reward', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Survival Comparison
    ax = axes[1]
    bars = ax.bar(range(len(policies)), survivals, yerr=survival_stds,
                   capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha='right')
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Policy Comparison: Survival', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 3: Action Distribution
    ax = axes[2]
    action_names = ['Mutate', 'Transfer', 'Stable']
    width = 0.15
    x = np.arange(len(policies))
    
    for i in range(3):
        action_probs = [r["action_dist"][i] for r in results]
        offset = (i - 1) * width
        ax.bar(x + offset, action_probs, width, 
               label=action_names[i], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right')
    ax.set_ylabel('Action Probability', fontsize=12)
    ax.set_title('Action Distribution by Policy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/baseline_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/baseline_comparison.png")
    plt.close()


if __name__ == "__main__":
    plot_comparison()