# scripts/analyze_results.py

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Load metrics from all seeds
results_dir = "results_icml/metrics"

for seed in range(3):
    print(f"\n{'='*60}")
    print(f"SEED {seed} ANALYSIS")
    print(f"{'='*60}")
    
    # Load metrics
    rewards = np.load(f"{results_dir}/rewards_seed{seed}.npy")
    survival = np.load(f"{results_dir}/survival_seed{seed}.npy")
    genes = np.load(f"{results_dir}/gene_counts_seed{seed}.npy")
    
    # Print statistics
    print(f"\nRewards:")
    print(f"  Final (last 50): {np.mean(rewards[-50:]):.3f} ± {np.std(rewards[-50:]):.3f}")
    print(f"  Best episode: {np.max(rewards):.3f}")
    print(f"  Worst episode: {np.min(rewards):.3f}")
    
    print(f"\nSurvival:")
    print(f"  Final (last 50): {np.mean(survival[-50:]):.3f} ± {np.std(survival[-50:]):.3f}")
    print(f"  Maximum: {np.max(survival):.3f}")
    
    print(f"\nGene Counts:")
    print(f"  Final (last 50): {np.mean(genes[-50:]):.1f} ± {np.std(genes[-50:]):.1f}")
    print(f"  Maximum: {np.max(genes):.1f}")

print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")

# Load all seeds
all_rewards = []
all_survival = []
for seed in range(3):
    all_rewards.append(np.load(f"{results_dir}/rewards_seed{seed}.npy"))
    all_survival.append(np.load(f"{results_dir}/survival_seed{seed}.npy"))

# Compute mean across seeds
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

print(f"\nFinal Performance (avg over 3 seeds):")
print(f"  Reward: {np.mean(mean_rewards[-50:]):.3f} ± {np.mean(std_rewards[-50:]):.3f}")
print(f"  Survival: {np.mean([np.mean(s[-50:]) for s in all_survival]):.3f}")

# Check convergence
reward_improvement = mean_rewards[-50:].mean() - mean_rewards[:50].mean()
print(f"\nLearning Progress:")
print(f"  Improvement: {reward_improvement:.3f} (first 50 → last 50 episodes)")
print(f"  Converged: {'YES' if abs(mean_rewards[-10:].std()) < 0.5 else '⚠️ Still learning'}")