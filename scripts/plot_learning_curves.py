# scripts/plot_learning_curves.py

import numpy as np
import matplotlib.pyplot as plt
import os

results_dir = "results_icml/metrics"
plots_dir = "results_icml/plots"
os.makedirs(plots_dir, exist_ok=True)

# Load all seeds
all_rewards = []
all_survival = []
all_genes = []

for seed in range(3):
    all_rewards.append(np.load(f"{results_dir}/rewards_seed{seed}.npy"))
    all_survival.append(np.load(f"{results_dir}/survival_seed{seed}.npy"))
    all_genes.append(np.load(f"{results_dir}/gene_counts_seed{seed}.npy"))

# Convert to arrays
all_rewards = np.array(all_rewards)
all_survival = np.array(all_survival)
all_genes = np.array(all_genes)

# Compute statistics
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)
mean_survival = np.mean(all_survival, axis=0)
std_survival = np.std(all_survival, axis=0)
mean_genes = np.mean(all_genes, axis=0)
std_genes = np.std(all_genes, axis=0)

# Smooth for visualization
from scipy.ndimage import uniform_filter1d
window = 10
smooth_rewards = uniform_filter1d(mean_rewards, size=window)
smooth_survival = uniform_filter1d(mean_survival, size=window)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Rewards
ax = axes[0, 0]
episodes = np.arange(len(mean_rewards))
ax.plot(episodes, smooth_rewards, linewidth=2, label='Mean', color='#2E86AB')
ax.fill_between(episodes, 
                 smooth_rewards - std_rewards, 
                 smooth_rewards + std_rewards, 
                 alpha=0.3, color='#2E86AB')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero reward')
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Total Reward', fontsize=12)
ax.set_title('Learning Curve: Reward over Episodes', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Survival
ax = axes[0, 1]
ax.plot(episodes, smooth_survival, linewidth=2, label='Mean', color='#06A77D')
ax.fill_between(episodes, 
                 smooth_survival - std_survival, 
                 smooth_survival + std_survival, 
                 alpha=0.3, color='#06A77D')
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('Survival Probability over Episodes', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Gene count
ax = axes[1, 0]
smooth_genes = uniform_filter1d(mean_genes, size=window)
ax.plot(episodes, smooth_genes, linewidth=2, label='Mean', color='#D62246')
ax.fill_between(episodes, 
                 smooth_genes - std_genes, 
                 smooth_genes + std_genes, 
                 alpha=0.3, color='#D62246')
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Average Gene Count', fontsize=12)
ax.set_title('Gene Acquisition over Episodes', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Distribution of final rewards
ax = axes[1, 1]
final_rewards_all = all_rewards[:, -50:].flatten()
ax.hist(final_rewards_all, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
ax.axvline(x=np.mean(final_rewards_all), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {np.mean(final_rewards_all):.2f}')
ax.set_xlabel('Reward', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Final Episode Rewards', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{plots_dir}/training_results.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plots_dir}/training_results.png")
plt.close()

print("\n Plots generated successfully!")