# ============================================
# Pathogen/scripts/run_metrics.py
# Evaluates BioPolicyNet checkpoint performance
# ============================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from tqdm import trange

from env.bacterial_evolution_env import BacterialEvolutionEnv
from models.policy_network import BioPolicyNet
from metrics.adaptation_efficiency_curve import plot_adaptation_efficiency
from metrics.evolutionary_stability_index import evolutionary_stability_index
from metrics.mutation_diversity_entropy import mutation_diversity_entropy
from metrics.policy_convergence_plots import plot_policy_convergence

# --------------------------------------------
# 1. Load model checkpoint
# --------------------------------------------
CHECKPOINT_DIR = "checkpoints"
os.makedirs("visualizations", exist_ok=True)

# Try final, then fallback to latest
model_path = None
for fname in ["final_policy.pt", "latest.pt"]:
    candidate = os.path.join(CHECKPOINT_DIR, fname)
    if os.path.exists(candidate):
        model_path = candidate
        break

if model_path is None:
    raise FileNotFoundError("No model checkpoint found in 'checkpoints/'.")

print(f"\nEvaluating PathoGen model performance using checkpoint: {model_path}\n")

# --------------------------------------------
# 2. Rebuild environment and model
# --------------------------------------------
env = BacterialEvolutionEnv("data/processed/merged_dataset.csv",
                            sample_size=100, max_steps=20)

x = torch.tensor(env.features.values, dtype=torch.float32)
try:
    edge_index = env.get_edge_index()
except Exception:
    from scripts.train_pathogen_agent import build_safe_edge_index
    edge_index = build_safe_edge_index(x.size(0))

input_dim = x.size(1)
num_actions = env.action_space.n
model = BioPolicyNet(input_dim=input_dim, hidden_dim=128, num_actions=num_actions)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# --------------------------------------------
# 3. Simulate rollouts for evaluation
# --------------------------------------------
num_eval_episodes = 50
rewards, all_actions, all_probs = [], [], []

for ep in trange(num_eval_episodes, desc="Evaluating"):
    state, _ = env.reset()
    done = False
    total_reward, ep_actions, ep_probs = 0.0, [], []

    while not done:
        with torch.no_grad():
            probs_all = model(x, edge_index)
        if torch.isnan(probs_all).any() or torch.isinf(probs_all).any():
            probs_all = torch.ones_like(probs_all) / probs_all.size(-1)

        node_idx = np.random.randint(0, x.size(0))
        probs = probs_all[node_idx]
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        try:
            next_state, reward, done, truncated, info = env.step(action.item())
        except ValueError:
            next_state, reward, done, info = env.step(action.item())

        total_reward += reward
        ep_actions.append(int(action.item()))
        ep_probs.append(probs.detach().cpu().numpy().tolist())

    rewards.append(total_reward)
    all_actions.extend(ep_actions)
    all_probs.extend(ep_probs)

# Convert to numpy
# Convert to numpy safely (ensure 2D)
rewards = np.array(rewards, dtype=np.float32)
actions = np.array(all_actions, dtype=np.int32)

# Handle policy probabilities — ensure 2D array (N x num_actions)
try:
    probs = np.array(all_probs, dtype=np.float32)
    if probs.ndim == 1:
        # reshape if flattened, assume 3 actions
        probs = probs.reshape(-1, 3)
except Exception:
    print("Could not reshape policy probabilities — using uniform fallback.")
    probs = np.ones((len(actions), 3)) / 3.0


# --------------------------------------------
# 4. Compute metrics
# --------------------------------------------
print("\nComputing metrics...\n")

adapt_curve_path = plot_adaptation_efficiency(rewards, "visualizations/adaptation_efficiency.png")
esi = evolutionary_stability_index(rewards)
entropy = mutation_diversity_entropy(actions)
policy_conv_path = plot_policy_convergence(probs, "visualizations/policy_convergence.png")

# --------------------------------------------
# 5. Print summary
# --------------------------------------------
print(f"Adaptation Efficiency Curve saved at: {adapt_curve_path}")
print(f"Policy Convergence Plot saved at: {policy_conv_path}")
print(f"Evolutionary Stability Index (ESI): {esi:.4f}")
print(f"Mutation Diversity Entropy: {entropy:.4f}")

if esi > 0.8:
    print("Model shows strong evolutionary stability.")
elif esi > 0.5:
    print("Moderate stability — consider PPO optimization next.")
else:
    print("High volatility detected — try longer horizon or tuned rewards.")
