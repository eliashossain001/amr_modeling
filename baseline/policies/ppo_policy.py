# baseline_experiments/policies/ppo_policy.py
"""
PPO (Proximal Policy Optimization) baseline using stable-baselines3.
Current standard for policy optimization in RL.
"""

import numpy as np
import torch

class PPOPolicy:
    """
    Wrapper for trained PPO model from stable-baselines3.
    """
    
    def __init__(self, model):
        """
        Args:
            model: Trained stable_baselines3 PPO model
        """
        self.model = model
    
    def predict(self, state):
        """
        Predict action using trained PPO model.
        """
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)
    
    def get_action_probs(self, state):
        """
        Get action probabilities from PPO policy.
        """
        # Get probabilities from policy network
        obs_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            # Access the policy network
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]
        return probs
    
    def __str__(self):
        return "PPO (stable-baselines3)"