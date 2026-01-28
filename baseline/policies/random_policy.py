# baseline_experiments/policies/random_policy.py
"""
Random policy: Uniformly random actions.
Baseline to show learning is happening.
"""

import numpy as np

class RandomPolicy:
    """Random action selection."""
    
    def __init__(self, num_actions=3):
        self.num_actions = num_actions
    
    def predict(self, state):
        """Return random action."""
        return np.random.randint(0, self.num_actions)
    
    def get_action_probs(self, state):
        """Return uniform probabilities."""
        return np.ones(self.num_actions) / self.num_actions
    
    def __str__(self):
        return "Random Policy"