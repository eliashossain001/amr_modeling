# baseline_experiments/policies/greedy_policy.py
"""
Greedy policy: Choose action that maximizes immediate survival.
"""

import numpy as np

class GreedyPolicy:
    """
    Greedy heuristic: Simulate each action one step ahead,
    choose the one with highest immediate survival.
    """
    
    def __init__(self, env):
        self.env = env
        self.num_actions = env.action_space.n
    
    def predict(self, state):
        """
        Try each action, pick best immediate survival.
        """
        best_action = 0
        best_survival = -float('inf')
        
        # Save current state
        current_genes = self.env.current_genes.copy()
        current_state = self.env.state.copy()
        
        for action in range(self.num_actions):
            # Reset to current state
            self.env.current_genes = current_genes.copy()
            self.env.state = current_state.copy()
            
            # Try this action
            _, reward, _, _, info = self.env.step(action)
            survival = info.get('survival_prob', 0)
            
            if survival > best_survival:
                best_survival = survival
                best_action = action
        
        # Restore state
        self.env.current_genes = current_genes
        self.env.state = current_state
        
        return best_action
    
    def get_action_probs(self, state):
        """Return one-hot for greedy action."""
        action = self.predict(state)
        probs = np.zeros(self.num_actions)
        probs[action] = 1.0
        return probs
    
    def __str__(self):
        return "Greedy Heuristic"