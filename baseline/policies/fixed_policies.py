# baseline_experiments/policies/fixed_policies.py
"""
Fixed policies: Always choose the same action.
"""

class AlwaysMutatePolicy:
    """Always choose action 0 (mutate)."""
    
    def predict(self, state):
        return 0
    
    def get_action_probs(self, state):
        return [1.0, 0.0, 0.0]
    
    def __str__(self):
        return "Always Mutate"


class AlwaysTransferPolicy:
    """Always choose action 1 (transfer)."""
    
    def predict(self, state):
        return 1
    
    def get_action_probs(self, state):
        return [0.0, 1.0, 0.0]
    
    def __str__(self):
        return "Always Transfer"


class AlwaysStablePolicy:
    """Always choose action 2 (stable)."""
    
    def predict(self, state):
        return 2
    
    def get_action_probs(self, state):
        return [0.0, 0.0, 1.0]
    
    def __str__(self):
        return "Always Stable"