# eval/metrics/aei.py
"""
Adaptive Efficiency Index (AEI)

Mathematical foundation: Optimal control theory and information theory
Measures how efficiently learned policies perform compared to theoretical optima using:
- Dynamic programming for theoretical optimal policy
- KL divergence for policy comparison
- Information-theoretic efficiency measures
- Bellman optimality analysis
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from typing import List, Dict, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class AdaptiveEfficiencyIndex:
    """
    Computes AEI using optimal control theory and information measures.
    
    The AEI quantifies how close learned policies are to theoretical optimum
    by deriving optimal policies via dynamic programming and comparing
    using information-theoretic measures.
    
    Mathematical Components:
    1. Theoretical optimal policy via Bellman equations
    2. Policy divergence via KL divergence
    3. Action efficiency via reward/cost analysis
    4. Temporal consistency via policy entropy
    """
    
    def __init__(self, 
                 horizon_length: int = 10,
                 discount_factor: float = 0.95,
                 action_costs: Dict[int, float] = None):
        """
        Initialize AEI computer.
        
        Args:
            horizon_length: Planning horizon for optimal policy
            discount_factor: Future reward discounting [0, 1]
            action_costs: Cost dictionary {action_id: cost}
        """
        self.horizon = horizon_length
        self.gamma = discount_factor
        self.action_costs = action_costs or {0: 0.1, 1: 0.05, 2: 0.0}  # mutate, transfer, stable
        
        # Biological parameters for theoretical model
        self.mutation_gene_gain = 1.0      # Expected genes from mutation
        self.transfer_gene_gain = 2.0      # Expected genes from transfer  
        self.mutation_survival_gain = 0.05  # Survival boost from mutation
        self.transfer_survival_gain = 0.10  # Survival boost from transfer
        
    def compute(self, 
               trajectories: List[Dict],
               model: Optional[Any] = None,
               environment: Optional[Any] = None) -> Dict[str, Any]:
        """
        Compute AEI for trajectory set.
        
        Args:
            trajectories: List of trajectory dictionaries
            model: Optional trained model for additional analysis
            environment: Optional environment for theoretical modeling
            
        Returns:
            AEI results including efficiency scores and comparisons
        """
        aei_scores = []
        detailed_results = []
        
        for i, trajectory in enumerate(trajectories):
            try:
                result = self._compute_single_trajectory_aei(trajectory)
                if result is not None:
                    aei_scores.append(result['aei_score'])
                    detailed_results.append(result)
            except Exception as e:
                print(f" AEI computation failed for trajectory {i}: {str(e)}")
                continue
        
        if not aei_scores:
            return {'error': 'No valid AEI scores computed'}
        
        # Overall statistics
        return {
            'mean_aei': np.mean(aei_scores),
            'std_aei': np.std(aei_scores),
            'median_aei': np.median(aei_scores),
            'min_aei': np.min(aei_scores),
            'max_aei': np.max(aei_scores),
            'aei_distribution': aei_scores,
            'detailed_scores': detailed_results,
            'n_valid_trajectories': len(aei_scores)
        }
    
    def _compute_single_trajectory_aei(self, trajectory: Dict) -> Dict[str, float]:
        """
        Compute AEI for a single trajectory.
        
        Args:
            trajectory: Single trajectory dictionary
            
        Returns:
            Dictionary with AEI score and components
        """
        # Extract trajectory data
        learned_actions = [step.get('action', 2) for step in trajectory['generations']]
        initial_genes = len(trajectory.get('initial_genes', []))
        initial_survival = 0.3  # Baseline survival
        
        if len(learned_actions) == 0:
            return None
        
        # 1. Compute theoretical optimal policy
        optimal_actions = self._compute_theoretical_optimal_policy(
            initial_genes, initial_survival, len(learned_actions)
        )
        
        # 2. Policy comparison metrics
        agreement_rate = self._compute_action_agreement(learned_actions, optimal_actions)
        kl_divergence = self._compute_policy_divergence(learned_actions, optimal_actions)
        
        # 3. Efficiency metrics
        action_efficiency = self._compute_action_efficiency(learned_actions, trajectory)
        temporal_consistency = self._compute_temporal_consistency(learned_actions)
        
        # 4. Combined AEI score
        # High agreement, low divergence, high efficiency, high consistency
        aei_score = (
            0.3 * agreement_rate +
            0.3 * (1.0 / (1.0 + kl_divergence)) +  # Inverse KL
            0.2 * action_efficiency +
            0.2 * temporal_consistency
        )
        
        return {
            'aei_score': aei_score,
            'action_agreement': agreement_rate,
            'policy_divergence': kl_divergence,
            'action_efficiency': action_efficiency,
            'temporal_consistency': temporal_consistency,
            'trajectory_length': len(learned_actions)
        }
    
    def _compute_theoretical_optimal_policy(self, 
                                          initial_genes: int,
                                          initial_survival: float,
                                          trajectory_length: int) -> List[int]:
        """
        Derive theoretical optimal policy using dynamic programming.
        
        Uses Bellman optimality principle to compute optimal action sequence
        for maximizing expected cumulative reward.
        
        Args:
            initial_genes: Starting number of genes
            initial_survival: Starting survival probability
            trajectory_length: Length of policy to compute
            
        Returns:
            List of optimal actions
        """
        # State: (genes, survival, remaining_steps)
        # Action: 0=mutate, 1=transfer, 2=stable
        
        memo = {}  # Memoization for dynamic programming
        
        def value_function(genes: int, survival: float, steps_remaining: int) -> float:
            """
            Compute optimal value function V*(s) via Bellman equation.
            """
            # Terminal condition
            if steps_remaining <= 0:
                return survival
            
            # Memoization key
            state_key = (genes, round(survival, 3), steps_remaining)
            if state_key in memo:
                return memo[state_key]
            
            # Compute Q-values for each action
            action_values = []
            
            # Action 0: Mutate
            new_genes_mut = genes + self.mutation_gene_gain
            new_survival_mut = min(1.0, survival + self.mutation_survival_gain)
            reward_mut = new_survival_mut - self.action_costs[0]
            value_mut = reward_mut + self.gamma * value_function(
                int(new_genes_mut), new_survival_mut, steps_remaining - 1
            )
            action_values.append(value_mut)
            
            # Action 1: Transfer
            new_genes_trans = genes + self.transfer_gene_gain
            new_survival_trans = min(1.0, survival + self.transfer_survival_gain)
            reward_trans = new_survival_trans - self.action_costs[1]
            value_trans = reward_trans + self.gamma * value_function(
                int(new_genes_trans), new_survival_trans, steps_remaining - 1
            )
            action_values.append(value_trans)
            
            # Action 2: Stable
            reward_stable = survival - self.action_costs[2]
            value_stable = reward_stable + self.gamma * value_function(
                genes, survival, steps_remaining - 1
            )
            action_values.append(value_stable)
            
            # Optimal value (Bellman optimality)
            optimal_value = max(action_values)
            memo[state_key] = optimal_value
            
            return optimal_value
        
        # Generate optimal action sequence
        optimal_actions = []
        current_genes = initial_genes
        current_survival = initial_survival
        
        for step in range(trajectory_length):
            remaining = trajectory_length - step
            
            # Compute Q-values for current state
            action_values = []
            
            # Q-value for mutate
            new_genes = current_genes + self.mutation_gene_gain
            new_survival = min(1.0, current_survival + self.mutation_survival_gain)
            q_mut = (new_survival - self.action_costs[0] + 
                    self.gamma * value_function(int(new_genes), new_survival, remaining - 1))
            action_values.append(q_mut)
            
            # Q-value for transfer
            new_genes = current_genes + self.transfer_gene_gain
            new_survival = min(1.0, current_survival + self.transfer_survival_gain)
            q_trans = (new_survival - self.action_costs[1] + 
                      self.gamma * value_function(int(new_genes), new_survival, remaining - 1))
            action_values.append(q_trans)
            
            # Q-value for stable
            q_stable = (current_survival - self.action_costs[2] + 
                       self.gamma * value_function(current_genes, current_survival, remaining - 1))
            action_values.append(q_stable)
            
            # Optimal action (argmax Q)
            optimal_action = np.argmax(action_values)
            optimal_actions.append(optimal_action)
            
            # Update state
            if optimal_action == 0:  # Mutate
                current_genes += self.mutation_gene_gain
                current_survival = min(1.0, current_survival + self.mutation_survival_gain)
            elif optimal_action == 1:  # Transfer
                current_genes += self.transfer_gene_gain  
                current_survival = min(1.0, current_survival + self.transfer_survival_gain)
            # Stable action doesn't change state
        
        return optimal_actions
    
    def _compute_action_agreement(self, learned: List[int], optimal: List[int]) -> float:
        """
        Compute agreement rate between learned and optimal actions.
        
        Args:
            learned: Learned action sequence
            optimal: Optimal action sequence
            
        Returns:
            Agreement rate [0, 1]
        """
        min_length = min(len(learned), len(optimal))
        if min_length == 0:
            return 0.0
        
        agreements = [l == o for l, o in zip(learned[:min_length], optimal[:min_length])]
        return np.mean(agreements)
    
    def _compute_policy_divergence(self, learned: List[int], optimal: List[int]) -> float:
        """
        Compute KL divergence between learned and optimal action distributions.
        
        Args:
            learned: Learned action sequence  
            optimal: Optimal action sequence
            
        Returns:
            KL divergence [0, ∞)
        """
        # Convert to probability distributions
        learned_probs = self._actions_to_distribution(learned)
        optimal_probs = self._actions_to_distribution(optimal)
        
        # Add small epsilon for numerical stability
        epsilon = 1e-8
        learned_probs += epsilon
        optimal_probs += epsilon
        learned_probs /= learned_probs.sum()
        optimal_probs /= optimal_probs.sum()
        
        # KL divergence: D_KL(learned || optimal)
        kl_div = entropy(learned_probs, optimal_probs)
        
        return kl_div if not np.isnan(kl_div) else np.inf
    
    def _actions_to_distribution(self, actions: List[int]) -> np.ndarray:
        """Convert action sequence to probability distribution."""
        if not actions:
            return np.array([1/3, 1/3, 1/3])  # Uniform default
        
        counts = np.bincount(actions, minlength=3)
        return counts / len(actions)
    
    def _compute_action_efficiency(self, actions: List[int], trajectory: Dict) -> float:
        """
        Compute efficiency of actions in achieving survival gains.
        
        Args:
            actions: Action sequence
            trajectory: Full trajectory with outcomes
            
        Returns:
            Efficiency score [0, 1]
        """
        if len(actions) == 0:
            return 0.0
        
        # Compute survival gains
        survival_probs = [step.get('survival_prob', 0.0) for step in trajectory['generations']]
        
        if len(survival_probs) < 2:
            return 0.0
        
        total_survival_gain = survival_probs[-1] - survival_probs[0]
        total_cost = sum(self.action_costs.get(action, 0.1) for action in actions)
        
        # Efficiency = gain per unit cost
        if total_cost > 0:
            efficiency = total_survival_gain / total_cost
            # Normalize to [0, 1] range
            normalized_efficiency = 1.0 / (1.0 + np.exp(-efficiency))  # Sigmoid
        else:
            normalized_efficiency = 1.0 if total_survival_gain > 0 else 0.0
        
        return normalized_efficiency
    
    def _compute_temporal_consistency(self, actions: List[int]) -> float:
        """
        Compute temporal consistency of action choices.
        
        Measures how consistent the policy is over time using entropy.
        Lower entropy = more consistent (but could indicate lack of adaptation).
        
        Args:
            actions: Action sequence
            
        Returns:
            Consistency score [0, 1]
        """
        if len(actions) < 2:
            return 1.0
        
        # Compute action distribution
        action_probs = self._actions_to_distribution(actions)
        
        # Shannon entropy of action distribution
        policy_entropy = entropy(action_probs)
        
        # Maximum entropy (uniform distribution over 3 actions)
        max_entropy = np.log(3)
        
        # Consistency = 1 - normalized entropy
        # High consistency = low entropy = more predictable policy
        if max_entropy > 0:
            consistency = 1.0 - (policy_entropy / max_entropy)
        else:
            consistency = 1.0
        
        return max(0.0, consistency)
