# ============================================
# Pathogen/utils/reward_functions.py
# Biologically-Grounded Reward Functions
# UPDATED v3: Amplified fitness gains + gene bonuses
# ============================================

import numpy as np
import pandas as pd

def normalize_state(state):
    """Safely normalize state vector."""
    state = np.nan_to_num(state)
    norm = np.linalg.norm(state)
    return state / (norm + 1e-8)


class BiologicalRewardCalculator:
    """
    Computes biologically-grounded rewards for bacterial evolution.
    
    UPDATED v3: Prioritizes survival via amplified fitness signal
    """
    
    def __init__(self, amr_data_path, antibiotic_pressure=0.5):
        """
        Args:
            amr_data_path: Path to amr_clean.csv
            antibiotic_pressure: Selection pressure [0=none, 1=lethal]
        """
        self.antibiotic_pressure = antibiotic_pressure
        
        # Load AMR gene data
        try:
            amr_df = pd.read_csv(amr_data_path)
            print(f"✓ Loaded {len(amr_df)} AMR gene records")
            
            # Build gene -> resistance strength mapping
            self.gene_to_resistance = {}
            if 'gene' in amr_df.columns:
                gene_counts = amr_df['gene'].value_counts()
                max_count = gene_counts.max()
                for gene, count in gene_counts.items():
                    self.gene_to_resistance[gene] = count / max_count
                
                print(f"✓ Mapped {len(self.gene_to_resistance)} resistance genes")
            else:
                print("⚠ 'gene' column not found in AMR data")
                
        except Exception as e:
            print(f"⚠ Error loading AMR data: {e}")
            self.gene_to_resistance = {}
    
    def compute_survival_probability(self, active_genes):
        """
        Calculate survival probability based on resistance genes.
        """
        n_genes = len(active_genes)
        
        if n_genes == 0:
            survival = max(0.0, 0.10 - self.antibiotic_pressure)
            return float(survival)
        
        # Base survival
        base_survival = 0.30
        
        # Gene count bonus (diminishing returns)
        count_bonus = 0.08 * n_genes * (1.0 - 0.05 * min(n_genes, 10))
        count_bonus = min(count_bonus, 0.40)
        
        # Gene quality bonus
        quality_score = sum(self.gene_to_resistance.get(g, 0.0) for g in active_genes)
        quality_bonus = min(quality_score * 0.10, 0.30)
        
        # Total survival
        survival = base_survival + count_bonus + quality_bonus - self.antibiotic_pressure
        
        return float(np.clip(survival, 0.0, 1.0))
    
    def compute_mutation_cost(self, action, state_complexity):
        """
        Calculate metabolic cost of evolutionary actions.
        """
        if action == 0:  # Mutation
            base_cost = 0.05
            complexity_penalty = 0.03 * state_complexity
            return base_cost + complexity_penalty
            
        elif action == 1:  # Plasmid Transfer
            return 0.04
            
        else:  # Stable
            return 0.0
    
    def compute_plasmid_burden(self, state_features):
        """
        Plasmid maintenance cost.
        """
        if len(state_features) > 0:
            plasmid_size_proxy = state_features[0]
            burden = 0.01 + 0.02 * plasmid_size_proxy
        else:
            burden = 0.01
        
        return burden
    
    def compute_reward(self, old_genes, new_genes, action, state_features):
        """
        Main reward function: AMPLIFIED fitness - costs + bonuses/penalties
        
        UPDATED v3: 
        - 3x fitness multiplier to prioritize survival
        - Rewards for gene acquisition
        - Penalties for gene loss
        """
        # Calculate survival change
        survival_before = self.compute_survival_probability(old_genes)
        survival_after = self.compute_survival_probability(new_genes)
        fitness_gain = survival_after - survival_before
        
        # CRITICAL FIX: Amplify fitness signal (3x weight)
        fitness_gain = fitness_gain * 3.0
        
        # Calculate costs
        state_complexity = np.mean(state_features) if len(state_features) > 0 else 0.5
        mutation_cost = self.compute_mutation_cost(action, state_complexity)
        plasmid_burden = self.compute_plasmid_burden(state_features)
        
        # Net reward
        reward = fitness_gain - mutation_cost - plasmid_burden
        
        # Bonus: First resistance gene (critical milestone)
        if len(old_genes) == 0 and len(new_genes) > 0:
            reward += 0.25  # Increased from 0.15
        
        # Bonus: Gene acquisition (encourage gaining genes)
        if len(new_genes) > len(old_genes):
            genes_gained = len(new_genes) - len(old_genes)
            reward += 0.05 * genes_gained  # 5% bonus per gene
            
            # Extra bonus for high-value genes
            new_gene_values = [
                self.gene_to_resistance.get(g, 0.0) 
                for g in new_genes if g not in old_genes
            ]
            if new_gene_values:
                avg_new_gene_value = np.mean(new_gene_values)
                if avg_new_gene_value > 0.5:  # High-value gene (e.g., blaTEM-1)
                    reward += 0.15
        
        # Penalty: Gene loss (discourage losing resistance)
        if len(new_genes) < len(old_genes):
            genes_lost = len(old_genes) - len(new_genes)
            reward -= 0.10 * genes_lost
        
        return float(reward)


# Legacy function (kept for compatibility)
def compute_reward(survival_prob, instability, alpha=1.0, beta=0.5):
    """Legacy reward function - use BiologicalRewardCalculator instead."""
    return alpha * survival_prob - beta * instability