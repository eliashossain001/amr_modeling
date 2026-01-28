# ablation_experiments/configs/ablation_configs.py
"""
Configuration definitions for all ablation experiments.
Each config defines a specific variant to test against the main model.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional


class AblationConfig:
    """Base configuration class for ablation experiments."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.base_config = self._get_base_config()
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Get the base configuration that matches your main model."""
        return {
            # Environment settings
            'data_path': "data/processed/merged_dataset.csv",
            'amr_path': "data/processed/amr_clean.csv",
            'sample_size': 1000,
            'max_steps': 20,
            'antibiotic_pressure': 0.5,
            
            # Model architecture
            'hidden_dim': 128,
            'num_actions': 3,
            'use_graph': True,
            'edge_types': ['gene_similarity', 'geographic', 'serovar', 'plasmid_compatibility'],
            'gnn_layers': 2,
            
            # Training settings
            'algorithm': 'reinforce',
            'learning_rate': 1e-4,
            'num_episodes': 500,
            'num_seeds': 3,
            
            # Reward settings
            'use_survival_reward': True,
            'use_gene_reward': True,
            'survival_weight': 1.0,
            'gene_weight': 0.1,
            'penalty_weight': 0.1,
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the modified configuration for this ablation."""
        config = self.base_config.copy()
        self._modify_config(config)
        return config
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        """Override this method to modify the configuration."""
        pass


# =============================================================================
# GRAPH STRUCTURE ABLATIONS
# =============================================================================

class NoGraphAblation(AblationConfig):
    """Remove graph structure entirely - use MLP only."""
    
    def __init__(self):
        super().__init__(
            name="no_graph",
            description="MLP baseline without any graph structure"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['use_graph'] = False
        config['edge_types'] = []
        config['gnn_layers'] = 0


class GeneSimilarityOnlyAblation(AblationConfig):
    """Use only gene similarity edges."""
    
    def __init__(self):
        super().__init__(
            name="gene_similarity_only",
            description="GCN with gene similarity edges only"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['edge_types'] = ['gene_similarity']


class GeographicOnlyAblation(AblationConfig):
    """Use only geographic proximity edges."""
    
    def __init__(self):
        super().__init__(
            name="geographic_only", 
            description="GCN with geographic proximity edges only"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['edge_types'] = ['geographic']


class SerovarOnlyAblation(AblationConfig):
    """Use only serovar similarity edges."""
    
    def __init__(self):
        super().__init__(
            name="serovar_only",
            description="GCN with serovar similarity edges only"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['edge_types'] = ['serovar']


class PlasmidOnlyAblation(AblationConfig):
    """Use only plasmid compatibility edges."""
    
    def __init__(self):
        super().__init__(
            name="plasmid_only",
            description="GCN with plasmid compatibility edges only"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['edge_types'] = ['plasmid_compatibility']


# =============================================================================
# ARCHITECTURE ABLATIONS
# =============================================================================

class SmallNetworkAblation(AblationConfig):
    """Smaller network with 64 hidden dimensions."""
    
    def __init__(self):
        super().__init__(
            name="small_network",
            description="64 hidden dimensions instead of 128"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['hidden_dim'] = 64


class LargeNetworkAblation(AblationConfig):
    """Larger network with 256 hidden dimensions."""
    
    def __init__(self):
        super().__init__(
            name="large_network",
            description="256 hidden dimensions instead of 128"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['hidden_dim'] = 256


class SingleLayerAblation(AblationConfig):
    """Single GNN layer instead of 2."""
    
    def __init__(self):
        super().__init__(
            name="single_layer",
            description="1 GNN layer instead of 2"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['gnn_layers'] = 1


class TripleLayerAblation(AblationConfig):
    """Three GNN layers instead of 2."""
    
    def __init__(self):
        super().__init__(
            name="triple_layer",
            description="3 GNN layers instead of 2"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['gnn_layers'] = 3


# =============================================================================
# ALGORITHM ABLATIONS
# =============================================================================

class PPOAblation(AblationConfig):
    """Use PPO instead of REINFORCE."""
    
    def __init__(self):
        super().__init__(
            name="ppo_algorithm",
            description="PPO training instead of REINFORCE"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['algorithm'] = 'ppo'


class HighLearningRateAblation(AblationConfig):
    """Higher learning rate."""
    
    def __init__(self):
        super().__init__(
            name="high_lr",
            description="Learning rate 1e-3 instead of 1e-4"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['learning_rate'] = 1e-3


class LowLearningRateAblation(AblationConfig):
    """Lower learning rate."""
    
    def __init__(self):
        super().__init__(
            name="low_lr", 
            description="Learning rate 1e-5 instead of 1e-4"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['learning_rate'] = 1e-5


# =============================================================================
# REWARD FUNCTION ABLATIONS  
# =============================================================================

class SurvivalOnlyAblation(AblationConfig):
    """Only survival-based rewards, no gene bonuses."""
    
    def __init__(self):
        super().__init__(
            name="survival_only",
            description="Survival rewards only, no gene acquisition bonuses"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['use_gene_reward'] = False
        config['gene_weight'] = 0.0


class GenesOnlyAblation(AblationConfig):
    """Only gene acquisition rewards, no survival component."""
    
    def __init__(self):
        super().__init__(
            name="genes_only",
            description="Gene acquisition rewards only, no survival component"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['use_survival_reward'] = False
        config['survival_weight'] = 0.0


class NoPenaltiesAblation(AblationConfig):
    """Remove all negative rewards/penalties."""
    
    def __init__(self):
        super().__init__(
            name="no_penalties",
            description="No negative rewards or penalties"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['penalty_weight'] = 0.0


class BalancedRewardsAblation(AblationConfig):
    """Equal weighting of survival and gene rewards."""
    
    def __init__(self):
        super().__init__(
            name="balanced_rewards",
            description="Equal weighting of survival and gene rewards"
        )
    
    def _modify_config(self, config: Dict[str, Any]) -> None:
        config['survival_weight'] = 1.0
        config['gene_weight'] = 1.0


# =============================================================================
# ABLATION REGISTRY
# =============================================================================

def get_all_ablations() -> Dict[str, AblationConfig]:
    """Get all defined ablation experiments."""
    
    ablations = {}
    
    # Graph structure ablations (High Priority)
    ablations['no_graph'] = NoGraphAblation()
    ablations['gene_similarity_only'] = GeneSimilarityOnlyAblation() 
    ablations['geographic_only'] = GeographicOnlyAblation()
    ablations['serovar_only'] = SerovarOnlyAblation()
    ablations['plasmid_only'] = PlasmidOnlyAblation()
    
    # Architecture ablations (Medium Priority)
    ablations['small_network'] = SmallNetworkAblation()
    ablations['large_network'] = LargeNetworkAblation()
    ablations['single_layer'] = SingleLayerAblation()
    ablations['triple_layer'] = TripleLayerAblation()
    
    # Algorithm ablations (Medium Priority)
    ablations['ppo_algorithm'] = PPOAblation()
    ablations['high_lr'] = HighLearningRateAblation()
    ablations['low_lr'] = LowLearningRateAblation()
    
    # Reward ablations (Lower Priority)
    ablations['survival_only'] = SurvivalOnlyAblation()
    ablations['genes_only'] = GenesOnlyAblation() 
    ablations['no_penalties'] = NoPenaltiesAblation()
    ablations['balanced_rewards'] = BalancedRewardsAblation()
    
    return ablations


def get_priority_ablations() -> Dict[str, AblationConfig]:
    """Get high-priority ablations for time-constrained evaluation."""
    
    all_ablations = get_all_ablations()
    
    priority_names = [
        'no_graph',           # Most important - shows graph value
        'gene_similarity_only', # Individual edge type analysis  
        'geographic_only',
        'serovar_only', 
        'plasmid_only',
        'small_network',      # Architecture sensitivity
        'large_network'
    ]
    
    return {name: all_ablations[name] for name in priority_names if name in all_ablations}


if __name__ == "__main__":
    # Test configuration generation
    ablations = get_all_ablations()
    
    print("Available Ablation Experiments:")
    print("=" * 50)
    
    for name, ablation in ablations.items():
        print(f"{name}: {ablation.description}")
        
    print(f"\nTotal ablations: {len(ablations)}")
    print(f"Priority ablations: {len(get_priority_ablations())}")