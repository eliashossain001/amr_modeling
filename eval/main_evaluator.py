# eval/main_evaluator.py
"""
Main evaluation orchestrator for bacterial evolution RL models.
Coordinates all mathematical metrics and provides unified interface.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Import metric modules
from .metrics.etci import EvolutionaryTrajectoryCoherenceIndex
from .metrics.gpac import GenotypicPhenotypicAlignmentCoefficient  
from .metrics.aei import AdaptiveEfficiencyIndex
from .metrics.bayesian_uncertainty import BayesianUncertaintyQuantifier
from .metrics.temporal_dynamics import TemporalDynamicsAnalyzer

# Import utilities
from .utils.math_utils import validate_trajectories, compute_confidence_intervals
from .utils.data_utils import preprocess_trajectories, extract_features
from .utils.visualization import create_evaluation_plots

warnings.filterwarnings('ignore', category=RuntimeWarning)


class BacterialEvolutionEvaluator:
    """
    Main evaluation class for bacterial evolution RL models.
    
    Provides comprehensive mathematical assessment using novel metrics:
    - ETCI: Trajectory coherence via dynamical systems theory
    - GPAC: Genotype-phenotype alignment via manifold learning
    - AEI: Policy efficiency via optimal control theory
    - Bayesian confidence intervals for statistical rigor
    - Temporal dynamics analysis across multiple scales
    
    Example:
        evaluator = BacterialEvolutionEvaluator(confidence_level=0.95)
        results = evaluator.evaluate_model(model, trajectories)
        evaluator.save_results(results, 'evaluation_output/')
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 random_seed: int = 42,
                 n_bootstrap_samples: int = 1000):
        """
        Initialize evaluator with configuration parameters.
        
        Args:
            confidence_level: Statistical confidence for intervals (0.90, 0.95, 0.99)
            random_seed: Seed for reproducible results
            n_bootstrap_samples: Number of bootstrap samples for uncertainty
        """
        self.confidence_level = confidence_level
        self.random_seed = random_seed  
        self.n_bootstrap_samples = n_bootstrap_samples
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Initialize metric computers
        self._initialize_metrics()
        
        # Validation parameters
        self.min_trajectories = 10
        self.min_trajectory_length = 3
        
    def _initialize_metrics(self):
        """Initialize all metric computation modules."""
        self.etci_computer = EvolutionaryTrajectoryCoherenceIndex()
        self.gpac_computer = GenotypicPhenotypicAlignmentCoefficient()
        self.aei_computer = AdaptiveEfficiencyIndex()
        self.uncertainty_quantifier = BayesianUncertaintyQuantifier(
            confidence_level=self.confidence_level,
            n_samples=self.n_bootstrap_samples
        )
        self.temporal_analyzer = TemporalDynamicsAnalyzer()
        
    def evaluate_model(self, 
                      trajectories: List[Dict],
                      model: Optional[Any] = None,
                      environment: Optional[Any] = None) -> Dict[str, Any]:
        """
        Main evaluation method - computes all metrics.
        
        Args:
            trajectories: List of trajectory dictionaries from model simulation
            model: Optional trained model for additional analysis
            environment: Optional environment for theoretical comparisons
            
        Returns:
            Comprehensive evaluation results dictionary
            
        Raises:
            ValueError: If trajectories are invalid or insufficient
            RuntimeError: If metric computation fails
        """
        
        print("\n" + "="*80)
        print("BACTERIAL EVOLUTION RL ASSESSMENT TOOLKIT (BERAT)")
        print("="*80)
        print(f"Evaluating {len(trajectories)} trajectories...")
        print(f"Confidence level: {self.confidence_level}")
        print(f"Random seed: {self.random_seed}")
        
        # 1. Validate and preprocess input
        print("\n[1/6] Validating trajectories...")
        try:
            validated_trajectories = self._validate_and_preprocess(trajectories)
            print(f" Validated {len(validated_trajectories)} trajectories")
        except Exception as e:
            raise ValueError(f"Trajectory validation failed: {str(e)}")
        
        # 2. Compute core metrics
        results = {}
        
        try:
            print("\n[2/6] Computing ETCI (Evolutionary Trajectory Coherence Index)...")
            results['etci'] = self.etci_computer.compute(validated_trajectories)
            print(f" ETCI Score: {results['etci']['mean_etci']:.4f}")
            
            print("\n[3/6] Computing GPAC (Genotypic-Phenotypic Alignment Coefficient)...")
            results['gpac'] = self.gpac_computer.compute(validated_trajectories)
            print(f" GPAC Score: {results['gpac']['gpac_score']:.4f}")
            
            print("\n[4/6] Computing AEI (Adaptive Efficiency Index)...")
            results['aei'] = self.aei_computer.compute(validated_trajectories, model, environment)
            print(f" AEI Score: {results['aei']['mean_aei']:.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Core metric computation failed: {str(e)}")
        
        # 3. Uncertainty quantification
        try:
            print("\n[5/6] Computing Bayesian uncertainty estimates...")
            results['uncertainty'] = self.uncertainty_quantifier.compute(results)
            print(" Confidence intervals computed")
        except Exception as e:
            print(f" Warning: Uncertainty quantification failed: {str(e)}")
            results['uncertainty'] = {'status': 'failed', 'error': str(e)}
        
        # 4. Temporal dynamics analysis  
        try:
            print("\n[6/6] Computing temporal dynamics analysis...")
            results['temporal'] = self.temporal_analyzer.compute(validated_trajectories)
            print(" Temporal analysis completed")
        except Exception as e:
            print(f" Warning: Temporal analysis failed: {str(e)}")
            results['temporal'] = {'status': 'failed', 'error': str(e)}
        
        # 5. Compute overall assessment
        results['overall_assessment'] = self._compute_overall_assessment(results)
        
        # 6. Add metadata
        results['metadata'] = self._generate_metadata(validated_trajectories)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Overall Score: {results['overall_assessment']['overall_score']:.4f}")
        print(f"Quality Rating: {results['overall_assessment']['quality_rating']}")
        print("="*80)
        
        return results
    
    def _validate_and_preprocess(self, trajectories: List[Dict]) -> List[Dict]:
        """
        Validate trajectory format and preprocess for metric computation.
        
        Args:
            trajectories: Raw trajectory data
            
        Returns:
            Validated and preprocessed trajectories
            
        Raises:
            ValueError: If validation fails
        """
        if not trajectories:
            raise ValueError("Empty trajectory list provided")
        
        if len(trajectories) < self.min_trajectories:
            raise ValueError(f"Need at least {self.min_trajectories} trajectories, got {len(trajectories)}")
        
        validated = []
        
        for i, trajectory in enumerate(trajectories):
            try:
                # Check required fields
                required_fields = ['initial_genes', 'final_genes', 'generations']
                for field in required_fields:
                    if field not in trajectory:
                        raise ValueError(f"Missing required field: {field}")
                
                # Check trajectory length
                if len(trajectory['generations']) < self.min_trajectory_length:
                    print(f" Skipping short trajectory {i}: {len(trajectory['generations'])} steps")
                    continue
                
                # Validate generation data
                for gen in trajectory['generations']:
                    if 'survival_prob' not in gen or 'genes' not in gen:
                        raise ValueError("Invalid generation data structure")
                
                validated.append(trajectory)
                
            except Exception as e:
                print(f" Skipping invalid trajectory {i}: {str(e)}")
                continue
        
        if len(validated) < self.min_trajectories:
            raise ValueError(f"Only {len(validated)} valid trajectories after filtering")
        
        return validated
    
    def _compute_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall model quality assessment from individual metrics.
        
        Args:
            results: Dictionary containing all computed metrics
            
        Returns:
            Overall assessment including score and quality rating
        """
        # Define weights for overall score
        weights = {
            'etci': 0.25,    # Trajectory coherence
            'gpac': 0.25,    # Genotype-phenotype alignment  
            'aei': 0.30,     # Adaptive efficiency
            'temporal': 0.20 # Temporal dynamics
        }
        
        # Extract scores (with defaults for failed metrics)
        scores = {
            'etci': results.get('etci', {}).get('mean_etci', 0.0),
            'gpac': results.get('gpac', {}).get('gpac_score', 0.0),
            'aei': results.get('aei', {}).get('mean_aei', 0.0),
            'temporal': results.get('temporal', {}).get('overall_temporal_score', 0.0)
        }
        
        # Compute weighted overall score
        overall_score = sum(weights[k] * scores[k] for k in weights.keys())
        
        # Determine quality rating
        if overall_score >= 0.8:
            quality_rating = "Excellent"
        elif overall_score >= 0.6:
            quality_rating = "Good"
        elif overall_score >= 0.4:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"
        
        return {
            'overall_score': overall_score,
            'quality_rating': quality_rating,
            'component_scores': scores,
            'weights': weights,
            'score_breakdown': {k: f"{weights[k]} × {scores[k]:.3f} = {weights[k] * scores[k]:.3f}" 
                              for k in weights.keys()}
        }
    
    def _generate_metadata(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """Generate metadata about the evaluation."""
        return {
            'timestamp': datetime.now().isoformat(),
            'berat_version': "1.0.0",
            'n_trajectories': len(trajectories),
            'evaluation_config': {
                'confidence_level': self.confidence_level,
                'random_seed': self.random_seed,
                'bootstrap_samples': self.n_bootstrap_samples
            }
        }
    
    def save_results(self, 
                    results: Dict[str, Any], 
                    output_dir: str,
                    create_plots: bool = True) -> None:
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results dictionary
            output_dir: Directory to save results
            create_plots: Whether to generate visualization plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        results_path = os.path.join(output_dir, 'detailed_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary as CSV
        summary = self._create_summary_table(results)
        summary_path = os.path.join(output_dir, 'evaluation_summary.csv')
        summary.to_csv(summary_path, index=False)
        
        # Create plots if requested
        if create_plots:
            try:
                plots_dir = os.path.join(output_dir, 'plots')
                create_evaluation_plots(results, plots_dir)
                print(f" Plots saved to {plots_dir}/")
            except Exception as e:
                print(f" Plot generation failed: {str(e)}")
        
        print(f" Results saved to {output_dir}/")
    
    def _create_summary_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary table of all metrics."""
        summary_data = {
            'Metric': ['ETCI', 'GPAC', 'AEI', 'Overall'],
            'Score': [
                results.get('etci', {}).get('mean_etci', 0.0),
                results.get('gpac', {}).get('gpac_score', 0.0),
                results.get('aei', {}).get('mean_aei', 0.0),
                results.get('overall_assessment', {}).get('overall_score', 0.0)
            ],
            'Description': [
                'Evolutionary Trajectory Coherence',
                'Genotype-Phenotype Alignment',
                'Adaptive Efficiency Index',
                'Weighted Overall Assessment'
            ]
        }
        
        return pd.DataFrame(summary_data)
