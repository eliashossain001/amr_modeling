# eval/tests/test_metrics.py
"""
Unit tests for evaluation metrics.

Tests all metric computations, edge cases, and numerical stability
to ensure robust evaluation framework performance.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics.etci import EvolutionaryTrajectoryCoherenceIndex
from metrics.gpac import GenotypicPhenotypicAlignmentCoefficient
from metrics.aei import AdaptiveEfficiencyIndex
from metrics.bayesian_uncertainty import BayesianUncertaintyQuantifier
from metrics.temporal_dynamics import TemporalDynamicsAnalyzer
from utils.math_utils import validate_trajectories, safe_entropy, robust_correlation
from utils.data_utils import preprocess_trajectories, extract_features


class TestETCI(unittest.TestCase):
    """Test Evolutionary Trajectory Coherence Index."""
    
    def setUp(self):
        self.etci_computer = EvolutionaryTrajectoryCoherenceIndex()
        self.sample_trajectories = self._create_sample_trajectories()
    
    def _create_sample_trajectories(self):
        """Create sample trajectories for testing."""
        trajectories = []
        
        for i in range(10):
            trajectory = {
                'initial_genes': [f'gene_{j}' for j in range(i % 5)],
                'final_genes': [f'gene_{j}' for j in range((i % 5) + 3)],
                'generations': []
            }
            
            # Create generations with varying complexity
            for step in range(5 + i % 3):
                gen = {
                    'genes': [f'gene_{j}' for j in range(step + 1)],
                    'survival_prob': 0.3 + 0.1 * step + 0.05 * np.sin(step),
                    'action': step % 3
                }
                trajectory['generations'].append(gen)
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def test_etci_computation(self):
        """Test basic ETCI computation."""
        result = self.etci_computer.compute(self.sample_trajectories)
        
        self.assertIsInstance(result, dict)
        self.assertIn('mean_etci', result)
        self.assertIn('std_etci', result)
        
        # Check value ranges
        self.assertGreaterEqual(result['mean_etci'], 0.0)
        self.assertLessEqual(result['mean_etci'], 1.0)
    
    def test_etci_empty_trajectories(self):
        """Test ETCI with empty trajectory list."""
        result = self.etci_computer.compute([])
        self.assertIn('error', result)
    
    def test_etci_short_trajectories(self):
        """Test ETCI with very short trajectories."""
        short_trajectories = []
        for i in range(3):
            trajectory = {
                'initial_genes': ['gene1'],
                'final_genes': ['gene1', 'gene2'],
                'generations': [
                    {'genes': ['gene1'], 'survival_prob': 0.3, 'action': 0},
                    {'genes': ['gene1', 'gene2'], 'survival_prob': 0.4, 'action': 1}
                ]
            }
            short_trajectories.append(trajectory)
        
        result = self.etci_computer.compute(short_trajectories)
        # Should handle short trajectories gracefully
        self.assertIsInstance(result, dict)
    
    def test_etci_deterministic(self):
        """Test ETCI deterministic behavior."""
        # Same trajectories should give same results
        result1 = self.etci_computer.compute(self.sample_trajectories)
        result2 = self.etci_computer.compute(self.sample_trajectories)
        
        self.assertAlmostEqual(result1['mean_etci'], result2['mean_etci'], places=6)


class TestGPAC(unittest.TestCase):
    """Test Genotypic-Phenotypic Alignment Coefficient."""
    
    def setUp(self):
        self.gpac_computer = GenotypicPhenotypicAlignmentCoefficient()
        self.sample_trajectories = self._create_diverse_trajectories()
    
    def _create_diverse_trajectories(self):
        """Create trajectories with diverse genotype-phenotype relationships."""
        trajectories = []
        
        for i in range(15):  # More trajectories for manifold learning
            # Create correlated genotype-phenotype relationship
            n_genes = i + 2
            survival_prob = 0.2 + 0.05 * n_genes + 0.1 * np.random.normal()
            survival_prob = max(0.0, min(1.0, survival_prob))
            
            trajectory = {
                'initial_genes': [f'gene_{j}' for j in range(2)],
                'final_genes': [f'gene_{j}' for j in range(n_genes)],
                'generations': [
                    {'genes': [f'gene_{j}' for j in range(n_genes)], 
                     'survival_prob': survival_prob, 
                     'action': i % 3}
                ]
            }
            trajectories.append(trajectory)
        
        return trajectories
    
    def test_gpac_computation(self):
        """Test basic GPAC computation."""
        result = self.gpac_computer.compute(self.sample_trajectories)
        
        self.assertIsInstance(result, dict)
        self.assertIn('gpac_score', result)
        
        if 'error' not in result:
            self.assertGreaterEqual(result['gpac_score'], 0.0)
            self.assertLessEqual(result['gpac_score'], 1.0)
    
    def test_gpac_insufficient_samples(self):
        """Test GPAC with insufficient samples."""
        small_trajectories = self.sample_trajectories[:3]
        result = self.gpac_computer.compute(small_trajectories)
        
        # Should handle gracefully
        self.assertIn('error', result)
    
    def test_gpac_identical_genotypes(self):
        """Test GPAC with identical genotypes."""
        identical_trajectories = []
        for i in range(12):
            trajectory = {
                'initial_genes': ['gene1', 'gene2'],
                'final_genes': ['gene1', 'gene2', 'gene3'],
                'generations': [
                    {'genes': ['gene1', 'gene2', 'gene3'], 
                     'survival_prob': 0.5 + 0.1 * i, 
                     'action': 1}
                ]
            }
            identical_trajectories.append(trajectory)
        
        result = self.gpac_computer.compute(identical_trajectories)
        # Should handle identical genotypes
        self.assertIsInstance(result, dict)


class TestAEI(unittest.TestCase):
    """Test Adaptive Efficiency Index."""
    
    def setUp(self):
        self.aei_computer = AdaptiveEfficiencyIndex()
        self.sample_trajectories = self._create_action_trajectories()
    
    def _create_action_trajectories(self):
        """Create trajectories with specific action patterns."""
        trajectories = []
        
        for i in range(8):
            trajectory = {
                'initial_genes': ['gene1'],
                'final_genes': ['gene1', 'gene2', 'gene3'],
                'generations': []
            }
            
            # Create action sequence
            for step in range(6):
                # Vary action patterns
                if i < 3:
                    action = 0  # Always mutate
                elif i < 6:
                    action = 1  # Always transfer
                else:
                    action = step % 3  # Mixed actions
                
                survival = 0.3 + 0.1 * step
                gen = {
                    'genes': [f'gene_{j}' for j in range(step + 1)],
                    'survival_prob': min(1.0, survival),
                    'action': action
                }
                trajectory['generations'].append(gen)
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def test_aei_computation(self):
        """Test basic AEI computation."""
        result = self.aei_computer.compute(self.sample_trajectories)
        
        self.assertIsInstance(result, dict)
        self.assertIn('mean_aei', result)
        self.assertIn('std_aei', result)
        
        if 'error' not in result:
            self.assertGreaterEqual(result['mean_aei'], 0.0)
            self.assertLessEqual(result['mean_aei'], 1.0)
    
    def test_aei_empty_actions(self):
        """Test AEI with empty action sequences."""
        empty_trajectories = [{
            'initial_genes': [],
            'final_genes': [],
            'generations': []
        }]
        
        result = self.aei_computer.compute(empty_trajectories)
        self.assertIn('error', result)
    
    def test_aei_single_action(self):
        """Test AEI with single action trajectories."""
        single_action_trajectories = []
        for action in [0, 1, 2]:
            trajectory = {
                'initial_genes': ['gene1'],
                'final_genes': ['gene1', 'gene2'],
                'generations': [
                    {'genes': ['gene1', 'gene2'], 'survival_prob': 0.6, 'action': action}
                ]
            }
            single_action_trajectories.append(trajectory)
        
        result = self.aei_computer.compute(single_action_trajectories)
        # Should handle single actions
        self.assertIsInstance(result, dict)


class TestBayesianUncertainty(unittest.TestCase):
    """Test Bayesian Uncertainty Quantification."""
    
    def setUp(self):
        self.uncertainty_quantifier = BayesianUncertaintyQuantifier()
        self.sample_results = self._create_sample_results()
    
    def _create_sample_results(self):
        """Create sample metric results for uncertainty testing."""
        return {
            'etci': {
                'mean_etci': 0.65,
                'std_etci': 0.12,
                'etci_distribution': np.random.beta(3, 2, 20).tolist()
            },
            'gpac': {
                'gpac_score': 0.72
            },
            'aei': {
                'mean_aei': 0.58,
                'detailed_scores': [
                    {'aei_score': 0.55 + 0.1 * np.random.normal()} for _ in range(15)
                ]
            }
        }
    
    def test_uncertainty_computation(self):
        """Test uncertainty quantification computation."""
        result = self.uncertainty_quantifier.compute(self.sample_results)
        
        self.assertIsInstance(result, dict)
        self.assertIn('etci', result)
        self.assertIn('aei', result)
        
        # Check ETCI uncertainty
        if 'etci' in result:
            etci_uncertainty = result['etci']
            self.assertIn('bayesian_ci', etci_uncertainty)
            self.assertIn('point_estimate', etci_uncertainty)
    
    def test_uncertainty_empty_data(self):
        """Test uncertainty with empty data."""
        empty_results = {}
        result = self.uncertainty_quantifier.compute(empty_results)
        
        self.assertIsInstance(result, dict)
        # Should handle empty gracefully
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        data = np.random.normal(0.6, 0.1, 100)
        ci = self.uncertainty_quantifier._bayesian_bootstrap_ci(data)
        
        self.assertIsInstance(ci, dict)
        self.assertIn('lower', ci)
        self.assertIn('upper', ci)
        self.assertLess(ci['lower'], ci['upper'])


class TestTemporalDynamics(unittest.TestCase):
    """Test Temporal Dynamics Analysis."""
    
    def setUp(self):
        self.temporal_analyzer = TemporalDynamicsAnalyzer()
        self.sample_trajectories = self._create_temporal_trajectories()
    
    def _create_temporal_trajectories(self):
        """Create trajectories with interesting temporal patterns."""
        trajectories = []
        
        for i in range(5):
            trajectory = {
                'initial_genes': ['gene1'],
                'final_genes': ['gene1', 'gene2', 'gene3'],
                'generations': []
            }
            
            # Create temporal patterns
            for step in range(10):
                if i == 0:
                    # Linear trend
                    survival = 0.3 + 0.05 * step
                elif i == 1:
                    # Oscillatory pattern
                    survival = 0.5 + 0.2 * np.sin(step * 0.5)
                elif i == 2:
                    # Exponential-like growth
                    survival = 0.3 + 0.4 * (1 - np.exp(-step * 0.3))
                else:
                    # Random walk
                    survival = 0.5 + 0.3 * np.cumsum(np.random.normal(0, 0.1, 1))[0]
                
                survival = max(0.0, min(1.0, survival))
                
                gen = {
                    'genes': [f'gene_{j}' for j in range(step // 2 + 1)],
                    'survival_prob': survival,
                    'action': step % 3
                }
                trajectory['generations'].append(gen)
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def test_temporal_analysis(self):
        """Test basic temporal analysis."""
        result = self.temporal_analyzer.compute(self.sample_trajectories)
        
        self.assertIsInstance(result, dict)
        
        if 'error' not in result:
            self.assertIn('frequency_analysis', result)
            self.assertIn('derivative_analysis', result)
            self.assertIn('overall_temporal_score', result)
    
    def test_temporal_short_trajectories(self):
        """Test temporal analysis with short trajectories."""
        short_trajectories = []
        for i in range(3):
            trajectory = {
                'initial_genes': ['gene1'],
                'final_genes': ['gene2'],
                'generations': [
                    {'genes': ['gene1'], 'survival_prob': 0.3, 'action': 0},
                    {'genes': ['gene2'], 'survival_prob': 0.4, 'action': 1}
                ]
            }
            short_trajectories.append(trajectory)
        
        result = self.temporal_analyzer.compute(short_trajectories)
        # Should handle gracefully
        self.assertIsInstance(result, dict)
    
    def test_temporal_empty(self):
        """Test temporal analysis with empty trajectories."""
        result = self.temporal_analyzer.compute([])
        self.assertIn('error', result)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_validate_trajectories(self):
        """Test trajectory validation."""
        valid_trajectory = {
            'initial_genes': ['gene1'],
            'final_genes': ['gene1', 'gene2'],
            'generations': [
                {'genes': ['gene1'], 'survival_prob': 0.3},
                {'genes': ['gene1', 'gene2'], 'survival_prob': 0.5}
            ]
        }
        
        invalid_trajectory = {
            'initial_genes': ['gene1'],
            # Missing final_genes
            'generations': []
        }
        
        valid, errors = validate_trajectories([valid_trajectory, invalid_trajectory])
        
        self.assertEqual(len(valid), 1)
        self.assertGreater(len(errors), 0)
    
    def test_safe_entropy(self):
        """Test numerically stable entropy computation."""
        # Normal probabilities
        probs1 = np.array([0.5, 0.3, 0.2])
        entropy1 = safe_entropy(probs1)
        self.assertGreater(entropy1, 0)
        
        # With zeros
        probs2 = np.array([0.7, 0.3, 0.0])
        entropy2 = safe_entropy(probs2)
        self.assertGreater(entropy2, 0)
        
        # Uniform distribution should have maximum entropy
        probs3 = np.array([1/3, 1/3, 1/3])
        entropy3 = safe_entropy(probs3)
        self.assertGreater(entropy3, entropy1)
    
    def test_robust_correlation(self):
        """Test robust correlation computation."""
        # Perfect correlation
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        corr1 = robust_correlation(x, y, 'pearson')
        self.assertAlmostEqual(corr1, 1.0, places=5)
        
        # No correlation
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 1, 1, 1, 1])
        corr2 = robust_correlation(x, y, 'pearson')
        self.assertEqual(corr2, 0.0)
        
        # With NaN values
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, np.nan, 10])
        corr3 = robust_correlation(x, y, 'pearson')
        self.assertIsInstance(corr3, float)
    
    def test_extract_features(self):
        """Test feature extraction from trajectories."""
        sample_trajectories = [{
            'initial_genes': ['gene1'],
            'final_genes': ['gene1', 'gene2', 'gene3'],
            'generations': [
                {'genes': ['gene1'], 'survival_prob': 0.3, 'action': 0},
                {'genes': ['gene1', 'gene2'], 'survival_prob': 0.5, 'action': 1},
                {'genes': ['gene1', 'gene2', 'gene3'], 'survival_prob': 0.7, 'action': 1}
            ]
        }]
        
        features_df = extract_features(sample_trajectories)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertGreater(len(features_df), 0)
        self.assertIn('length', features_df.columns)
        self.assertIn('initial_gene_count', features_df.columns)
        self.assertIn('final_gene_count', features_df.columns)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete evaluation pipeline."""
    
    def test_complete_evaluation_pipeline(self):
        """Test the complete evaluation pipeline."""
        from main_evaluator import BacterialEvolutionEvaluator
        
        # Create comprehensive test trajectories
        trajectories = self._create_comprehensive_test_trajectories()
        
        # Initialize evaluator
        evaluator = BacterialEvolutionEvaluator(confidence_level=0.95)
        
        # Run evaluation
        try:
            results = evaluator.evaluate_model(trajectories)
            
            # Check structure
            self.assertIsInstance(results, dict)
            self.assertIn('metadata', results)
            
            # Check for at least some successful metrics
            successful_metrics = sum([
                'etci' in results and 'error' not in results['etci'],
                'gpac' in results and 'error' not in results['gpac'],
                'aei' in results and 'error' not in results['aei']
            ])
            
            self.assertGreater(successful_metrics, 0, "At least one metric should succeed")
            
        except Exception as e:
            self.fail(f"Complete evaluation pipeline failed: {str(e)}")
    
    def _create_comprehensive_test_trajectories(self):
        """Create comprehensive test trajectories for integration testing."""
        trajectories = []
        
        for i in range(20):
            trajectory = {
                'initial_genes': [f'gene_{j}' for j in range(i % 3 + 1)],
                'final_genes': [f'gene_{j}' for j in range((i % 3 + 1) + (i % 5))],
                'generations': []
            }
            
            # Create realistic trajectory
            current_genes = trajectory['initial_genes'].copy()
            current_survival = 0.2 + 0.1 * np.random.random()
            
            for step in range(8 + i % 5):
                # Simulate gene acquisition
                if step > 0 and np.random.random() < 0.7:
                    new_gene = f'acquired_gene_{step}_{i}'
                    if new_gene not in current_genes:
                        current_genes.append(new_gene)
                
                # Simulate survival change
                action = np.random.choice([0, 1, 2])
                if action == 0:  # Mutate
                    current_survival += 0.02 + np.random.normal(0, 0.02)
                elif action == 1:  # Transfer
                    current_survival += 0.05 + np.random.normal(0, 0.03)
                else:  # Stable
                    current_survival += np.random.normal(0, 0.01)
                
                current_survival = max(0.0, min(1.0, current_survival))
                
                gen = {
                    'genes': current_genes.copy(),
                    'survival_prob': current_survival,
                    'action': action
                }
                trajectory['generations'].append(gen)
            
            # Update final genes
            trajectory['final_genes'] = current_genes.copy()
            
            trajectories.append(trajectory)
        
        return trajectories


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)