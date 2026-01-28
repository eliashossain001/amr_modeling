# eval/metrics/bayesian_uncertainty.py
"""
Bayesian Uncertainty Quantification

Mathematical foundation: Bayesian bootstrap and probabilistic inference
Provides statistical uncertainty estimates for all evaluation metrics using:
- Bayesian bootstrap for non-parametric confidence intervals
- Credible intervals via posterior sampling
- Uncertainty propagation through metric calculations
- Statistical significance testing
"""

import numpy as np
from scipy import stats
from scipy.special import gamma, digamma
from typing import Dict, List, Any, Tuple
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class BayesianUncertaintyQuantifier:
    """
    Provides Bayesian uncertainty quantification for evaluation metrics.
    
    Uses non-parametric Bayesian methods to estimate uncertainty without
    assuming specific distributional forms, providing robust confidence
    intervals and credible regions.
    
    Mathematical Components:
    1. Bayesian bootstrap for empirical distributions
    2. Dirichlet process for non-parametric inference
    3. Monte Carlo sampling for posterior estimates
    4. Credible interval computation via quantiles
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 n_samples: int = 1000,
                 random_seed: int = 42):
        """
        Initialize uncertainty quantifier.
        
        Args:
            confidence_level: Confidence level for intervals [0, 1]
            n_samples: Number of bootstrap/posterior samples
            random_seed: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.n_samples = n_samples
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Compute alpha for confidence intervals
        self.alpha = 1.0 - confidence_level
        
    def compute(self, metric_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute uncertainty estimates for all available metrics.
        
        Args:
            metric_results: Dictionary containing metric computation results
            
        Returns:
            Uncertainty analysis results with confidence intervals
        """
        uncertainty_results = {}
        
        # ETCI uncertainty
        if 'etci' in metric_results and 'etci_distribution' in metric_results['etci']:
            etci_data = metric_results['etci']['etci_distribution']
            uncertainty_results['etci'] = self._compute_metric_uncertainty(
                etci_data, metric_name='ETCI'
            )
        
        # GPAC uncertainty (single value - use theoretical bounds)
        if 'gpac' in metric_results and 'gpac_score' in metric_results['gpac']:
            gpac_score = metric_results['gpac']['gpac_score']
            uncertainty_results['gpac'] = self._compute_single_value_uncertainty(
                gpac_score, metric_name='GPAC'
            )
        
        # AEI uncertainty
        if 'aei' in metric_results and 'detailed_scores' in metric_results['aei']:
            aei_data = [score['aei_score'] for score in metric_results['aei']['detailed_scores']]
            uncertainty_results['aei'] = self._compute_metric_uncertainty(
                aei_data, metric_name='AEI'
            )
        
        # Overall uncertainty propagation
        if len(uncertainty_results) > 1:
            uncertainty_results['overall'] = self._propagate_uncertainty(uncertainty_results)
        
        return uncertainty_results
    
    def _compute_metric_uncertainty(self, data: List[float], metric_name: str) -> Dict[str, Any]:
        """
        Compute uncertainty for a metric with multiple observations.
        
        Args:
            data: List of metric values
            metric_name: Name of the metric for reporting
            
        Returns:
            Dictionary with uncertainty statistics
        """
        if len(data) < 2:
            return {
                'error': f'Insufficient data for uncertainty analysis: {len(data)} observations',
                'metric_name': metric_name
            }
        
        data_array = np.array(data)
        
        # 1. Bayesian bootstrap confidence intervals
        bayesian_ci = self._bayesian_bootstrap_ci(data_array)
        
        # 2. Parametric confidence intervals (for comparison)
        parametric_ci = self._parametric_ci(data_array)
        
        # 3. Posterior statistics
        posterior_stats = self._compute_posterior_statistics(data_array)
        
        # 4. Uncertainty metrics
        uncertainty_metrics = self._compute_uncertainty_metrics(data_array)
        
        return {
            'metric_name': metric_name,
            'n_observations': len(data),
            'point_estimate': np.mean(data_array),
            'bayesian_ci': bayesian_ci,
            'parametric_ci': parametric_ci,
            'posterior_stats': posterior_stats,
            'uncertainty_metrics': uncertainty_metrics,
            'confidence_level': self.confidence_level
        }
    
    def _bayesian_bootstrap_ci(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute confidence intervals using Bayesian bootstrap.
        
        The Bayesian bootstrap treats empirical distribution as posterior
        over true distribution, sampling via Dirichlet weights.
        
        Args:
            data: Array of observations
            
        Returns:
            Dictionary with confidence interval bounds
        """
        n = len(data)
        bootstrap_samples = []
        
        for _ in range(self.n_samples):
            # Sample Dirichlet weights (uniform Dirichlet)
            weights = np.random.dirichlet([1] * n)
            
            # Weighted sample mean
            bootstrap_mean = np.sum(weights * data)
            bootstrap_samples.append(bootstrap_mean)
        
        bootstrap_samples = np.array(bootstrap_samples)
        
        # Compute percentiles for confidence interval
        lower_percentile = 100 * (self.alpha / 2)
        upper_percentile = 100 * (1 - self.alpha / 2)
        
        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)
        
        return {
            'lower': ci_lower,
            'upper': ci_upper,
            'width': ci_upper - ci_lower,
            'method': 'bayesian_bootstrap'
        }
    
    def _parametric_ci(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute parametric confidence intervals assuming normality.
        
        Args:
            data: Array of observations
            
        Returns:
            Dictionary with parametric confidence interval
        """
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)  # Standard error of mean
        
        # t-distribution critical value
        t_critical = stats.t.ppf(1 - self.alpha / 2, df=n-1)
        
        # Confidence interval
        margin_error = t_critical * std_err
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        return {
            'lower': ci_lower,
            'upper': ci_upper,
            'width': ci_upper - ci_lower,
            'method': 'parametric_t',
            't_statistic': mean / std_err if std_err > 0 else np.inf,
            'degrees_freedom': n - 1
        }
    
    def _compute_posterior_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute posterior statistics for the metric.
        
        Args:
            data: Array of observations
            
        Returns:
            Dictionary with posterior statistics
        """
        # Fit Beta distribution to bounded metrics (assuming [0,1] range)
        if np.all((data >= 0) & (data <= 1)) and len(np.unique(data)) > 2:
            try:
                # Beta distribution MLE
                alpha_mle, beta_mle, _, _ = stats.beta.fit(data, floc=0, fscale=1)
                
                # Posterior predictive statistics
                posterior_mean = alpha_mle / (alpha_mle + beta_mle)
                posterior_var = (alpha_mle * beta_mle) / ((alpha_mle + beta_mle)**2 * (alpha_mle + beta_mle + 1))
                
                return {
                    'posterior_mean': posterior_mean,
                    'posterior_variance': posterior_var,
                    'posterior_std': np.sqrt(posterior_var),
                    'alpha_parameter': alpha_mle,
                    'beta_parameter': beta_mle,
                    'distribution': 'beta'
                }
                
            except Exception:
                pass
        
        # Fallback to normal approximation
        return {
            'posterior_mean': np.mean(data),
            'posterior_variance': np.var(data, ddof=1),
            'posterior_std': np.std(data, ddof=1),
            'distribution': 'normal_approximation'
        }
    
    def _compute_uncertainty_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute various uncertainty quantification metrics.
        
        Args:
            data: Array of observations
            
        Returns:
            Dictionary with uncertainty metrics
        """
        n = len(data)
        mean = np.mean(data)
        
        # Coefficient of variation
        cv = np.std(data) / np.abs(mean) if mean != 0 else np.inf
        
        # Relative standard error
        relative_se = stats.sem(data) / np.abs(mean) if mean != 0 else np.inf
        
        # Effective sample size (accounting for potential autocorrelation)
        eff_n = self._compute_effective_sample_size(data)
        
        # Monte Carlo standard error
        mc_se = np.std(data) / np.sqrt(eff_n)
        
        return {
            'coefficient_variation': cv,
            'relative_standard_error': relative_se,
            'effective_sample_size': eff_n,
            'monte_carlo_se': mc_se,
            'precision': 1.0 / (1.0 + cv) if cv != np.inf else 0.0
        }
    
    def _compute_effective_sample_size(self, data: np.ndarray) -> float:
        """
        Compute effective sample size accounting for autocorrelation.
        
        Args:
            data: Array of observations
            
        Returns:
            Effective sample size
        """
        try:
            # Compute autocorrelation at lag 1
            if len(data) > 2:
                autocorr_1 = np.corrcoef(data[:-1], data[1:])[0, 1]
                
                if not np.isnan(autocorr_1) and autocorr_1 > 0:
                    # Effective sample size formula for AR(1) process
                    eff_n = len(data) * (1 - autocorr_1) / (1 + autocorr_1)
                    return max(1, eff_n)
            
            return float(len(data))
            
        except Exception:
            return float(len(data))
    
    def _compute_single_value_uncertainty(self, value: float, metric_name: str) -> Dict[str, Any]:
        """
        Handle uncertainty for single-value metrics (like GPAC).
        
        Args:
            value: Single metric value
            metric_name: Name of the metric
            
        Returns:
            Dictionary with uncertainty bounds based on metric properties
        """
        # For bounded metrics [0,1], use Beta prior uncertainty
        if 0 <= value <= 1:
            # Assume weak prior: Beta(1,1) = Uniform[0,1]
            # With one observation, posterior is Beta(1+x, 1+(1-x))
            alpha_post = 1 + value
            beta_post = 1 + (1 - value)
            
            # Credible interval from Beta distribution
            ci_lower = stats.beta.ppf(self.alpha / 2, alpha_post, beta_post)
            ci_upper = stats.beta.ppf(1 - self.alpha / 2, alpha_post, beta_post)
            
            return {
                'metric_name': metric_name,
                'point_estimate': value,
                'credible_interval': {
                    'lower': ci_lower,
                    'upper': ci_upper,
                    'width': ci_upper - ci_lower,
                    'method': 'beta_posterior'
                },
                'posterior_mean': alpha_post / (alpha_post + beta_post),
                'note': 'Single observation - using Beta prior uncertainty'
            }
        
        # For unbounded metrics, use normal approximation with conservative width
        else:
            # Conservative confidence interval (±20% of value)
            width = 0.2 * abs(value)
            
            return {
                'metric_name': metric_name,
                'point_estimate': value,
                'approximate_interval': {
                    'lower': value - width,
                    'upper': value + width,
                    'width': 2 * width,
                    'method': 'conservative_approximation'
                },
                'note': 'Single observation - using conservative approximation'
            }
    
    def _propagate_uncertainty(self, metric_uncertainties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propagate uncertainty across multiple metrics for overall assessment.
        
        Args:
            metric_uncertainties: Dictionary of individual metric uncertainties
            
        Returns:
            Combined uncertainty analysis
        """
        # Extract point estimates and confidence intervals
        point_estimates = {}
        ci_widths = {}
        
        for metric, uncertainty in metric_uncertainties.items():
            if 'point_estimate' in uncertainty:
                point_estimates[metric] = uncertainty['point_estimate']
                
                # Extract confidence interval width
                if 'bayesian_ci' in uncertainty:
                    ci_widths[metric] = uncertainty['bayesian_ci']['width']
                elif 'credible_interval' in uncertainty:
                    ci_widths[metric] = uncertainty['credible_interval']['width']
        
        if not point_estimates:
            return {'error': 'No point estimates available for uncertainty propagation'}
        
        # Simple uncertainty propagation (assuming independence)
        # For weighted average: Var(∑wᵢXᵢ) = ∑wᵢ²Var(Xᵢ)
        weights = {'etci': 0.25, 'gpac': 0.25, 'aei': 0.30, 'temporal': 0.20}
        
        weighted_mean = 0
        propagated_variance = 0
        
        for metric, estimate in point_estimates.items():
            if metric in weights:
                weight = weights[metric]
                weighted_mean += weight * estimate
                
                # Approximate variance from confidence interval width
                if metric in ci_widths:
                    # CI width ≈ 2 * 1.96 * std for normal distribution
                    approx_std = ci_widths[metric] / (2 * 1.96)
                    propagated_variance += (weight ** 2) * (approx_std ** 2)
        
        propagated_std = np.sqrt(propagated_variance)
        
        # Overall confidence interval
        overall_ci_lower = weighted_mean - 1.96 * propagated_std
        overall_ci_upper = weighted_mean + 1.96 * propagated_std
        
        return {
            'overall_estimate': weighted_mean,
            'overall_uncertainty': propagated_std,
            'overall_ci': {
                'lower': overall_ci_lower,
                'upper': overall_ci_upper,
                'width': overall_ci_upper - overall_ci_lower,
                'method': 'uncertainty_propagation'
            },
            'component_weights': weights,
            'assumptions': 'Independence between metrics assumed for propagation'
        }