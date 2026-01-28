# eval/utils/math_utils.py
"""
Mathematical utility functions for evaluation metrics.

Provides robust mathematical operations, statistical functions,
and numerical utilities used across the evaluation framework.
"""

import numpy as np
from scipy import stats, linalg
from scipy.special import gamma, digamma, logsumexp
from typing import List, Dict, Any, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def validate_trajectories(trajectories: List[Dict], 
                         min_length: int = 3,
                         required_fields: List[str] = None) -> Tuple[List[Dict], List[str]]:
    """
    Validate trajectory data structure and content.
    
    Args:
        trajectories: List of trajectory dictionaries
        min_length: Minimum trajectory length required
        required_fields: List of required fields in each trajectory
        
    Returns:
        Tuple of (valid_trajectories, validation_errors)
    """
    if required_fields is None:
        required_fields = ['initial_genes', 'final_genes', 'generations']
    
    valid_trajectories = []
    validation_errors = []
    
    for i, trajectory in enumerate(trajectories):
        errors = []
        
        # Check basic structure
        if not isinstance(trajectory, dict):
            errors.append(f"Trajectory {i}: Not a dictionary")
            continue
        
        # Check required fields
        for field in required_fields:
            if field not in trajectory:
                errors.append(f"Trajectory {i}: Missing field '{field}'")
        
        # Check generations structure
        if 'generations' in trajectory:
            generations = trajectory['generations']
            
            if not isinstance(generations, list):
                errors.append(f"Trajectory {i}: 'generations' is not a list")
            elif len(generations) < min_length:
                errors.append(f"Trajectory {i}: Too short ({len(generations)} < {min_length})")
            else:
                # Check generation structure
                for j, gen in enumerate(generations):
                    if not isinstance(gen, dict):
                        errors.append(f"Trajectory {i}, generation {j}: Not a dictionary")
                        break
                    
                    # Check required generation fields
                    gen_fields = ['survival_prob', 'genes']
                    for field in gen_fields:
                        if field not in gen:
                            errors.append(f"Trajectory {i}, generation {j}: Missing '{field}'")
                    
                    # Validate survival probability
                    if 'survival_prob' in gen:
                        surv_prob = gen['survival_prob']
                        if not isinstance(surv_prob, (int, float)) or not (0 <= surv_prob <= 1):
                            errors.append(f"Trajectory {i}, generation {j}: Invalid survival_prob")
        
        if errors:
            validation_errors.extend(errors)
        else:
            valid_trajectories.append(trajectory)
    
    return valid_trajectories, validation_errors


def compute_confidence_intervals(data: np.ndarray, 
                               confidence_level: float = 0.95,
                               method: str = 'bootstrap') -> Dict[str, float]:
    """
    Compute confidence intervals using various methods.
    
    Args:
        data: Array of observations
        confidence_level: Confidence level [0, 1]
        method: Method to use ('bootstrap', 'parametric', 'percentile')
        
    Returns:
        Dictionary with confidence interval bounds
    """
    if len(data) == 0:
        return {'lower': 0.0, 'upper': 0.0, 'width': 0.0, 'method': method}
    
    alpha = 1.0 - confidence_level
    
    if method == 'bootstrap':
        return _bootstrap_ci(data, alpha)
    elif method == 'parametric':
        return _parametric_ci(data, alpha)
    elif method == 'percentile':
        return _percentile_ci(data, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")


def _bootstrap_ci(data: np.ndarray, alpha: float, n_bootstrap: int = 1000) -> Dict[str, float]:
    """Bootstrap confidence interval."""
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return {
        'lower': lower,
        'upper': upper, 
        'width': upper - lower,
        'method': 'bootstrap'
    }


def _parametric_ci(data: np.ndarray, alpha: float) -> Dict[str, float]:
    """Parametric confidence interval assuming normality."""
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of mean
    
    # t-distribution critical value
    df = len(data) - 1
    t_critical = stats.t.ppf(1 - alpha / 2, df)
    
    margin = t_critical * sem
    lower = mean - margin
    upper = mean + margin
    
    return {
        'lower': lower,
        'upper': upper,
        'width': upper - lower,
        'method': 'parametric'
    }


def _percentile_ci(data: np.ndarray, alpha: float) -> Dict[str, float]:
    """Percentile-based confidence interval."""
    lower = np.percentile(data, 100 * alpha / 2)
    upper = np.percentile(data, 100 * (1 - alpha / 2))
    
    return {
        'lower': lower,
        'upper': upper,
        'width': upper - lower,
        'method': 'percentile'
    }


def normalize_matrix(matrix: np.ndarray, 
                    method: str = 'standardize',
                    axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize matrix using various methods.
    
    Args:
        matrix: Input matrix
        method: Normalization method ('standardize', 'minmax', 'robust', 'unit_norm')
        axis: Axis along which to normalize (None for entire matrix)
        
    Returns:
        Normalized matrix
    """
    matrix = np.asarray(matrix, dtype=float)
    
    if method == 'standardize':
        mean = np.mean(matrix, axis=axis, keepdims=True)
        std = np.std(matrix, axis=axis, keepdims=True)
        return np.where(std > 1e-8, (matrix - mean) / std, 0.0)
    
    elif method == 'minmax':
        min_val = np.min(matrix, axis=axis, keepdims=True)
        max_val = np.max(matrix, axis=axis, keepdims=True)
        range_val = max_val - min_val
        return np.where(range_val > 1e-8, (matrix - min_val) / range_val, 0.0)
    
    elif method == 'robust':
        median = np.median(matrix, axis=axis, keepdims=True)
        mad = np.median(np.abs(matrix - median), axis=axis, keepdims=True)
        return np.where(mad > 1e-8, (matrix - median) / (1.4826 * mad), 0.0)
    
    elif method == 'unit_norm':
        norm = np.linalg.norm(matrix, axis=axis, keepdims=True)
        return np.where(norm > 1e-8, matrix / norm, 0.0)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def safe_entropy(probabilities: np.ndarray, base: float = np.e) -> float:
    """
    Compute entropy with numerical stability.
    
    Args:
        probabilities: Array of probabilities
        base: Logarithm base (e for natural log, 2 for bits)
        
    Returns:
        Entropy value
    """
    # Ensure probabilities are valid
    probs = np.asarray(probabilities, dtype=float)
    probs = np.maximum(probs, 1e-16)  # Avoid log(0)
    
    # Normalize if needed
    if not np.isclose(np.sum(probs), 1.0):
        probs = probs / np.sum(probs)
    
    # Compute entropy
    if base == np.e:
        return -np.sum(probs * np.log(probs))
    else:
        return -np.sum(probs * np.log(probs)) / np.log(base)


def robust_correlation(x: np.ndarray, y: np.ndarray, method: str = 'spearman') -> float:
    """
    Compute robust correlation coefficient.
    
    Args:
        x, y: Arrays to correlate
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation coefficient
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) < 2:
        return 0.0
    
    try:
        if method == 'pearson':
            corr, _ = stats.pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr, _ = stats.spearmanr(x_clean, y_clean)
        elif method == 'kendall':
            corr, _ = stats.kendalltau(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return corr if not np.isnan(corr) else 0.0
        
    except Exception:
        return 0.0


def matrix_condition_number(matrix: np.ndarray, rcond: float = 1e-12) -> float:
    """
    Compute condition number of matrix with numerical stability.
    
    Args:
        matrix: Input matrix
        rcond: Relative condition number threshold
        
    Returns:
        Condition number (or np.inf if singular)
    """
    try:
        # Compute SVD
        U, s, Vt = linalg.svd(matrix)
        
        # Condition number = largest singular value / smallest singular value
        s_max = np.max(s)
        s_min = np.max(s[s > rcond * s_max]) if np.any(s > rcond * s_max) else 0.0
        
        if s_min > 0:
            return s_max / s_min
        else:
            return np.inf
            
    except Exception:
        return np.inf


def weighted_quantile(values: np.ndarray, 
                     weights: np.ndarray, 
                     quantile: float) -> float:
    """
    Compute weighted quantile.
    
    Args:
        values: Array of values
        weights: Array of weights
        quantile: Quantile to compute [0, 1]
        
    Returns:
        Weighted quantile value
    """
    # Sort values and corresponding weights
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Compute cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    
    # Find quantile position
    quantile_weight = quantile * total_weight
    
    # Find insertion point
    idx = np.searchsorted(cum_weights, quantile_weight)
    
    if idx == 0:
        return sorted_values[0]
    elif idx >= len(sorted_values):
        return sorted_values[-1]
    else:
        # Linear interpolation
        w1 = cum_weights[idx - 1]
        w2 = cum_weights[idx]
        v1 = sorted_values[idx - 1]
        v2 = sorted_values[idx]
        
        alpha = (quantile_weight - w1) / (w2 - w1)
        return v1 + alpha * (v2 - v1)


def numerical_gradient(func, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
    """
    Compute numerical gradient using central differences.
    
    Args:
        func: Function to differentiate
        x: Point at which to compute gradient
        h: Step size
        
    Returns:
        Gradient vector
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        try:
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        except Exception:
            grad[i] = 0.0
    
    return grad


def safe_log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Numerically stable log-sum-exp computation.
    
    Args:
        x: Array of values
        axis: Axis along which to compute
        
    Returns:
        Log-sum-exp value(s)
    """
    return logsumexp(x, axis=axis)


def multivariate_normal_pdf(x: np.ndarray, 
                           mean: np.ndarray, 
                           cov: np.ndarray) -> float:
    """
    Compute multivariate normal PDF with numerical stability.
    
    Args:
        x: Point to evaluate
        mean: Mean vector
        cov: Covariance matrix
        
    Returns:
        PDF value
    """
    try:
        x = np.asarray(x)
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        
        k = len(mean)
        diff = x - mean
        
        # Compute log determinant and inverse via Cholesky decomposition
        L = linalg.cholesky(cov, lower=True)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        # Solve L * y = diff for y, then compute y^T * y
        y = linalg.solve_triangular(L, diff, lower=True)
        mahalanobis_sq = np.dot(y, y)
        
        # Log PDF
        log_pdf = -0.5 * (k * np.log(2 * np.pi) + log_det + mahalanobis_sq)
        
        return np.exp(log_pdf)
        
    except Exception:
        return 0.0


def matrix_rank_deficient_svd(matrix: np.ndarray, 
                             threshold: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD decomposition handling rank-deficient matrices.
    
    Args:
        matrix: Input matrix
        threshold: Threshold for determining rank
        
    Returns:
        Tuple of (U, s, Vt) where s contains only significant singular values
    """
    U, s, Vt = linalg.svd(matrix, full_matrices=False)
    
    # Filter out small singular values
    significant_mask = s > threshold * s[0] if len(s) > 0 else np.array([])
    
    if np.any(significant_mask):
        U_filtered = U[:, significant_mask]
        s_filtered = s[significant_mask]
        Vt_filtered = Vt[significant_mask, :]
        
        return U_filtered, s_filtered, Vt_filtered
    else:
        # Return minimal rank-1 approximation
        return U[:, :1], s[:1], Vt[:1, :]