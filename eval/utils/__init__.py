# eval/utils/__init__.py
"""
Utility functions for bacterial evolution RL evaluation framework.
"""

from .math_utils import (
    validate_trajectories, 
    compute_confidence_intervals, 
    normalize_matrix, 
    safe_entropy,
    robust_correlation
)

from .data_utils import (
    preprocess_trajectories, 
    extract_features, 
    filter_outlier_trajectories,  # ← Fixed function name
    standardize_data_types
)

from .visualization import (
    create_evaluation_plots, 
    plot_metric_distribution, 
    plot_uncertainty_bands
)

__all__ = [
    'validate_trajectories',
    'compute_confidence_intervals', 
    'normalize_matrix',
    'safe_entropy',
    'robust_correlation',
    'preprocess_trajectories',
    'extract_features', 
    'filter_outlier_trajectories',  # ← Fixed function name
    'standardize_data_types',
    'create_evaluation_plots',
    'plot_metric_distribution',
    'plot_uncertainty_bands'
]