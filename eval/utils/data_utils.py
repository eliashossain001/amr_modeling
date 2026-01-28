# eval/utils/data_utils.py
"""
Data processing utilities for trajectory analysis.

Provides functions for preprocessing trajectories, extracting features,
filtering outliers, and preparing data for metric computation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Tuple, Optional, Set
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def preprocess_trajectories(trajectories: List[Dict], 
                          remove_outliers: bool = True,
                          normalize_features: bool = True) -> List[Dict]:
    """
    Preprocess trajectories for analysis.
    
    Args:
        trajectories: List of trajectory dictionaries
        remove_outliers: Whether to remove outlier trajectories
        normalize_features: Whether to normalize trajectory features
        
    Returns:
        Preprocessed trajectories
    """
    if not trajectories:
        return []
    
    processed = trajectories.copy()
    
    # 1. Remove outliers if requested
    if remove_outliers:
        processed = filter_outlier_trajectories(processed)
    
    # 2. Normalize features if requested
    if normalize_features:
        processed = normalize_trajectory_features(processed)
    
    # 3. Ensure consistent data types
    processed = standardize_data_types(processed)
    
    return processed


def extract_features(trajectories: List[Dict]) -> pd.DataFrame:
    """
    Extract numerical features from trajectories for analysis.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        DataFrame with extracted features
    """
    features = []
    
    for i, trajectory in enumerate(trajectories):
        try:
            # Basic trajectory features
            feature_dict = {
                'trajectory_id': i,
                'length': len(trajectory.get('generations', [])),
                'initial_gene_count': len(trajectory.get('initial_genes', [])),
                'final_gene_count': len(trajectory.get('final_genes', [])),
                'gene_gain': len(trajectory.get('final_genes', [])) - len(trajectory.get('initial_genes', []))
            }
            
            # Extract temporal features
            generations = trajectory.get('generations', [])
            if generations:
                survival_probs = [gen.get('survival_prob', 0.0) for gen in generations]
                gene_counts = [len(gen.get('genes', [])) for gen in generations]
                actions = [gen.get('action', 2) for gen in generations]
                
                # Survival features
                feature_dict.update({
                    'initial_survival': survival_probs[0] if survival_probs else 0.0,
                    'final_survival': survival_probs[-1] if survival_probs else 0.0,
                    'survival_gain': survival_probs[-1] - survival_probs[0] if len(survival_probs) > 1 else 0.0,
                    'max_survival': max(survival_probs) if survival_probs else 0.0,
                    'min_survival': min(survival_probs) if survival_probs else 0.0,
                    'survival_variance': np.var(survival_probs) if len(survival_probs) > 1 else 0.0
                })
                
                # Gene evolution features
                feature_dict.update({
                    'max_genes': max(gene_counts) if gene_counts else 0,
                    'gene_variance': np.var(gene_counts) if len(gene_counts) > 1 else 0.0,
                    'gene_trend': _compute_trend(gene_counts),
                    'survival_trend': _compute_trend(survival_probs)
                })
                
                # Action pattern features
                action_counts = np.bincount(actions, minlength=3)
                total_actions = len(actions)
                
                feature_dict.update({
                    'action_0_freq': action_counts[0] / total_actions if total_actions > 0 else 0.0,
                    'action_1_freq': action_counts[1] / total_actions if total_actions > 0 else 0.0,
                    'action_2_freq': action_counts[2] / total_actions if total_actions > 0 else 0.0,
                    'action_entropy': _compute_action_entropy(actions),
                    'action_switches': _count_action_switches(actions)
                })
            
            features.append(feature_dict)
            
        except Exception as e:
            # Create minimal feature dict for failed trajectories
            features.append({
                'trajectory_id': i,
                'length': 0,
                'initial_gene_count': 0,
                'final_gene_count': 0,
                'gene_gain': 0,
                'error': str(e)
            })
    
    return pd.DataFrame(features)


def _compute_trend(values: List[float]) -> float:
    """Compute linear trend (slope) of time series."""
    if len(values) < 2:
        return 0.0
    
    x = np.arange(len(values))
    try:
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope if not np.isnan(slope) else 0.0
    except Exception:
        return 0.0


def _compute_action_entropy(actions: List[int]) -> float:
    """Compute entropy of action distribution."""
    if not actions:
        return 0.0
    
    # Count action frequencies
    action_counts = np.bincount(actions, minlength=3)
    total_actions = len(actions)
    
    # Compute probabilities
    probs = action_counts / total_actions
    
    # Compute entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def _count_action_switches(actions: List[int]) -> int:
    """Count number of action switches in sequence."""
    if len(actions) < 2:
        return 0
    
    switches = 0
    for i in range(1, len(actions)):
        if actions[i] != actions[i-1]:
            switches += 1
    
    return switches


def filter_outlier_trajectories(trajectories: List[Dict], 
                               method: str = 'iqr',
                               contamination: float = 0.1) -> List[Dict]:
    """
    Filter out outlier trajectories based on extracted features.
    
    Args:
        trajectories: List of trajectory dictionaries
        method: Outlier detection method ('iqr', 'zscore', 'isolation', 'dbscan')
        contamination: Expected fraction of outliers
        
    Returns:
        Filtered trajectories without outliers
    """
    if len(trajectories) < 10:  # Too few trajectories to filter
        return trajectories
    
    # Extract features for outlier detection
    features_df = extract_features(trajectories)
    
    # Select numerical features for outlier detection
    numerical_cols = features_df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col not in ['trajectory_id']]
    
    if len(numerical_cols) == 0:
        return trajectories
    
    X = features_df[numerical_cols].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Detect outliers
    if method == 'iqr':
        outlier_mask = _detect_outliers_iqr(X)
    elif method == 'zscore':
        outlier_mask = _detect_outliers_zscore(X)
    elif method == 'isolation':
        outlier_mask = _detect_outliers_isolation(X, contamination)
    elif method == 'dbscan':
        outlier_mask = _detect_outliers_dbscan(X)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Filter trajectories
    filtered_trajectories = [traj for i, traj in enumerate(trajectories) if not outlier_mask[i]]
    
    print(f"Outlier filtering: {len(trajectories)} → {len(filtered_trajectories)} trajectories")
    
    return filtered_trajectories


def _detect_outliers_iqr(X: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Detect outliers using IQR method."""
    outlier_mask = np.zeros(X.shape[0], dtype=bool)
    
    for col in range(X.shape[1]):
        data = X[:, col]
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        col_outliers = (data < lower_bound) | (data > upper_bound)
        outlier_mask |= col_outliers
    
    return outlier_mask


def _detect_outliers_zscore(X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using Z-score method."""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute Z-scores
    z_scores = np.abs(X_scaled)
    
    # Mark outliers (any feature with |z-score| > threshold)
    outlier_mask = np.any(z_scores > threshold, axis=1)
    
    return outlier_mask


def _detect_outliers_isolation(X: np.ndarray, contamination: float) -> np.ndarray:
    """Detect outliers using Isolation Forest."""
    try:
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        
        # Convert labels (-1 for outliers, 1 for inliers) to boolean mask
        outlier_mask = outlier_labels == -1
        
        return outlier_mask
        
    except ImportError:
        print("Warning: sklearn not available for Isolation Forest, using IQR method")
        return _detect_outliers_iqr(X)


def _detect_outliers_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """Detect outliers using DBSCAN clustering."""
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X_scaled)
        
        # Points labeled as -1 are outliers
        outlier_mask = cluster_labels == -1
        
        return outlier_mask
        
    except Exception:
        print("Warning: DBSCAN failed, using IQR method")
        return _detect_outliers_iqr(X)


def normalize_trajectory_features(trajectories: List[Dict]) -> List[Dict]:
    """
    Normalize numerical features within trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        Trajectories with normalized features
    """
    normalized_trajectories = []
    
    for trajectory in trajectories:
        normalized_traj = trajectory.copy()
        
        # Normalize generation-level features
        generations = trajectory.get('generations', [])
        if generations:
            # Extract survival probabilities for normalization
            survival_probs = [gen.get('survival_prob', 0.0) for gen in generations]
            
            # Robust normalization for survival probabilities
            if len(survival_probs) > 1:
                scaler = RobustScaler()
                survival_normalized = scaler.fit_transform(np.array(survival_probs).reshape(-1, 1)).flatten()
                
                # Update generations with normalized values
                normalized_generations = []
                for i, gen in enumerate(generations):
                    normalized_gen = gen.copy()
                    normalized_gen['survival_prob_normalized'] = survival_normalized[i]
                    normalized_generations.append(normalized_gen)
                
                normalized_traj['generations'] = normalized_generations
        
        normalized_trajectories.append(normalized_traj)
    
    return normalized_trajectories


def standardize_data_types(trajectories: List[Dict]) -> List[Dict]:
    """
    Ensure consistent data types across trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        Trajectories with standardized data types
    """
    standardized_trajectories = []
    
    for trajectory in trajectories:
        standardized_traj = trajectory.copy()
        
        # Ensure gene lists are lists of strings
        for gene_field in ['initial_genes', 'final_genes']:
            if gene_field in standardized_traj:
                genes = standardized_traj[gene_field]
                if genes is None:
                    standardized_traj[gene_field] = []
                else:
                    standardized_traj[gene_field] = [str(gene) for gene in genes]
        
        # Standardize generation data
        generations = trajectory.get('generations', [])
        standardized_generations = []
        
        for gen in generations:
            standardized_gen = gen.copy()
            
            # Ensure survival_prob is float
            if 'survival_prob' in standardized_gen:
                try:
                    standardized_gen['survival_prob'] = float(standardized_gen['survival_prob'])
                except (ValueError, TypeError):
                    standardized_gen['survival_prob'] = 0.0
            
            # Ensure action is integer
            if 'action' in standardized_gen:
                try:
                    standardized_gen['action'] = int(standardized_gen['action'])
                except (ValueError, TypeError):
                    standardized_gen['action'] = 2  # Default to stable
            
            # Ensure genes is list of strings
            if 'genes' in standardized_gen:
                genes = standardized_gen['genes']
                if genes is None:
                    standardized_gen['genes'] = []
                else:
                    standardized_gen['genes'] = [str(gene) for gene in genes]
            
            standardized_generations.append(standardized_gen)
        
        standardized_traj['generations'] = standardized_generations
        standardized_trajectories.append(standardized_traj)
    
    return standardized_trajectories


def compute_trajectory_similarities(trajectories: List[Dict], 
                                  method: str = 'dtw') -> np.ndarray:
    """
    Compute pairwise similarities between trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        method: Similarity method ('dtw', 'euclidean', 'cosine')
        
    Returns:
        Similarity matrix
    """
    n = len(trajectories)
    similarity_matrix = np.zeros((n, n))
    
    # Extract trajectory features
    features = []
    for trajectory in trajectories:
        generations = trajectory.get('generations', [])
        survival_probs = [gen.get('survival_prob', 0.0) for gen in generations]
        features.append(np.array(survival_probs))
    
    # Compute similarities
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                if method == 'dtw':
                    sim = _compute_dtw_similarity(features[i], features[j])
                elif method == 'euclidean':
                    sim = _compute_euclidean_similarity(features[i], features[j])
                elif method == 'cosine':
                    sim = _compute_cosine_similarity(features[i], features[j])
                else:
                    sim = 0.0
                
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric
    
    return similarity_matrix


def _compute_dtw_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute Dynamic Time Warping similarity."""
    try:
        # Simple DTW implementation
        n, m = len(seq1), len(seq2)
        
        if n == 0 or m == 0:
            return 0.0
        
        # DTW distance matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0.0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        # Convert distance to similarity
        dtw_distance = dtw_matrix[n, m]
        similarity = 1.0 / (1.0 + dtw_distance)
        
        return similarity
        
    except Exception:
        return 0.0


def _compute_euclidean_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute Euclidean similarity."""
    try:
        # Pad shorter sequence
        max_len = max(len(seq1), len(seq2))
        seq1_padded = np.pad(seq1, (0, max_len - len(seq1)), 'constant', constant_values=seq1[-1] if len(seq1) > 0 else 0)
        seq2_padded = np.pad(seq2, (0, max_len - len(seq2)), 'constant', constant_values=seq2[-1] if len(seq2) > 0 else 0)
        
        # Euclidean distance
        distance = np.linalg.norm(seq1_padded - seq2_padded)
        
        # Convert to similarity
        similarity = 1.0 / (1.0 + distance)
        
        return similarity
        
    except Exception:
        return 0.0


def _compute_cosine_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute cosine similarity."""
    try:
        # Pad shorter sequence
        max_len = max(len(seq1), len(seq2))
        seq1_padded = np.pad(seq1, (0, max_len - len(seq1)), 'constant', constant_values=seq1[-1] if len(seq1) > 0 else 0)
        seq2_padded = np.pad(seq2, (0, max_len - len(seq2)), 'constant', constant_values=seq2[-1] if len(seq2) > 0 else 0)
        
        # Cosine similarity
        dot_product = np.dot(seq1_padded, seq2_padded)
        norms = np.linalg.norm(seq1_padded) * np.linalg.norm(seq2_padded)
        
        if norms > 0:
            similarity = dot_product / norms
        else:
            similarity = 1.0 if len(seq1) == len(seq2) == 0 else 0.0
        
        return max(0.0, similarity)  # Ensure non-negative
        
    except Exception:
        return 0.0