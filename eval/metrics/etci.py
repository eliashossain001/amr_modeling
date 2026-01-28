# eval/metrics/etci.py
"""
Evolutionary Trajectory Coherence Index (ETCI)

Mathematical foundation: Dynamical systems theory and information geometry
Measures coherence of evolutionary trajectories using:
- State space embedding (Takens' theorem)
- Lyapunov exponent estimation  
- Information-theoretic coherence
- Trajectory smoothness analysis
"""

import numpy as np
from scipy import linalg
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any


class EvolutionaryTrajectoryCoherenceIndex:
    """
    Computes ETCI using rigorous dynamical systems mathematics.
    
    The ETCI quantifies how coherent and predictable evolutionary trajectories are
    by analyzing them as dynamical systems in state space.
    
    Mathematical Components:
    1. Phase space reconstruction via delay embedding
    2. Lyapunov exponent for stability analysis  
    3. Mutual information for predictability
    4. Total variation for smoothness
    """
    
    def __init__(self, 
                 embedding_dimension: int = 3,
                 time_delay: int = 1,
                 min_trajectory_length: int = 5):
        """
        Initialize ETCI computer.
        
        Args:
            embedding_dimension: Dimension for phase space reconstruction
            time_delay: Time delay for embedding (Takens' theorem)
            min_trajectory_length: Minimum trajectory length for analysis
        """
        self.embedding_dim = embedding_dimension
        self.time_delay = time_delay
        self.min_length = min_trajectory_length
        
    def compute(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """
        Compute ETCI for all trajectories.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            ETCI results including individual components and statistics
        """
        etci_scores = []
        detailed_results = []
        
        for i, trajectory in enumerate(trajectories):
            try:
                result = self._compute_single_trajectory_etci(trajectory)
                if result is not None:
                    etci_scores.append(result['etci_score'])
                    detailed_results.append(result)
            except Exception as e:
                print(f"⚠ ETCI computation failed for trajectory {i}: {str(e)}")
                continue
        
        if not etci_scores:
            return {'error': 'No valid ETCI scores computed'}
        
        return {
            'mean_etci': np.mean(etci_scores),
            'std_etci': np.std(etci_scores),
            'median_etci': np.median(etci_scores),
            'min_etci': np.min(etci_scores),
            'max_etci': np.max(etci_scores),
            'etci_distribution': etci_scores,
            'detailed_results': detailed_results,
            'n_valid_trajectories': len(etci_scores)
        }
    
    def _compute_single_trajectory_etci(self, trajectory: Dict) -> Dict[str, float]:
        """
        Compute ETCI for a single trajectory.
        
        Args:
            trajectory: Single trajectory dictionary
            
        Returns:
            Dictionary with ETCI score and components
        """
        # Extract gene count time series
        gene_counts = []
        survival_probs = []
        
        for generation in trajectory['generations']:
            gene_counts.append(len(generation.get('genes', [])))
            survival_probs.append(generation.get('survival_prob', 0.0))
        
        if len(gene_counts) < self.min_length:
            return None
        
        # 1. Phase space reconstruction
        embedded_trajectory = self._embed_trajectory(gene_counts)
        
        # 2. Compute Lyapunov exponent
        lyapunov_exp = self._compute_lyapunov_exponent(embedded_trajectory)
        
        # 3. Information-theoretic coherence
        mi_coherence = self._compute_mutual_information_coherence(gene_counts, survival_probs)
        
        # 4. Trajectory smoothness  
        smoothness = self._compute_trajectory_smoothness(gene_counts)
        
        # 5. Combine into ETCI score
        # Stability (sigmoid of negative Lyapunov), high MI, high smoothness
        stability_score = 1.0 / (1.0 + np.exp(lyapunov_exp))  # Higher = more stable
        
        etci_score = 0.4 * stability_score + 0.3 * mi_coherence + 0.3 * smoothness
        
        return {
            'etci_score': etci_score,
            'stability_component': stability_score,
            'coherence_component': mi_coherence,
            'smoothness_component': smoothness,
            'lyapunov_exponent': lyapunov_exp,
            'trajectory_length': len(gene_counts)
        }
    
    def _embed_trajectory(self, time_series: List[float]) -> np.ndarray:
        """
        Phase space reconstruction using Takens' delay embedding.
        
        Takens' Theorem: A d-dimensional dynamical system can be reconstructed
        from scalar time series using delay coordinates.
        
        Args:
            time_series: 1D time series data
            
        Returns:
            Embedded trajectory in higher-dimensional space
        """
        series = np.array(time_series, dtype=float)
        n = len(series)
        
        # Number of embedded points
        m = n - (self.embedding_dim - 1) * self.time_delay
        
        if m <= 0:
            raise ValueError("Time series too short for embedding")
        
        # Create delay coordinate matrix
        embedded = np.zeros((m, self.embedding_dim))
        
        for i in range(m):
            for j in range(self.embedding_dim):
                embedded[i, j] = series[i + j * self.time_delay]
        
        return embedded
    
    def _compute_lyapunov_exponent(self, embedded_trajectory: np.ndarray) -> float:
        """
        Estimate largest Lyapunov exponent using Wolf et al. algorithm.
        
        The Lyapunov exponent λ measures sensitive dependence on initial conditions:
        - λ > 0: Chaotic behavior
        - λ = 0: Marginal stability  
        - λ < 0: Stable behavior
        
        Args:
            embedded_trajectory: Phase space points
            
        Returns:
            Estimated largest Lyapunov exponent
        """
        if len(embedded_trajectory) < 10:
            return 0.0
        
        # Use nearest neighbors for divergence estimation
        nbrs = NearestNeighbors(n_neighbors=2).fit(embedded_trajectory)
        
        log_divergences = []
        
        for i in range(len(embedded_trajectory) - 1):
            current_point = embedded_trajectory[i:i+1]
            next_point = embedded_trajectory[i+1:i+2]
            
            # Find nearest neighbor
            distances, indices = nbrs.kneighbors(current_point)
            
            if len(indices[0]) > 1:
                neighbor_idx = indices[0][1]  # Second closest (first is self)
                
                if neighbor_idx < len(embedded_trajectory) - 1:
                    neighbor_current = embedded_trajectory[neighbor_idx:neighbor_idx+1]
                    neighbor_next = embedded_trajectory[neighbor_idx+1:neighbor_idx+2]
                    
                    # Compute separation distances
                    d0 = linalg.norm(current_point - neighbor_current)
                    d1 = linalg.norm(next_point - neighbor_next)
                    
                    # Avoid division by zero or log of zero
                    if d0 > 1e-8 and d1 > 1e-8:
                        log_divergences.append(np.log(d1 / d0))
        
        # Estimate Lyapunov exponent
        if log_divergences:
            lyapunov_est = np.mean(log_divergences)
        else:
            lyapunov_est = 0.0
        
        return lyapunov_est
    
    def _compute_mutual_information_coherence(self, 
                                            gene_counts: List[float], 
                                            survival_probs: List[float]) -> float:
        """
        Compute mutual information between gene states and survival outcomes.
        
        MI measures how much information one variable contains about another.
        High MI indicates coherent gene-survival relationship.
        
        Args:
            gene_counts: Gene count time series
            survival_probs: Survival probability time series
            
        Returns:
            Normalized mutual information [0, 1]
        """
        if len(gene_counts) != len(survival_probs) or len(gene_counts) < 3:
            return 0.0
        
        # Discretize for MI computation
        n_bins = min(5, len(gene_counts) // 2)
        
        try:
            gene_bins = np.digitize(gene_counts, 
                                  bins=np.linspace(min(gene_counts), max(gene_counts)+1e-8, n_bins))
            survival_bins = np.digitize(survival_probs,
                                      bins=np.linspace(0, 1, n_bins))
            
            # Compute mutual information
            mi = mutual_info_score(gene_bins, survival_bins)
            
            # Normalize by joint entropy (0 ≤ normalized MI ≤ 1)
            joint_entropy = entropy(np.histogram2d(gene_bins, survival_bins)[0].ravel())
            
            if joint_entropy > 0:
                normalized_mi = mi / joint_entropy
            else:
                normalized_mi = 0.0
                
            return min(1.0, normalized_mi)
            
        except Exception:
            return 0.0
    
    def _compute_trajectory_smoothness(self, gene_counts: List[float]) -> float:
        """
        Compute trajectory smoothness using total variation.
        
        Total Variation measures how much a function oscillates.
        Lower TV indicates smoother, more biologically plausible evolution.
        
        Args:
            gene_counts: Gene count time series
            
        Returns:
            Smoothness score [0, 1], where 1 = perfectly smooth
        """
        if len(gene_counts) < 2:
            return 1.0
        
        counts = np.array(gene_counts, dtype=float)
        
        # Compute total variation
        differences = np.diff(counts)
        tv_norm = np.sum(np.abs(differences))
        
        # Normalize by trajectory length and scale
        trajectory_scale = np.std(counts) if np.std(counts) > 0 else 1.0
        normalized_tv = tv_norm / (len(counts) * trajectory_scale)
        
        # Convert to smoothness score (higher = smoother)
        smoothness = 1.0 / (1.0 + normalized_tv)
        
        return smoothness