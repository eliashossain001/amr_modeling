# eval/metrics/gpac.py
"""
Genotypic-Phenotypic Alignment Coefficient (GPAC)

Mathematical foundation: Manifold learning and graph theory
Measures alignment between genotypic changes and phenotypic outcomes using:
- Manifold embedding of genotype space
- Phenotype landscape modeling
- Procrustes analysis for spatial alignment
- Graph structural similarity analysis
"""

import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
import networkx as nx
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class GenotypicPhenotypicAlignmentCoefficient:
    """
    Computes GPAC using manifold learning and geometric analysis.
    
    The GPAC quantifies how well genotypic changes correspond to phenotypic
    outcomes by analyzing the geometric relationship between genotype and
    phenotype spaces.
    
    Mathematical Components:
    1. Genotype manifold embedding via Isomap
    2. Phenotype landscape via t-SNE  
    3. Procrustes analysis for spatial alignment
    4. Mantel test for distance correlation
    """
    
    def __init__(self, 
                 manifold_components: int = 2,
                 min_samples: int = 10,
                 distance_metric: str = 'jaccard'):
        """
        Initialize GPAC computer.
        
        Args:
            manifold_components: Dimensions for manifold embedding
            min_samples: Minimum samples needed for analysis
            distance_metric: Distance metric for genotype comparison
        """
        self.n_components = manifold_components
        self.min_samples = min_samples
        self.distance_metric = distance_metric
        
    def compute(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """
        Compute GPAC for trajectory set.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            GPAC results including alignment scores and embeddings
        """
        try:
            # Extract final states
            final_genotypes = [traj['final_genes'] for traj in trajectories]
            final_survivals = [traj['generations'][-1]['survival_prob'] 
                             for traj in trajectories]
            
            if len(final_genotypes) < self.min_samples:
                return {
                    'gpac_score': 0.0,
                    'error': f'Insufficient samples: {len(final_genotypes)} < {self.min_samples}'
                }
            
            # 1. Build genotype manifold
            genotype_embedding, genotype_distances = self._build_genotype_manifold(final_genotypes)
            
            # 2. Build phenotype landscape
            phenotype_embedding, phenotype_distances = self._build_phenotype_landscape(final_survivals)
            
            # 3. Compute alignments
            procrustes_score = self._compute_procrustes_alignment(
                genotype_embedding, phenotype_embedding
            )
            
            correlation_score = self._compute_distance_correlation(
                genotype_distances, phenotype_distances
            )
            
            # 4. Combined GPAC score
            gpac_score = 0.6 * procrustes_score + 0.4 * abs(correlation_score)
            
            return {
                'gpac_score': gpac_score,
                'procrustes_alignment': procrustes_score,
                'distance_correlation': correlation_score,
                'genotype_embedding': genotype_embedding,
                'phenotype_embedding': phenotype_embedding,
                'n_samples': len(final_genotypes)
            }
            
        except Exception as e:
            return {
                'gpac_score': 0.0,
                'error': f'GPAC computation failed: {str(e)}'
            }
    
    def _build_genotype_manifold(self, genotypes: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create genotype manifold using Jaccard similarity and Isomap embedding.
        
        Args:
            genotypes: List of gene lists for each sample
            
        Returns:
            Tuple of (embedding coordinates, distance matrix)
        """
        n_samples = len(genotypes)
        
        # Compute Jaccard similarity matrix
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                genes_i = set(genotypes[i]) if genotypes[i] else set()
                genes_j = set(genotypes[j]) if genotypes[j] else set()
                
                if len(genes_i.union(genes_j)) == 0:
                    similarity = 1.0  # Both empty
                else:
                    intersection = len(genes_i.intersection(genes_j))
                    union = len(genes_i.union(genes_j))
                    similarity = intersection / union
                
                similarity_matrix[i, j] = similarity
        
        # Convert to distance matrix
        distance_matrix = 1.0 - similarity_matrix
        
        # Ensure distance matrix is valid (symmetric, zero diagonal)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        # Manifold embedding using Isomap
        try:
            # Use fewer neighbors if we have few samples
            n_neighbors = min(5, n_samples - 1)
            embedding = Isomap(n_components=self.n_components, 
                             n_neighbors=n_neighbors, 
                             metric='precomputed')
            genotype_coords = embedding.fit_transform(distance_matrix)
            
        except Exception as e:
            # Fallback to MDS if Isomap fails
            from sklearn.manifold import MDS
            mds = MDS(n_components=self.n_components, 
                     dissimilarity='precomputed', 
                     random_state=42)
            genotype_coords = mds.fit_transform(distance_matrix)
        
        return genotype_coords, distance_matrix
    
    def _build_phenotype_landscape(self, survivals: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model phenotype space as survival landscape.
        
        Args:
            survivals: List of survival probabilities
            
        Returns:
            Tuple of (embedding coordinates, distance matrix)
        """
        # Convert to 2D array for distance computation
        survival_array = np.array(survivals).reshape(-1, 1)
        
        # Compute phenotype distances (Euclidean on survival)
        phenotype_distances = squareform(pdist(survival_array, metric='euclidean'))
        
        # If all survivals are identical, create artificial spread
        if np.all(phenotype_distances == 0):
            phenotype_distances = np.random.uniform(0, 0.1, phenotype_distances.shape)
            np.fill_diagonal(phenotype_distances, 0)
        
        # 2D embedding using t-SNE or MDS
        try:
            if len(survivals) > 4:  # t-SNE needs at least 4 points
                tsne = TSNE(n_components=self.n_components, 
                           random_state=42, 
                           metric='precomputed',
                           perplexity=min(3, len(survivals)-1))
                phenotype_coords = tsne.fit_transform(phenotype_distances)
            else:
                raise ValueError("Too few points for t-SNE")
                
        except Exception:
            # Fallback to MDS
            from sklearn.manifold import MDS
            mds = MDS(n_components=self.n_components, 
                     dissimilarity='precomputed', 
                     random_state=42)
            phenotype_coords = mds.fit_transform(phenotype_distances)
        
        return phenotype_coords, phenotype_distances
    
    def _compute_procrustes_alignment(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Procrustes alignment between two point configurations.
        
        Procrustes analysis finds the optimal similarity transformation 
        (translation, rotation, scaling) to align two point sets.
        
        Args:
            X: First point configuration (genotype embedding)
            Y: Second point configuration (phenotype embedding)
            
        Returns:
            Alignment score [0, 1], where 1 = perfect alignment
        """
        try:
            # Use scipy's procrustes for standardized analysis
            _, _, disparity = procrustes(X, Y)
            
            # Convert disparity to alignment score
            # Disparity ranges from 0 (perfect) to ~sqrt(2) (worst)
            alignment_score = 1.0 - min(disparity / np.sqrt(2), 1.0)
            
            return max(0.0, alignment_score)
            
        except Exception as e:
            # Manual Procrustes if scipy fails
            return self._manual_procrustes(X, Y)
    
    def _manual_procrustes(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Manual Procrustes analysis implementation.
        
        Args:
            X, Y: Point configurations to align
            
        Returns:
            Alignment score [0, 1]
        """
        try:
            # Center the configurations
            X_centered = X - X.mean(axis=0)
            Y_centered = Y - Y.mean(axis=0)
            
            # Normalize (Frobenius norm)
            X_norm = np.linalg.norm(X_centered, 'fro')
            Y_norm = np.linalg.norm(Y_centered, 'fro')
            
            if X_norm == 0 or Y_norm == 0:
                return 0.0
            
            X_normalized = X_centered / X_norm
            Y_normalized = Y_centered / Y_norm
            
            # Optimal rotation via SVD
            H = Y_normalized.T @ X_normalized
            U, _, Vt = np.linalg.svd(H)
            R = U @ Vt
            
            # Align Y to X
            Y_aligned = Y_normalized @ R
            
            # Compute alignment error
            error = np.linalg.norm(X_normalized - Y_aligned, 'fro')
            
            # Convert to alignment score
            alignment_score = 1.0 - min(error / np.sqrt(2), 1.0)
            
            return max(0.0, alignment_score)
            
        except Exception:
            return 0.0
    
    def _compute_distance_correlation(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Compute correlation between distance matrices (Mantel test).
        
        Args:
            dist1: First distance matrix (genotype)
            dist2: Second distance matrix (phenotype)
            
        Returns:
            Correlation coefficient [-1, 1]
        """
        try:
            # Extract upper triangular elements (avoid diagonal)
            n = len(dist1)
            indices = np.triu_indices(n, k=1)
            
            dist1_vec = dist1[indices]
            dist2_vec = dist2[indices]
            
            if len(dist1_vec) < 2:
                return 0.0
            
            # Spearman rank correlation (robust to non-linearity)
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(dist1_vec, dist2_vec)
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0