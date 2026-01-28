# eval/metrics/temporal_dynamics.py
"""
Temporal Dynamics Analysis

Mathematical foundation: Signal processing and time series analysis
Analyzes multi-scale temporal patterns in evolutionary trajectories using:
- Fourier analysis for frequency domain characterization
- Wavelet transforms for time-frequency analysis  
- Phase space reconstruction for dynamical analysis
- Spectral entropy for temporal complexity
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class TemporalDynamicsAnalyzer:
    """
    Analyzes temporal dynamics of evolutionary trajectories.
    
    Provides multi-scale analysis of temporal patterns to understand
    the dynamical properties of learned evolutionary strategies.
    
    Mathematical Components:
    1. Fourier analysis for frequency characterization
    2. Wavelet transforms for multi-scale decomposition
    3. Temporal derivatives for velocity/acceleration analysis
    4. Spectral entropy for complexity quantification
    """
    
    def __init__(self, 
                 min_trajectory_length: int = 5,
                 sampling_rate: float = 1.0):
        """
        Initialize temporal dynamics analyzer.
        
        Args:
            min_trajectory_length: Minimum length for temporal analysis
            sampling_rate: Sampling rate for time series (steps per unit time)
        """
        self.min_length = min_trajectory_length
        self.sampling_rate = sampling_rate
        
    def compute(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """
        Compute temporal dynamics analysis for trajectory set.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Temporal dynamics analysis results
        """
        try:
            # Extract time series data
            time_series_data = self._extract_time_series(trajectories)
            
            if not time_series_data:
                return {'error': 'No valid time series extracted'}
            
            # 1. Frequency domain analysis
            frequency_analysis = self._analyze_frequency_domain(time_series_data)
            
            # 2. Temporal derivative analysis
            derivative_analysis = self._analyze_temporal_derivatives(time_series_data)
            
            # 3. Multi-scale wavelet analysis
            wavelet_analysis = self._analyze_wavelets(time_series_data)
            
            # 4. Spectral complexity analysis
            complexity_analysis = self._analyze_spectral_complexity(time_series_data)
            
            # 5. Overall temporal score
            overall_score = self._compute_overall_temporal_score({
                'frequency': frequency_analysis,
                'derivatives': derivative_analysis,
                'wavelets': wavelet_analysis,
                'complexity': complexity_analysis
            })
            
            return {
                'frequency_analysis': frequency_analysis,
                'derivative_analysis': derivative_analysis,
                'wavelet_analysis': wavelet_analysis,
                'complexity_analysis': complexity_analysis,
                'overall_temporal_score': overall_score,
                'n_trajectories': len(time_series_data)
            }
            
        except Exception as e:
            return {
                'error': f'Temporal analysis failed: {str(e)}',
                'overall_temporal_score': 0.0
            }
    
    def _extract_time_series(self, trajectories: List[Dict]) -> List[Dict[str, np.ndarray]]:
        """
        Extract time series data from trajectories.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            List of time series dictionaries
        """
        time_series_data = []
        
        for trajectory in trajectories:
            generations = trajectory.get('generations', [])
            
            if len(generations) < self.min_length:
                continue
            
            # Extract multiple time series
            gene_counts = []
            survival_probs = []
            actions = []
            
            for generation in generations:
                gene_counts.append(len(generation.get('genes', [])))
                survival_probs.append(generation.get('survival_prob', 0.0))
                actions.append(generation.get('action', 2))
            
            time_series_data.append({
                'gene_counts': np.array(gene_counts),
                'survival_probs': np.array(survival_probs),
                'actions': np.array(actions),
                'length': len(gene_counts)
            })
        
        return time_series_data
    
    def _analyze_frequency_domain(self, time_series_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze frequency domain characteristics using Fourier analysis.
        
        Args:
            time_series_data: List of time series dictionaries
            
        Returns:
            Frequency domain analysis results
        """
        dominant_frequencies = []
        spectral_centroids = []
        spectral_bandwidths = []
        
        for ts_data in time_series_data:
            gene_counts = ts_data['gene_counts']
            
            if len(gene_counts) < 4:  # Need minimum length for FFT
                continue
            
            # Compute FFT
            fft_values = fft(gene_counts)
            fft_freq = fftfreq(len(gene_counts), d=1.0/self.sampling_rate)
            
            # Power spectral density
            power_spectrum = np.abs(fft_values) ** 2
            
            # Only consider positive frequencies
            positive_freq_idx = fft_freq > 0
            positive_freqs = fft_freq[positive_freq_idx]
            positive_power = power_spectrum[positive_freq_idx]
            
            if len(positive_freqs) == 0:
                continue
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(positive_power)
            dominant_freq = positive_freqs[dominant_freq_idx]
            dominant_frequencies.append(dominant_freq)
            
            # Spectral centroid (center of mass of spectrum)
            if np.sum(positive_power) > 0:
                spectral_centroid = np.sum(positive_freqs * positive_power) / np.sum(positive_power)
                spectral_centroids.append(spectral_centroid)
            
            # Spectral bandwidth (spread of spectrum)
            if len(spectral_centroids) > 0:
                centroid = spectral_centroids[-1]
                bandwidth = np.sqrt(np.sum(((positive_freqs - centroid) ** 2) * positive_power) / 
                                  np.sum(positive_power))
                spectral_bandwidths.append(bandwidth)
        
        return {
            'mean_dominant_frequency': np.mean(dominant_frequencies) if dominant_frequencies else 0.0,
            'mean_spectral_centroid': np.mean(spectral_centroids) if spectral_centroids else 0.0,
            'mean_spectral_bandwidth': np.mean(spectral_bandwidths) if spectral_bandwidths else 0.0,
            'frequency_variability': np.std(dominant_frequencies) if dominant_frequencies else 0.0,
            'n_analyzed': len(dominant_frequencies)
        }
    
    def _analyze_temporal_derivatives(self, time_series_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze temporal derivatives (velocity, acceleration).
        
        Args:
            time_series_data: List of time series dictionaries
            
        Returns:
            Temporal derivative analysis results
        """
        velocities = []
        accelerations = []
        velocity_changes = []
        
        for ts_data in time_series_data:
            gene_counts = ts_data['gene_counts'].astype(float)
            
            if len(gene_counts) < 3:
                continue
            
            # First derivative (velocity)
            velocity = np.gradient(gene_counts)
            velocities.extend(velocity)
            
            # Second derivative (acceleration)
            acceleration = np.gradient(velocity)
            accelerations.extend(acceleration)
            
            # Velocity change patterns
            velocity_changes.extend(np.diff(velocity))
        
        return {
            'mean_velocity': np.mean(velocities) if velocities else 0.0,
            'velocity_variability': np.std(velocities) if velocities else 0.0,
            'mean_acceleration': np.mean(accelerations) if accelerations else 0.0,
            'acceleration_variability': np.std(accelerations) if accelerations else 0.0,
            'velocity_change_magnitude': np.mean(np.abs(velocity_changes)) if velocity_changes else 0.0,
            'temporal_smoothness': self._compute_temporal_smoothness(velocities),
            'n_analyzed': len(time_series_data)
        }
    
    def _compute_temporal_smoothness(self, velocities: List[float]) -> float:
        """
        Compute temporal smoothness from velocity profile.
        
        Args:
            velocities: List of velocity values
            
        Returns:
            Smoothness score [0, 1]
        """
        if len(velocities) < 2:
            return 1.0
        
        # Total variation of velocity
        velocity_tv = np.sum(np.abs(np.diff(velocities)))
        
        # Normalize by length and scale
        velocity_scale = np.std(velocities) if np.std(velocities) > 0 else 1.0
        normalized_tv = velocity_tv / (len(velocities) * velocity_scale)
        
        # Smoothness score (higher = smoother)
        smoothness = 1.0 / (1.0 + normalized_tv)
        
        return smoothness
    
    def _analyze_wavelets(self, time_series_data: List[Dict]) -> Dict[str, Any]:
        """
        Perform wavelet analysis for multi-scale decomposition.
        
        Args:
            time_series_data: List of time series dictionaries
            
        Returns:
            Wavelet analysis results
        """
        try:
            import pywt
        except ImportError:
            # Fallback without pywt
            return {
                'error': 'PyWavelets not available',
                'multi_scale_score': 0.5
            }
        
        multi_scale_energies = []
        wavelet_entropies = []
        
        for ts_data in time_series_data:
            gene_counts = ts_data['gene_counts'].astype(float)
            
            if len(gene_counts) < 8:  # Minimum for wavelet analysis
                continue
            
            try:
                # Wavelet decomposition using Daubechies wavelet
                coeffs = pywt.wavedec(gene_counts, 'db4', level=3)
                
                # Energy at each scale
                energies = [np.sum(c ** 2) for c in coeffs]
                multi_scale_energies.append(energies)
                
                # Wavelet entropy (measure of signal complexity)
                total_energy = sum(energies)
                if total_energy > 0:
                    energy_ratios = [e / total_energy for e in energies]
                    wavelet_entropy = entropy(energy_ratios)
                    wavelet_entropies.append(wavelet_entropy)
                    
            except Exception:
                continue
        
        if multi_scale_energies:
            # Average energy distribution across scales
            max_scales = max(len(energies) for energies in multi_scale_energies)
            scale_energies = [[] for _ in range(max_scales)]
            
            for energies in multi_scale_energies:
                for i, energy in enumerate(energies):
                    scale_energies[i].append(energy)
            
            mean_scale_energies = [np.mean(scale) if scale else 0.0 for scale in scale_energies]
            
            return {
                'mean_scale_energies': mean_scale_energies,
                'mean_wavelet_entropy': np.mean(wavelet_entropies) if wavelet_entropies else 0.0,
                'multi_scale_score': self._compute_multi_scale_score(mean_scale_energies),
                'n_analyzed': len(multi_scale_energies)
            }
        
        return {
            'mean_scale_energies': [],
            'mean_wavelet_entropy': 0.0,
            'multi_scale_score': 0.0,
            'n_analyzed': 0
        }
    
    def _compute_multi_scale_score(self, scale_energies: List[float]) -> float:
        """
        Compute multi-scale organization score.
        
        Args:
            scale_energies: Energy distribution across scales
            
        Returns:
            Multi-scale score [0, 1]
        """
        if not scale_energies or sum(scale_energies) == 0:
            return 0.0
        
        # Normalize energies
        total_energy = sum(scale_energies)
        normalized_energies = [e / total_energy for e in scale_energies]
        
        # Multi-scale organization = inverse of energy entropy
        # Low entropy = energy concentrated in few scales = organized
        energy_entropy = entropy(normalized_energies)
        max_entropy = np.log(len(scale_energies))
        
        if max_entropy > 0:
            organization = 1.0 - (energy_entropy / max_entropy)
        else:
            organization = 1.0
        
        return max(0.0, organization)
    
    def _analyze_spectral_complexity(self, time_series_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze spectral complexity and irregularity.
        
        Args:
            time_series_data: List of time series dictionaries
            
        Returns:
            Spectral complexity analysis results
        """
        spectral_entropies = []
        spectral_flatnesses = []
        
        for ts_data in time_series_data:
            gene_counts = ts_data['gene_counts']
            
            if len(gene_counts) < 4:
                continue
            
            # Power spectral density
            frequencies, psd = signal.periodogram(gene_counts, fs=self.sampling_rate)
            
            # Remove DC component
            if len(psd) > 1:
                psd = psd[1:]
                frequencies = frequencies[1:]
            
            if len(psd) == 0 or np.sum(psd) == 0:
                continue
            
            # Normalize power spectrum
            psd_normalized = psd / np.sum(psd)
            
            # Spectral entropy (complexity measure)
            spec_entropy = entropy(psd_normalized)
            spectral_entropies.append(spec_entropy)
            
            # Spectral flatness (measure of noise-likeness)
            geometric_mean = np.exp(np.mean(np.log(psd + 1e-12)))
            arithmetic_mean = np.mean(psd)
            
            if arithmetic_mean > 0:
                flatness = geometric_mean / arithmetic_mean
                spectral_flatnesses.append(flatness)
        
        return {
            'mean_spectral_entropy': np.mean(spectral_entropies) if spectral_entropies else 0.0,
            'spectral_entropy_variability': np.std(spectral_entropies) if spectral_entropies else 0.0,
            'mean_spectral_flatness': np.mean(spectral_flatnesses) if spectral_flatnesses else 0.0,
            'complexity_score': self._compute_complexity_score(spectral_entropies, spectral_flatnesses),
            'n_analyzed': len(spectral_entropies)
        }
    
    def _compute_complexity_score(self, entropies: List[float], flatnesses: List[float]) -> float:
        """
        Compute overall complexity score from spectral measures.
        
        Args:
            entropies: List of spectral entropy values
            flatnesses: List of spectral flatness values
            
        Returns:
            Complexity score [0, 1]
        """
        if not entropies:
            return 0.0
        
        # Normalize entropy (higher = more complex)
        max_entropy = np.log(10)  # Assumed maximum based on typical trajectory lengths
        normalized_entropy = np.mean(entropies) / max_entropy
        
        # Flatness score (lower flatness = more structured = higher score)
        if flatnesses:
            flatness_score = 1.0 - np.mean(flatnesses)
        else:
            flatness_score = 0.5
        
        # Combined complexity score
        complexity = 0.6 * min(normalized_entropy, 1.0) + 0.4 * max(flatness_score, 0.0)
        
        return complexity
    
    def _compute_overall_temporal_score(self, analyses: Dict[str, Any]) -> float:
        """
        Compute overall temporal dynamics score from all analyses.
        
        Args:
            analyses: Dictionary of analysis results
            
        Returns:
            Overall temporal score [0, 1]
        """
        scores = []
        weights = []
        
        # Frequency domain score
        freq_analysis = analyses.get('frequency', {})
        if 'mean_spectral_centroid' in freq_analysis:
            # Higher spectral centroid suggests more dynamic behavior
            centroid = freq_analysis['mean_spectral_centroid']
            freq_score = min(centroid / 0.5, 1.0)  # Normalize to [0,1]
            scores.append(freq_score)
            weights.append(0.25)
        
        # Derivative score
        deriv_analysis = analyses.get('derivatives', {})
        if 'temporal_smoothness' in deriv_analysis:
            smoothness_score = deriv_analysis['temporal_smoothness']
            scores.append(smoothness_score)
            weights.append(0.25)
        
        # Wavelet score
        wavelet_analysis = analyses.get('wavelets', {})
        if 'multi_scale_score' in wavelet_analysis:
            wavelet_score = wavelet_analysis['multi_scale_score']
            scores.append(wavelet_score)
            weights.append(0.25)
        
        # Complexity score
        complexity_analysis = analyses.get('complexity', {})
        if 'complexity_score' in complexity_analysis:
            complexity_score = complexity_analysis['complexity_score']
            scores.append(complexity_score)
            weights.append(0.25)
        
        # Weighted average
        if scores:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return weighted_score
        
        return 0.0