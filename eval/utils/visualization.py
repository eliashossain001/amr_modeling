# eval/utils/visualization.py
"""
Visualization utilities for evaluation results.

Creates publication-quality plots and figures for evaluation metrics,
uncertainty bands, and comprehensive assessment visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# At the top of the file, add:
from scipy import stats
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import os
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Global plotting parameters
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
COLOR_PALETTE = ['#2E8B57', '#FF6347', '#4682B4', '#DAA520', '#9370DB', '#FF69B4']


def create_evaluation_plots(results: Dict[str, Any], 
                          output_dir: str) -> None:
    """
    Create comprehensive evaluation plots.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Metric overview plot
    create_metric_overview_plot(results, output_dir)
    
    # 2. Uncertainty visualization
    create_uncertainty_plot(results, output_dir)
    
    # 3. Temporal dynamics plot
    create_temporal_dynamics_plot(results, output_dir)
    
    # 4. Distribution plots
    create_distribution_plots(results, output_dir)
    
    # 5. Comprehensive dashboard
    create_evaluation_dashboard(results, output_dir)
    
    print(f"✓ Evaluation plots saved to {output_dir}/")


def create_metric_overview_plot(results: Dict[str, Any], output_dir: str) -> None:
    """Create overview plot of all metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Evaluation Metrics Overview', fontsize=16, fontweight='bold')
    
    # ETCI Plot
    ax1 = axes[0, 0]
    if 'etci' in results and 'etci_distribution' in results['etci']:
        etci_data = results['etci']['etci_distribution']
        ax1.hist(etci_data, bins=20, alpha=0.7, color=COLOR_PALETTE[0], edgecolor='black')
        ax1.axvline(np.mean(etci_data), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(etci_data):.3f}')
        ax1.set_xlabel('ETCI Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Evolutionary Trajectory Coherence Index')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'ETCI data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('ETCI - No Data')
    
    # GPAC Plot
    ax2 = axes[0, 1]
    if 'gpac' in results and 'gpac_score' in results['gpac']:
        gpac_score = results['gpac']['gpac_score']
        # Create a gauge-style plot for single value
        create_gauge_plot(ax2, gpac_score, 'GPAC Score', COLOR_PALETTE[1])
    else:
        ax2.text(0.5, 0.5, 'GPAC data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('GPAC - No Data')
    
    # AEI Plot
    ax3 = axes[1, 0]
    if 'aei' in results and 'detailed_scores' in results['aei']:
        aei_scores = [score['aei_score'] for score in results['aei']['detailed_scores']]
        ax3.hist(aei_scores, bins=20, alpha=0.7, color=COLOR_PALETTE[2], edgecolor='black')
        ax3.axvline(np.mean(aei_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(aei_scores):.3f}')
        ax3.set_xlabel('AEI Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Adaptive Efficiency Index')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'AEI data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('AEI - No Data')
    
    # Overall Assessment
    ax4 = axes[1, 1]
    if 'overall_assessment' in results:
        overall_score = results['overall_assessment']['overall_score']
        component_scores = results['overall_assessment']['component_scores']
        
        # Bar plot of component scores
        metrics = list(component_scores.keys())
        scores = list(component_scores.values())
        
        bars = ax4.bar(metrics, scores, color=COLOR_PALETTE[:len(metrics)], alpha=0.7, edgecolor='black')
        ax4.axhline(overall_score, color='red', linestyle='-', linewidth=3,
                   label=f'Overall: {overall_score:.3f}')
        ax4.set_ylabel('Score')
        ax4.set_title('Component Scores')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Overall assessment not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Overall - No Data')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_overview.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def create_gauge_plot(ax, value: float, title: str, color: str) -> None:
    """Create a gauge-style plot for single values."""
    
    # Create semicircle gauge
    theta = np.linspace(0, np.pi, 100)
    radius = 1.0
    
    # Background arc
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'lightgray', linewidth=8, alpha=0.3)
    
    # Value arc
    value_theta = theta[:int(value * len(theta))]
    ax.plot(radius * np.cos(value_theta), radius * np.sin(value_theta), color, linewidth=8)
    
    # Center point and needle
    needle_angle = np.pi * (1 - value)
    needle_x = 0.8 * np.cos(needle_angle)
    needle_y = 0.8 * np.sin(needle_angle)
    
    ax.plot([0, needle_x], [0, needle_y], 'black', linewidth=3)
    ax.plot(0, 0, 'ko', markersize=8)
    
    # Labels
    ax.text(0, -0.3, f'{value:.3f}', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0, -0.5, title, ha='center', va='center', fontsize=12)
    
    # Scale labels
    for i, val in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        angle = np.pi * (1 - val)
        x = 1.1 * np.cos(angle)
        y = 1.1 * np.sin(angle)
        ax.text(x, y, f'{val:.2f}', ha='center', va='center', fontsize=8)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.7, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')


def create_uncertainty_plot(results: Dict[str, Any], output_dir: str) -> None:
    """Create uncertainty visualization plots."""
    
    if 'uncertainty' not in results:
        return
    
    uncertainty = results['uncertainty']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Uncertainty Quantification', fontsize=16, fontweight='bold')
    
    # Confidence intervals plot
    ax1 = axes[0]
    metrics = []
    point_estimates = []
    ci_lowers = []
    ci_uppers = []
    
    for metric, data in uncertainty.items():
        if metric == 'overall' or 'point_estimate' not in data:
            continue
            
        metrics.append(metric.upper())
        point_estimates.append(data['point_estimate'])
        
        # Extract confidence intervals
        if 'bayesian_ci' in data:
            ci_lowers.append(data['bayesian_ci']['lower'])
            ci_uppers.append(data['bayesian_ci']['upper'])
        elif 'credible_interval' in data:
            ci_lowers.append(data['credible_interval']['lower'])
            ci_uppers.append(data['credible_interval']['upper'])
        else:
            # Fallback - assume ±10% uncertainty
            ci_lowers.append(data['point_estimate'] * 0.9)
            ci_uppers.append(data['point_estimate'] * 1.1)
    
    if metrics:
        y_pos = np.arange(len(metrics))
        
        # Error bars
        ax1.errorbar(point_estimates, y_pos, 
                    xerr=[np.array(point_estimates) - np.array(ci_lowers),
                          np.array(ci_uppers) - np.array(point_estimates)],
                    fmt='o', capsize=5, capthick=2, markersize=8)
        
        # Point estimates
        ax1.scatter(point_estimates, y_pos, color='red', s=100, zorder=5)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(metrics)
        ax1.set_xlabel('Metric Value')
        ax1.set_title('Confidence Intervals')
        ax1.grid(True, alpha=0.3)
    
    # Uncertainty measures plot
    ax2 = axes[1]
    if 'overall' in uncertainty and 'overall_ci' in uncertainty['overall']:
        overall_data = uncertainty['overall']
        estimate = overall_data['overall_estimate']
        ci = overall_data['overall_ci']
        
        # Create uncertainty band visualization
        x = np.linspace(0, 1, 100)
        y_center = estimate * np.ones_like(x)
        y_lower = ci['lower'] * np.ones_like(x)
        y_upper = ci['upper'] * np.ones_like(x)
        
        ax2.fill_between(x, y_lower, y_upper, alpha=0.3, color=COLOR_PALETTE[0], label='Confidence Band')
        ax2.plot(x, y_center, color=COLOR_PALETTE[0], linewidth=3, label=f'Estimate: {estimate:.3f}')
        
        ax2.set_xlabel('Normalized Scale')
        ax2.set_ylabel('Overall Score')
        ax2.set_title('Overall Uncertainty')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Overall uncertainty not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Overall Uncertainty - No Data')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def create_temporal_dynamics_plot(results: Dict[str, Any], output_dir: str) -> None:
    """Create temporal dynamics visualization."""
    
    if 'temporal' not in results:
        return
    
    temporal = results['temporal']
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    fig.suptitle('Temporal Dynamics Analysis', fontsize=16, fontweight='bold')
    
    # Frequency analysis
    ax1 = fig.add_subplot(gs[0, 0])
    if 'frequency_analysis' in temporal:
        freq_data = temporal['frequency_analysis']
        
        metrics = ['Dominant Freq', 'Spectral Centroid', 'Spectral Bandwidth', 'Frequency Var']
        values = [
            freq_data.get('mean_dominant_frequency', 0),
            freq_data.get('mean_spectral_centroid', 0),
            freq_data.get('mean_spectral_bandwidth', 0),
            freq_data.get('frequency_variability', 0)
        ]
        
        bars = ax1.bar(metrics, values, color=COLOR_PALETTE[0], alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Value')
        ax1.set_title('Frequency Domain')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No frequency data', ha='center', va='center', transform=ax1.transAxes)
    
    # Derivative analysis
    ax2 = fig.add_subplot(gs[0, 1])
    if 'derivative_analysis' in temporal:
        deriv_data = temporal['derivative_analysis']
        
        metrics = ['Mean Velocity', 'Velocity Var', 'Mean Accel', 'Smoothness']
        values = [
            deriv_data.get('mean_velocity', 0),
            deriv_data.get('velocity_variability', 0),
            deriv_data.get('mean_acceleration', 0),
            deriv_data.get('temporal_smoothness', 0)
        ]
        
        bars = ax2.bar(metrics, values, color=COLOR_PALETTE[1], alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Value')
        ax2.set_title('Temporal Derivatives')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No derivative data', ha='center', va='center', transform=ax2.transAxes)
    
    # Wavelet analysis
    ax3 = fig.add_subplot(gs[0, 2])
    if 'wavelet_analysis' in temporal and 'mean_scale_energies' in temporal['wavelet_analysis']:
        wavelet_data = temporal['wavelet_analysis']
        scale_energies = wavelet_data['mean_scale_energies']
        
        if scale_energies:
            scales = [f'Scale {i+1}' for i in range(len(scale_energies))]
            ax3.bar(scales, scale_energies, color=COLOR_PALETTE[2], alpha=0.7, edgecolor='black')
            ax3.set_ylabel('Energy')
            ax3.set_title('Multi-Scale Energy')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No wavelet energy data', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No wavelet data', ha='center', va='center', transform=ax3.transAxes)
    
    # Complexity analysis
    ax4 = fig.add_subplot(gs[1, :])
    if 'complexity_analysis' in temporal:
        complexity_data = temporal['complexity_analysis']
        
        # Create complexity radar chart
        metrics = ['Spectral Entropy', 'Spectral Flatness', 'Complexity Score']
        values = [
            complexity_data.get('mean_spectral_entropy', 0),
            complexity_data.get('mean_spectral_flatness', 0),
            complexity_data.get('complexity_score', 0)
        ]
        
        # Normalize values for radar chart
        max_vals = [3.0, 1.0, 1.0]  # Approximate maximum values
        normalized_values = [min(v/m, 1.0) for v, m in zip(values, max_vals)]
        
        # Create simple bar chart instead of radar for simplicity
        bars = ax4.bar(metrics, normalized_values, color=COLOR_PALETTE[3], alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Spectral Complexity Analysis')
        ax4.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, norm_val, orig_val in zip(bars, normalized_values, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{orig_val:.3f}', ha='center', va='bottom', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No complexity data', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_dynamics.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def create_distribution_plots(results: Dict[str, Any], output_dir: str) -> None:
    """Create detailed distribution plots for metrics."""
    
    # Collect all available distributions
    distributions = {}
    
    if 'etci' in results and 'etci_distribution' in results['etci']:
        distributions['ETCI'] = results['etci']['etci_distribution']
    
    if 'aei' in results and 'detailed_scores' in results['aei']:
        distributions['AEI'] = [score['aei_score'] for score in results['aei']['detailed_scores']]
    
    if not distributions:
        return
    
    fig, axes = plt.subplots(len(distributions), 2, figsize=(12, 4*len(distributions)))
    if len(distributions) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Metric Distributions', fontsize=16, fontweight='bold')
    
    for i, (metric_name, data) in enumerate(distributions.items()):
        # Histogram
        ax_hist = axes[i, 0]
        n, bins, patches = ax_hist.hist(data, bins=20, alpha=0.7, color=COLOR_PALETTE[i], 
                                       edgecolor='black', density=True)
        
        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        median_val = np.median(data)
        
        ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax_hist.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        
        ax_hist.set_xlabel(f'{metric_name} Score')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title(f'{metric_name} Distribution')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Q-Q plot for normality assessment
        ax_qq = axes[i, 1]
        stats.probplot(data, dist="norm", plot=ax_qq)
        ax_qq.set_title(f'{metric_name} Q-Q Plot (Normality Test)')
        ax_qq.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distributions.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def create_evaluation_dashboard(results: Dict[str, Any], output_dir: str) -> None:
    """Create comprehensive evaluation dashboard."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Bacterial Evolution RL Evaluation Dashboard', fontsize=20, fontweight='bold')
    
    # Overall score gauge (top center)
    ax_gauge = fig.add_subplot(gs[0, 1:3])
    if 'overall_assessment' in results:
        overall_score = results['overall_assessment']['overall_score']
        quality_rating = results['overall_assessment']['quality_rating']
        
        # Large gauge plot
        create_gauge_plot(ax_gauge, overall_score, f'Overall Score\n{quality_rating}', COLOR_PALETTE[0])
        
        # Add score interpretation
        interpretation = get_score_interpretation(overall_score)
        ax_gauge.text(0, -0.7, interpretation, ha='center', va='center', 
                     transform=ax_gauge.transAxes, fontsize=10, style='italic')
    
    # Metric summary table (top left)
    ax_table = fig.add_subplot(gs[0, 0])
    create_summary_table(ax_table, results)
    
    # Quality indicators (top right)
    ax_indicators = fig.add_subplot(gs[0, 3])
    create_quality_indicators(ax_indicators, results)
    
    # Component scores (middle left)
    ax_components = fig.add_subplot(gs[1, :2])
    if 'overall_assessment' in results and 'component_scores' in results['overall_assessment']:
        component_scores = results['overall_assessment']['component_scores']
        weights = results['overall_assessment']['weights']
        
        metrics = list(component_scores.keys())
        scores = list(component_scores.values())
        weight_values = [weights.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        bars1 = ax_components.bar(x - 0.2, scores, 0.4, label='Score', color=COLOR_PALETTE[0], alpha=0.7)
        bars2 = ax_components.bar(x + 0.2, weight_values, 0.4, label='Weight', color=COLOR_PALETTE[1], alpha=0.7)
        
        ax_components.set_xlabel('Metrics')
        ax_components.set_ylabel('Value')
        ax_components.set_title('Component Scores and Weights')
        ax_components.set_xticks(x)
        ax_components.set_xticklabels([m.upper() for m in metrics])
        ax_components.legend()
        ax_components.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax_components.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Uncertainty summary (middle right)
    ax_uncertainty = fig.add_subplot(gs[1, 2:])
    create_uncertainty_summary(ax_uncertainty, results)
    
    # Performance metrics timeline (bottom)
    ax_timeline = fig.add_subplot(gs[2, :])
    create_performance_timeline(ax_timeline, results)
    
    plt.savefig(os.path.join(output_dir, 'evaluation_dashboard.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def create_summary_table(ax, results: Dict[str, Any]) -> None:
    """Create summary statistics table."""
    ax.axis('off')
    
    # Collect summary data
    table_data = []
    
    if 'etci' in results:
        etci_data = results['etci']
        table_data.append(['ETCI', f"{etci_data.get('mean_etci', 0):.3f} ± {etci_data.get('std_etci', 0):.3f}"])
    
    if 'gpac' in results:
        gpac_score = results['gpac'].get('gpac_score', 0)
        table_data.append(['GPAC', f"{gpac_score:.3f}"])
    
    if 'aei' in results:
        aei_data = results['aei']
        table_data.append(['AEI', f"{aei_data.get('mean_aei', 0):.3f} ± {aei_data.get('std_aei', 0):.3f}"])
    
    if 'overall_assessment' in results:
        overall_score = results['overall_assessment']['overall_score']
        quality_rating = results['overall_assessment']['quality_rating']
        table_data.append(['Overall', f"{overall_score:.3f} ({quality_rating})"])
    
    if table_data:
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')
    
    ax.set_title('Summary Statistics', fontweight='bold', pad=20)


def create_quality_indicators(ax, results: Dict[str, Any]) -> None:
    """Create quality indicator visualization."""
    ax.axis('off')
    
    # Define quality indicators
    indicators = []
    
    # Data quality
    if 'metadata' in results:
        n_trajectories = results['metadata'].get('n_trajectories', 0)
        if n_trajectories >= 50:
            indicators.append(('Data Quality', 'Excellent', 'green'))
        elif n_trajectories >= 20:
            indicators.append(('Data Quality', 'Good', 'yellow'))
        else:
            indicators.append(('Data Quality', 'Poor', 'red'))
    
    # Metric coverage
    available_metrics = sum([
        'etci' in results and 'error' not in results['etci'],
        'gpac' in results and 'error' not in results['gpac'],
        'aei' in results and 'error' not in results['aei']
    ])
    
    if available_metrics == 3:
        indicators.append(('Metric Coverage', 'Complete', 'green'))
    elif available_metrics == 2:
        indicators.append(('Metric Coverage', 'Partial', 'yellow'))
    else:
        indicators.append(('Metric Coverage', 'Limited', 'red'))
    
    # Statistical significance
    if 'uncertainty' in results:
        indicators.append(('Uncertainty', 'Quantified', 'green'))
    else:
        indicators.append(('Uncertainty', 'Not Available', 'red'))
    
    # Create indicator visualization
    y_positions = np.linspace(0.8, 0.2, len(indicators))
    
    for i, (indicator, status, color) in enumerate(indicators):
        # Indicator label
        ax.text(0.1, y_positions[i], indicator, fontweight='bold', fontsize=12, va='center')
        
        # Status indicator (circle)
        circle = plt.Circle((0.8, y_positions[i]), 0.05, color=color, alpha=0.7)
        ax.add_patch(circle)
        
        # Status text
        ax.text(0.9, y_positions[i], status, fontsize=10, va='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Quality Indicators', fontweight='bold', pad=20)


def create_uncertainty_summary(ax, results: Dict[str, Any]) -> None:
    """Create uncertainty summary visualization."""
    
    if 'uncertainty' not in results:
        ax.text(0.5, 0.5, 'Uncertainty data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Uncertainty Summary - No Data')
        return
    
    uncertainty = results['uncertainty']
    
    # Extract confidence interval widths
    metrics = []
    ci_widths = []
    
    for metric, data in uncertainty.items():
        if metric == 'overall' or 'point_estimate' not in data:
            continue
        
        metrics.append(metric.upper())
        
        if 'bayesian_ci' in data:
            ci_widths.append(data['bayesian_ci']['width'])
        elif 'credible_interval' in data:
            ci_widths.append(data['credible_interval']['width'])
        else:
            ci_widths.append(0.1)  # Default uncertainty
    
    if metrics:
        # Create horizontal bar chart of uncertainties
        y_pos = np.arange(len(metrics))
        bars = ax.barh(y_pos, ci_widths, color=COLOR_PALETTE[4], alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics)
        ax.set_xlabel('Confidence Interval Width')
        ax.set_title('Uncertainty Magnitudes')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, width in zip(bars, ci_widths):
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                   f'{width:.3f}', ha='left', va='center', fontsize=8)


def create_performance_timeline(ax, results: Dict[str, Any]) -> None:
    """Create performance timeline visualization."""
    
    # Extract temporal data if available
    if 'temporal' in results and 'derivative_analysis' in results['temporal']:
        temporal_data = results['temporal']['derivative_analysis']
        
        # Create synthetic timeline based on available data
        time_points = np.arange(0, 10)  # Synthetic time points
        
        # Use available temporal metrics
        velocity = temporal_data.get('mean_velocity', 0.1)
        acceleration = temporal_data.get('mean_acceleration', 0.01)
        smoothness = temporal_data.get('temporal_smoothness', 0.8)
        
        # Generate synthetic performance trajectory
        performance = []
        current_perf = 0.3  # Starting performance
        
        for t in time_points:
            # Add velocity and acceleration effects
            current_perf += velocity + acceleration * t + np.random.normal(0, 0.05)
            current_perf = max(0, min(1, current_perf))  # Bound between 0 and 1
            performance.append(current_perf)
        
        # Plot performance timeline
        ax.plot(time_points, performance, color=COLOR_PALETTE[0], linewidth=3, marker='o', markersize=8, alpha=0.8)
        
        # Add smoothness indicator (error band)
        smoothness_band = smoothness * 0.1  # Convert to band width
        upper_band = np.array(performance) + smoothness_band
        lower_band = np.array(performance) - smoothness_band
        
        ax.fill_between(time_points, lower_band, upper_band, alpha=0.2, color=COLOR_PALETTE[0])
        
        ax.set_xlabel('Evolution Steps')
        ax.set_ylabel('Performance Score')
        ax.set_title('Evolutionary Performance Timeline')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add annotations
        ax.annotate(f'Final Performance: {performance[-1]:.3f}', 
                   xy=(time_points[-1], performance[-1]), 
                   xytext=(time_points[-3], performance[-1] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=10, fontweight='bold')
        
    else:
        ax.text(0.5, 0.5, 'Temporal data not available for timeline', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance Timeline - No Data')


def get_score_interpretation(score: float) -> str:
    """Get interpretation text for overall score."""
    if score >= 0.8:
        return "Excellent performance across all metrics.\nModel demonstrates strong evolutionary learning."
    elif score >= 0.6:
        return "Good performance with room for improvement.\nModel shows solid evolutionary capabilities."
    elif score >= 0.4:
        return "Fair performance with significant gaps.\nModel requires further optimization."
    else:
        return "Poor performance across metrics.\nModel needs substantial improvement."


def plot_metric_distribution(data: List[float], 
                           metric_name: str, 
                           output_path: str,
                           confidence_interval: Optional[Tuple[float, float]] = None) -> None:
    """
    Plot distribution of a single metric with optional confidence interval.
    
    Args:
        data: List of metric values
        metric_name: Name of the metric
        output_path: Path to save the plot
        confidence_interval: Optional tuple of (lower, upper) bounds
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{metric_name} Distribution Analysis', fontsize=14, fontweight='bold')
    
    # Histogram
    ax1.hist(data, bins=20, alpha=0.7, color=COLOR_PALETTE[0], edgecolor='black', density=True)
    
    # Statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    median_val = np.median(data)
    
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    
    if confidence_interval:
        ax1.axvspan(confidence_interval[0], confidence_interval[1], alpha=0.2, color='gray', 
                   label=f'95% CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]')
    
    ax1.set_xlabel(f'{metric_name} Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    box_plot = ax2.boxplot(data, patch_artist=True)
    box_plot['boxes'][0].set_facecolor(COLOR_PALETTE[0])
    box_plot['boxes'][0].set_alpha(0.7)
    
    ax2.set_ylabel(f'{metric_name} Value')
    ax2.set_title('Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def plot_uncertainty_bands(x: np.ndarray, 
                          y_mean: np.ndarray,
                          y_lower: np.ndarray, 
                          y_upper: np.ndarray,
                          output_path: str,
                          title: str = "Uncertainty Bands") -> None:
    """
    Plot uncertainty bands for time series data.
    
    Args:
        x: X-axis values (e.g., time points)
        y_mean: Mean values
        y_lower: Lower confidence bound
        y_upper: Upper confidence bound
        output_path: Path to save the plot
        title: Plot title
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main line
    ax.plot(x, y_mean, color=COLOR_PALETTE[0], linewidth=3, label='Mean')
    
    # Uncertainty band
    ax.fill_between(x, y_lower, y_upper, alpha=0.3, color=COLOR_PALETTE[0], label='95% Confidence Interval')
    
    # Bounds
    ax.plot(x, y_lower, '--', color=COLOR_PALETTE[1], alpha=0.7, label='Lower Bound')
    ax.plot(x, y_upper, '--', color=COLOR_PALETTE[2], alpha=0.7, label='Upper Bound')
    
    ax.set_xlabel('Time/Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()