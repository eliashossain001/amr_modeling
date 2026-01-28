# ablation_experiments/compare_ablations.py
"""
Comparison and visualization script for ablation experiments.
Creates publication-quality plots comparing all ablation variants.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Tuple
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Global plotting parameters
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'


def load_evaluation_results(results_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    
    with open(results_file, 'r') as f:
        return json.load(f)


def create_ablation_comparison_plot(results: Dict[str, Any], 
                                  output_dir: str) -> None:
    """Create comprehensive ablation comparison visualization."""
    
    # Prepare data for plotting
    ablation_data = []
    
    for ablation_name, result in results.items():
        if result is None:
            continue
            
        ablation_data.append({
            'Ablation': ablation_name.replace('_', ' ').title(),
            'Overall': result['overall_score_mean'],
            'Overall_Std': result['overall_score_std'],
            'ETCI': result['etci_score_mean'],
            'ETCI_Std': result['etci_score_std'],
            'GPAC': result['gpac_score_mean'],
            'GPAC_Std': result['gpac_score_std'],
            'AEI': result['aei_score_mean'],
            'AEI_Std': result['aei_score_std'],
            'Temporal': result['temporal_score_mean'],
            'Temporal_Std': result['temporal_score_std'],
            'Category': categorize_ablation(ablation_name)
        })
    
    df = pd.DataFrame(ablation_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ablation Study Results: Component Analysis', fontsize=20, fontweight='bold')
    
    # Plot 1: Overall Performance Comparison
    ax1 = axes[0, 0]
    create_metric_comparison_plot(ax1, df, 'Overall', 'Overall Performance', 
                                 include_baseline=True)
    
    # Plot 2: Individual Metrics Comparison
    ax2 = axes[0, 1]
    create_radar_comparison(ax2, df)
    
    # Plot 3: Category-wise Analysis
    ax3 = axes[1, 0]
    create_category_analysis(ax3, df)
    
    # Plot 4: Metric Correlation Analysis
    ax4 = axes[1, 1]
    create_correlation_analysis(ax4, df)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f" Ablation comparison plot saved: {output_path}")


def create_metric_comparison_plot(ax, df: pd.DataFrame, metric: str, title: str,
                                include_baseline: bool = False) -> None:
    """Create bar plot comparing single metric across ablations."""
    
    # Sort by performance
    df_sorted = df.sort_values(metric, ascending=False)
    
    # Create bar plot with error bars
    bars = ax.bar(range(len(df_sorted)), 
                  df_sorted[metric], 
                  yerr=df_sorted[f'{metric}_Std'],
                  capsize=5,
                  alpha=0.8,
                  edgecolor='black')
    
    # Color bars by category
    colors = {'Graph': '#2E8B57', 'Architecture': '#4682B4', 
             'Algorithm': '#DAA520', 'Reward': '#9370DB', 'Baseline': '#FF6347'}
    
    for i, (bar, category) in enumerate(zip(bars, df_sorted['Category'])):
        bar.set_color(colors.get(category, '#808080'))
    
    # Add baseline line if requested
    if include_baseline:
        # Assuming baseline is your main model result (around 0.597)
        baseline_score = 0.597  # Update with actual baseline
        ax.axhline(baseline_score, color='red', linestyle='--', linewidth=2,
                  label=f'Main Model: {baseline_score:.3f}')
        ax.legend()
    
    # Formatting
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['Ablation'], rotation=45, ha='right')
    ax.set_ylabel(f'{metric} Score')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value, std) in enumerate(zip(bars, df_sorted[metric], df_sorted[f'{metric}_Std'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=8)


def create_radar_comparison(ax, df: pd.DataFrame) -> None:
    """Create radar plot comparing top ablations across all metrics."""
    
    # Select top 5 ablations by overall performance
    top_ablations = df.nlargest(5, 'Overall')
    
    # Metrics for radar plot
    metrics = ['ETCI', 'GPAC', 'AEI', 'Temporal']
    
    # Number of metrics
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Plot each ablation
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_ablations)))
    
    for i, (_, ablation) in enumerate(top_ablations.iterrows()):
        values = [ablation[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=ablation['Ablation'], color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Ablations: Multi-Metric Comparison', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)


def create_category_analysis(ax, df: pd.DataFrame) -> None:
    """Create category-wise performance analysis."""
    
    # Group by category and compute statistics
    category_stats = df.groupby('Category').agg({
        'Overall': ['mean', 'std', 'count'],
        'ETCI': 'mean',
        'GPAC': 'mean', 
        'AEI': 'mean',
        'Temporal': 'mean'
    }).round(3)
    
    # Flatten column names
    category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
    
    # Create grouped bar plot
    categories = category_stats.index
    overall_means = category_stats['Overall_mean']
    overall_stds = category_stats['Overall_std']
    
    bars = ax.bar(categories, overall_means, yerr=overall_stds, 
                  capsize=5, alpha=0.8, edgecolor='black')
    
    # Color bars
    colors = ['#2E8B57', '#4682B4', '#DAA520', '#9370DB']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars, category_stats['Overall_count'])):
        height = bar.get_height() + overall_stds.iloc[i] + 0.01
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'n={int(count)}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Overall Score')
    ax.set_title('Performance by Ablation Category', fontweight='bold')
    ax.grid(True, alpha=0.3)


def create_correlation_analysis(ax, df: pd.DataFrame) -> None:
    """Create correlation matrix of metrics."""
    
    # Select metric columns
    metric_cols = ['Overall', 'ETCI', 'GPAC', 'AEI', 'Temporal']
    corr_matrix = df[metric_cols].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    
    ax.set_title('Metric Correlation Matrix', fontweight='bold')


def categorize_ablation(ablation_name: str) -> str:
    """Categorize ablation by type."""
    
    if 'graph' in ablation_name or any(x in ablation_name for x in 
                                      ['gene_similarity', 'geographic', 'serovar', 'plasmid']):
        return 'Graph'
    elif any(x in ablation_name for x in ['network', 'layer', 'hidden']):
        return 'Architecture'
    elif any(x in ablation_name for x in ['ppo', 'lr', 'learning']):
        return 'Algorithm'
    elif any(x in ablation_name for x in ['reward', 'survival', 'genes', 'penalties']):
        return 'Reward'
    else:
        return 'Other'


def create_detailed_comparison_table(results: Dict[str, Any], 
                                   output_dir: str) -> None:
    """Create detailed comparison table for paper."""
    
    # Prepare data
    table_data = []
    
    for ablation_name, result in results.items():
        if result is None:
            continue
            
        table_data.append({
            'Ablation': ablation_name.replace('_', ' ').title(),
            'Description': result['description'],
            'Overall': f"{result['overall_score_mean']:.3f} ± {result['overall_score_std']:.3f}",
            'ETCI': f"{result['etci_score_mean']:.3f} ± {result['etci_score_std']:.3f}",
            'GPAC': f"{result['gpac_score_mean']:.3f} ± {result['gpac_score_std']:.3f}",
            'AEI': f"{result['aei_score_mean']:.3f} ± {result['aei_score_std']:.3f}",
            'Temporal': f"{result['temporal_score_mean']:.3f} ± {result['temporal_score_std']:.3f}",
            'Seeds': result['n_seeds']
        })
    
    # Sort by overall performance
    table_data.sort(key=lambda x: float(x['Overall'].split(' ')[0]), reverse=True)
    
    # Create DataFrame and save
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'ablation_detailed_comparison.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as LaTeX table
    latex_path = os.path.join(output_dir, 'ablation_table.tex')
    with open(latex_path, 'w') as f:
        f.write("% Ablation Study Results Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Study Results: Performance across BERAT metrics}\n")
        f.write("\\label{tab:ablation_results}\n")
        f.write("\\begin{tabular}{lllllll}\n")
        f.write("\\toprule\n")
        f.write("Ablation & Overall & ETCI & GPAC & AEI & Temporal & Seeds \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['Ablation']} & {row['Overall']} & {row['ETCI']} & "
                   f"{row['GPAC']} & {row['AEI']} & {row['Temporal']} & {row['Seeds']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f" Detailed comparison saved:")
    print(f" {csv_path}")
    print(f" {latex_path}")
    
    return df


def create_performance_ranking(results: Dict[str, Any], output_dir: str) -> None:
    """Create performance ranking visualization."""
    
    # Extract performance data
    performance_data = []
    
    for ablation_name, result in results.items():
        if result is None:
            continue
            
        performance_data.append({
            'Ablation': ablation_name.replace('_', ' ').title(),
            'Overall': result['overall_score_mean'],
            'Category': categorize_ablation(ablation_name)
        })
    
    # Sort by performance
    performance_data.sort(key=lambda x: x['Overall'], reverse=True)
    
    # Create ranking plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ablations = [item['Ablation'] for item in performance_data]
    scores = [item['Overall'] for item in performance_data]
    categories = [item['Category'] for item in performance_data]
    
    # Color map
    colors = {'Graph': '#2E8B57', 'Architecture': '#4682B4', 
             'Algorithm': '#DAA520', 'Reward': '#9370DB', 'Other': '#808080'}
    bar_colors = [colors.get(cat, '#808080') for cat in categories]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(ablations)), scores, color=bar_colors, alpha=0.8, edgecolor='black')
    
    # Add baseline line
    baseline = 0.597  # Your main model performance
    ax.axvline(baseline, color='red', linestyle='--', linewidth=3, 
              label=f'Main Model: {baseline:.3f}')
    
    # Formatting
    ax.set_yticks(range(len(ablations)))
    ax.set_yticklabels(ablations)
    ax.set_xlabel('Overall BERAT Score')
    ax.set_title('Ablation Study: Performance Ranking', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
               f'{score:.3f}', ha='left', va='center', fontsize=9)
    
    # Add category legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                   label=category) for category, color in colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', title='Category')
    
    plt.tight_layout()
    
    # Save plot
    ranking_path = os.path.join(output_dir, 'ablation_ranking.png')
    plt.savefig(ranking_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f" Performance ranking saved: {ranking_path}")


def generate_ablation_report(results: Dict[str, Any], output_dir: str) -> None:
    """Generate comprehensive ablation study report."""
    
    report_path = os.path.join(output_dir, 'ablation_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Ablation Study Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Ablations:** {len(results)}\n\n")
        
        # Performance ranking
        f.write("## Performance Ranking\n\n")
        
        performance_data = []
        for ablation_name, result in results.items():
            if result is None:
                continue
            performance_data.append((ablation_name, result['overall_score_mean']))
        
        performance_data.sort(key=lambda x: x[1], reverse=True)
        
        f.write("| Rank | Ablation | Overall Score | Description |\n")
        f.write("|------|----------|---------------|-------------|\n")
        
        for i, (ablation_name, score) in enumerate(performance_data):
            result = results[ablation_name]
            f.write(f"| {i+1} | {ablation_name.replace('_', ' ').title()} | {score:.3f} | {result['description']} |\n")
        
        # Key findings
        f.write("\n## Key Findings\n\n")
        
        best_ablation = performance_data[0][0]
        worst_ablation = performance_data[-1][0]
        
        f.write(f"- **Best performing ablation:** {best_ablation.replace('_', ' ').title()} "
               f"({performance_data[0][1]:.3f})\n")
        f.write(f"- **Worst performing ablation:** {worst_ablation.replace('_', ' ').title()} "
               f"({performance_data[-1][1]:.3f})\n")
        
        # Category analysis
        categories = {}
        for ablation_name, result in results.items():
            if result is None:
                continue
            cat = categorize_ablation(ablation_name)
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['overall_score_mean'])
        
        f.write(f"\n### Category Performance\n\n")
        for category, scores in categories.items():
            avg_score = np.mean(scores)
            f.write(f"- **{category}:** {avg_score:.3f} (n={len(scores)})\n")
        
        f.write("\n## Methodology\n\n")
        f.write("- Evaluation framework: BERAT (Bacterial Evolution RL Assessment Toolkit)\n")
        f.write("- Metrics: ETCI, GPAC, AEI, Temporal Dynamics\n")
        f.write("- Statistical confidence: 95% intervals\n")
        f.write("- Multiple random seeds per ablation\n")
    
    print(f" Ablation report saved: {report_path}")


def main():
    """Main comparison function."""
    
    parser = argparse.ArgumentParser(description="Compare and visualize ablation results")
    parser.add_argument('--results_file', 
                       default='results_icml/ablation_evaluation/ablation_evaluation_results.json',
                       help='Path to evaluation results JSON file')
    parser.add_argument('--output_dir', default='results_icml/ablation_comparison',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f" Results file not found: {args.results_file}")
        print("  Run evaluate_ablations.py first!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(" Loading evaluation results...")
    results = load_evaluation_results(args.results_file)
    
    print(f" Creating comparison visualizations...")
    
    # Generate all comparison outputs
    create_ablation_comparison_plot(results, args.output_dir)
    create_performance_ranking(results, args.output_dir)
    comparison_df = create_detailed_comparison_table(results, args.output_dir)
    generate_ablation_report(results, args.output_dir)
    
    print(f"\n Ablation comparison complete!")
    print(f" Results saved to: {args.output_dir}")
    
    # Print top 3 ablations
    print(f"\n Top 3 Performing Ablations:")
    top_3 = comparison_df.head(3)
    for i, (_, row) in enumerate(top_3.iterrows()):
        print(f"   {i+1}. {row['Ablation']}: {row['Overall']}")


if __name__ == "__main__":
    main()