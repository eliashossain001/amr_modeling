# ablation_experiments/run_complete_ablation_study.py
"""
Complete ablation study runner.
Executes training, evaluation, and comparison in one comprehensive workflow.
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ablation_experiments.configs.ablation_configs import get_all_ablations, get_priority_ablations


def run_command(command: str, description: str) -> bool:
    """
    Run a command and handle errors.
    
    Args:
        command: Command to execute
        description: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    
    print(f"\n {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        
        if result.stdout:
            print(f" Output: {result.stdout.strip()}")
        
        print(f" {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f" {description} failed")
        print(f"  Error: {e.stderr}")
        return False


def run_complete_ablation_study(ablation_names: list = None,
                               num_seeds: int = 3,
                               base_output_dir: str = "results_icml/ablation_study",
                               skip_training: bool = False,
                               skip_evaluation: bool = False,
                               skip_comparison: bool = False) -> bool:
    """
    Run complete ablation study workflow.
    
    Args:
        ablation_names: List of ablation names to run
        num_seeds: Number of seeds per ablation
        base_output_dir: Base output directory
        skip_training: Skip training phase
        skip_evaluation: Skip evaluation phase  
        skip_comparison: Skip comparison phase
        
    Returns:
        True if successful, False otherwise
    """
    
    print(" COMPLETE ABLATION STUDY")
    print("="*60)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Output directory: {base_output_dir}")
    
    # Setup directories
    training_dir = os.path.join(base_output_dir, "training_results")
    evaluation_dir = os.path.join(base_output_dir, "evaluation_results")
    comparison_dir = os.path.join(base_output_dir, "comparison_results")
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Determine ablations to run
    if ablation_names is None:
        available_ablations = get_priority_ablations()
        ablation_names = list(available_ablations.keys())
    
    print(f" Ablations to run: {ablation_names}")
    print(f" Seeds per ablation: {num_seeds}")
    print(f" Total experiments: {len(ablation_names) * num_seeds}")
    print("="*60)
    
    success = True
    
    # Phase 1: Training
    if not skip_training:
        print("\n PHASE 1: TRAINING ABLATIONS")
        print("-" * 40)
        
        train_command = (
            f"python ablation_experiments/train_ablations.py "
            f"--ablations {' '.join(ablation_names)} "
            f"--output_dir {training_dir} "
            f"--num_seeds {num_seeds}"
        )
        
        if not run_command(train_command, "Training all ablations"):
            success = False
            print(" Training phase failed!")
            if not input("Continue anyway? [y/n]: ").lower().startswith('y'):
                return False
    else:
        print("\n Skipping training phase")
    
    # Phase 2: Evaluation
    if not skip_evaluation:
        print("\n PHASE 2: EVALUATING ABLATIONS")
        print("-" * 40)
        
        eval_command = (
            f"python ablation_experiments/evaluate_ablations.py "
            f"--results_dir {training_dir} "
            f"--output_dir {evaluation_dir}"
        )
        
        if ablation_names:
            eval_command += f" --ablations {' '.join(ablation_names)}"
        
        if not run_command(eval_command, "Evaluating all ablations"):
            success = False
            print(" Evaluation phase failed!")
            if not input("Continue anyway? [y/n]: ").lower().startswith('y'):
                return False
    else:
        print("\n Skipping evaluation phase")
    
    # Phase 3: Comparison
    if not skip_comparison:
        print("\n PHASE 3: COMPARING RESULTS")
        print("-" * 40)
        
        results_file = os.path.join(evaluation_dir, "ablation_evaluation_results.json")
        
        compare_command = (
            f"python ablation_experiments/compare_ablations.py "
            f"--results_file {results_file} "
            f"--output_dir {comparison_dir}"
        )
        
        if not run_command(compare_command, "Comparing ablation results"):
            success = False
            print(" Comparison phase failed!")
    else:
        print("\n Skipping comparison phase")
    
    # Summary
    print("\n" + "="*60)
    if success:
        print(" ABLATION STUDY COMPLETED SUCCESSFULLY!")
        print(f" All results saved to: {base_output_dir}")
        print(f" Key outputs:")
        print(f" Training models: {training_dir}/")
        print(f" Evaluation results: {evaluation_dir}/")
        print(f" Comparison plots: {comparison_dir}/")
        
        # Print quick summary
        try:
            summary_file = os.path.join(comparison_dir, "ablation_detailed_comparison.csv")
            if os.path.exists(summary_file):
                import pandas as pd
                df = pd.read_csv(summary_file)
                print(f"\n Top 3 Ablations:")
                for i, (_, row) in enumerate(df.head(3).iterrows()):
                    print(f"   {i+1}. {row['Ablation']}: {row['Overall']}")
        except Exception:
            pass
            
    else:
        print("  ABLATION STUDY COMPLETED WITH ERRORS")
        print(f" Partial results saved to: {base_output_dir}")
    
    print("="*60)
    
    return success


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description="Run complete ablation study")
    
    # Experiment configuration
    parser.add_argument('--ablations', nargs='+', default=None,
                       help='Specific ablations to run (default: priority ablations)')
    parser.add_argument('--num_seeds', type=int, default=3,
                       help='Number of seeds per ablation')
    parser.add_argument('--output_dir', default='results_icml/ablation_study',
                       help='Base output directory')
    
    # Phase control
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training phase (use existing models)')
    parser.add_argument('--skip_evaluation', action='store_true', 
                       help='Skip evaluation phase (use existing results)')
    parser.add_argument('--skip_comparison', action='store_true',
                       help='Skip comparison phase')
    
    # Preset configurations
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with minimal ablations')
    parser.add_argument('--priority_only', action='store_true',
                       help='Run only priority ablations')
    parser.add_argument('--full_study', action='store_true',
                       help='Run complete ablation study (all ablations)')
    
    args = parser.parse_args()
    
    # Handle preset configurations
    if args.quick_test:
        ablation_names = ['no_graph', 'gene_similarity_only']
        num_seeds = 1
        print(" Quick test mode: 2 ablations, 1 seed each")
        
    elif args.priority_only:
        ablation_names = list(get_priority_ablations().keys())
        num_seeds = args.num_seeds
        print(f" Priority ablations mode: {len(ablation_names)} ablations")
        
    elif args.full_study:
        ablation_names = list(get_all_ablations().keys())
        num_seeds = args.num_seeds
        print(f" Full study mode: {len(ablation_names)} ablations")
        
    else:
        ablation_names = args.ablations
        num_seeds = args.num_seeds
        
        if ablation_names is None:
            # Interactive mode
            print("Available ablation presets:")
            print("  1. Quick test (2 ablations, 1 seed) - ~30 minutes")
            print("  2. Priority only (7 ablations, 3 seeds) - ~6 hours") 
            print("  3. Full study (16 ablations, 3 seeds) - ~12 hours")
            print("  4. Custom selection")
            
            choice = input("Choose option [1-4]: ").strip()
            
            if choice == '1':
                ablation_names = ['no_graph', 'gene_similarity_only']
                num_seeds = 1
            elif choice == '2':
                ablation_names = list(get_priority_ablations().keys())
            elif choice == '3':
                ablation_names = list(get_all_ablations().keys())
            else:
                print("Available ablations:")
                all_ablations = get_all_ablations()
                for name, config in all_ablations.items():
                    print(f"  • {name}: {config.description}")
                
                ablation_input = input("Enter ablation names (space-separated): ").strip()
                ablation_names = ablation_input.split() if ablation_input else None
    
    # Estimate time
    if ablation_names:
        estimated_hours = len(ablation_names) * num_seeds * 0.5  # ~30 min per experiment
        print(f" Estimated time: {estimated_hours:.1f} hours")
        
        if estimated_hours > 2:
            confirm = input(f"This will take ~{estimated_hours:.1f} hours. Continue? [y/n]: ")
            if not confirm.lower().startswith('y'):
                print("Aborted.")
                return
    
    # Run ablation study
    run_complete_ablation_study(
        ablation_names=ablation_names,
        num_seeds=num_seeds,
        base_output_dir=args.output_dir,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation,
        skip_comparison=args.skip_comparison
    )


if __name__ == "__main__":
    main()