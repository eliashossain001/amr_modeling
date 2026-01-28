# run_comprehensive_evaluation.py
"""
Run comprehensive evaluation of the trained bacterial evolution RL model.
Uses the BERAT (Bacterial Evolution RL Assessment Toolkit) framework.
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime

# Add paths
sys.path.append('.')  # Current directory
sys.path.append('eval')  # Evaluation framework

# Import your existing components
from env.bacterial_evolution_env import BacterialEvolutionEnv
from models.policy_network import BioPolicyNet
from digital_twin.twin_interface import DigitalTwin

# Import evaluation framework
from eval.main_evaluator import BacterialEvolutionEvaluator


def generate_evaluation_trajectories(model_path: str, 
                                   n_trajectories: int = 50,
                                   max_generations: int = 10) -> list:
    """
    Generate trajectories from trained model for evaluation.
    
    Args:
        model_path: Path to trained model checkpoint
        n_trajectories: Number of trajectories to generate
        max_generations: Number of evolution steps per trajectory
        
    Returns:
        List of trajectory dictionaries
    """
    
    print(f"\n Generating {n_trajectories} evaluation trajectories...")
    
    # 1. Initialize environment
    env = BacterialEvolutionEnv(
        data_path="data/processed/merged_dataset.csv",
        amr_path="data/processed/amr_clean.csv",
        sample_size=1000,
        max_steps=20,
        antibiotic_pressure=0.5
    )
    
    # 2. Load trained model
    print(f" Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    input_dim = env.features.shape[1]
    model = BioPolicyNet(input_dim=input_dim, hidden_dim=128, num_actions=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f" Model loaded (trained for {checkpoint.get('episode', 'unknown')} episodes)")
    
    # 3. Initialize Digital Twin
    twin = DigitalTwin(model, env)
    
    # 4. Generate trajectories
    trajectories = []
    
    # Get number of available samples from environment - FIXED
    n_available_samples = len(env.features)
    print(f" Available samples: {n_available_samples}")
    
    print(" Generating trajectories...")
    for i in range(n_trajectories):
        # Use different isolate indices for diversity - FIXED
        isolate_idx = i % min(100, n_available_samples)
        
        try:
            # Generate trajectory using Digital Twin
            trajectory = twin.simulate_evolution(
                isolate_idx=isolate_idx, 
                generations=max_generations,
                antibiotic_pressure=0.5
            )
            
            # Add trajectory ID for tracking
            trajectory['trajectory_id'] = i
            trajectory['isolate_idx'] = isolate_idx
            
            trajectories.append(trajectory)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{n_trajectories} trajectories...")
                
        except Exception as e:
            print(f" Failed to generate trajectory {i}: {str(e)}")
            continue
    
    print(f" Successfully generated {len(trajectories)} trajectories")
    return trajectories


def run_evaluation():
    """
    Main evaluation runner.
    """
    
    print("\n" + "="*80)
    print(" BACTERIAL EVOLUTION RL COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" Using BERAT (Bacterial Evolution RL Assessment Toolkit)")
    print("="*80)
    
    # Configuration
    MODEL_PATH = "results_icml/checkpoints/latest_seed0.pt"
    OUTPUT_DIR = "results_icml/comprehensive_evaluation"
    N_TRAJECTORIES = 50  # Enough for statistical significance
    MAX_GENERATIONS = 10  # Reasonable evolution length
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f" Model not found at: {MODEL_PATH}")
        print("   Make sure you have trained the model first!")
        available_checkpoints = []
        checkpoint_dir = "results_icml/checkpoints/"
        if os.path.exists(checkpoint_dir):
            available_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if available_checkpoints:
                print(f"   Available checkpoints: {available_checkpoints}")
        return
    
    try:
        # Step 1: Generate trajectories
        trajectories = generate_evaluation_trajectories(
            model_path=MODEL_PATH,
            n_trajectories=N_TRAJECTORIES,
            max_generations=MAX_GENERATIONS
        )
        
        if len(trajectories) < 10:
            print(f" Too few trajectories generated: {len(trajectories)}")
            print("   Need at least 10 trajectories for meaningful evaluation")
            print("   This might be due to Digital Twin simulation issues.")
            
            # Print some debug info
            if len(trajectories) > 0:
                print(f"   Sample trajectory keys: {list(trajectories[0].keys())}")
            return
        
        # Step 2: Initialize comprehensive evaluator
        print(f"\ Initializing BERAT evaluator...")
        evaluator = BacterialEvolutionEvaluator(
            confidence_level=0.95,
            random_seed=42,
            n_bootstrap_samples=1000
        )
        print(f" Evaluator initialized")
        
        # Step 3: Run comprehensive evaluation
        print(f"\n Running comprehensive evaluation on {len(trajectories)} trajectories...")
        print("   This may take 5-10 minutes...")
        
        results = evaluator.evaluate_model(trajectories)
        
        # Step 4: Save results and create plots
        print(f"\n Saving results to: {OUTPUT_DIR}/")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        evaluator.save_results(results, OUTPUT_DIR, create_plots=True)
        
        # Step 5: Print summary
        print_evaluation_summary(results)
        
        print(f"\n Evaluation complete! Results saved to: {OUTPUT_DIR}/")
        print(f"\n Check these files:")
        print(f"   • {OUTPUT_DIR}/detailed_evaluation.json")
        print(f"   • {OUTPUT_DIR}/evaluation_summary.csv")
        print(f"   • {OUTPUT_DIR}/plots/evaluation_dashboard.png")
        
    except Exception as e:
        print(f"\n Evaluation failed: {str(e)}")
        import traceback
        print("\n Full error traceback:")
        traceback.print_exc()
        
        print(f"\n Debugging tips:")
        print(f"   1. Check that your model checkpoint exists and loads correctly")
        print(f"   2. Verify Digital Twin is working: python scripts/test_digital_twin.py")
        print(f"   3. Try reducing N_TRAJECTORIES to 20 if memory issues")
        print(f"   4. Check that all eval/ framework files are in place")


def print_evaluation_summary(results: dict):
    """Print a nice summary of evaluation results."""
    
    print("\n" + "="*80)
    print(" EVALUATION SUMMARY")
    print("="*80)
    
    # Check if evaluation was successful
    if not results:
        print("  No results to display - evaluation may have failed")
        return
    
    # Overall assessment
    if 'overall_assessment' in results:
        overall = results['overall_assessment']
        print(f" Overall Score: {overall['overall_score']:.3f}")
        print(f" Quality Rating: {overall['quality_rating']}")
        
        print(f"\ Component Scores:")
        for metric, score in overall['component_scores'].items():
            print(f"   • {metric.upper()}: {score:.3f}")
    else:
        print(" Overall assessment not available")
    
    # Individual metrics
    print(f"\n Detailed Metrics:")
    
    if 'etci' in results and 'error' not in results['etci']:
        etci = results['etci']
        print(f"    ETCI (Trajectory Coherence): {etci['mean_etci']:.3f} ± {etci['std_etci']:.3f}")
        print(f"      Analyzed trajectories: {etci.get('n_valid_trajectories', 'unknown')}")
    else:
        error_msg = results.get('etci', {}).get('error', 'Unknown error')
        print(f"   ETCI: Failed ({error_msg})")
    
    if 'gpac' in results and 'error' not in results['gpac']:
        gpac = results['gpac']
        print(f"    GPAC (Genotype-Phenotype Alignment): {gpac['gpac_score']:.3f}")
        print(f"     Samples analyzed: {gpac.get('n_samples', 'unknown')}")
    else:
        error_msg = results.get('gpac', {}).get('error', 'Unknown error')
        print(f"    GPAC: Failed ({error_msg})")
    
    if 'aei' in results and 'error' not in results['aei']:
        aei = results['aei']
        print(f"    AEI (Adaptive Efficiency): {aei['mean_aei']:.3f} ± {aei['std_aei']:.3f}")
        print(f"    Analyzed trajectories: {aei.get('n_valid_trajectories', 'unknown')}")
    else:
        error_msg = results.get('aei', {}).get('error', 'Unknown error')
        print(f"    AEI:  Failed ({error_msg})")
    
    # Temporal dynamics
    if 'temporal' in results and 'error' not in results['temporal']:
        temporal = results['temporal']
        overall_temporal = temporal.get('overall_temporal_score', 0.0)
        print(f"    Temporal Dynamics: {overall_temporal:.3f}")
        print(f"    Analyzed trajectories: {temporal.get('n_trajectories', 'unknown')}")
    else:
        error_msg = results.get('temporal', {}).get('error', 'Unknown error')
        print(f"    Temporal Dynamics: Failed ({error_msg})")
    
    # Uncertainty
    if 'uncertainty' in results:
        print(f"\n Statistical Confidence: 95% intervals computed")
        
        # Count successful uncertainty estimates
        successful_uncertainty = sum([
            'etci' in results['uncertainty'],
            'gpac' in results['uncertainty'], 
            'aei' in results['uncertainty']
        ])
        print(f"   Uncertainty estimates: {successful_uncertainty}/3 metrics")
    else:
        print(f"\n Statistical Confidence: Not available")
    
    # Metadata
    if 'metadata' in results:
        metadata = results['metadata']
        print(f"\n Evaluation Metadata:")
        print(f"   • Trajectories processed: {metadata.get('n_trajectories', 'unknown')}")
        print(f"   • BERAT version: {metadata.get('berat_version', 'unknown')}")
        print(f"   • Timestamp: {metadata.get('timestamp', 'unknown')}")
    
    # Files generated
    print(f"\n Generated Files:")
    print(f"   • detailed_evaluation.json - Complete results")
    print(f"   • evaluation_summary.csv - Summary table")
    print(f"   • plots/ - Comprehensive visualizations")
    print(f"     - metric_overview.png")
    print(f"     - uncertainty_analysis.png") 
    print(f"     - temporal_dynamics.png")
    print(f"     - evaluation_dashboard.png")
    
    # Success rate summary
    successful_metrics = sum([
        'etci' in results and 'error' not in results['etci'],
        'gpac' in results and 'error' not in results['gpac'],
        'aei' in results and 'error' not in results['aei'],
        'temporal' in results and 'error' not in results['temporal']
    ])
    
    print(f"\n Success Rate: {successful_metrics}/4 metrics completed successfully")
    
    if successful_metrics >= 3:
        print(" Excellent! Most metrics computed successfully.")
    elif successful_metrics >= 2:
        print(" Good! Majority of metrics computed successfully.")
    elif successful_metrics >= 1:
        print(" Partial success. Some metrics may need debugging.")
    else:
        print(" Evaluation mostly failed. Check trajectory generation and framework setup.")
    
    print("="*80)


def test_setup():
    """Test if everything is set up correctly before running evaluation."""
    
    print("\n Testing setup...")
    
    # Test imports
    try:
        from eval.main_evaluator import BacterialEvolutionEvaluator
        print(" Evaluation framework imports successfully")
    except Exception as e:
        print(f" Import error: {e}")
        return False
    
    # Test model exists
    model_path = "results_icml/checkpoints/latest_seed0.pt"
    if os.path.exists(model_path):
        print(f" Model checkpoint found: {model_path}")
    else:
        print(f" Model not found: {model_path}")
        return False
    
    # Test data files
    data_files = [
        "data/processed/merged_dataset.csv",
        "data/processed/amr_clean.csv"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f" Data file found: {file_path}")
        else:
            print(f" Data file missing: {file_path}")
            return False
    
    print(" Setup test completed successfully!")
    return True


if __name__ == "__main__":
    # Optional: Test setup first
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_setup()
    else:
        run_evaluation()
