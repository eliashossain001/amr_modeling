# scripts/test_digital_twin.py
"""
Test script for Digital Twin functionality
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import json
from env.bacterial_evolution_env import BacterialEvolutionEnv
from models.policy_network import BioPolicyNet
from digital_twin.twin_interface import DigitalTwin

def main():
    print("\n" + "="*70)
    print("PathoGen Digital Twin - Test")
    print("="*70)
    
    # ============================================
    # 1. Load Environment
    # ============================================
    print("\n[1/4] Loading environment...")
    env = BacterialEvolutionEnv(
        data_path="data/processed/merged_dataset.csv",
        amr_path="data/processed/amr_clean.csv",
        sample_size=1000,
        max_steps=20,
        antibiotic_pressure=0.5
    )
    
    # ============================================
    # 2. Load Trained Model
    # ============================================
    print("\n[2/4] Loading trained model...")
    
    # Load checkpoint
    checkpoint_path = "results_icml/checkpoints/latest_seed0.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model
    input_dim = env.features.shape[1]
    num_actions = env.action_space.n
    model = BioPolicyNet(input_dim=input_dim, hidden_dim=128, num_actions=num_actions)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"  ✓ Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")
    
    # ============================================
    # 3. Initialize Digital Twin
    # ============================================
    print("\n[3/4] Initializing Digital Twin...")
    twin = DigitalTwin(model, env, device='cpu')
    
    # ============================================
    # 4. Run Tests
    # ============================================
    print("\n[4/4] Running simulations...")
    
    # Test 1: Single isolate simulation
    print("\n" + "-"*70)
    print("TEST 1: Simulate Single Isolate")
    print("-"*70)
    
    test_isolate = 0
    trajectory = twin.simulate_evolution(test_isolate, generations=10)
    
    print(f"\nIsolate {test_isolate}:")
    print(f"  Initial genes: {len(trajectory['initial_genes'])}")
    print(f"  Final genes: {len(trajectory['final_genes'])}")
    print(f"  Genes acquired: {trajectory['genes_acquired'][:5]}")  # Show first 5
    print(f"  Final survival: {trajectory['generations'][-1]['survival_prob']:.3f}")
    
    # Test 2: Compare antibiotic pressures
    print("\n" + "-"*70)
    print("TEST 2: Compare Different Antibiotic Pressures")
    print("-"*70)
    
    comparison = twin.compare_scenarios(
        isolate_idx=test_isolate,
        pressures=[0.3, 0.5, 0.7],
        generations=10
    )
    
    print(f"\nIsolate {test_isolate} under different pressures:")
    for scenario, results in comparison.items():
        pressure = scenario.split("_")[1]
        print(f"  Pressure {pressure}:")
        print(f"    Final survival: {results['final_survival']:.3f}")
        print(f"    Final genes: {results['final_genes']}")
        print(f"    Total reward: {results['total_reward']:.3f}")
    
    # Test 3: Batch predictions
    print("\n" + "-"*70)
    print("TEST 3: Batch Predictions")
    print("-"*70)
    
    test_isolates = list(range(0, 10))  # First 10 isolates
    predictions = twin.batch_predict(test_isolates)
    
    print(f"\nPredictions for {len(test_isolates)} isolates:")
    print(predictions.to_string(index=False))
    
    # Save predictions
    output_dir = "results_icml/predictions"
    os.makedirs(output_dir, exist_ok=True)
    predictions.to_csv(f"{output_dir}/batch_predictions.csv", index=False)
    print(f"\n✓ Saved predictions to: {output_dir}/batch_predictions.csv")
    
    # ============================================
    # 5. Save Example Trajectory
    # ============================================
    # Save Example Trajectory (with numpy type conversion)
    output_file = f"{output_dir}/example_trajectory.json"

    # Convert numpy types to Python native types
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python types."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.str_, np.bytes_)):
            return str(obj)
        else:
            return obj

    trajectory_json = convert_to_json_serializable(trajectory)

    with open(output_file, 'w') as f:
        json.dump(trajectory_json, f, indent=2)

if __name__ == "__main__":
    main()