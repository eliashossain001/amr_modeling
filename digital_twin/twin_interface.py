# digital_twin/twin_interface.py
"""
PathoGen Digital Twin Interface
Enables isolate-specific simulation and prediction
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class DigitalTwin:
    """
    Digital Twin for bacterial evolution prediction.
    Uses trained BioPolicyNet to simulate specific isolates.
    """
    
    def __init__(self, model, env, device='cpu'):
        """
        Args:
            model: Trained BioPolicyNet
            env: BacterialEvolutionEnv instance
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.env = env
        self.device = device
        
        # Prepare graph data
        self.x = torch.tensor(env.features.values, dtype=torch.float32).to(device)
        self.edge_index = env.get_edge_index().to(device)
        
        print("Digital Twin initialized")
        print(f" Model: BioPolicyNet")
        print(f" Isolates: {len(env.features)}")
        print(f" Device: {device}")
    
    def predict_action(self, isolate_idx: int) -> Tuple[int, np.ndarray]:
        """
        Predict optimal action for a specific isolate.
        
        Args:
            isolate_idx: Index of isolate in dataset
        Returns:
            (action, probabilities)
        """
        with torch.no_grad():
            probs_all = self.model(self.x, self.edge_index)
            probs = probs_all[isolate_idx].cpu().numpy()
            action = np.argmax(probs)
        
        return action, probs
    
    def simulate_evolution(self, isolate_idx: int, generations: int = 10,
                          antibiotic_pressure: float = None) -> Dict:
        """
        Simulate evolution of a specific isolate over time.
        
        Args:
            isolate_idx: Which isolate to simulate
            generations: Number of evolutionary steps
            antibiotic_pressure: Override environment pressure (optional)
        Returns:
            Dictionary with trajectory data
        """
        # Store original pressure
        original_pressure = self.env.antibiotic_pressure
        if antibiotic_pressure is not None:
            self.env.antibiotic_pressure = antibiotic_pressure
            self.env.reward_calculator.antibiotic_pressure = antibiotic_pressure
        
        # Initialize from isolate's state
        self.env.current_sample_idx = isolate_idx
        self.env.state = self.env.features.iloc[isolate_idx].values.astype(np.float32)
        self.env.current_genes = self.env.sample_to_genes.get(isolate_idx, []).copy()
        self.env.current_step = 0
        
        # Track trajectory
        trajectory = {
            "isolate_id": isolate_idx,
            "initial_genes": self.env.current_genes.copy(),
            "generations": [],
            "antibiotic_pressure": self.env.antibiotic_pressure
        }
        
        for gen in range(generations):
            # Predict action
            action, probs = self.predict_action(isolate_idx)
            
            # Execute action
            old_genes = self.env.current_genes.copy()
            next_state, reward, done, _, info = self.env.step(action)
            
            # Record this generation
            gen_data = {
                "generation": gen,
                "action": action,
                "action_name": ["mutate", "transfer", "stable"][action],
                "action_probs": probs.tolist(),
                "genes": self.env.current_genes.copy(),
                "n_genes": len(self.env.current_genes),
                "genes_gained": len(self.env.current_genes) - len(old_genes),
                "survival_prob": info.get('survival_prob', 0),
                "reward": reward
            }
            
            trajectory["generations"].append(gen_data)
            
            if done:
                break
        
        trajectory["final_genes"] = self.env.current_genes
        trajectory["genes_acquired"] = list(set(trajectory["final_genes"]) - set(trajectory["initial_genes"]))
        
        # Restore original pressure
        if antibiotic_pressure is not None:
            self.env.antibiotic_pressure = original_pressure
            self.env.reward_calculator.antibiotic_pressure = original_pressure
        
        return trajectory
    
    def compare_scenarios(self, isolate_idx: int, 
                         pressures: List[float], 
                         generations: int = 10) -> Dict:
        """
        Compare evolution under different antibiotic pressures.
        
        Args:
            isolate_idx: Which isolate to test
            pressures: List of pressure values to test
            generations: Simulation length
        Returns:
            Comparison results
        """
        results = {}
        
        for pressure in pressures:
            traj = self.simulate_evolution(isolate_idx, generations, pressure)
            results[f"pressure_{pressure}"] = {
                "final_survival": traj["generations"][-1]["survival_prob"],
                "final_genes": len(traj["final_genes"]),
                "genes_acquired": traj["genes_acquired"],
                "total_reward": sum(g["reward"] for g in traj["generations"])
            }
        
        return results
    
    def batch_predict(self, isolate_indices: List[int]) -> pd.DataFrame:
        """
        Predict outcomes for multiple isolates.
        
        Args:
            isolate_indices: List of isolate indices
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for idx in isolate_indices:
            traj = self.simulate_evolution(idx, generations=10)
            
            results.append({
                "isolate_idx": idx,
                "initial_genes": len(traj["initial_genes"]),
                "final_genes": len(traj["final_genes"]),
                "genes_acquired": len(traj["genes_acquired"]),
                "acquired_gene_names": ", ".join(traj["genes_acquired"][:5]),  # Top 5
                "final_survival": traj["generations"][-1]["survival_prob"],
                "total_reward": sum(g["reward"] for g in traj["generations"])
            })
        
        return pd.DataFrame(results)