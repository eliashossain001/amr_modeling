# baseline_experiments/train_ppo.py
"""
Train PPO baseline using official stable-baselines3 style.
Updated to follow official documentation patterns.
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env  # ← Official approach
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym

from env.bacterial_evolution_env import BacterialEvolutionEnv


class FlattenedEnvWrapper(gym.Env):
    """
    Wrapper to make BacterialEvolutionEnv compatible with stable-baselines3.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Initialize the underlying environment
        self.env = BacterialEvolutionEnv(**kwargs)
        
        # SB3 needs these attributes
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action):
        """Execute action."""
        return self.env.step(action)
    
    def render(self):
        """Render (not implemented)."""
        pass
    
    def close(self):
        """Close environment."""
        pass


def make_pathogen_env():
    """Factory function to create PathogenEnv (required by make_vec_env)."""
    return FlattenedEnvWrapper(
        data_path="data/processed/merged_dataset.csv",
        amr_path="data/processed/amr_clean.csv",
        sample_size=1000,
        max_steps=20,
        antibiotic_pressure=0.5
    )


def train_ppo(num_timesteps=10000, save_path="results_icml/baseline_results/ppo"):
    """
    Train PPO model using official stable-baselines3 style.
    """
    
    print("\n" + "="*70)
    print("Training PPO Baseline (Official SB3 Style)")
    print("="*70)
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # ============================================
    # 1. Create Vectorized Environment
    # ============================================
    print("\n[1/4] Creating vectorized environment...")
    
    # Using official make_vec_env function
    env = make_vec_env(make_pathogen_env, n_envs=1)  # Official pattern!
    
    print(" Environment created using make_vec_env (official method)")
    
    # ============================================
    # 2. Initialize PPO 
    # ============================================
    print("\n[2/4] Initializing PPO...")
    
    model = PPO(
        "MlpPolicy",                  # Official policy name
        env,                          # Vectorized env
        learning_rate=3e-4,           # Standard hyperparameters
        n_steps=20,                   # Match episode length
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,                    # Show training progress
        tensorboard_log=f"{save_path}/tensorboard/",
        seed=42
    )
    
    print(" PPO initialized with standard hyperparameters")
    print(f" Policy: MlpPolicy")
    print(f" Learning rate: 3e-4")
    print(f" Clip range: 0.2")
    
    # ============================================
    # 3. Setup Callbacks
    # ============================================
    print("\n[3/4] Setting up callbacks...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=f"{save_path}/checkpoints/",
        name_prefix="ppo_model"
    )
    
    print(" Checkpoint callback configured")
    
    # ============================================
    # 4. Train PPO (Official Style)
    # ============================================
    print("\n[4/4] Training PPO...")
    print(f" Total timesteps: {num_timesteps}")
    print(f" Approximate episodes: {num_timesteps // 20}")
    print()
    
    # Train model (exactly like official docs)
    model.learn(
        total_timesteps=num_timesteps,
        callback=checkpoint_callback,
        progress_bar=True  # Show progress bar
    )
    
    # ============================================
    # 5. Save Final Model 
    # ============================================
    final_path = f"{save_path}/ppo_final"
    model.save(final_path)  # Official save method
    
    print("\n" + "="*70)
    print("PPO Training Complete!")
    print("="*70)
    print(f" Final model saved: {final_path}.zip")
    print(f" Checkpoints: {save_path}/checkpoints/")
    print(f" TensorBoard logs: {save_path}/tensorboard/")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {save_path}/tensorboard/")
    print("="*70 + "\n")
    
    return model


def main():
    """Train PPO for comparison with REINFORCE."""
    
    # Train for same number of timesteps as REINFORCE
    # 500 episodes × 20 steps = 10,000 timesteps
    num_timesteps = 10000
    
    model = train_ppo(num_timesteps=num_timesteps)


if __name__ == "__main__":
    main()