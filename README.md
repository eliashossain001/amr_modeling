# PathoGen: Graph-Based Reinforcement Learning for Bacterial Evolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


> Novel framework for modeling antimicrobial resistance evolution through graph-based reinforcement learning with comprehensive biological evaluation.

## Overview

PathoGen introduces a groundbreaking approach to modeling bacterial evolution by combining graph neural networks with reinforcement learning. The framework includes:

- **Digital Twin Environment**: Realistic simulation of bacterial adaptation under antibiotic pressure
- **BERAT Evaluation Toolkit**: First comprehensive assessment framework for biological RL systems  
- **Multi-Relational Graph Learning**: Integration of genomic, geographic, and phylogenetic relationships
- **Comprehensive Ablation Framework**: Systematic analysis of component contributions

## Key Features

### 🧬 BERAT (Bacterial Evolution RL Assessment Toolkit)
- **ETCI**: Evolutionary Trajectory Coherence Index
- **GPAC**: Genotypic-Phenotypic Alignment Coefficient  
- **AEI**: Adaptive Efficiency Index
- **Temporal Dynamics**: Time-series analysis of evolutionary patterns
- **Statistical Rigor**: Bayesian bootstrap with 95% confidence intervals

### 🕸️ Multi-Relational Graph Construction
- **Gene Similarity Networks**: AMR gene profile relationships
- **Geographic Proximity**: Spatial transmission modeling
- **Serovar Compatibility**: Phylogenetic relationship encoding
- **Plasmid Compatibility**: Horizontal gene transfer potential

### 🔬 Digital Twin Environment
- **Biologically-Grounded Rewards**: Empirical AMR data integration
- **Realistic Evolutionary Dynamics**: Mutation and transfer probabilities
- **Comprehensive State Representation**: Genomic + epidemiological features

## 📦 Installation

### Prerequisites
```bash
# Python 3.8+ required
python --version

# GPU support (optional but recommended)
nvidia-smi
```

### Setup Environment
```bash
# Clone repository
git clone https://github.com/your-username/pathogen.git
cd pathogen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Required Dependencies
```
torch>=1.12.0
torch-geometric>=2.1.0
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
networkx>=2.8.0
tqdm>=4.64.0
```

## 🎮 Quick Start

### 1. Train a Model
```bash
# Basic training with default settings
python train_pathogen.py

# Train with specific configuration
python train_pathogen.py --config configs/pathogen_config.yaml --episodes 1000
```

### 2. Evaluate with BERAT
```bash
# Comprehensive evaluation
python run_comprehensive_evaluation.py --model_path checkpoints/pathogen_model.pt

# Generate evaluation report
python eval/generate_report.py --results_path results_icml/comprehensive_evaluation/
```

### 3. Run Ablation Studies
```bash
# Quick test (2 ablations, 1 seed each)
python ablation_experiments/run_complete_ablation_study.py --quick_test

# Full ablation study (16 experiments, 3 seeds each)  
python ablation_experiments/run_complete_ablation_study.py --full_study

# Priority ablations only
python ablation_experiments/run_complete_ablation_study.py --priority_only
```

### 4. Interactive Digital Twin
```bash
# Launch digital twin interface
python digital_twin/twin_interface.py

# Example usage in Python
from digital_twin import DigitalTwin
twin = DigitalTwin(model_path="checkpoints/pathogen_model.pt")
prediction = twin.predict_evolution(isolate_data)
```

## 📁 Project Structure

```
pathogen/
├── 📊 ablation_experiments/          # Comprehensive ablation study framework
│   ├── configs/                      # Ablation configurations
│   ├── train_ablations.py           # Training pipeline
│   ├── evaluate_ablations.py        # BERAT evaluation
│   └── compare_ablations.py         # Results comparison
├── 🧬 env/                          # Digital twin environment
│   └── bacterial_evolution_env.py   # Main environment class
├── 🕸️ models/                       # Neural network architectures  
│   └── policy_network.py           # BioPolicyNet implementation
├── 📈 eval/                         # BERAT evaluation framework
│   ├── evaluator.py                # Main evaluator class
│   ├── metrics/                     # BERAT metric implementations
│   └── visualization.py            # Evaluation visualizations
├── 🔬 digital_twin/                 # Interactive digital twin interface
│   └── twin_interface.py           # User interface for predictions
├── 🛠️ utils/                        # Utility functions
│   ├── reward_functions.py         # Biological reward calculation
│   └── data_processing.py          # Data preprocessing pipeline
├── 📊 results_icml/                 # Experimental results
│   ├── comprehensive_evaluation/   # BERAT evaluation results
│   ├── ablation_study/             # Ablation experiment outputs
│   └── baseline_results/           # Baseline comparisons
├── 📓 notebooks/                    # Jupyter analysis notebooks
├── 📋 configs/                      # Configuration files
└── 🧪 scripts/                      # Utility scripts
```

## 🔬 Usage Examples

### Training a Custom Model
```python
from env.bacterial_evolution_env import BacterialEvolutionEnv
from models.policy_network import BioPolicyNet
from utils.training import train_reinforce

# Initialize environment
env = BacterialEvolutionEnv(
    data_path="data/amr_data.csv",
    sample_size=1000,
    antibiotic_pressure=0.5
)

# Create model
model = BioPolicyNet(
    input_dim=env.features.shape[1],
    hidden_dim=128,
    num_actions=3
)

# Train with REINFORCE
trained_model = train_reinforce(model, env, episodes=500)
```

### BERAT Evaluation
```python
from eval.evaluator import BacterialEvolutionEvaluator
from digital_twin import DigitalTwin

# Initialize evaluator
evaluator = BacterialEvolutionEvaluator(confidence_level=0.95)

# Load trained model
twin = DigitalTwin(model_path="trained_model.pt")

# Generate evaluation trajectories
trajectories = twin.generate_trajectories(n_trajectories=50)

# Comprehensive evaluation
results = evaluator.evaluate(trajectories)
print(f"Overall Score: {results['overall_score']:.3f}")
print(f"AEI Score: {results['aei_score']:.3f}")
```

## 📊 Results

### Comprehensive Evaluation Results
- **Overall BERAT Score**: 0.597 (Good quality)
- **Adaptive Efficiency Index**: 0.740 (Excellent) 
- **Genotypic-Phenotypic Alignment**: 0.600 (Good)
- **Statistical Confidence**: 95% intervals via Bayesian bootstrap

### Key Findings
- Graph-based policies show enhanced training efficiency
- BERAT provides independent validation beyond training rewards
- Multi-relational graphs capture essential biological relationships
- Framework enables systematic biological RL evaluation

## 🔧 Configuration

### Environment Configuration
```yaml
# configs/pathogen_config.yaml
environment:
  data_path: "data/amr_data.csv"
  amr_path: "data/amr_clean.csv" 
  sample_size: 1000
  max_steps: 20
  antibiotic_pressure: 0.5

model:
  hidden_dim: 128
  num_layers: 2
  num_actions: 3

training:
  algorithm: "reinforce"
  learning_rate: 1e-4
  episodes: 500
  gamma: 0.99
```

### BERAT Configuration
```python
# Custom BERAT evaluation
evaluator = BacterialEvolutionEvaluator(
    confidence_level=0.95,
    n_bootstrap=1000,
    metrics=['etci', 'gpac', 'aei', 'temporal'],
    min_trajectories=10
)
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black pathogen/
flake8 pathogen/
```


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- AMR surveillance data providers
- Open-source bioinformatics community
- PyTorch Geometric development team


# sync test Wed Jan 28 01:49:29 EST 2026
