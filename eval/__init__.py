# eval/__init__.py
"""
Bacterial Evolution RL Assessment Toolkit (BERAT)
A comprehensive mathematical evaluation framework for bacterial evolution RL models.

Novel Contributions:
- ETCI: Dynamical systems approach to trajectory analysis
- GPAC: Manifold learning for genotype-phenotype alignment
- AEI: Information-theoretic policy efficiency analysis
- Bayesian uncertainty quantification
- Multi-scale temporal dynamics analysis
"""

__version__ = "1.0.0"
__author__ = "PathoGen Research Team"

from .main_evaluator import BacterialEvolutionEvaluator

# Don't import utils with * to avoid conflicts
# Let main_evaluator handle its own imports

__all__ = [
    'BacterialEvolutionEvaluator'
]