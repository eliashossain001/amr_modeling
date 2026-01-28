# scripts/check_gene_values.py

import pandas as pd
from utils.reward_functions import BiologicalRewardCalculator

# Initialize calculator
calc = BiologicalRewardCalculator("data/processed/amr_clean.csv", antibiotic_pressure=0.7)

# Check gene resistance values
print("="*60)
print("TOP 20 RESISTANCE GENES (by value)")
print("="*60)
sorted_genes = sorted(calc.gene_to_resistance.items(), key=lambda x: x[1], reverse=True)
for gene, value in sorted_genes[:20]:
    print(f"{gene:30s} → {value:.4f}")

print("\n" + "="*60)
print("SURVIVAL PROBABILITY SIMULATION")
print("="*60)

# Simulate survival for different gene counts
for n_genes in [1, 3, 5, 7, 10, 15]:
    # Take top N genes
    top_genes = [g for g, v in sorted_genes[:n_genes]]
    survival = calc.compute_survival_probability(top_genes)
    print(f"{n_genes:2d} genes → Survival: {survival:.3f}")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
# Calculate needed genes for 50% survival
target_survival = 0.5
needed_resistance = (target_survival - 0.2 + 0.7) / 0.1  # From formula
print(f"For 50% survival at pressure=0.7:")
print(f"  Need total gene value: {needed_resistance:.2f}")
print(f"  With avg gene value 0.5: need ~{int(needed_resistance/0.5)} genes")

print("\n" + "="*60)
print("AVERAGE GENE VALUE")
print("="*60)
avg_value = sum(calc.gene_to_resistance.values()) / len(calc.gene_to_resistance)
print(f"Average gene resistance value: {avg_value:.4f}")
print(f"Total genes in database: {len(calc.gene_to_resistance)}")
