# Ablation Study Report

**Generated:** 2025-11-23 18:16:22

**Total Ablations:** 2

## Performance Ranking

| Rank | Ablation | Overall Score | Description |
|------|----------|---------------|-------------|
| 1 | No Graph | 0.606 | MLP baseline without any graph structure |
| 2 | Gene Similarity Only | 0.606 | GCN with gene similarity edges only |

## Key Findings

- **Best performing ablation:** No Graph (0.606)
- **Worst performing ablation:** Gene Similarity Only (0.606)

### Category Performance

- **Graph:** 0.606 (n=2)

## Methodology

- Evaluation framework: BERAT (Bacterial Evolution RL Assessment Toolkit)
- Metrics: ETCI, GPAC, AEI, Temporal Dynamics
- Statistical confidence: 95% intervals
- Multiple random seeds per ablation
