# scripts/diagnose_data.py
import pandas as pd

# Load your data
merged = pd.read_csv("data/processed/merged_dataset.csv")
amr = pd.read_csv("data/processed/amr_clean.csv")

print("="*60)
print("MERGED DATASET STRUCTURE")
print("="*60)
print(f"Columns: {merged.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(merged.head())
print(f"\nSample column unique values: {merged['sample'].nunique() if 'sample' in merged.columns else 'NO SAMPLE COLUMN'}")
print(f"Gene column unique values: {merged['gene'].nunique() if 'gene' in merged.columns else 'NO GENE COLUMN'}")

print("\n" + "="*60)
print("AMR CLEAN STRUCTURE")
print("="*60)
print(f"Columns: {amr.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(amr.head())
print(f"\nSample column unique values: {amr['sample'].nunique() if 'sample' in amr.columns else 'NO SAMPLE COLUMN'}")
print(f"Gene column unique values: {amr['gene'].nunique() if 'gene' in amr.columns else 'NO GENE COLUMN'}")

print("\n" + "="*60)
print("GENE-SAMPLE MAPPING")
print("="*60)
if 'gene' in merged.columns and 'sample' in merged.columns:
    sample_gene_counts = merged.groupby('sample')['gene'].apply(lambda x: x.dropna().nunique())
    print(f"Samples with 0 genes: {(sample_gene_counts == 0).sum()}")
    print(f"Samples with 1+ genes: {(sample_gene_counts > 0).sum()}")
    print(f"Average genes per sample: {sample_gene_counts.mean():.2f}")
    print(f"\nTop 5 samples by gene count:")
    print(sample_gene_counts.sort_values(ascending=False).head())
else:
    print("Cannot analyze: missing 'gene' or 'sample' column in merged dataset")
