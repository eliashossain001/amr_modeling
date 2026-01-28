"""
=========================================================
 PathoGen Project – Data Cleaning Script (Final Version)

=========================================================
"""

import pandas as pd
from pathlib import Path

# --------------------------------------------------------
# 1. Define paths and create folder structure
# --------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print(" Directory structure verified.")

# --------------------------------------------------------
# 2. Load raw datasets
# --------------------------------------------------------
def load_file(filename):
    path = DATA_RAW / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {filename}")
    return pd.read_csv(path, sep="\t", low_memory=False)

metadata = load_file("Salmonella-sample-metadata.txt.gz")
amr = load_file("Salmonella-abricate-ncbi-amr-genes.txt.gz")
mob = load_file("Salmonella-mobtyper-results.txt.gz")
mobrecon = load_file("Salmonella-mobrecon-contig-reports.txt.gz")

print(" All datasets loaded successfully.")
print(f"metadata: {metadata.shape}, amr: {amr.shape}, mob: {mob.shape}, mobrecon: {mobrecon.shape}")

# --------------------------------------------------------
# 3. Standardize column names and remove blanks
# --------------------------------------------------------
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df.dropna(how="all")

metadata = clean_columns(metadata)
amr = clean_columns(amr)
mob = clean_columns(mob)
mobrecon = clean_columns(mobrecon)

# Rename the '#file' column in AMR dataset to match others
if "#file" in amr.columns:
    amr.rename(columns={"#file": "sample_id"}, inplace=True)

print(" Column names standardized and #FILE renamed to sample_id.")

# --------------------------------------------------------
# 4. Unify sample identifier across datasets
# --------------------------------------------------------
def unify_sample_column(df):
    if "sample_id" in df.columns:
        df.rename(columns={"sample_id": "sample"}, inplace=True)
    return df

metadata = unify_sample_column(metadata)
amr = unify_sample_column(amr)
mob = unify_sample_column(mob)
mobrecon = unify_sample_column(mobrecon)

print(" Unified identifier column as 'sample' across all datasets.")

# --------------------------------------------------------
# 5. Select relevant columns (subset of useful features)
# --------------------------------------------------------
def select_cols(df, cols):
    existing = [c for c in cols if c in df.columns]
    return df[existing]

metadata_clean = select_cols(metadata, ["sample", "host", "country", "year", "serovar"])
amr_clean = select_cols(amr, ["sample", "gene", "product", "resistance", "percent_identity", "percent_coverage"])
mob_clean = select_cols(mob, ["sample", "predicted_mobility", "rep_type(s)", "relaxase_type(s)"])
mobrecon_clean = select_cols(mobrecon, ["sample", "rep_type(s)", "size", "gc"])

# --------------------------------------------------------
# 6. Merge all datasets
# --------------------------------------------------------
merged = (
    metadata_clean
    .merge(amr_clean, on="sample", how="left")
    .merge(mob_clean, on="sample", how="left")
    .merge(mobrecon_clean, on="sample", how="left")
)

print(f" Merged dataset created: {merged.shape[0]} rows, {merged.shape[1]} columns")

# --------------------------------------------------------
# 7. Save cleaned outputs
# --------------------------------------------------------
metadata_clean.to_csv(DATA_PROCESSED / "metadata_clean.csv", index=False)
amr_clean.to_csv(DATA_PROCESSED / "amr_clean.csv", index=False)
mob_clean.to_csv(DATA_PROCESSED / "mobtyper_clean.csv", index=False)
mobrecon_clean.to_csv(DATA_PROCESSED / "mobrecon_clean.csv", index=False)
merged.to_csv(DATA_PROCESSED / "merged_dataset.csv", index=False)

print(" Cleaned datasets saved in 'data/processed/'.")

# --------------------------------------------------------
# 8. Quick integrity summary
# --------------------------------------------------------
meta_ids = set(metadata["sample"])
amr_ids = set(amr["sample"])
mob_ids = set(mob["sample"])
mobrecon_ids = set(mobrecon["sample"])

intersection = len(meta_ids & amr_ids & mob_ids & mobrecon_ids)
print(f"\n Common samples across all datasets: {intersection}")

if "gene" in amr_clean.columns:
    print("\nTop 10 most common AMR genes:")
    print(amr_clean["gene"].value_counts().head(10))

if "predicted_mobility" in mob_clean.columns:
    print("\nMobility type distribution:")
    print(mob_clean["predicted_mobility"].value_counts())

print("\n Data cleaning and integration complete — ready for Stage I simulation.")
