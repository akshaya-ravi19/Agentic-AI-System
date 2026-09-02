import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import pandas as pd, numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import *

df_311   = pd.read_csv(DATA_RAW / "nyc_311_food_complaints.csv", parse_dates=["created_date"], low_memory=False)
df_dohmh = pd.read_csv(DATA_RAW / "dohmh_inspections.csv", parse_dates=["inspection_date"], low_memory=False)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
print(f"311: {len(df_311):,} rows | DOHMH: {len(df_dohmh):,} rows")

# 1. Complaint volume over time
fig, ax = plt.subplots(figsize=(12,4))
df_311.set_index("created_date").resample("ME")["unique_key"].count().plot(ax=ax)
ax.set_title("NYC 311 Food Complaints — Monthly Volume")
ax.set_xlabel("Date"); ax.set_ylabel("Complaints / month")
plt.tight_layout(); plt.savefig(DATA_PROCESSED / "fig_volume.png", dpi=150)
plt.close(fig)

# 2. Borough breakdown
print("\nComplaints by borough:")
print(df_311["borough"].value_counts())

# 3. Top complaint descriptors
print("\nTop 15 complaint descriptors:")
print(df_311["descriptor"].value_counts().head(15))

# 4. DOHMH critical vs general
print("\nDOHMH critical flag breakdown:")
print(df_dohmh["critical_flag"].value_counts())
crit_pct = (df_dohmh["critical_flag"]=="Critical").mean()*100
print(f"Critical violations: {crit_pct:.1f}% of all findings")

# 5. Complaint text length
df_311["word_count"] = df_311["descriptor"].fillna("").str.split().str.len()
print(f"\nComplaint word count:\n{df_311['word_count'].describe().round(1)}")

# 6. Expected class imbalance preview
print(f"\nExpected ~{crit_pct:.0f}% severe labels after joining datasets")
print("Class imbalance — will compare SMOTE vs class weighting vs random oversampling")

# 7. Missing values audit
print("\nMissing values in key columns:")
for col in ["descriptor","latitude","longitude","created_date","borough"]:
    if col in df_311.columns:
        print(f"  311  {col}: {df_311[col].isna().sum():,} missing")
for col in ["camis","inspection_date","critical_flag","grade"]:
    if col in df_dohmh.columns:
        print(f"  DOHMH {col}: {df_dohmh[col].isna().sum():,} missing")

print("\nEDA complete.")