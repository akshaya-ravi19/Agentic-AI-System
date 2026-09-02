"""
notebooks/03_labelling/03_digital_health_ground_truth.py
---------------------------------------------------------
Constructs a Clinical/Syndromic Digital Health Ground Truth.

Aligned with:
1. US CDC Foodborne Disease Active Surveillance Network (FoodNet)
2. FDA Model Food Code Priority Biological & Chemical Hazards
3. WHO Syndromic Surveillance Guidelines

Creates a multi-tier target:
  - syndromic_hazard_level:
      2 = Critical Biohazard / Acute Illness (Vermin on food contact, Acute food poisoning, Temperature abuse, Pesticides)
      1 = Moderate Hazard (Hygiene, Sanitation, Foreign object, Unprotected food)
      0 = Low / Administrative (Letter grade display, Billing, Facility maintenance, Odor)
  - binary_priority_label: 1 if syndromic_hazard_level >= 1 else 0
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.config import DATA_RAW, DATA_LABELLED, RANDOM_SEED

print("Loading raw 311 complaints and DOHMH inspections...")
df_311 = pd.read_csv(
    DATA_RAW / "nyc_311_food_complaints.csv",
    parse_dates=["created_date"],
    dtype={"unique_key": str, "incident_zip": str},
    low_memory=False
)
df_dohmh = pd.read_csv(
    DATA_RAW / "dohmh_inspections.csv",
    parse_dates=["inspection_date"],
    dtype={"camis": str, "zipcode": str},
    low_memory=False
)

# ── 1. Comprehensive Syndromic Hazard Taxonomy ───────────────────
# Level 2: Acute Pathogen / Immediate Biohazard / Clinical Symptoms
CRITICAL_HAZARDS = [
    "food poisoned", "food poisoning", "vomit", "sick", "diarrhea", "fever", "nausea",
    "rodent infestation", "mice", "rats", "roaches", "pesticide", "chemical", "sewage",
    "temperature", "undercooked", "raw chicken", "raw meat", "spoilage"
]

# Level 1: Moderate Hazard / Cross-Contamination Risk
MODERATE_HAZARDS = [
    "food spoiled", "food contaminated", "food contains foreign object",
    "bare hands in contact w/ food", "food worker hygiene", "kitchen/food prep area",
    "unsanitary condition", "insects", "flies", "filth flies", "glove"
]

# Level 0: Low Risk / Administrative / Non-Clinical
LOW_HAZARDS = [
    "letter grading", "no permit or license", "toilet facility", "odor",
    "pet/animal", "facility condition", "signage", "card missing"
]

def score_syndromic_hazard(row):
    text_corpus = f"{row.get('descriptor', '')} {row.get('descriptor_2', '')} {row.get('resolution_description', '')}".lower()
    
    # Check level 2
    for kw in CRITICAL_HAZARDS:
        if kw in text_corpus:
            return 2
    # Check level 1
    for kw in MODERATE_HAZARDS:
        if kw in text_corpus:
            return 1
    # Check level 0
    return 0

print("Evaluating syndromic public health hazard scores...")
df_311["syndromic_hazard_level"] = df_311.apply(score_syndromic_hazard, axis=1)
df_311["binary_priority_label"] = (df_311["syndromic_hazard_level"] >= 1).astype(int)

# ── 2. Match to DOHMH Establishment History ───────────────────
df_311["address_clean"] = df_311["incident_address"].fillna("").astype(str).str.strip().str.lower()
df_311["zip5"] = df_311["incident_zip"].fillna("").astype(str).str[:5]

dohmh_lookup = df_dohmh[["camis", "dba", "building", "street", "zipcode", "score", "grade", "action"]].copy()
dohmh_lookup["zip5"] = dohmh_lookup["zipcode"].fillna("").astype(str).str[:5]
dohmh_lookup["address_clean"] = (dohmh_lookup["building"].fillna("").astype(str) + " " + dohmh_lookup["street"].fillna("").astype(str)).str.strip().str.lower()

merged = pd.merge(
    df_311,
    dohmh_lookup.drop_duplicates(subset=["camis"]),
    on=["zip5", "address_clean"],
    how="left"
)

# ── 3. Save Output ─────────────────────────────────────────────
DATA_LABELLED.mkdir(parents=True, exist_ok=True)
out_file = DATA_LABELLED / "labelled_complaints_digital_health.csv"
merged.to_csv(out_file, index=False)

print(f"\n[Success] Digital Health Ground Truth Saved -> {out_file}")
print(f"Total Dataset Size: {len(merged):,} records")
print("\nSyndromic Hazard Level Distribution:")
print(merged["syndromic_hazard_level"].value_counts(normalize=True).round(3).to_dict())
print("\nBinary Priority Label Distribution (1=Actionable Hazard, 0=Administrative):")
print(merged["binary_priority_label"].value_counts(normalize=True).round(3).to_dict())
