# ============================================================
# NOTEBOOK 03 (REGULATORY STANDARD) — Ground Truth Label Construction
# 
# Aligned with:
# 1. US FDA Model Food Code (Priority Item Hazards)
# 2. NYC Health Code Title 24 (DOHMH Restaurant Scoring & Penalty System)
#
# Generates two explicit targets:
#   - intrinsic_hazard_label (1 = High Hazard, 0 = Low Hazard) -> Target for NLP
#   - inspection_risk_label (1 = Critical Violation/Closure, 0 = Compliant) -> Target for Agent/Evidence
# ============================================================
import pandas as pd, numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.config import DATA_RAW, DATA_LABELLED, DATA_PROCESSED

# Load raw datasets with strict string dtypes
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

# ── 1. FDA / NYC Health Code Intrinsic Hazard Classification ───
PRIORITY_HAZARDS = {
    "Rodents/Insects/Garbage": 1,
    "Food Spoiled": 1,
    "Food Contaminated": 1,
    "Food Temperature": 1,
    "Bare Hands in Contact w/ Food": 1,
    "Food Contains Foreign Object": 1,
    "Pesticide": 1,
    "Food Worker Hygiene": 1,
    "Sewage": 1,
    "Unsanitary Condition": 1
}

CORE_GENERAL = {
    "Letter Grading": 0,
    "No Permit or License": 0,
    "Toilet Facility": 0,
    "Facility Condition": 0,
    "Pet/Animal": 0,
    "Odor": 0,
    "Food Protection": 0,
    "Kitchen/Food Prep Area": 0
}

def derive_intrinsic_hazard(row):
    desc = str(row.get("descriptor", "")).strip()
    desc2 = str(row.get("descriptor_2", "")).strip()
    
    # Priority check
    for hazard, lbl in PRIORITY_HAZARDS.items():
        if hazard.lower() in desc.lower() or hazard.lower() in desc2.lower():
            return 1
    # Core check
    for general, lbl in CORE_GENERAL.items():
        if general.lower() in desc.lower() or general.lower() in desc2.lower():
            return 0
    return 0

print("Assigning FDA/NYC Health Code intrinsic hazard labels...")
df_311["intrinsic_hazard_label"] = df_311.apply(derive_intrinsic_hazard, axis=1)

# ── 2. Address Matching to Link Inspection Records ────────────
# Clean and prepare match keys
df_311["address_clean"] = df_311["incident_address"].fillna("").astype(str).str.strip().str.lower()
df_311["zip5"] = df_311["incident_zip"].fillna("").astype(str).str[:5]

dohmh_lookup = df_dohmh[["camis", "building", "street", "zipcode", "score", "grade", "action", "critical_flag"]].copy()
dohmh_lookup["zip5"] = dohmh_lookup["zipcode"].fillna("").astype(str).str[:5]
dohmh_lookup["address_clean"] = (dohmh_lookup["building"].fillna("").astype(str) + " " + dohmh_lookup["street"].fillna("").astype(str)).str.strip().str.lower()

# Fast exact ZIP + Address join
merged = pd.merge(
    df_311,
    dohmh_lookup.drop_duplicates(subset=["camis"]),
    on=["zip5", "address_clean"],
    how="left"
)

# ── 3. Regulatory Consequence Ground Truth (DOHMH Inspection Outcome) ──
def derive_inspection_risk(row):
    grade = str(row.get("grade", "")).strip().upper()
    action = str(row.get("action", "")).lower()
    score = pd.to_numeric(row.get("score"), errors="coerce")
    
    # Critical Risk: Grade C, Closure, or Score >= 28 points (NYC Health Code standard)
    if grade == "C" or "closed" in action or (pd.notna(score) and score >= 28):
        return 1
    return 0

merged["inspection_risk_label"] = merged.apply(derive_inspection_risk, axis=1)

# ── 4. Save Clean Regulatory Dataset ──────────────────────────
DATA_LABELLED.mkdir(parents=True, exist_ok=True)
out_file = DATA_LABELLED / "labelled_complaints_regulatory.csv"
merged[["unique_key", "created_date", "descriptor", "descriptor_2", "incident_address", 
        "zip5", "camis", "intrinsic_hazard_label", "inspection_risk_label"]].to_csv(out_file, index=False)

print(f"\nRegulatory Ground Truth Saved -> {out_file}")
print(f"Total Records: {len(merged):,}")
print(f"Intrinsic Hazard Distribution: {merged['intrinsic_hazard_label'].value_counts(normalize=True).round(3).to_dict()}")
print(f"Inspection Risk Distribution: {merged['inspection_risk_label'].value_counts(normalize=True).round(3).to_dict()}")
