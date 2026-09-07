"""
notebooks/03_labelling/03_reframed_ground_truth.py
--------------------------------------------------
Option 1: Reframed Ground Truth - Complaint Prioritisation Ranker

CONCEPT:
  Instead of asking "did the linked inspection find a Grade C or closure?"
  (which decouples the label from the complaint text), we ask:
  "Based on the complaint category and description, does this complaint
  represent a genuine public health hazard that should be prioritised?"

  Academically defensible because:
  1. Descriptor categories are NYC DOHMH's own standardised taxonomy
  2. Mapping follows FDA Model Food Code Priority Item definitions
     and CDC FoodNet syndromic surveillance guidelines
  3. Label derives FROM the complaint itself -- no text/label decoupling
  4. Reframes model as a TRIAGE RANKER, not a severity predictor

OUTPUTS:
  data/labelled/labelled_complaints_reframed.csv
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.config import DATA_RAW, DATA_LABELLED, RANDOM_SEED

# ── Tier 2: Critical Biohazard (FDA Priority Items) ──────────────
TIER_2_DESCRIPTORS = {
    "Rodents/Insects/Garbage",
    "Food Temperature",
    "Bare Hands in Contact w/ Food",
    "Food Worker Hygiene",
    "Food Worker Activity",
    "Sewage",
    "Food Source",
}

# ── Tier 1: Moderate Hazard (FDA Priority Foundation Items) ──────
TIER_1_DESCRIPTORS = {
    "Food Spoiled",
    "Food Contaminated",
    "Food Contains Foreign Object",
    "Kitchen/Food Prep Area",
    "Food Protection",
    "Food Preparation Location",
    "Dishwashing/Utensils",
    "Handwashing",
    "Cross Contamination",
}

# ── Tier 2 keyword fallback for descriptor_2 ─────────────────────
TIER_2_KEYWORDS = [
    "food poisoning","food poisoned","vomit","vomiting","sick",
    "diarrhea","diarrhoea","nausea","fever","cramps","hospital",
    "hospitalized","undercooked","raw chicken","raw meat",
    "temperature","rodent","mice","rats","roach","pest",
    "chemical","sewage","pesticide","spoilage","mold on food",
]
TIER_1_KEYWORDS = [
    "food spoiled","food contaminated","foreign object",
    "bare hand","glove","unsanitary","flies","insects",
    "dirty kitchen","cross contamina","cutting board",
]

def assign_hazard_tier(row):
    desc1 = str(row.get("descriptor", "")).strip()
    desc2 = str(row.get("descriptor_2", "")).strip().lower()
    res   = str(row.get("resolution_description", "")).strip().lower()
    combined = desc2 + " " + res

    if desc1 in TIER_2_DESCRIPTORS:
        return 2, "Critical Biohazard"
    if desc1 in TIER_1_DESCRIPTORS:
        return 1, "Moderate Hazard"
    for kw in TIER_2_KEYWORDS:
        if kw in combined:
            return 2, "Critical Biohazard (keyword)"
    for kw in TIER_1_KEYWORDS:
        if kw in combined:
            return 1, "Moderate Hazard (keyword)"
    return 0, "Administrative/Non-pathogenic"

print("Loading NYC 311 complaints...")
df = pd.read_csv(
    DATA_RAW / "nyc_311_food_complaints.csv",
    parse_dates=["created_date"],
    dtype={"unique_key": str, "incident_zip": str},
    low_memory=False,
)
print(f"Loaded {len(df):,} complaints")
df = df.dropna(subset=["descriptor"]).copy()
print(f"After dropping missing descriptors: {len(df):,}")

print("\nApplying FDA/CDC/WHO hazard taxonomy...")
tier_results = df.apply(assign_hazard_tier, axis=1)
df["hazard_tier"]    = [r[0] for r in tier_results]
df["hazard_category"]= [r[1] for r in tier_results]
df["priority_label"] = (df["hazard_tier"] >= 1).astype(int)

df["complaint_text"] = (
    df["descriptor"].fillna("") + " | " +
    df["descriptor_2"].fillna("") + " | " +
    df["resolution_description"].fillna("")
).str.strip()

total = len(df)
tier_counts = df["hazard_tier"].value_counts().sort_index()
print("\n" + "="*60)
print("HAZARD TIER DISTRIBUTION")
print("="*60)
tier_names = {2:"Critical Biohazard", 1:"Moderate Hazard", 0:"Administrative"}
for tier, count in tier_counts.items():
    print(f"  Tier {tier} ({tier_names[tier]}): {count:,} ({count/total:.1%})")

pos_rate = df["priority_label"].mean()
print(f"\nBinary label: {pos_rate:.1%} actionable hazard")
if 0.20 <= pos_rate <= 0.80:
    print("  >> WELL BALANCED - good for classifier training")
elif 0.10 <= pos_rate < 0.20:
    print("  >> MODERATELY IMBALANCED - SMOTE/class-weighting will help")
else:
    print("  >> CHECK TAXONOMY THRESHOLDS")

print("\nDescriptor -> Tier breakdown (top 25):")
breakdown = df.groupby(["descriptor","hazard_tier"]).size().reset_index(name="n").sort_values("n",ascending=False).head(25)
print(breakdown.to_string(index=False))

DATA_LABELLED.mkdir(parents=True, exist_ok=True)
out_cols = [c for c in ["unique_key","created_date","descriptor","descriptor_2",
    "resolution_description","incident_address","incident_zip","borough",
    "hazard_tier","priority_label","hazard_category","complaint_text"] if c in df.columns]
out_file = DATA_LABELLED / "labelled_complaints_reframed.csv"
df[out_cols].to_csv(out_file, index=False)
print(f"\n[SUCCESS] Saved -> {out_file}  ({len(df):,} records)")
print("Next: run 04_reframed_classifier.py")
