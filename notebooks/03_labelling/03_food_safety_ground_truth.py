"""
notebooks/03_labelling/03_food_safety_ground_truth.py
------------------------------------------------------
Food Safety Complaint Prioritization Ground Truth

Constructs a 3-tier municipal hazard prioritization ground truth based on NYC DOHMH
food establishment violations and complaint taxonomy (independent of CDC/FDA labels):

- Tier 2 (Critical Hazard): Acute pathogen transmission, vermin infestation, temperature abuse, sewage
- Tier 1 (Moderate Hazard): Food contamination, worker hygiene, spoiled food, unsanitary prep area
- Tier 0 (Routine / Low Hazard): Administrative, letter grading, licensing, general facility condition

Produces:
  data/labelled/labelled_complaints_ground_truth.csv
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.config import DATA_RAW, DATA_LABELLED, RANDOM_SEED

# ── 1. Municipal Hazard Categories (NYC Health Code Standard) ──
CRITICAL_DESCRIPTORS = {
    "Rodents/Insects/Garbage",
    "Food Temperature",
    "Bare Hands in Contact w/ Food",
    "Food Worker Hygiene",
    "Food Worker Activity",
    "Sewage",
    "Food Source",
}

MODERATE_DESCRIPTORS = {
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

CRITICAL_KEYWORDS = [
    "food poisoning", "food poisoned", "vomit", "vomiting", "sick",
    "diarrhea", "diarrhoea", "nausea", "fever", "cramps", "hospital",
    "hospitalized", "undercooked", "raw chicken", "raw meat",
    "temperature", "rodent", "mice", "rats", "roach", "pest",
    "chemical", "sewage", "pesticide", "spoilage", "acute illness"
]

MODERATE_KEYWORDS = [
    "food spoiled", "food contaminated", "foreign object",
    "bare hand", "glove", "unsanitary", "flies", "insects",
    "dirty kitchen", "cross contamina", "cutting board"
]

def assign_hazard_tier(row):
    desc1 = str(row.get("descriptor", "")).strip()
    desc2 = str(row.get("descriptor_2", "")).strip().lower()
    res = str(row.get("resolution_description", "")).strip().lower()
    combined = f"{desc2} {res}"

    if desc1 in CRITICAL_DESCRIPTORS:
        return 2, "Critical Hazard"
    if desc1 in MODERATE_DESCRIPTORS:
        return 1, "Moderate Hazard"
    for kw in CRITICAL_KEYWORDS:
        if kw in combined:
            return 2, "Critical Hazard"
    for kw in MODERATE_KEYWORDS:
        if kw in combined:
            return 1, "Moderate Hazard"
    return 0, "Routine / Administrative"

print("Loading NYC 311 complaints...")
df = pd.read_csv(
    DATA_RAW / "nyc_311_food_complaints.csv",
    parse_dates=["created_date"],
    dtype={"unique_key": str, "incident_zip": str},
    low_memory=False,
)
print(f"Loaded {len(df):,} complaints")
df = df.dropna(subset=["descriptor"]).copy()

print("Applying 3-Tier Municipal Food Safety Hazard Taxonomy...")
tier_results = df.apply(assign_hazard_tier, axis=1)
df["hazard_tier"] = [r[0] for r in tier_results]
df["hazard_category"] = [r[1] for r in tier_results]
df["priority_label"] = (df["hazard_tier"] >= 1).astype(int)

df["complaint_text"] = (
    df["descriptor"].fillna("") + " | " +
    df["descriptor_2"].fillna("") + " | " +
    df["resolution_description"].fillna("")
).str.strip()

print("\n" + "="*60)
print("GROUND TRUTH HAZARD TIER DISTRIBUTION (3-Class)")
print("="*60)
tier_names = {2: "Tier 2 (Critical Hazard)", 1: "Tier 1 (Moderate Hazard)", 0: "Tier 0 (Routine / Low)"}
total = len(df)
for tier, count in df["hazard_tier"].value_counts().sort_index().items():
    print(f"  {tier_names[tier]}: {count:,} ({count/total:.1%})")

print(f"\nActionable Public Health Priority Rate: {df['priority_label'].mean():.1%}")

DATA_LABELLED.mkdir(parents=True, exist_ok=True)
out_file = DATA_LABELLED / "labelled_complaints_ground_truth.csv"
df.to_csv(out_file, index=False)
print(f"\n[SUCCESS] Ground Truth Saved -> {out_file} ({len(df):,} records)")
