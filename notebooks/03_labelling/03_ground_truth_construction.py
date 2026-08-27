# ============================================================
# NOTEBOOK 03 — Ground Truth Label Construction (FIXED)
#
# THREE BUGS FIXED FROM THE ORIGINAL VERSION:
#
# BUG 1 — CAMIS dtype corruption:
#   pd.read_csv() auto-detects all-digit ID columns (camis) as
#   numbers. "50002628" silently becomes 50002628.0, and later
#   str() conversions produce "50002628.0" -- which never matches
#   DOHMH's clean "50002628". FIX: dtype=str on every read of an
#   ID column, everywhere, always.
#
# BUG 2 — Matching on the wrong field:
#   The original matched against df_311["location"]. In the NYC
#   311 Socrata feed, "location" is a nested Location object (lat/
#   lon + a human_address sub-object), which pandas reads back from
#   CSV as a stringified Python dict, e.g.
#   "{'latitude': '40.71', ..., 'human_address': '{\"address\":..}'}"
#   Fuzzy-matching restaurant names against that garbled string
#   barely works. FIX: build the match key from the real address
#   fields (incident_address / street_name / cross streets +
#   incident_zip), same idea as DOHMH's building+street+zip.
#
# BUG 3 — O(n × m) matching is too slow to ever finish comfortably:
#   Calling thefuzz.process.extractOne() inside a per-row .apply()
#   compares every single complaint against every single restaurant
#   from scratch -- with ~70K complaints x ~25K restaurants that's
#   over a billion string comparisons. FIX: first narrow candidates
#   by ZIP code (a restaurant in Brooklyn can never be the right
#   match for a complaint in the Bronx), which cuts the comparison
#   set for each row from ~25,000 down to a few hundred.
#
# DESIGN CHANGE — severity definition redone (Aug 2026):
#   The original rule ("severe" = any single Critical violation
#   found within the window) produced a test set that was 88%
#   severe. That's not real signal -- DOHMH cites at least one
#   Critical violation in the large majority of ALL inspections, so
#   this rule barely separated anything. It also broke model
#   evaluation: a trivial majority-class baseline scored a HIGHER
#   PR-AUC (0.94) than every real trained model, since PR-AUC of a
#   constant classifier tracks positive-class prevalence.
#   NEW RULE: "severe" = the linked inspection resulted in a grade
#   of C, OR the establishment was closed by DOHMH (action field
#   contains "Closed"). These are DOHMH's own consequential
#   outcomes, not a threshold we invented -- easier to defend, and
#   should produce a test set with real class separation instead of
#   near-total prevalence in one class.
# ============================================================
import pandas as pd, numpy as np
from thefuzz import process, fuzz
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import *
from tqdm import tqdm

tqdm.pandas()

# dtype=str is the fix for BUG 1 -- applied at the very first read,
# before anything else has a chance to corrupt these ID-like columns.
df_311 = pd.read_csv(
    DATA_RAW / "nyc_311_food_complaints.csv",
    parse_dates=["created_date"],
    dtype={"unique_key": str, "incident_zip": str},
)
df_dohmh = pd.read_csv(
    DATA_RAW / "dohmh_inspections.csv",
    parse_dates=["inspection_date"],
    dtype={"camis": str, "zipcode": str},
)

# ── STEP 1: See exactly what columns you actually have ───────
# Column names sometimes vary slightly (capitalisation, snake_case
# vs original Socrata names) depending on how the download was run.
# Always check before assuming a name -- this cost real debugging
# time in earlier runs of this project.
print("=== NYC 311 relevant columns present ===")
addr_candidates_311 = ["incident_address", "street_name", "cross_street_1",
                        "cross_street_2", "incident_zip", "location"]
for c in addr_candidates_311:
    print(f"  {c:<20} present: {c in df_311.columns}")

print("\n=== DOHMH relevant columns present ===")
addr_candidates_dohmh = ["dba", "building", "street", "zipcode"]
for c in addr_candidates_dohmh:
    print(f"  {c:<20} present: {c in df_dohmh.columns}")

# ── STEP 2: Clean ──────────────────────────────────────────────
df_311 = df_311.dropna(subset=["descriptor", "created_date"]).copy()
df_dohmh = df_dohmh.dropna(subset=["camis", "inspection_date", "critical_flag"]).copy()
df_dohmh["camis"] = df_dohmh["camis"].astype(str).str.strip()
# grade/action are used by the new severity rule below -- fill
# missing values with empty string rather than dropping those rows,
# since most inspections legitimately have a blank grade (only
# certain inspection types get graded at all).
for col in ["grade", "action"]:
    if col not in df_dohmh.columns:
        df_dohmh[col] = ""
    df_dohmh[col] = df_dohmh[col].fillna("")
print(f"\nCleaned: 311={len(df_311):,} | DOHMH={len(df_dohmh):,}")

# ── STEP 3: Build DOHMH restaurant lookup (BUG 2 fix, DOHMH side) ─
def build_dohmh_key(row):
    parts = []
    for col in ["dba", "building", "street"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]).strip().lower())
    return " ".join(parts)

dohmh_unique = df_dohmh[
    ["camis"] + [c for c in ["dba", "building", "street", "zipcode"] if c in df_dohmh.columns]
].drop_duplicates("camis").copy()
dohmh_unique["match_key"] = dohmh_unique.apply(build_dohmh_key, axis=1)
dohmh_unique = dohmh_unique[dohmh_unique["match_key"].str.len() > 3].reset_index(drop=True)
dohmh_unique["zip5"] = dohmh_unique.get("zipcode", pd.Series(dtype=str)).astype(str).str[:5]
print(f"\nDOHMH restaurant index: {len(dohmh_unique):,} restaurants")
print(f"Sample key: {dohmh_unique['match_key'].iloc[0]!r}")

# ── STEP 4: Build 311 address key (BUG 2 fix, 311 side) ──────────
# Build from real address fields, never from the raw "location"
# object -- that field is not usable text for fuzzy matching.
def build_311_key(row):
    parts = []
    if pd.notna(row.get("incident_address")):
        parts.append(str(row["incident_address"]).strip().lower())
    elif pd.notna(row.get("street_name")):
        parts.append(str(row["street_name"]).strip().lower())
    return " ".join(parts)

df_311["match_key"] = df_311.apply(build_311_key, axis=1)
df_311["zip5"] = df_311.get("incident_zip", pd.Series(dtype=str)).astype(str).str[:5]
df_311 = df_311[df_311["match_key"].str.len() > 3].reset_index(drop=True)

print("\n=== Sample 311 address keys built ===")
for i in range(min(5, len(df_311))):
    print(f"  Row {i}: {df_311['match_key'].iloc[i]!r}  (zip {df_311['zip5'].iloc[i]})")

# ── STEP 5: ZIP-narrowed fuzzy match (BUG 3 fix) ──────────────────
# Group DOHMH restaurants by ZIP so each complaint only gets compared
# against restaurants in the same ZIP (a few hundred, not ~25,000).
dohmh_by_zip = {
    z: (g["match_key"].tolist(), g["camis"].tolist())
    for z, g in dohmh_unique.groupby("zip5")
}

def match_to_camis(row, threshold=75):
    key, zip5 = row["match_key"], row["zip5"]
    if not key:
        return None
    keys, camis_ids = dohmh_by_zip.get(zip5, (None, None))
    if not keys:
        # Fall back to the full restaurant list only if the ZIP has
        # no candidates at all (e.g. a bad/missing 311 ZIP code).
        keys, camis_ids = dohmh_unique["match_key"].tolist(), dohmh_unique["camis"].tolist()
        if not keys:
            return None
    result = process.extractOne(key, keys, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return camis_ids[keys.index(result[0])]
    return None

# Quick check on 50 rows first so you see a match-rate estimate in
# seconds, before committing to the full run.
print("\nTesting match on 50 rows...")
sample = df_311.head(50).copy()
sample["matched_camis"] = sample.apply(match_to_camis, axis=1)
rate = sample["matched_camis"].notna().mean()
print(f"Match rate on sample: {rate:.0%}")
print("(If this is well under 50%, check the address key samples above --")
print(" something about the address fields may look different than expected.)")

# ── STEP 6: Full run ────────────────────────────────────────────
print(f"\nRunning full match on {len(df_311):,} complaints "
      f"against {len(dohmh_unique):,} restaurants (ZIP-narrowed, a few minutes)...")
df_311["matched_camis"] = df_311.progress_apply(match_to_camis, axis=1)
match_rate_full = df_311["matched_camis"].notna().mean()
print(f"Full match rate: {match_rate_full:.1%} "
      f"({df_311['matched_camis'].notna().sum():,} of {len(df_311):,})")

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
df_311.to_csv(DATA_PROCESSED / "311_with_camis.csv", index=False)
print(f"Saved → {DATA_PROCESSED / '311_with_camis.csv'}")

# ── STEP 7: Core labelling function ─────────────────────────────
def is_severe_inspection(hits: pd.DataFrame) -> bool:
    """
    NEW severity rule (see DESIGN CHANGE note at top of file).
    Severe = at least one linked inspection resulted in:
      - a grade of C, OR
      - the establishment being closed by DOHMH
    Both are DOHMH's own consequential outcomes -- not a threshold
    we invented -- which makes this far more defensible than "any
    single Critical violation" (which nearly every inspection has).
    """
    graded_c = (hits["grade"].str.strip().str.upper() == "C").any()
    closed = hits["action"].str.lower().str.contains("closed", na=False).any()
    return bool(graded_c or closed)


def assign_label(complaint_date, camis, inspections_sorted_by_camis, window_days=LABEL_WINDOW_DAYS):
    """
    Returns: 1 (severe), 0 (non-severe), or None (censored/excluded).
    See is_severe_inspection() for the current severity definition.
    """
    end = complaint_date + pd.Timedelta(days=window_days)
    hits = inspections_sorted_by_camis[
        (inspections_sorted_by_camis["camis"] == str(camis)) &
        (inspections_sorted_by_camis["inspection_date"] > complaint_date) &
        (inspections_sorted_by_camis["inspection_date"] <= end)
    ]
    if hits.empty:
        return None
    if is_severe_inspection(hits):
        return SEVERE_LABEL
    return NON_SEVERE_LABEL

# ── STEP 8: ID overlap sanity check BEFORE the slow labelling pass ─
# This takes under a second and would have caught the earlier CAMIS
# bug immediately instead of after a 67-minute run.
matched_df = df_311[df_311["matched_camis"].notna()].copy()
matched_ids = set(matched_df["matched_camis"].astype(str))
dohmh_ids = set(df_dohmh["camis"].astype(str))
overlap = matched_ids & dohmh_ids
print(f"\n=== ID overlap check ===")
print(f"  Unique matched_camis values: {len(matched_ids):,}")
print(f"  Unique DOHMH camis values:   {len(dohmh_ids):,}")
print(f"  Overlapping IDs:             {len(overlap):,}")
if len(overlap) == 0:
    print("  *** 0 overlap -- stopping before the slow labelling pass. ***")
    print(f"  Sample matched_camis: {list(matched_ids)[:5]}")
    print(f"  Sample DOHMH camis:   {list(dohmh_ids)[:5]}")
    sys.exit(1)

# ── STEP 9: Window sensitivity analysis (on a sample, fast) ──────
print("\n=== Window sensitivity analysis ===")
sens_sample = matched_df.sample(min(500, len(matched_df)), random_state=RANDOM_SEED)
for window in LABEL_WINDOW_SENSITIVITY:
    count, severe, non_severe = 0, 0, 0
    for _, row in sens_sample.iterrows():
        lbl = assign_label(row["created_date"], row["matched_camis"], df_dohmh, window)
        if lbl is None:
            continue
        count += 1
        severe += (lbl == SEVERE_LABEL)
        non_severe += (lbl == NON_SEVERE_LABEL)
    if count > 0:
        print(f"  Window {window:2d} days: {count} labelled "
              f"| severe={severe} ({severe/count:.0%}) "
              f"| non-severe={non_severe} ({non_severe/count:.0%})")
    else:
        print(f"  Window {window:2d} days: 0 labels found")

# ── STEP 10: Apply labels to the full matched set ────────────────
print(f"\nApplying {LABEL_WINDOW_DAYS}-day window labels to {len(matched_df):,} complaints...")
matched_df["label"] = matched_df.progress_apply(
    lambda r: assign_label(r["created_date"], r["matched_camis"], df_dohmh), axis=1
)

total = len(matched_df)
labelled = matched_df["label"].notna().sum()
censored = matched_df["label"].isna().sum()

print(f"\n=== LABELLING RESULTS ===")
print(f"Total matched complaints:  {total:,}")
print(f"Successfully labelled:     {labelled:,} ({labelled/total:.1%})")
print(f"Censored (no inspection):  {censored:,} ({censored/total:.1%})")

if labelled > 0:
    df_labelled = matched_df[matched_df["label"].notna()].copy()
    severe = (df_labelled["label"] == SEVERE_LABEL).sum()
    non_severe = (df_labelled["label"] == NON_SEVERE_LABEL).sum()
    print(f"\nOf labelled complaints:")
    print(f"  Severe   (1): {severe:,} ({severe/labelled:.1%})")
    print(f"  Non-sev  (0): {non_severe:,} ({non_severe/labelled:.1%})")

    DATA_LABELLED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_LABELLED / "labelled_complaints.csv"
    df_labelled.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_labelled):,} labelled complaints to: {out_path}")

    # ── Manual validation subset ──────────────────────────────────
    # A held-out sample you hand-label yourself against a rubric, to
    # sanity-check the distant-supervision labels -- important for
    # your methodology chapter, since these labels are a proxy, not
    # verified ground truth.
    review_n = min(EVAL_SAMPLE_SIZE, len(df_labelled))
    review_sample = df_labelled.sample(review_n, random_state=RANDOM_SEED).copy()
    review_sample["manual_severity_label"] = ""
    review_sample["reviewer_notes"] = ""
    review_path = DATA_LABELLED / "manual_review_sample.csv"
    review_sample[["descriptor", "matched_camis", "created_date", "label",
                    "manual_severity_label", "reviewer_notes"]].to_csv(review_path, index=False)
    print(f"Saved {review_n} rows for manual review to: {review_path}")
    print("Fill in 'manual_severity_label' by hand, then compare against 'label'")
    print("(e.g. with sklearn.metrics.cohen_kappa_score) to report label reliability.")

    print(f"\nClass balance: {'manageable' if 0.2 <= severe/labelled <= 0.8 else 'IMBALANCED'} "
          f"({severe/labelled:.0%} severe)")
    print("Next: notebooks/02_preprocessing/")
else:
    print("\n*** 0 labels assigned despite non-zero ID overlap. ***")
    print("Check inspection_date range vs created_date range, and critical_flag values.")