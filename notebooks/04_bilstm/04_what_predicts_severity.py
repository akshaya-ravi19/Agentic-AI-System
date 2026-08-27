"""
notebooks/04_bilstm/04b_what_predicts_severity.py
------------------------------------------------------
Investigates the text/label decoupling finding from the Streamlit
demo: the same complaint descriptor (e.g. "Rodent Infestation")
appeared under both severe and non-severe labels, suggesting text
alone has a ceiling on how well it can predict this label (since the
label is really about the LINKED INSPECTION's outcome, not the
complaint itself).

This script checks whether non-text features predict the label
better than text does:
  1. Prior violation count at that establishment (from DOHMH history
     BEFORE the complaint date)
  2. Complaint frequency at that establishment (how many other 311
     complaints hit the same place recently)
  3. Days since that establishment's last inspection

Compares each feature's standalone predictive power (via a simple
logistic regression on JUST that feature) against the existing
text-based classifier's performance from notebook 04, so you can
report which signal actually carries more information about this
label -- useful evidence either way for your discussion chapter.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import DATA_LABELLED, DATA_RAW, RANDOM_SEED

df = pd.read_csv(DATA_LABELLED / "labelled_complaints.csv", parse_dates=["created_date"])
df_dohmh = pd.read_csv(DATA_RAW / "dohmh_inspections.csv", parse_dates=["inspection_date"],
                        dtype={"camis": str})
df["matched_camis"] = df["matched_camis"].astype(str)

print(f"Loaded {len(df):,} labelled complaints")

# ── Feature 1: prior violation count at that establishment ───────
# Only counts inspections BEFORE the complaint date -- using later
# inspections here would leak the very thing we're trying to predict.
def prior_violation_count(row):
    hist = df_dohmh[
        (df_dohmh["camis"] == row["matched_camis"]) &
        (df_dohmh["inspection_date"] < row["created_date"])
    ]
    return len(hist)

print("Computing prior violation counts (this loops over all rows, may take a minute)...")
df["prior_violation_count"] = df.apply(prior_violation_count, axis=1)

# ── Feature 2: complaint frequency at that establishment ─────────
camis_freq = df["matched_camis"].value_counts()
df["complaint_frequency"] = df["matched_camis"].map(camis_freq)

# ── Feature 3: days since last inspection before the complaint ───
def days_since_last_inspection(row):
    hist = df_dohmh[
        (df_dohmh["camis"] == row["matched_camis"]) &
        (df_dohmh["inspection_date"] < row["created_date"])
    ]
    if hist.empty:
        return np.nan
    return (row["created_date"] - hist["inspection_date"].max()).days

print("Computing days-since-last-inspection...")
df["days_since_last_inspection"] = df.apply(days_since_last_inspection, axis=1)
df["days_since_last_inspection"] = df["days_since_last_inspection"].fillna(
    df["days_since_last_inspection"].median()
)

features = ["prior_violation_count", "complaint_frequency", "days_since_last_inspection"]
X = df[features].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"\n{'='*55}\nHOW WELL DO NON-TEXT FEATURES PREDICT THE LABEL?\n{'='*55}")

# Combined model with all three features
clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"\nAll 3 non-text features combined:")
print(f"  ROC-AUC: {auc:.4f}")
print(classification_report(y_test, clf.predict(X_test), target_names=["non-severe", "severe"]))

# Each feature standalone, to see which one carries the most signal
print(f"\n{'='*55}\nEACH FEATURE STANDALONE\n{'='*55}")
for i, fname in enumerate(features):
    clf_single = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(
        X_train[:, [i]], y_train
    )
    auc_single = roc_auc_score(y_test, clf_single.predict_proba(X_test[:, [i]])[:, 1])
    print(f"  {fname:<28} ROC-AUC: {auc_single:.4f}")

print(f"\n{'='*55}")
print("COMPARE against your text-based classifier's PR-AUC/ROC-AUC")
print("from notebook 04's comparison.csv. If these non-text features")
print("score comparably or higher, that's strong evidence the label")
print("depends more on establishment history than on how the")
print("complaint itself was worded -- worth reporting explicitly in")
print("your discussion chapter alongside the text/label decoupling")
print("example (same descriptor, different labels) from the demo.")
print(f"{'='*55}")

out_path = DATA_LABELLED / "non_text_feature_analysis.csv"
df[["descriptor", "matched_camis", "label"] + features].to_csv(out_path, index=False)
print(f"\nSaved feature table to {out_path}")