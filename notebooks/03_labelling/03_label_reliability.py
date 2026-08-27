"""
notebooks/03_labelling/03c_label_reliability.py
---------------------------------------------------
Run this AFTER you've filled in manual_severity_label in
data/labelled/manual_review_sample.csv by hand.

Computes Cohen's Kappa between the automated distant-supervision
label ('label', from linked inspection outcomes) and your own manual
judgement ('manual_severity_label') on the held-out review sample.
This is the label-reliability number for your methodology chapter --
it tells you how much a human agrees with the automated labelling
rule, which matters because that rule is itself only a proxy for
true severity, not verified ground truth.

INTERPRETING THE KAPPA VALUE (standard Landis & Koch bands):
  < 0.00        no agreement
  0.00 - 0.20   slight
  0.21 - 0.40   fair
  0.41 - 0.60   moderate
  0.61 - 0.80   substantial
  0.81 - 1.00   almost perfect
Report both the number and the band in your write-up -- a raw kappa
number means little to a reader without that context.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import DATA_LABELLED

path = DATA_LABELLED / "manual_review_sample.csv"
df = pd.read_csv(path)

df["manual_severity_label"] = df["manual_severity_label"].astype(str).str.strip()
missing = df["manual_severity_label"].isin(["", "nan", "NaN"])
if missing.any():
    print(f"WARNING: {missing.sum()} of {len(df)} rows still have manual_severity_label "
          f"blank -- fill these in before trusting this result. Excluding them for now.")
    df = df[~missing]

if len(df) == 0:
    print("No labelled rows found. Fill in manual_severity_label in "
          f"{path} first, then re-run this script.")
    sys.exit(1)

df["manual_severity_label"] = df["manual_severity_label"].astype(int)
df["label"] = df["label"].astype(int)

kappa = cohen_kappa_score(df["label"], df["manual_severity_label"])

if kappa < 0:
    band = "no agreement (worse than chance)"
elif kappa <= 0.20:
    band = "slight agreement"
elif kappa <= 0.40:
    band = "fair agreement"
elif kappa <= 0.60:
    band = "moderate agreement"
elif kappa <= 0.80:
    band = "substantial agreement"
else:
    band = "almost perfect agreement"

print(f"\n{'='*50}")
print(f"LABEL RELIABILITY (n={len(df)})")
print(f"{'='*50}")
print(f"Cohen's Kappa: {kappa:.3f}  ->  {band}")

cm = confusion_matrix(df["label"], df["manual_severity_label"])
print("\nConfusion matrix (rows=automated label, cols=your manual label):")
print(f"                  manual=0   manual=1")
print(f"  automated=0     {cm[0][0]:>8}   {cm[0][1]:>8}")
print(f"  automated=1     {cm[1][0]:>8}   {cm[1][1]:>8}")

disagreements = df[df["label"] != df["manual_severity_label"]]
if len(disagreements):
    print(f"\n{len(disagreements)} disagreements -- worth quoting a couple in your "
          f"limitations discussion:")
    for _, row in disagreements.head(5).iterrows():
        print(f"  - {row.get('descriptor', '(no descriptor column)')!r}: "
              f"automated={row['label']}, you said={row['manual_severity_label']}")

out_path = DATA_LABELLED / "label_reliability_report.txt"
with open(out_path, "w") as f:
    f.write(f"Cohen's Kappa: {kappa:.3f} ({band})\n")
    f.write(f"n = {len(df)}\n")
    f.write(f"Confusion matrix:\n{cm}\n")
print(f"\nSaved summary to {out_path}")
