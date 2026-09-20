"""
Ablation study: systematically disables each pipeline component
and measures impact on Cohen's Kappa vs expert labels.

Versions:
  A: No BiLSTM — all complaints assumed severe (bilstm_pred=1)
  B: No cluster context tool (cluster signal = neutral)
  C: No inspection history (days_since=0, grade='A', count=0)
  D: Full pipeline (reference)
"""
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.config import (
    EVAL_DIR, TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE, KAPPA_THRESHOLD
)
from pipeline.routing.rule_based_triage import rule_based_triage

# Load expert cases
df_cases  = pd.read_csv(EVAL_DIR / "agent" / "expert_labelled_50_cases.csv")
df_agent  = pd.read_csv(EVAL_DIR / "agent" / "agent_results.csv")

level_map = {TRIAGE_LOG: 0, TRIAGE_REVIEW: 1, TRIAGE_ESCALATE: 2}
expert_num = df_cases["expert_triage"].map(level_map)

def run_ablation(name, bilstm_fn=None, grade_fn=None, days_fn=None, count_fn=None):
    """Apply ablation overrides and compute Kappa vs expert."""
    preds = []
    for _, row in df_cases.iterrows():
        bp   = bilstm_fn(row) if bilstm_fn else int(row["bilstm_pred"])
        gr   = grade_fn(row)  if grade_fn  else str(row["last_grade"])
        days = days_fn(row)   if days_fn   else int(row["days_since_inspection"])
        cnt  = count_fn(row)  if count_fn  else int(row["complaint_count_30d"])
        preds.append(rule_based_triage(bp, gr, days, cnt))
    pred_num = pd.Series(preds).map(level_map)
    kappa = cohen_kappa_score(expert_num, pred_num)
    acc = (pred_num == expert_num).mean()
    return preds, kappa, acc

ablation_rows = []

# Version D: Full pipeline (baseline for comparison)
preds_d, kappa_d, acc_d = run_ablation("D: Full pipeline")
ablation_rows.append({
    "version": "D: Full pipeline (all components)",
    "kappa": round(kappa_d, 4),
    "accuracy": round(acc_d, 4),
    "delta_kappa": 0.0,
})

# Version A: No BiLSTM — assume all complaints are non-severe (bilstm_pred=0)
# This simulates routing without the neural classifier
preds_a, kappa_a, acc_a = run_ablation("A", bilstm_fn=lambda r: 0)
ablation_rows.append({
    "version": "A: No BiLSTM (bilstm_pred=0 for all)",
    "kappa": round(kappa_a, 4),
    "accuracy": round(acc_a, 4),
    "delta_kappa": round(kappa_a - kappa_d, 4),
})

# Version A2: No BiLSTM — assume all severe (bilstm_pred=1)
preds_a2, kappa_a2, acc_a2 = run_ablation("A2", bilstm_fn=lambda r: 1)
ablation_rows.append({
    "version": "A2: No BiLSTM (bilstm_pred=1 for all)",
    "kappa": round(kappa_a2, 4),
    "accuracy": round(acc_a2, 4),
    "delta_kappa": round(kappa_a2 - kappa_d, 4),
})

# Version B: No cluster context — neutral grade (A) when no cluster signal
# Simulates system without spatiotemporal grouping informing routing
preds_b, kappa_b, acc_b = run_ablation("B", grade_fn=lambda r: "A")
ablation_rows.append({
    "version": "B: No spatiotemporal signal (grade forced to A)",
    "kappa": round(kappa_b, 4),
    "accuracy": round(acc_b, 4),
    "delta_kappa": round(kappa_b - kappa_d, 4),
})

# Version C: No inspection history — blank inspection context
preds_c, kappa_c, acc_c = run_ablation(
    "C",
    grade_fn=lambda r: "Unknown",
    days_fn=lambda r: 0,
    count_fn=lambda r: 0
)
ablation_rows.append({
    "version": "C: No inspection history (unknown grade, 0 days, 0 complaints)",
    "kappa": round(kappa_c, 4),
    "accuracy": round(acc_c, 4),
    "delta_kappa": round(kappa_c - kappa_d, 4),
})

df_abl = pd.DataFrame(ablation_rows)

print("\n" + "=" * 65)
print("ABLATION STUDY RESULTS")
print("=" * 65)
print(df_abl.to_string(index=False))
print(f"\nReference (Full pipeline): Kappa={kappa_d:.4f}, Accuracy={acc_d:.4f}")
print("\nInterpretation:")
for _, row in df_abl.iterrows():
    delta = row["delta_kappa"]
    if delta < 0:
        print(f"  {row['version']}: Kappa drops {abs(delta):.4f} -- component ADDS value")
    elif delta > 0:
        print(f"  {row['version']}: Kappa improves {delta:.4f} -- component HURTS or is redundant")
    else:
        print(f"  {row['version']}: No change -- component has no marginal effect in this dataset")

# Save
out = EVAL_DIR / "ablation_results.csv"
df_abl.to_csv(out, index=False)
print(f"\nSaved -> {out}")

# Detailed per-level F1 for version D vs A (no BiLSTM)
print("\n--- Per-level F1: Full Pipeline vs No-BiLSTM ---")
pred_d_num = pd.Series([level_map[p] for p in preds_d])
pred_a_num = pd.Series([level_map[p] for p in preds_a])
print("\nFull Pipeline:")
print(classification_report(expert_num, pred_d_num,
      target_names=[TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE]))
print("No BiLSTM (all non-severe):")
print(classification_report(expert_num, pred_a_num,
      target_names=[TRIAGE_LOG, TRIAGE_REVIEW, TRIAGE_ESCALATE]))
